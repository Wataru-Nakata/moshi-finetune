"""Annotate podcast_crawl WebDataset shards with Whisper transcripts.

Reads webdataset shards (e.g. data/wds_ja_filtered/**/*.tar.gz from
podcast_crawl), decodes each sample's stereo MP3 with sphn, runs Whisper
independently on channel 0 and channel 1, and writes one JSONL file per
input shard containing:

    {"key": "<__key__>", "duration_sec": <float>, "sample_rate": <int>,
     "alignments_ch0": [[word, [start, end], "SPEAKER_A"], ...],
     "alignments_ch1": [[word, [start, end], "SPEAKER_B"], ...]}

The output JSONL files mirror the input shard layout under --output so the
dataset loader can pair them back up by relative path:

    <input>/1613530/7/000003.tar.gz
    <output>/1613530/7/000003.jsonl

Usage:

    uv run python3 scripts/annotate_wds.py \\
        --input /work/.../data/wds_ja_filtered \\
        --output /work/.../data/wds_ja_transcripts \\
        --lang ja --whisper-model medium

Distributed (split across N workers, all reading the same input):

    --shard $PBS_ARRAY_INDEX --num-shards 64
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import sphn
import torch
import torchaudio.functional as F_audio
import webdataset as wds
import whisper_timestamped as whisper

# `whisper_timestamped.transcribe` is shadowed by the top-level function of
# the same name, so import the submodule explicitly to get at
# `get_vad_segments`. Matches annotate.py's pattern.
transcribe = importlib.import_module("whisper_timestamped.transcribe")

LOGGER = logging.getLogger("annotate_wds")

WHISPER_SAMPLE_RATE = 16_000
OLD_GET_VAD_SEGMENTS = transcribe.get_vad_segments


def build_vad_patch(keep_silence_seconds: float):
    def new_get_vad_segments(*args, **kwargs):
        segs = OLD_GET_VAD_SEGMENTS(*args, **kwargs)
        outs = []
        last_end = 0
        d = int(WHISPER_SAMPLE_RATE * keep_silence_seconds)
        for seg in segs:
            outs.append(
                {"start": max(last_end, seg["start"] - d), "end": seg["end"] + d}
            )
            last_end = outs[-1]["end"]
        return outs

    return new_get_vad_segments


def transcribe_channel(
    wav_chan: np.ndarray, sr: int, w_model, language: str, speaker_label: str,
) -> list[list]:
    """Run whisper on a single audio channel. Returns alignments list."""
    tensor = torch.from_numpy(wav_chan).cuda()[None]
    if sr != WHISPER_SAMPLE_RATE:
        tensor = F_audio.resample(tensor, sr, WHISPER_SAMPLE_RATE)
    vocals = tensor.cpu().numpy()[0]

    this_duration = vocals.shape[-1] / WHISPER_SAMPLE_RATE
    pipe_output = whisper.transcribe(
        w_model,
        vocals,
        language=language,
        vad="auditok" if this_duration > 10 else None,
        best_of=5,
        beam_size=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        verbose=None,
    )

    alignments: list[list] = []
    for segment in pipe_output["segments"]:
        for word in segment.get("words") or []:
            try:
                alignments.append([
                    word["text"],
                    [float(word["start"]), float(word["end"])],
                    speaker_label,
                ])
            except KeyError:
                continue
    return alignments


def decode_stereo(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode MP3 bytes with sphn.

    sphn only accepts file paths, so we stream the bytes through a
    NamedTemporaryFile. Returns (wav[channels, T], sample_rate).
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as fp:
        fp.write(audio_bytes)
        fp.flush()
        wav, sr = sphn.read(fp.name)
    return wav, sr


def iter_shards(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.tar.gz"))


def process_shard(
    shard_path: Path,
    output_path: Path,
    w_model,
    language: str,
    rerun_errors: bool,
    max_samples: int | None = None,
) -> tuple[int, int]:
    """Transcribe one shard. Returns (processed, errors).

    If `max_samples` is set, stop after transcribing that many samples in
    this shard (useful for smoke tests).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: if output exists and is valid, count lines and skip.
    done_keys: set[str] = set()
    if output_path.exists() and not rerun_errors:
        with open(output_path) as f:
            for line in f:
                try:
                    done_keys.add(json.loads(line)["key"])
                except Exception:
                    continue
        LOGGER.info("resume %s: %d already transcribed", output_path, len(done_keys))

    processed = 0
    errors = 0
    # Append to the JSONL so restarts are cheap.
    with open(output_path, "a", encoding="utf-8") as out_fp:
        ds = wds.WebDataset(
            str(shard_path),
            shardshuffle=False,
            handler=wds.handlers.warn_and_continue,
        )
        for sample in ds:
            key = sample.get("__key__", "")
            if key in done_keys:
                continue
            audio_bytes = sample.get("audio.mp3")
            if not audio_bytes:
                continue

            try:
                wav, sr = decode_stereo(audio_bytes)
            except Exception as exc:
                LOGGER.warning("decode failed %s: %s", key, exc)
                errors += 1
                continue

            if wav.ndim == 1 or wav.shape[0] == 1:
                # Mono — fall back to single-channel transcription.
                ch0 = wav if wav.ndim == 1 else wav[0]
                ch1 = None
            else:
                ch0 = wav[0]
                ch1 = wav[1]

            try:
                align_a = transcribe_channel(ch0, sr, w_model, language, "SPEAKER_A")
                align_b = transcribe_channel(ch1, sr, w_model, language, "SPEAKER_B") \
                    if ch1 is not None else []
            except RuntimeError as exc:
                if "cuda" in repr(exc).lower():
                    raise
                LOGGER.warning("transcribe failed %s: %s", key, exc)
                errors += 1
                continue

            record = {
                "key": key,
                "duration_sec": float(ch0.shape[-1] / sr),
                "sample_rate": int(sr),
                "alignments_ch0": align_a,
                "alignments_ch1": align_b,
            }
            out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fp.flush()
            processed += 1

            if max_samples is not None and processed >= max_samples:
                break

    return processed, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Root directory containing *.tar.gz webdataset shards.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Root directory for transcript .jsonl files (mirrors input layout).")
    parser.add_argument("--lang", default="ja", help="Whisper language code (default: ja).")
    parser.add_argument("--whisper-model", default="medium",
                        help="Whisper model name (default: medium — recommended for stereo).")
    parser.add_argument("--keep-silence-seconds", type=float, default=0.5,
                        help="Silence padding around VAD segments (default: 0.5).")
    parser.add_argument("--shard", type=int, default=0,
                        help="Rank of this worker (for distributed processing).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of workers (for distributed processing).")
    parser.add_argument("--rerun-errors", action="store_true",
                        help="Re-process shards that already have a transcript file.")
    parser.add_argument("--local-rank", type=int, default=0,
                        help="Local GPU index to bind to (default: 0).")
    parser.add_argument("--max-samples-per-shard", type=int, default=None,
                        help="Stop after transcribing N samples per shard (smoke test).")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Stop after this many shards (smoke test).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    torch.cuda.set_device(args.local_rank)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.local_rank))

    # Patch whisper-timestamped VAD to keep surrounding silence (matches annotate.py).
    transcribe.get_vad_segments = build_vad_patch(args.keep_silence_seconds)  # type: ignore

    LOGGER.info("loading whisper model %s", args.whisper_model)
    w_model = whisper.load_model(args.whisper_model, device=f"cuda:{args.local_rank}")

    shards = iter_shards(args.input)
    if not shards:
        LOGGER.error("no shards found under %s", args.input)
        sys.exit(1)
    shards = shards[args.shard :: args.num_shards]
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    LOGGER.info(
        "worker %d/%d processing %d shards",
        args.shard, args.num_shards, len(shards),
    )

    total_processed = 0
    total_errors = 0
    for i, shard_path in enumerate(shards):
        rel = shard_path.relative_to(args.input)
        out_path = args.output / rel.with_suffix(".jsonl")
        LOGGER.info("[%d/%d] %s -> %s", i + 1, len(shards), shard_path, out_path)
        try:
            n, e = process_shard(
                shard_path, out_path, w_model, args.lang, args.rerun_errors,
                max_samples=args.max_samples_per_shard,
            )
        except RuntimeError as exc:
            if "cuda" in repr(exc).lower():
                raise
            LOGGER.exception("shard %s failed: %s", shard_path, exc)
            continue
        total_processed += n
        total_errors += e
        LOGGER.info(
            "shard done: processed=%d errors=%d (total processed=%d)",
            n, e, total_processed,
        )

    LOGGER.info("ALL DONE processed=%d errors=%d", total_processed, total_errors)


if __name__ == "__main__":
    main()
