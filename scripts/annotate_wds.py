"""Annotate podcast_crawl WebDataset shards with WhisperX transcripts.

Uses WhisperX (faster-whisper + batched inference + wav2vec2 alignment)
for fast, GPU-efficient transcription of both stereo channels.

Output: one JSONL per input shard under --output, mirroring the input layout.

Usage:
    .venv/bin/python3 scripts/annotate_wds.py \
        --input /path/to/wds_ja_filtered \
        --output /path/to/wds_ja_transcripts \
        --lang ja --whisper-model medium --batch-size 128

Distributed:
    --shard $INDEX --num-shards 8
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import queue
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

# PyTorch 2.6 defaults weights_only=True in torch.load, which breaks
# pyannote's VAD model (contains omegaconf objects). Patch globally.
_orig_torch_load = torch.load
torch.load = lambda *args, **kwargs: _orig_torch_load(
    *args, **{**kwargs, "weights_only": False}
)

import whisperx  # noqa: E402 — must come after torch.load patch

LOGGER = logging.getLogger("annotate_wds")


# ── Audio decoding ───────────────────────────────────────────────────────

def decode_stereo(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    import sphn
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as fp:
        fp.write(audio_bytes)
        fp.flush()
        wav, sr = sphn.read(fp.name)
    return wav, sr


# ── WhisperX transcription ──────────────────────────────────────────────

def transcribe_channel(
    audio: np.ndarray, sr: int,
    model, align_model, align_metadata,
    language: str, speaker_label: str,
    batch_size: int, device: str,
) -> list[list]:
    if audio.ndim > 1:
        audio = audio[0]
    audio = audio.astype(np.float32)

    result = model.transcribe(audio, batch_size=batch_size, language=language)

    if align_model is not None:
        result = whisperx.align(
            result["segments"], align_model, align_metadata,
            audio, device, return_char_alignments=False,
        )

    alignments: list[list] = []
    for seg in result.get("segments") or []:
        for word in seg.get("words") or []:
            start = word.get("start")
            end = word.get("end")
            text = word.get("word", "")
            if start is not None and end is not None and text:
                alignments.append([text, [float(start), float(end)], speaker_label])
    return alignments


# ── Shard reading (tolerant tarfile) ─────────────────────────────────────

def read_shard_samples(
    shard_path: Path, done_keys: set[str],
) -> list[tuple[str, bytes]]:
    """Read (key, audio_bytes) pairs from a .tar.gz, skipping done_keys."""
    samples: list[tuple[str, bytes]] = []
    try:
        with gzip.open(shard_path, "rb") as gz:
            with tarfile.open(fileobj=gz, mode="r|") as tf:
                current_key: str | None = None
                current_audio: bytes | None = None
                for member in tf:
                    if not member.isfile():
                        continue
                    try:
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        data = f.read()
                    except Exception:
                        break
                    name = member.name
                    dot = name.find(".")
                    if dot < 0:
                        continue
                    key = name[:dot]
                    ext = name[dot + 1:]
                    if key != current_key:
                        if current_key and current_audio and current_key not in done_keys:
                            samples.append((current_key, current_audio))
                        current_key = key
                        current_audio = None
                    if ext == "audio.mp3":
                        current_audio = data
                if current_key and current_audio and current_key not in done_keys:
                    samples.append((current_key, current_audio))
    except (EOFError, OSError):
        pass
    return samples


def load_done_keys(output_path: Path) -> set[str]:
    done: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:
                    continue
    return done


# ── Process samples with prefetch ────────────────────────────────────────

def process_samples(
    samples: list[tuple[str, bytes]],
    output_path: Path,
    model, align_model, align_metadata,
    language: str, batch_size: int, device: str,
) -> tuple[int, int]:
    """Transcribe samples with background audio decoding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _DONE = object()
    decode_q: queue.Queue = queue.Queue(maxsize=8)

    def _decode_loop():
        for key, audio_bytes in samples:
            try:
                wav, sr = decode_stereo(audio_bytes)
                ch0 = wav[0] if wav.ndim > 1 else wav
                ch1 = wav[1] if wav.ndim > 1 and wav.shape[0] > 1 else None
                decode_q.put((key, ch0, ch1, sr))
            except Exception as exc:
                decode_q.put((key, None, None, exc))
        decode_q.put(_DONE)

    decoder = threading.Thread(target=_decode_loop, daemon=True)
    decoder.start()

    processed = 0
    errors = 0
    with open(output_path, "a", encoding="utf-8") as out_fp:
        while True:
            item = decode_q.get()
            if item is _DONE:
                break
            key, ch0, ch1, sr_or_exc = item

            if ch0 is None:
                LOGGER.warning("decode failed %s: %s", key, sr_or_exc)
                errors += 1
                continue

            sr = sr_or_exc
            try:
                align_a = transcribe_channel(
                    ch0, sr, model, align_model, align_metadata,
                    language, "SPEAKER_A", batch_size, device,
                )
                align_b = transcribe_channel(
                    ch1, sr, model, align_model, align_metadata,
                    language, "SPEAKER_B", batch_size, device,
                ) if ch1 is not None else []
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

    decoder.join()
    return processed, errors


# ── CLI ──────────────────────────────────────────────────────────────────

def iter_shards(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.tar.gz"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lang", default="ja")
    parser.add_argument("--whisper-model", default="medium")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--rerun-errors", action="store_true")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--max-samples-per-shard", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    device = "cuda"
    device_index = args.local_rank
    align_device = f"cuda:{device_index}"
    torch.cuda.set_device(device_index)

    LOGGER.info("loading whisperx model %s (compute_type=%s, batch_size=%d, device_index=%d)",
                args.whisper_model, args.compute_type, args.batch_size, device_index)
    model = whisperx.load_model(
        args.whisper_model, device=device, device_index=device_index,
        compute_type=args.compute_type, language=args.lang,
    )

    LOGGER.info("loading alignment model for %s", args.lang)
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=args.lang, device=align_device,
        )
    except Exception as exc:
        LOGGER.warning("alignment model unavailable for %s: %s", args.lang, exc)
        align_model = None
        align_metadata = None

    shards = iter_shards(args.input)
    if not shards:
        LOGGER.error("no shards under %s", args.input)
        sys.exit(1)
    shards = shards[args.shard :: args.num_shards]
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    LOGGER.info("worker %d/%d processing %d shards", args.shard, args.num_shards, len(shards))

    # Prefetch next shard's tar reading while GPU processes current shard.
    total_processed = 0
    total_errors = 0
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        next_future = None
        for i, shard_path in enumerate(shards):
            rel = shard_path.relative_to(args.input)
            out_path = args.output / rel.with_suffix(".jsonl")
            done_keys = load_done_keys(out_path) if not args.rerun_errors else set()

            # Wait for prefetched samples or read now.
            if next_future is not None:
                samples = next_future.result()
            else:
                samples = read_shard_samples(shard_path, done_keys)

            # Prefetch next shard.
            if i + 1 < len(shards):
                next_rel = shards[i + 1].relative_to(args.input)
                next_out = args.output / next_rel.with_suffix(".jsonl")
                next_done = load_done_keys(next_out) if not args.rerun_errors else set()
                next_future = prefetch.submit(read_shard_samples, shards[i + 1], next_done)
            else:
                next_future = None

            if args.max_samples_per_shard is not None:
                samples = samples[: args.max_samples_per_shard]

            LOGGER.info("[%d/%d] %s (%d samples)", i + 1, len(shards), shard_path, len(samples))
            if not samples:
                continue

            try:
                n, e = process_samples(
                    samples, out_path, model, align_model, align_metadata,
                    args.lang, args.batch_size, align_device,
                )
            except RuntimeError as exc:
                if "cuda" in repr(exc).lower():
                    raise
                LOGGER.exception("shard failed: %s", exc)
                continue

            total_processed += n
            total_errors += e
            LOGGER.info("shard done: processed=%d errors=%d (total=%d)", n, e, total_processed)

    LOGGER.info("ALL DONE processed=%d errors=%d", total_processed, total_errors)


if __name__ == "__main__":
    main()
