"""Pre-compute mimi audio tokens for podcast_crawl WebDataset shards.

Reads filtered shards + transcript JSONLs, runs mimi.encode on both
channel orientations, and saves tokenized samples as JSONL. Training
then reads tokens directly — no audio decoding or mimi encoding needed.

Output: one .jsonl per input shard under --output:
    {"key": "abc_w000", "audio_tokens_ch0main": [[...], ...],
     "audio_tokens_ch1main": [[...], ...], "alignments_ch0": [...],
     "alignments_ch1": [...], "num_real_frames": 3750, "duration_sec": 300}

Usage:
    .venv/bin/python3 scripts/tokenize_wds.py \
        --input data/wds_ja_filtered \
        --transcripts data/wds_ja_transcripts \
        --output data/wds_ja_tokens \
        --duration-sec 300

PBS array: --shard $INDEX --num-shards 64
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

LOGGER = logging.getLogger("tokenize_wds")
NUM_DECODE_WORKERS = 4


# ── Audio helpers (same as wds_dataset.py) ───────────────────────────────

def _decode_stereo(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    import sphn
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as fp:
        fp.write(audio_bytes)
        fp.flush()
        wav, sr = sphn.read(fp.name)
    return wav, sr


def _prepare_stereo(wav: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    import torchaudio.functional as F_audio
    if src_sr != target_sr:
        t = torch.from_numpy(wav)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        t = F_audio.resample(t, src_sr, target_sr)
        wav = t.numpy()
    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=0)
    elif wav.shape[0] == 1:
        wav = np.concatenate([wav, wav], axis=0)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    return wav.astype(np.float32, copy=False)


def _read_shard_audio(shard_path: Path, done_keys: set[str]) -> list[tuple[str, bytes]]:
    samples: list[tuple[str, bytes]] = []
    try:
        with gzip.open(shard_path, "rb") as gz:
            with tarfile.open(fileobj=gz, mode="r|") as tf:
                cur_key: str | None = None
                cur_audio: bytes | None = None
                for m in tf:
                    if not m.isfile():
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                    except Exception:
                        break
                    name = m.name
                    dot = name.find(".")
                    if dot < 0:
                        continue
                    key = name[:dot]
                    ext = name[dot + 1:]
                    if key != cur_key:
                        if cur_key and cur_audio and cur_key not in done_keys:
                            samples.append((cur_key, cur_audio))
                        cur_key = key
                        cur_audio = None
                    if ext == "audio.mp3":
                        cur_audio = data
                if cur_key and cur_audio and cur_key not in done_keys:
                    samples.append((cur_key, cur_audio))
    except (EOFError, OSError):
        pass
    return samples


def _load_transcripts(jsonl_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not jsonl_path.exists():
        return out
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                out[rec["key"]] = rec
            except Exception:
                continue
    return out


# ── Tokenize one shard ───────────────────────────────────────────────────

@torch.inference_mode()
def tokenize_shard(
    shard_path: Path,
    transcript_path: Path,
    output_path: Path,
    mimi,
    target_sr: int,
    duration_sec: float,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_rate = mimi.frame_rate
    num_frames = int(duration_sec * frame_rate)

    transcripts = _load_transcripts(transcript_path)

    # Resume: skip already-tokenized keys
    done_keys: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done_keys.add(json.loads(line)["key"])
                except Exception:
                    continue

    raw_samples = _read_shard_audio(shard_path, done_keys)

    # ── Background decode (CPU) ──────────────────────────────────────
    _DONE = object()
    decode_q: queue.Queue = queue.Queue(maxsize=8)

    def _decode_worker():
        for key, audio_bytes in raw_samples:
            if key not in transcripts:
                decode_q.put((key, None, None, None, "no_transcript"))
                continue
            try:
                wav, sr = _decode_stereo(audio_bytes)
                stereo = _prepare_stereo(wav, sr, target_sr)
                decode_q.put((key, stereo, transcripts[key], None, None))
            except Exception as exc:
                decode_q.put((key, None, None, None, str(exc)))
        decode_q.put(_DONE)

    decode_thread = threading.Thread(target=_decode_worker, daemon=True)
    decode_thread.start()

    # ── Background writer (I/O) ──────────────────────────────────────
    write_q: queue.Queue = queue.Queue(maxsize=32)
    _WRITE_DONE = object()

    def _write_worker():
        with open(output_path, "a", encoding="utf-8") as fp:
            while True:
                item = write_q.get()
                if item is _WRITE_DONE:
                    return
                fp.write(item + "\n")
                fp.flush()

    write_thread = threading.Thread(target=_write_worker, daemon=True)
    write_thread.start()

    # ── Main loop: mimi encode (GPU) + enqueue writes ────────────────
    processed = 0
    errors = 0

    while True:
        item = decode_q.get()
        if item is _DONE:
            break
        key, stereo, rec, _, err = item
        if stereo is None:
            if err and err != "no_transcript":
                LOGGER.warning("decode failed %s: %s", key, err)
                errors += 1
            continue

        alignments_ch0 = rec.get("alignments_ch0") or []
        alignments_ch1 = rec.get("alignments_ch1") or []
        total_sec = stereo.shape[-1] / target_sr

        start_sec = 0.0
        win_idx = 0
        while start_sec < total_sec:
            end_sec = min(start_sec + duration_sec, total_sec)
            s0 = int(start_sec * target_sr)
            s1 = int(end_sec * target_sr)
            if s1 <= s0:
                break

            window = stereo[:, s0:s1]
            swapped = window[[1, 0], :]

            # mimi encode both orientations
            w_t = torch.from_numpy(window).cuda()
            s_t = torch.from_numpy(swapped).cuda()
            tok_ch0 = mimi.encode(w_t[:, None])   # (batch, n_cb, T)
            tok_ch1 = mimi.encode(s_t[:, None])

            real_frames = tok_ch0.shape[-1]

            # No padding — store actual length. Packing happens at training time.
            tok_ch0 = tok_ch0.squeeze(0).cpu().tolist()
            tok_ch1 = tok_ch1.squeeze(0).cpu().tolist()

            # Shift alignments to window-relative
            def shift(aligns, offset, dur):
                return [
                    [a[0], [a[1][0] - offset, a[1][1] - offset], a[2]]
                    for a in aligns
                    if a[1][1] > offset and a[1][0] < offset + dur
                ]

            record = {
                "key": f"{key}_w{win_idx:03d}",
                "audio_tokens_ch0main": tok_ch0,
                "audio_tokens_ch1main": tok_ch1,
                "alignments_ch0": shift(alignments_ch0, start_sec, duration_sec),
                "alignments_ch1": shift(alignments_ch1, start_sec, duration_sec),
                "num_real_frames": real_frames,
                "duration_sec": end_sec - start_sec,
            }
            write_q.put(json.dumps(record, ensure_ascii=False))

            start_sec += duration_sec
            win_idx += 1

        processed += 1

    write_q.put(_WRITE_DONE)
    write_thread.join()
    decode_thread.join()
    return processed, errors


# ── CLI ──────────────────────────────────────────────────────────────────

def iter_shards(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.tar.gz"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--moshi-repo", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--duration-sec", type=float, default=300)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    LOGGER.info("Loading mimi from %s", args.moshi_repo)
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders
    mimi_path = hf_hub_download(repo_id=args.moshi_repo, filename="tokenizer-e351c8d8-checkpoint125.safetensors")
    mimi = loaders.get_mimi(mimi_path, device="cuda")
    mimi.eval()
    target_sr = mimi.sample_rate
    LOGGER.info("mimi: sr=%d, frame_rate=%.1f", target_sr, mimi.frame_rate)

    shards = iter_shards(args.input)
    if not shards:
        LOGGER.error("no shards under %s", args.input)
        sys.exit(1)
    shards = shards[args.shard :: args.num_shards]
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    LOGGER.info("worker %d/%d processing %d shards", args.shard, args.num_shards, len(shards))

    # Prefetch next shard's tar reading while GPU tokenizes current
    total_p, total_e = 0, 0
    for i, shard_path in enumerate(shards):
        rel = shard_path.relative_to(args.input)
        transcript_path = args.transcripts / rel.with_suffix(".jsonl")
        output_path = args.output / rel.with_suffix(".jsonl")

        if not transcript_path.exists():
            continue

        LOGGER.info("[%d/%d] %s", i + 1, len(shards), shard_path)
        try:
            n, e = tokenize_shard(
                shard_path, transcript_path, output_path,
                mimi, target_sr, args.duration_sec,
            )
        except RuntimeError as exc:
            if "cuda" in repr(exc).lower():
                raise
            LOGGER.exception("shard failed: %s", exc)
            continue
        total_p += n
        total_e += e
        LOGGER.info("shard done: processed=%d errors=%d (total=%d)", n, e, total_p)

    LOGGER.info("ALL DONE processed=%d errors=%d", total_p, total_e)


if __name__ == "__main__":
    main()
