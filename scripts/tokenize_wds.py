"""Pre-compute mimi audio tokens for podcast_crawl WebDataset shards.

Single-process, pipelined:
  [Reader threads] → decode_queue → [GPU: batch mimi.encode] → write_queue → [Writer thread]

- Reader threads: read tar.gz + decode MP3 (CPU, parallel)
- Main thread: batch mimi.encode with streaming for long audio (GPU)
- Writer thread: serialize JSON + write to Lustre (async)

No per-sample padding — actual frame lengths are stored. Packing
happens at training time in tokenized_dataset.py.

Usage:
    .venv/bin/python3 scripts/tokenize_wds.py \
        --input data/wds_ja_filtered \
        --transcripts data/wds_ja_transcripts \
        --output data/wds_ja_tokens

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

ENCODE_BATCH = 8          # samples per mimi.encode batch
STREAM_CHUNK_SEC = 30.0   # streaming encode chunk size for long audio
READER_WORKERS = 4        # CPU decode threads
DECODE_QUEUE_SIZE = 32    # pre-decoded samples buffer
WRITE_QUEUE_SIZE = 64     # pending JSONL lines buffer


# ── Audio helpers ────────────────────────────────────────────────────────

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


# ── Streaming mimi encode ────────────────────────────────────────────────

@torch.inference_mode()
def _encode_streaming(mimi, stereo: np.ndarray, target_sr: int) -> torch.Tensor:
    """Encode stereo audio with streaming to limit peak GPU memory.

    Splits audio into STREAM_CHUNK_SEC chunks and encodes sequentially
    using mimi's streaming mode, then concatenates the token chunks.
    Returns (n_channels_codebooks, total_frames) on CPU.
    """
    chunk_samples = int(STREAM_CHUNK_SEC * target_sr)
    total_samples = stereo.shape[-1]

    if total_samples <= chunk_samples * 2:
        # Short enough — encode in one shot
        t = torch.from_numpy(stereo).cuda()
        tokens = mimi.encode(t[:, None])  # (2, n_cb, T_frames)
        return tokens.cpu()

    # Streaming encode for long audio
    token_chunks = []
    with mimi.streaming(batch_size=2):  # batch=2 for stereo channels
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = stereo[:, start:end]
            t = torch.from_numpy(chunk).cuda()
            tokens = mimi.encode(t[:, None])
            if tokens.shape[-1] > 0:
                token_chunks.append(tokens.cpu())

    if not token_chunks:
        return torch.zeros(2, 8, 0, dtype=torch.long)
    return torch.cat(token_chunks, dim=-1)


@torch.inference_mode()
def _encode_batch(mimi, items: list[tuple[np.ndarray, np.ndarray]], target_sr: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Batch-encode multiple (stereo, swapped) pairs.

    Short samples are batched together for a single mimi.encode call.
    Long samples fall back to streaming encode.
    Returns list of (tokens_ch0main, tokens_ch1main) on CPU.
    """
    chunk_samples = int(STREAM_CHUNK_SEC * target_sr * 2)
    results: list[tuple[torch.Tensor, torch.Tensor]] = []

    # Separate into short (batchable) and long (streaming)
    short_items: list[tuple[int, np.ndarray, np.ndarray]] = []
    for idx, (stereo, swapped) in enumerate(items):
        if stereo.shape[-1] > chunk_samples:
            # Encode long audio with streaming
            tok_ch0 = _encode_streaming(mimi, stereo, target_sr)
            tok_ch1 = _encode_streaming(mimi, swapped, target_sr)
            results.append((tok_ch0, tok_ch1))
        else:
            short_items.append((idx, stereo, swapped))
            results.append((None, None))  # placeholder

    if short_items:
        # Batch encode short items: pad to max length, stack
        max_len = max(s.shape[-1] for _, s, _ in short_items)

        # Batch ch0-main
        batch_ch0 = []
        batch_ch1 = []
        for _, stereo, swapped in short_items:
            padded = np.zeros((2, max_len), dtype=np.float32)
            padded[:, :stereo.shape[-1]] = stereo
            batch_ch0.append(padded)

            padded_s = np.zeros((2, max_len), dtype=np.float32)
            padded_s[:, :swapped.shape[-1]] = swapped
            batch_ch1.append(padded_s)

        # Stack: (N*2, 1, max_len) for mimi
        all_audio = np.concatenate(batch_ch0 + batch_ch1, axis=0)  # (N*2*2, max_len)
        # Reshape for mimi: treat as batch of mono? No — mimi expects (batch, channels, T)
        # Each item is (2, T) stereo. Stack N items: (N, 2, T)
        ch0_batch = np.stack(batch_ch0, axis=0)  # (N, 2, max_len)
        ch1_batch = np.stack(batch_ch1, axis=0)  # (N, 2, max_len)

        # Encode: mimi expects (batch*2, 1, T) — flatten channels into batch
        ch0_t = torch.from_numpy(ch0_batch.reshape(-1, 1, max_len)).cuda()
        ch1_t = torch.from_numpy(ch1_batch.reshape(-1, 1, max_len)).cuda()

        tok_ch0_all = mimi.encode(ch0_t)  # (N*2, n_cb, T_frames)
        tok_ch1_all = mimi.encode(ch1_t)

        # Split back into per-item, reshaping (2, n_cb, T)
        n = len(short_items)
        n_cb = tok_ch0_all.shape[1]
        t_frames = tok_ch0_all.shape[2]
        tok_ch0_all = tok_ch0_all.view(n, 2, n_cb, t_frames)
        tok_ch1_all = tok_ch1_all.view(n, 2, n_cb, t_frames)

        for i, (idx, stereo, _) in enumerate(short_items):
            real_frames = int(stereo.shape[-1] / max_len * t_frames) if max_len > 0 else 0
            # Reshape (2, n_cb, T) → keep only real frames
            t0 = tok_ch0_all[i, :, :, :].cpu()
            t1 = tok_ch1_all[i, :, :, :].cpu()
            results[idx] = (t0, t1)

    return results


# ── Per-shard tokenize ───────────────────────────────────────────────────

@torch.inference_mode()
def tokenize_shard(
    shard_path: Path,
    transcript_path: Path,
    output_path: Path,
    mimi,
    target_sr: int,
    duration_sec: float,
    decode_queue: queue.Queue,
    write_queue: queue.Queue,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_rate = mimi.frame_rate

    transcripts = _load_transcripts(transcript_path)

    # Resume
    done_keys: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done_keys.add(json.loads(line)["key"])
                except Exception:
                    continue

    raw_samples = _read_shard_audio(shard_path, done_keys)

    # ── Background decode ────────────────────────────────────────────
    _DONE = object()

    def _decode_worker():
        for key, audio_bytes in raw_samples:
            if key not in transcripts:
                continue
            try:
                wav, sr = _decode_stereo(audio_bytes)
                stereo = _prepare_stereo(wav, sr, target_sr)
                decode_queue.put((key, stereo, transcripts[key]))
            except Exception as exc:
                LOGGER.warning("decode failed %s: %s", key, exc)
        decode_queue.put(_DONE)

    decoder = threading.Thread(target=_decode_worker, daemon=True)
    decoder.start()

    # ── Main loop: batch mimi encode ─────────────────────────────────
    processed = 0
    errors = 0
    batch_items: list[tuple[str, np.ndarray, np.ndarray, dict]] = []

    def _flush_batch():
        nonlocal processed, errors
        if not batch_items:
            return

        audio_pairs = [(stereo, stereo[[1, 0], :]) for _, stereo, _, _ in batch_items]
        try:
            encoded = _encode_batch(mimi, audio_pairs, target_sr)
        except RuntimeError as exc:
            LOGGER.warning("encode batch failed: %s", exc)
            errors += len(batch_items)
            batch_items.clear()
            return

        for (key, stereo, _, rec), (tok_ch0, tok_ch1) in zip(batch_items, encoded):
            if tok_ch0 is None:
                errors += 1
                continue

            alignments_ch0 = rec.get("alignments_ch0") or []
            alignments_ch1 = rec.get("alignments_ch1") or []
            total_sec = stereo.shape[-1] / target_sr

            # Slide windows
            start_sec = 0.0
            win_idx = 0
            while start_sec < total_sec:
                end_sec = min(start_sec + duration_sec, total_sec)
                s_frame = int(start_sec * frame_rate)
                e_frame = int(end_sec * frame_rate)
                if e_frame <= s_frame:
                    break

                t0 = tok_ch0[:, :, s_frame:e_frame].tolist()
                t1 = tok_ch1[:, :, s_frame:e_frame].tolist()
                real_frames = e_frame - s_frame

                def shift(aligns, offset, dur):
                    return [
                        [a[0], [a[1][0] - offset, a[1][1] - offset], a[2]]
                        for a in aligns
                        if a[1][1] > offset and a[1][0] < offset + dur
                    ]

                record = json.dumps({
                    "key": f"{key}_w{win_idx:03d}",
                    "audio_tokens_ch0main": t0,
                    "audio_tokens_ch1main": t1,
                    "alignments_ch0": shift(alignments_ch0, start_sec, duration_sec),
                    "alignments_ch1": shift(alignments_ch1, start_sec, duration_sec),
                    "num_real_frames": real_frames,
                    "duration_sec": end_sec - start_sec,
                }, ensure_ascii=False)
                write_queue.put((output_path, record))

                start_sec += duration_sec
                win_idx += 1

            processed += 1

        batch_items.clear()

    while True:
        item = decode_queue.get()
        if item is _DONE:
            break
        key, stereo, rec = item
        batch_items.append((key, stereo, stereo[[1, 0], :], rec))
        if len(batch_items) >= ENCODE_BATCH:
            _flush_batch()

    _flush_batch()
    decoder.join()
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
    mimi_path = hf_hub_download(
        repo_id=args.moshi_repo,
        filename="tokenizer-e351c8d8-checkpoint125.safetensors",
    )
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

    # Shared queues
    decode_q: queue.Queue = queue.Queue(maxsize=DECODE_QUEUE_SIZE)

    # Background writer: receives (output_path, json_line) tuples
    write_q: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
    _WRITE_DONE = object()
    open_files: dict[Path, object] = {}

    def _writer_loop():
        while True:
            item = write_q.get()
            if item is _WRITE_DONE:
                break
            path, line = item
            if path not in open_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                open_files[path] = open(path, "a", encoding="utf-8")
            open_files[path].write(line + "\n")
            open_files[path].flush()
        for fp in open_files.values():
            fp.close()

    writer = threading.Thread(target=_writer_loop, daemon=True)
    writer.start()

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
                decode_q, write_q,
            )
        except RuntimeError as exc:
            if "cuda" in repr(exc).lower():
                raise
            LOGGER.exception("shard failed: %s", exc)
            continue
        total_p += n
        total_e += e
        LOGGER.info("shard done: processed=%d errors=%d (total=%d)", n, e, total_p)

    write_q.put(_WRITE_DONE)
    writer.join()
    LOGGER.info("ALL DONE processed=%d errors=%d", total_p, total_e)


if __name__ == "__main__":
    main()
