"""WebDataset-backed dataset loader for moshi-finetune.

Reads podcast_crawl filtered shards produced by
`podcast_crawl/scripts/filter_shards.py`, pairs each shard with a transcript
JSONL produced by `moshi-finetune/scripts/annotate_wds.py`, and yields
`Sample` objects in the same format as the existing sphn-based loader.

For every audio sample we emit TWO training samples, one per channel-speaker
assignment:

    1. ch0 stays as is -> main_speaker = "SPEAKER_A" (uses ch0 alignments)
    2. channels swapped -> main_speaker = "SPEAKER_B" (uses ch1 alignments)

This doubles the effective dataset and lets Moshi see each dialogue from
both "sides".

Layout expected:

    audio_root/<job_id>/<node_index>/NNNNNN.tar.gz   (sample audio.mp3 + meta.json)
    transcript_root/<job_id>/<node_index>/NNNNNN.jsonl  (one JSON per sample key)

Each transcript JSONL entry:

    {"key": "<__key__>",
     "duration_sec": <float>,
     "alignments_ch0": [[word, [start, end], "SPEAKER_A"], ...],
     "alignments_ch1": [[word, [start, end], "SPEAKER_B"], ...]}
"""

from __future__ import annotations

import gzip
import json
import logging
import queue
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import sphn
import torch
import torchaudio.functional as F_audio

from .interleaver import Alignment, InterleavedTokenizer, Sample

logger = logging.getLogger("wds_dataset")


MAIN_SPEAKER_A = "SPEAKER_A"
MAIN_SPEAKER_B = "SPEAKER_B"


@dataclass
class WDSDataSource:
    """A webdataset root: pairs audio shards with transcript JSONLs."""

    audio_root: Path
    transcript_root: Path

    def pairs(self) -> list[tuple[Path, Path]]:
        """List (shard, transcript) pairs where both files exist."""
        assert self.audio_root.exists(), f"missing {self.audio_root}"
        assert self.transcript_root.exists(), f"missing {self.transcript_root}"
        pairs: list[tuple[Path, Path]] = []
        for shard in sorted(self.audio_root.rglob("*.tar.gz")):
            rel = shard.relative_to(self.audio_root).with_suffix(".jsonl")
            transcript = self.transcript_root / rel
            if transcript.exists():
                pairs.append((shard, transcript))
            else:
                logger.warning("no transcript for %s (expected %s)", shard, transcript)
        if not pairs:
            raise FileNotFoundError(
                f"No shard/transcript pairs found under {self.audio_root}"
            )
        return pairs


def _load_transcripts(jsonl_path: Path) -> dict[str, dict]:
    """Load one transcript jsonl into a {key: record} dict."""
    out: dict[str, dict] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["key"]] = rec
            except Exception as exc:
                logger.warning("bad transcript line in %s: %s", jsonl_path, exc)
    return out


def _decode_stereo(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    # sphn only accepts file paths, so stream bytes through a temp file.
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as fp:
        fp.write(audio_bytes)
        fp.flush()
        wav, sr = sphn.read(fp.name)
    return wav, sr


def _resample(wav: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return wav
    tensor = torch.from_numpy(wav)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    tensor = F_audio.resample(tensor, src_sr, target_sr)
    return tensor.numpy()


def _prepare_stereo(wav: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Normalize to a (2, T) float32 array at target_sr."""
    wav = _resample(wav, src_sr, target_sr)
    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=0)
    elif wav.shape[0] == 1:
        wav = np.concatenate([wav, wav], axis=0)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    return wav.astype(np.float32, copy=False)


def _read_shard_audio(shard_path: Path) -> list[tuple[str, bytes]]:
    """Read (key, audio_bytes) from a .tar.gz using tolerant tarfile reader."""
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
                        if current_key and current_audio:
                            samples.append((current_key, current_audio))
                        current_key = key
                        current_audio = None
                    if ext == "audio.mp3":
                        current_audio = data
                if current_key and current_audio:
                    samples.append((current_key, current_audio))
    except (EOFError, OSError):
        pass
    return samples


@dataclass
class _DecodedItem:
    """CPU-decoded item ready for GPU encoding."""
    key: str
    stereo: np.ndarray  # (2, T) at target_sr
    alignments: list[Alignment]
    duration_sec: float
    target_sr: int


def _decode_shard_items(
    shard_path: Path,
    transcript_path: Path,
    target_sr: int,
) -> list[_DecodedItem]:
    """CPU-only: read shard, decode MP3, resample. No GPU ops."""
    transcripts = _load_transcripts(transcript_path)
    items: list[_DecodedItem] = []

    for key, audio_bytes in _read_shard_audio(shard_path):
        if key not in transcripts:
            continue
        rec = transcripts[key]
        align_a = _to_alignments(rec.get("alignments_ch0") or [], MAIN_SPEAKER_A)
        align_b = _to_alignments(rec.get("alignments_ch1") or [], MAIN_SPEAKER_B)
        if not align_a and not align_b:
            continue
        try:
            wav, src_sr = _decode_stereo(audio_bytes)
        except Exception as exc:
            logger.warning("decode failed %s: %s", key, exc)
            continue
        stereo = _prepare_stereo(wav, src_sr, target_sr)
        items.append(_DecodedItem(
            key=key,
            stereo=stereo,
            alignments=list(align_a) + list(align_b),
            duration_sec=float(stereo.shape[-1] / target_sr),
            target_sr=target_sr,
        ))
    return items


def _iter_decoded_items(
    items: list[_DecodedItem],
    tokenizer: InterleavedTokenizer,
) -> Iterator[Sample]:
    """GPU: mimi.encode + build_sample for pre-decoded items."""
    duration_sec = tokenizer.duration_sec

    for item in items:
        start_sec = 0.0
        while start_sec < item.duration_sec:
            end_sec = min(start_sec + duration_sec, item.duration_sec)
            start_samp = int(start_sec * item.target_sr)
            end_samp = int(end_sec * item.target_sr)
            if end_samp <= start_samp:
                break
            window = item.stereo[:, start_samp:end_samp]

            yield _build_sample(
                tokenizer, window, item.alignments,
                start_sec, MAIN_SPEAKER_A, item.key,
            )
            swapped = window[[1, 0], :]
            yield _build_sample(
                tokenizer, swapped, item.alignments,
                start_sec, MAIN_SPEAKER_B, item.key,
            )
            start_sec += duration_sec


def _to_alignments(raw: list, speaker_label: str) -> list[Alignment]:
    """Force the speaker label (in case the transcript used a different one)."""
    out: list[Alignment] = []
    for item in raw:
        if not item or len(item) < 2:
            continue
        text = item[0]
        ts = item[1]
        if not ts or len(ts) != 2:
            continue
        out.append((str(text), (float(ts[0]), float(ts[1])), speaker_label))
    return out


def _build_sample(
    tokenizer: InterleavedTokenizer,
    window: np.ndarray,
    alignments: list[Alignment],
    start_sec: float,
    main_speaker: str,
    key: str,
) -> Sample:
    """Run the InterleavedTokenizer on one window + alignments pair.

    Mirrors InterleavedTokenizer.__call__ but bypasses its sidecar-JSON
    loading and uses our in-memory alignments with an explicit main_speaker.
    """
    with torch.no_grad():
        audio_tensor = torch.Tensor(window).cuda()
        audio_tokens = tokenizer.mimi.encode(audio_tensor[:, None])
        audio_tokens = audio_tokens[..., : tokenizer.num_audio_frames]
        this_num_audio_frames = audio_tokens.shape[-1]
        audio_tokens = torch.nn.functional.pad(
            audio_tokens[..., : tokenizer.num_audio_frames],
            (0, tokenizer.num_audio_frames - this_num_audio_frames),
            value=tokenizer.interleaver.zero_padding,
        )
        audio_tokens = audio_tokens.view(1, -1, tokenizer.num_audio_frames)

        shifted = [
            (a[0], (a[1][0] - start_sec, a[1][1] - start_sec), a[2])
            for a in alignments
            if a[1][1] > start_sec and a[1][0] < start_sec + tokenizer.duration_sec
        ]

        text_tokens = tokenizer.interleaver.prepare_item(
            shifted, this_num_audio_frames, main_speaker=main_speaker,
        )
        text_tokens = torch.nn.functional.pad(
            text_tokens,
            (0, tokenizer.num_audio_frames - text_tokens.shape[-1]),
            value=tokenizer.interleaver.zero_padding,
        )

        codes = torch.cat([text_tokens, audio_tokens], dim=1)
        return Sample(codes, None)


NUM_DECODE_WORKERS = 4  # CPU threads for tar reading + MP3 decoding
PREFETCH_SHARDS = 2     # How many shards to pre-decode ahead


def iter_wds_dataset(
    source: WDSDataSource,
    tokenizer: InterleavedTokenizer,
    rank: int,
    world_size: int,
    is_finite: bool,
    shuffle_at_epoch: bool,
    seed: int | None,
) -> Iterator[Sample]:
    """Main entry point: yields Samples for a WDSDataSource.

    Shards are round-robined across data-parallel ranks so each rank owns a
    disjoint subset. CPU-heavy decoding is done in background threads so
    GPU doesn't wait for I/O.
    """
    pairs = source.pairs()
    target_sr = tokenizer.mimi.sample_rate
    epoch = 1
    rng: np.random.RandomState | None = None
    if shuffle_at_epoch:
        rng = np.random.RandomState(seed if seed is not None else 0)

    while True:
        shard_order = list(range(len(pairs)))
        if shuffle_at_epoch and rng is not None:
            rng.shuffle(shard_order)

        my_shards = shard_order[rank::world_size]

        # Pre-decode shards in background threads. The main thread only
        # does mimi.encode + _build_sample (GPU ops).
        with ThreadPoolExecutor(max_workers=NUM_DECODE_WORKERS) as pool:
            # Submit first batch of prefetch jobs.
            futures = {}
            for i, idx in enumerate(my_shards[:PREFETCH_SHARDS]):
                sp, tp = pairs[idx]
                futures[i] = pool.submit(_decode_shard_items, sp, tp, target_sr)

            for i, idx in enumerate(my_shards):
                # Wait for the current shard's decoded items.
                if i in futures:
                    items = futures.pop(i).result()
                else:
                    # Shouldn't happen, but fallback to sync decode.
                    sp, tp = pairs[idx]
                    items = _decode_shard_items(sp, tp, target_sr)

                # Submit prefetch for a future shard.
                next_i = i + PREFETCH_SHARDS
                if next_i < len(my_shards):
                    next_idx = my_shards[next_i]
                    sp, tp = pairs[next_idx]
                    futures[next_i] = pool.submit(
                        _decode_shard_items, sp, tp, target_sr,
                    )

                # GPU: mimi encode + build samples (main thread).
                for sample in _iter_decoded_items(items, tokenizer):
                    yield sample

        if is_finite:
            return
        logger.info("rank %d finished WDS epoch %d", rank, epoch)
        epoch += 1
