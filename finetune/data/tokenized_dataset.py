"""Training data loader for pre-tokenized JSONL files.

Reads JSONL files produced by `scripts/tokenize_wds.py`. Each line
contains pre-computed mimi audio tokens (both channel orientations) and
word-level alignments. No audio decoding or mimi.encode needed during
training — the GPU focuses on Moshi forward/backward only.

Reading is parallelized: a thread pool pre-loads upcoming JSONL files
while the main thread builds Samples from the current file.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .interleaver import Alignment, InterleavedTokenizer, Sample

logger = logging.getLogger("tokenized_dataset")

MAIN_SPEAKER_A = "SPEAKER_A"
MAIN_SPEAKER_B = "SPEAKER_B"
PREFETCH_LINES = 16  # lines to pre-parse ahead of GPU consumption
SHUFFLE_BUFFER = 1000  # samples to buffer before yielding in random order


@dataclass
class TokenizedDataSource:
    """Points to a directory of pre-tokenized JSONL files."""
    root: Path

    def jsonl_files(self) -> list[Path]:
        files = sorted(self.root.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files under {self.root}")
        return files


def _to_alignment(raw: list, speaker_label: str) -> list[Alignment]:
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


def _build_sample_from_tokens(
    audio_tokens: list[list[int]],
    alignments: list[Alignment],
    main_speaker: str,
    tokenizer: InterleavedTokenizer,
    num_real_frames: int,
) -> Sample:
    """Build a training Sample from pre-computed mimi tokens + alignments."""
    with torch.no_grad():
        # Reconstruct audio token tensor: (1, n_codebooks, T)
        tokens_tensor = torch.tensor(audio_tokens, dtype=torch.long, device="cuda")
        tokens_tensor = tokens_tensor.unsqueeze(0)  # (1, n_cb, T)
        tokens_tensor = tokens_tensor[..., : tokenizer.num_audio_frames]

        this_num_frames = min(num_real_frames, tokenizer.num_audio_frames)

        # Pad if needed
        if tokens_tensor.shape[-1] < tokenizer.num_audio_frames:
            tokens_tensor = torch.nn.functional.pad(
                tokens_tensor,
                (0, tokenizer.num_audio_frames - tokens_tensor.shape[-1]),
                value=tokenizer.interleaver.zero_padding,
            )
        tokens_tensor = tokens_tensor.view(1, -1, tokenizer.num_audio_frames)

        # Build text token stream from alignments
        text_tokens = tokenizer.interleaver.prepare_item(
            alignments, this_num_frames, main_speaker=main_speaker,
        )
        text_tokens = torch.nn.functional.pad(
            text_tokens,
            (0, tokenizer.num_audio_frames - text_tokens.shape[-1]),
            value=tokenizer.interleaver.zero_padding,
        )

        codes = torch.cat([text_tokens, tokens_tensor], dim=1)
        return Sample(codes, None)


def _stream_jsonl_files(
    files: list[Path],
    prefetch: int = PREFETCH_LINES,
) -> Iterator[dict]:
    """Stream parsed dicts from multiple JSONL files with background prefetch.

    A background thread reads and parses JSON lines into a bounded queue
    so I/O + JSON parsing overlaps with GPU compute in the main thread.
    """
    _DONE = object()
    q: queue.Queue = queue.Queue(maxsize=prefetch)

    def _reader():
        for path in files:
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            q.put(json.loads(line))
                        except Exception:
                            continue
            except Exception as exc:
                logger.warning("failed to read %s: %s", path, exc)
        q.put(_DONE)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    while True:
        item = q.get()
        if item is _DONE:
            break
        yield item

    reader.join()


def iter_tokenized_dataset(
    source: TokenizedDataSource,
    tokenizer: InterleavedTokenizer,
    rank: int,
    world_size: int,
    is_finite: bool,
    shuffle_at_epoch: bool,
    seed: int | None,
) -> Iterator[Sample]:
    """Yield Samples from pre-tokenized JSONL files.

    Each item yields TWO samples (ch0-main and ch1-main). JSONL lines
    are streamed one at a time via a background reader thread — no need
    to load entire files into memory. First sample is available after
    parsing just one line (~200KB), not the whole 1.4GB file.
    """
    all_files = source.jsonl_files()
    epoch = 1
    rng: np.random.RandomState | None = None
    if shuffle_at_epoch:
        rng = np.random.RandomState(seed if seed is not None else 0)

    import random as _random
    sample_rng = _random.Random(seed if seed is not None else 0)

    @dataclass
    class _PackItem:
        """Lightweight pre-sample for packing."""
        audio_tokens: list[list[int]]  # (n_codebooks, real_frames)
        alignments: list[Alignment]
        main_speaker: str
        num_real_frames: int

    def _pack_and_yield(items: list[_PackItem]) -> Sample:
        """Pack multiple short samples into one fixed-length Sample by
        concatenating audio tokens along the time axis."""
        target_frames = tokenizer.num_audio_frames
        n_codebooks = len(items[0].audio_tokens)

        # Concatenate audio tokens from all items
        packed_tokens = [[] for _ in range(n_codebooks)]
        packed_aligns: list[Alignment] = []
        offset_frames = 0

        for item in items:
            real = item.num_real_frames
            for cb in range(n_codebooks):
                packed_tokens[cb].extend(item.audio_tokens[cb][:real])

            # Shift alignments by current offset
            offset_sec = offset_frames / tokenizer.interleaver.audio_frame_rate
            for text, (s, e), spk in item.alignments:
                packed_aligns.append((text, (s + offset_sec, e + offset_sec), spk))

            offset_frames += real

        # Pad to target length
        total_real = min(offset_frames, target_frames)
        for cb in range(n_codebooks):
            packed_tokens[cb] = packed_tokens[cb][:target_frames]
            if len(packed_tokens[cb]) < target_frames:
                packed_tokens[cb].extend(
                    [0] * (target_frames - len(packed_tokens[cb]))
                )

        return _build_sample_from_tokens(
            packed_tokens, packed_aligns, items[0].main_speaker,
            tokenizer, total_real,
        )

    def _shuffled_packed_samples(files: list[Path]) -> Iterator[Sample]:
        """Yield packed + shuffled Samples.

        Short samples are bin-packed into 300s windows to minimize padding
        waste. A shuffle buffer ensures consecutive orientations of the same
        audio don't appear together.
        """
        target_frames = tokenizer.num_audio_frames
        pack_buf: list[_PackItem] = []
        pack_frames = 0
        yield_buf: list[Sample] = []

        def _flush_pack():
            nonlocal pack_buf, pack_frames
            if pack_buf:
                yield_buf.append(_pack_and_yield(pack_buf))
                pack_buf = []
                pack_frames = 0

        for rec in _stream_jsonl_files(files):
            tokens_ch0 = rec.get("audio_tokens_ch0main")
            tokens_ch1 = rec.get("audio_tokens_ch1main")
            aligns_ch0 = _to_alignment(
                rec.get("alignments_ch0") or [], MAIN_SPEAKER_A,
            )
            aligns_ch1 = _to_alignment(
                rec.get("alignments_ch1") or [], MAIN_SPEAKER_B,
            )
            num_real = rec.get("num_real_frames", target_frames)

            if not tokens_ch0 or not aligns_ch0:
                continue

            combined = list(aligns_ch0) + list(aligns_ch1)

            # Create pack items for both orientations
            for tokens, main_spk in [
                (tokens_ch0, MAIN_SPEAKER_A),
                (tokens_ch1, MAIN_SPEAKER_B),
            ]:
                if not tokens:
                    continue
                item = _PackItem(
                    audio_tokens=tokens,
                    alignments=combined,
                    main_speaker=main_spk,
                    num_real_frames=num_real,
                )

                # If this item alone fills the window, yield directly
                if num_real >= target_frames:
                    _flush_pack()
                    yield_buf.append(_build_sample_from_tokens(
                        tokens, combined, main_spk, tokenizer, num_real,
                    ))
                # If adding to pack would overflow, flush current pack first
                elif pack_frames + num_real > target_frames:
                    _flush_pack()
                    pack_buf.append(item)
                    pack_frames = num_real
                else:
                    pack_buf.append(item)
                    pack_frames += num_real

            # Shuffle buffer: when enough samples accumulated, shuffle and drain
            while len(yield_buf) >= SHUFFLE_BUFFER:
                sample_rng.shuffle(yield_buf)
                half = len(yield_buf) // 2
                for s in yield_buf[:half]:
                    yield s
                yield_buf = yield_buf[half:]

        # Flush remaining pack and yield buffer
        _flush_pack()
        sample_rng.shuffle(yield_buf)
        for s in yield_buf:
            yield s

    while True:
        file_order = list(range(len(all_files)))
        if shuffle_at_epoch and rng is not None:
            rng.shuffle(file_order)

        my_files = [all_files[i] for i in file_order[rank::world_size]]

        yield from _shuffled_packed_samples(my_files)

        if is_finite:
            return
        logger.info("rank %d finished tokenized epoch %d", rank, epoch)
        epoch += 1
