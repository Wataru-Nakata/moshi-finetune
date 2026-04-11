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

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import sphn
import torch
import torchaudio.functional as F_audio
import webdataset as wds

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


def _iter_shard(
    shard_path: Path,
    transcript_path: Path,
    tokenizer: InterleavedTokenizer,
) -> Iterator[Sample]:
    """Yield Sample objects for every audio entry in `shard_path`.

    Each entry yields TWO samples: one with ch0 as main, one with swapped
    channels and ch1 as main. The duration slicing logic mirrors the sphn
    loader's fixed-window behavior.
    """
    transcripts = _load_transcripts(transcript_path)
    duration_sec = tokenizer.duration_sec
    target_sr = tokenizer.mimi.sample_rate

    ds = wds.WebDataset(
        str(shard_path),
        shardshuffle=False,
        handler=wds.handlers.warn_and_continue,
    )
    for sample in ds:
        key = sample.get("__key__", "")
        audio_bytes = sample.get("audio.mp3")
        if not audio_bytes or key not in transcripts:
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
        total_sec = stereo.shape[-1] / target_sr

        combined_alignments = list(align_a) + list(align_b)

        # Slide a fixed-length window across the dialogue, emitting both
        # channel orderings for every window.
        start_sec = 0.0
        while start_sec < total_sec:
            end_sec = min(start_sec + duration_sec, total_sec)
            start_samp = int(start_sec * target_sr)
            end_samp = int(end_sec * target_sr)
            if end_samp <= start_samp:
                break
            window = stereo[:, start_samp:end_samp]

            # Orientation 1: ch0 is main (SPEAKER_A)
            yield _build_sample(
                tokenizer, window, combined_alignments,
                start_sec, MAIN_SPEAKER_A, key,
            )

            # Orientation 2: swap channels, ch0 becomes the original ch1
            # which we labeled SPEAKER_B — use it as main.
            swapped = window[[1, 0], :]
            yield _build_sample(
                tokenizer, swapped, combined_alignments,
                start_sec, MAIN_SPEAKER_B, key,
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
    disjoint subset.
    """
    pairs = source.pairs()
    epoch = 1
    rng: np.random.RandomState | None = None
    if shuffle_at_epoch:
        rng = np.random.RandomState(seed if seed is not None else 0)

    while True:
        shard_order = list(range(len(pairs)))
        if shuffle_at_epoch and rng is not None:
            rng.shuffle(shard_order)

        for idx in shard_order[rank::world_size]:
            shard_path, transcript_path = pairs[idx]
            for sample in _iter_shard(shard_path, transcript_path, tokenizer):
                yield sample

        if is_finite:
            return
        logger.info("rank %d finished WDS epoch %d", rank, epoch)
        epoch += 1
