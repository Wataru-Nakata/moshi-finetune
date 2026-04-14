"""Make two Moshi models talk to each other.

Each model is a full-duplex speaker. Model A's audio output is fed to
Model B as the "other speaker" input, and vice versa. Both models run
in streaming mode, processing one audio frame (~80ms) at a time.

The conversation proceeds in lockstep:
  1. Model A generates one frame of audio + text
  2. Model A's output audio → Model B's input (other speaker channel)
  3. Model B generates one frame of audio + text
  4. Model B's output audio → Model A's input (other speaker channel)
  5. Repeat for --duration seconds

Output: stereo WAV with model A on ch0, model B on ch1 + text transcript.

Usage:
    .venv/bin/python3 scripts/moshi_dialogue.py \
        --hf-repo kyutai/moshiko-pytorch-bf16 \
        --duration 30 \
        --output dialogue.wav

With LoRA checkpoint:
    --lora-path runs/moshi_ja_wds/checkpoints/checkpoint_00100/consolidated/lora.safetensors
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import sentencepiece
import torch
import sphn

from moshi.models import loaders, LMGen

LOGGER = logging.getLogger("moshi_dialogue")


def seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


@torch.inference_mode()
def run_dialogue(
    mimi_a,
    mimi_b,
    lm_a: LMGen,
    lm_b: LMGen,
    text_tokenizer: sentencepiece.SentencePieceProcessor,
    duration_sec: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run a dialogue between two Moshi models.

    Each model gets its own mimi instance so streaming state stays clean.
    Returns (audio_a, audio_b, transcript).
    """
    sr = mimi_a.sample_rate
    frame_size = int(sr / mimi_a.frame_rate)
    n_frames = int(duration_sec * mimi_a.frame_rate)
    n_codebooks = mimi_a.num_codebooks

    # Each mimi has independent streaming state
    mimi_a.streaming_forever(batch_size=1)
    mimi_b.streaming_forever(batch_size=1)
    lm_a.streaming_forever(batch_size=1)
    lm_b.streaming_forever(batch_size=1)

    # Start with silence — each is (1, 1, frame_size) mono
    silence = torch.zeros(1, 1, frame_size, device=device)
    prev_audio_a = silence  # A's last generated audio
    prev_audio_b = silence  # B's last generated audio

    out_audio_a: list[torch.Tensor] = []
    out_audio_b: list[torch.Tensor] = []
    transcript: list[dict] = []

    LOGGER.info("Starting dialogue for %.1f seconds (%d frames)", duration_sec, n_frames)
    start_time = time.time()

    for frame_idx in range(n_frames + 2):  # +2 for initial context
        # ── Model A step: encode B's audio → feed to A's LM → decode A's output ──
        codes_b_for_a = mimi_a.encode(prev_audio_b)  # (1, n_codebooks, 1)

        tokens_a = lm_a.step(codes_b_for_a)
        if tokens_a is not None:
            text_token_a = tokens_a[0, 0, 0].item()
            audio_codes_a = tokens_a[:, 1:]
            pcm_a = mimi_a.decode(audio_codes_a)  # (1, 1, frame_size)
            prev_audio_a = pcm_a
            out_audio_a.append(pcm_a.cpu())

            if text_token_a not in [0, 3]:
                text = text_tokenizer.id_to_piece(text_token_a).replace("▁", " ")
                transcript.append({"speaker": "A", "text": text, "frame": frame_idx})

        # ── Model B step: encode A's audio → feed to B's LM → decode B's output ──
        codes_a_for_b = mimi_b.encode(prev_audio_a)  # (1, n_codebooks, 1)

        tokens_b = lm_b.step(codes_a_for_b)
        if tokens_b is not None:
            text_token_b = tokens_b[0, 0, 0].item()
            audio_codes_b = tokens_b[:, 1:]
            pcm_b = mimi_b.decode(audio_codes_b)
            prev_audio_b = pcm_b
            out_audio_b.append(pcm_b.cpu())

            if text_token_b not in [0, 3]:
                text = text_tokenizer.id_to_piece(text_token_b).replace("▁", " ")
                transcript.append({"speaker": "B", "text": text, "frame": frame_idx})

        if frame_idx % 100 == 0 and frame_idx > 0:
            elapsed = time.time() - start_time
            LOGGER.info("frame %d/%d (%.1f%%), %.1fms/frame",
                        frame_idx, n_frames, frame_idx / n_frames * 100,
                        elapsed / frame_idx * 1000)

    dt = time.time() - start_time
    LOGGER.info("Generated %d frames in %.1fs (%.1f× realtime)",
                n_frames, dt, duration_sec / dt)

    # Concatenate audio
    if out_audio_a:
        audio_a = torch.cat(out_audio_a, dim=-1)[0, 0].numpy()
    else:
        audio_a = np.zeros(0, dtype=np.float32)
    if out_audio_b:
        audio_b = torch.cat(out_audio_b, dim=-1)[0, 0].numpy()
    else:
        audio_b = np.zeros(0, dtype=np.float32)

    return audio_a, audio_b, transcript


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--lora-path", type=Path, default=None,
                        help="Path to LoRA safetensors checkpoint (optional).")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Dialogue duration in seconds (default: 30).")
    parser.add_argument("--output", type=Path, default=Path("dialogue.wav"),
                        help="Output stereo WAV file.")
    parser.add_argument("--transcript", type=Path, default=None,
                        help="Output transcript JSON (default: <output>.json).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--temp-text", type=float, default=0.7)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    seed_all(args.seed)

    LOGGER.info("Loading models from %s", args.hf_repo)
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)

    mimi_a = checkpoint_info.get_mimi(device=args.device)
    mimi_a.eval()
    mimi_b = checkpoint_info.get_mimi(device=args.device)
    mimi_b.eval()
    text_tokenizer = checkpoint_info.get_text_tokenizer()

    # Load two separate LM instances (they need independent streaming state)
    lm_kwargs = {}
    if args.lora_path:
        lm_kwargs = {"lora": True, "lora_rank": 128, "lora_scaling": 2.0}

    LOGGER.info("Loading Model A")
    lm_a = checkpoint_info.get_moshi(device=args.device, dtype=torch.bfloat16,
                                      lm_kwargs_overrides=lm_kwargs)
    lm_a.eval()

    LOGGER.info("Loading Model B")
    lm_b = checkpoint_info.get_moshi(device=args.device, dtype=torch.bfloat16,
                                      lm_kwargs_overrides=lm_kwargs)
    lm_b.eval()

    if args.lora_path:
        import safetensors.torch
        LOGGER.info("Loading LoRA from %s", args.lora_path)
        lora_state = safetensors.torch.load_file(str(args.lora_path))
        missing_a, unexpected_a = lm_a.load_state_dict(lora_state, strict=False)
        missing_b, _ = lm_b.load_state_dict(lora_state, strict=False)
        LOGGER.info("LoRA loaded: %d keys applied, %d missing, %d unexpected",
                     len(lora_state) - len(unexpected_a), len(missing_a), len(unexpected_a))

    lm_gen_a = LMGen(lm_a, temp=args.temp, temp_text=args.temp_text,
                      **checkpoint_info.lm_gen_config)
    lm_gen_b = LMGen(lm_b, temp=args.temp, temp_text=args.temp_text,
                      **checkpoint_info.lm_gen_config)

    audio_a, audio_b, transcript = run_dialogue(
        mimi_a, mimi_b, lm_gen_a, lm_gen_b, text_tokenizer,
        args.duration, args.device,
    )

    # Save stereo WAV (A on ch0, B on ch1)
    stereo = np.stack([audio_a[:min(len(audio_a), len(audio_b))],
                       audio_b[:min(len(audio_a), len(audio_b))]], axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(str(args.output), stereo, sample_rate=mimi_a.sample_rate)
    LOGGER.info("Wrote %s (%.1fs stereo)", args.output, stereo.shape[1] / mimi_a.sample_rate)

    # Save transcript
    transcript_path = args.transcript or args.output.with_suffix(".json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump({
            "duration_sec": args.duration,
            "sample_rate": mimi_a.sample_rate,
            "transcript": transcript,
            "text_a": "".join(t["text"] for t in transcript if t["speaker"] == "A"),
            "text_b": "".join(t["text"] for t in transcript if t["speaker"] == "B"),
        }, f, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote %s", transcript_path)


if __name__ == "__main__":
    main()
