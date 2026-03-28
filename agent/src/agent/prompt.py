"""System prompt and analysis logic for the guessing game agent.

=== EDIT THIS FILE ===

Strategy (two-stage pipeline with history):
1. Buffer 3 frames over ~3 seconds
2. Send each frame to a fast model (haiku) in parallel → 3 initial guesses
3. Send all 3 frames + initial guesses + history to a strong model (sonnet) → 1 final guess
4. If wrong, the next iteration carries forward all prior guesses as context
"""

from __future__ import annotations

import asyncio
import io

from pydantic_ai import Agent, BinaryContent

from core import Frame

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

FAST_MODEL = "anthropic:claude-3-5-haiku-20241022"
STRONG_MODEL = "anthropic:claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

FAST_SYSTEM_PROMPT = """\
You are playing a visual guessing game. You will receive a screenshot from a
live camera feed. Your goal is to identify what is being shown as quickly and
accurately as possible.

Rules:
- Give your best guess as a short, specific answer (1-5 words).
- If you truly cannot tell what is being shown, respond with exactly "SKIP".
- Be specific: "golden retriever" is better than "dog".
- Focus on the main subject of the image.
"""

STRONG_SYSTEM_PROMPT = """\
You are the final judge in a visual guessing game. You will receive:
- 3 frames captured over ~3 seconds from a live camera feed
- 3 initial guesses from a fast screening model (one per frame)
- A history of all previous guesses and outcomes from prior rounds (if any)

Your job is to synthesize all of this information and produce the single best,
most accurate guess for what is being shown.

Rules:
- Give exactly ONE guess as a short, specific answer (1-5 words).
- Be specific: "golden retriever" is better than "dog".
- Consider the consensus across the 3 initial guesses — if they agree, that's
  a strong signal.
- If they disagree, use the frames themselves to decide.
- Use the history to AVOID repeating wrong guesses. If a guess was already
  tried and was wrong, pick something different.
- Respond with ONLY your guess, nothing else.
"""

# ---------------------------------------------------------------------------
# Agents (lazily initialized — env vars must be loaded before first use)
# ---------------------------------------------------------------------------

_fast_agent: Agent | None = None
_strong_agent: Agent | None = None


def _get_fast_agent() -> Agent:
    global _fast_agent
    if _fast_agent is None:
        _fast_agent = Agent(FAST_MODEL, system_prompt=FAST_SYSTEM_PROMPT)
    return _fast_agent


def _get_strong_agent() -> Agent:
    global _strong_agent
    if _strong_agent is None:
        _strong_agent = Agent(STRONG_MODEL, system_prompt=STRONG_SYSTEM_PROMPT)
    return _strong_agent


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Buffer for collecting frames before running the pipeline
_frame_buffer: list[Frame] = []

# History of all iterations: list of {"initial_guesses": [...], "final_guess": str}
_history: list[dict] = []

# How many frames to collect before running the pipeline
_FRAMES_PER_BATCH = 3


def _image_to_bytes(image) -> bytes:
    """Convert a PIL Image to JPEG bytes for sending to the LLM."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


async def _fast_guess(frame: Frame) -> str:
    """Send a single frame to the fast model and get a guess."""
    image_bytes = _image_to_bytes(frame.image)
    result = await _get_fast_agent().run([
        BinaryContent(data=image_bytes, media_type="image/jpeg"),
        "What is being shown in this image? Give your best guess.",
    ])
    return result.output.strip()


async def _strong_guess(
    frames: list[Frame],
    initial_guesses: list[str],
) -> str:
    """Send all frames + initial guesses + history to the strong model."""
    # Build the message parts: all 3 images first
    parts: list = []
    for i, frame in enumerate(frames):
        image_bytes = _image_to_bytes(frame.image)
        parts.append(BinaryContent(data=image_bytes, media_type="image/jpeg"))

    # Build the text prompt with initial guesses and history
    text_lines = ["Here are 3 frames from the live camera feed (attached above)."]
    text_lines.append("")
    text_lines.append("Initial guesses from the fast screening model:")
    for i, guess in enumerate(initial_guesses, 1):
        text_lines.append(f"  Frame {i}: {guess}")

    if _history:
        text_lines.append("")
        text_lines.append("=== HISTORY OF PREVIOUS ATTEMPTS ===")
        for iteration_num, entry in enumerate(_history, 1):
            text_lines.append(f"Iteration {iteration_num}:")
            text_lines.append(f"  Initial guesses: {', '.join(entry['initial_guesses'])}")
            text_lines.append(f"  Final guess submitted: {entry['final_guess']} (WRONG)")
        text_lines.append("")
        text_lines.append(
            "All previous final guesses were WRONG. Do NOT repeat them. "
            "Try a different, more specific or more creative answer."
        )

    text_lines.append("")
    text_lines.append(
        "Based on the frames, the initial guesses, and the history above, "
        "what is your single best guess? Respond with ONLY the guess (1-5 words)."
    )

    parts.append("\n".join(text_lines))

    result = await _get_strong_agent().run(parts)
    return result.output.strip()


async def analyze(frame: Frame) -> str | None:
    """Analyze a single frame and return a guess, or None to skip.

    Internally buffers 3 frames, then runs the two-stage pipeline:
    1. Fast model (haiku) on each frame in parallel → 3 initial guesses
    2. Strong model (sonnet) synthesizes a final guess from all data + history

    Returns None while buffering, returns the final guess on the 3rd frame.
    """
    global _frame_buffer

    # --- Accumulate frames ---
    _frame_buffer.append(frame)

    if len(_frame_buffer) < _FRAMES_PER_BATCH:
        remaining = _FRAMES_PER_BATCH - len(_frame_buffer)
        print(f"  [agent] Buffering frame {len(_frame_buffer)}/{_FRAMES_PER_BATCH} "
              f"({remaining} more to go)")
        return None

    # --- We have 3 frames, run the pipeline ---
    frames = _frame_buffer[:]
    _frame_buffer.clear()

    print(f"  [agent] Running two-stage pipeline (iteration {len(_history) + 1})...")

    # Stage 1: Fast model — 3 parallel calls
    print("  [agent] Stage 1: Fast model (haiku) on 3 frames...")
    initial_guesses_raw = await asyncio.gather(
        _fast_guess(frames[0]),
        _fast_guess(frames[1]),
        _fast_guess(frames[2]),
    )
    initial_guesses = list(initial_guesses_raw)

    for i, guess in enumerate(initial_guesses, 1):
        print(f"  [agent]   Frame {i} → {guess}")

    # Filter out SKIPs for display, but pass all to strong model
    non_skip = [g for g in initial_guesses if g.upper() != "SKIP"]
    if not non_skip:
        print("  [agent] All 3 fast guesses were SKIP — skipping this batch.")
        return None

    # Stage 2: Strong model — synthesize final guess
    print("  [agent] Stage 2: Strong model (sonnet) synthesizing final guess...")
    final_guess = await _strong_guess(frames, initial_guesses)
    print(f"  [agent] Final guess: {final_guess}")

    # Record in history
    _history.append({
        "initial_guesses": initial_guesses,
        "final_guess": final_guess,
    })

    # Don't return SKIP from the strong model
    if final_guess.upper() == "SKIP":
        print("  [agent] Strong model said SKIP — skipping.")
        return None

    return final_guess
