"""System prompt and analysis logic for the guessing game agent.

=== EDIT THIS FILE ===

Strategy (flipped two-stage pipeline, strict wall-clock timing):
1. A background pipeline runs on strict wall-clock schedule
2. Every _WINDOW_DURATION_S seconds, grab all buffered frames and fire off
   a strong model (o3) call as a background task (fire-and-forget)
3. After _WINDOWS_PER_CYCLE windows, wait for all o3 tasks to finish, then
   send the text guesses to the fast model (Gemini 3.1 Pro) for consensus
4. If wrong, the next cycle carries forward all prior guesses as context
"""

from __future__ import annotations

import asyncio
import io
import time

# ---------------------------------------------------------------------------
# Monkey-patch: pydantic-ai 1.73.0's OpenRouter provider rejects
# service_tier="standard" returned by OpenRouter. Patch the validation
# model to accept it before any agent call is made.
# ---------------------------------------------------------------------------
from openai.types.chat import ChatCompletion as _ChatCompletion

_orig_service_tier = _ChatCompletion.model_fields["service_tier"]
if "standard" not in str(_orig_service_tier.annotation):
    from typing import Literal, Optional

    _ChatCompletion.model_fields["service_tier"].annotation = Optional[
        Literal["auto", "default", "flex", "scale", "priority", "standard"]
    ]
    _ChatCompletion.model_rebuild()
# ---------------------------------------------------------------------------

from pydantic_ai import Agent, BinaryContent

from core import Frame

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

INITIAL_MODEL = "openrouter:openai/o3"                       # strong — reads frames
FINAL_MODEL = "openrouter:google/gemini-3.1-pro-preview"     # fast — text consensus

# ---------------------------------------------------------------------------
# Timing constants (placeholders — tune these during practice)
# ---------------------------------------------------------------------------

_WINDOW_DURATION_S = 1        # seconds of frame collection per window
_WINDOWS_PER_CYCLE = 3        # number of initial guesses before final guess

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

INITIAL_SYSTEM_PROMPT = """\
You are playing a charades guessing game. A human person is standing in front
of a camera acting out a word or phrase using only body language, gestures, and
pantomime — no speaking, no written words, no props with text.

You will receive a sequence of frames captured over a short window from the
live camera feed. The frames show the person's movements in chronological order,
so you can observe how their gestures evolve over time.

Important — distinguish between two types of acting:
1. STATIC (symbol/pose): The person holds a pose or forms a shape with their
   hands or body to represent a concept or object. If the pose is roughly the
   same across the frames, it is likely a static gesture. Examples: hands
   forming a heart shape = "love", arms spread wide and still = "airplane",
   flexing biceps = "strong", hands on head like antlers = "deer".
2. DYNAMIC (action/movement): The person is performing a movement or imitating
   an activity or animal. If the pose changes significantly across the frames,
   it is likely a dynamic gesture. Examples: swinging arms = "swimming",
   pretending to kick a ball = "soccer", crawling = "baby" or "snake",
   swinging an imaginary bat = "baseball".

Recognizing which type you are seeing helps narrow your guess.

Rules:
- Give your best guess as a short answer (1-5 words).
- Guesses should be things commonly used in charades: everyday words, actions,
  animals, emotions, movie titles, book titles, song titles, famous people,
  occupations, sports, or common phrases.
- Focus entirely on the person's gestures, poses, and movements — ignore
  background objects, furniture, and the room itself.
- Think about what concept or thing the body language represents, not what you
  literally see.
- If you truly cannot interpret what is being acted out, respond with exactly
  "SKIP".
"""

FINAL_SYSTEM_PROMPT = """\
You are the final judge in a charades guessing game. A human person has been
acting out a word or phrase in front of a camera.

You will receive:
- A set of initial guesses from o3 (a vision model) that watched consecutive
  windows of the person's performance (one guess per window)
- A history of all previous guesses and outcomes from prior cycles (if any)

Your job is to find the consensus across the initial guesses and produce the
single best, most accurate charades answer.

Rules:
- Give exactly ONE guess as a short answer (1-5 words).
- Guesses must be things commonly acted out in charades: everyday words,
  actions, animals, emotions, movie titles, book titles, song titles, famous
  people, occupations, sports, or common phrases.
- Look for patterns in the initial guesses — if most agree on a word or
  theme, that is a strong signal.
- If the guesses are split, consider which answer best fits the charades
  context.
- Use the history to AVOID repeating wrong guesses. If a guess was already
  tried and was wrong, pick something different or more specific.
- Respond with ONLY your guess, nothing else.
"""

# ---------------------------------------------------------------------------
# Agents (lazily initialized — env vars must be loaded before first use)
# ---------------------------------------------------------------------------

_initial_agent: Agent | None = None
_final_agent: Agent | None = None


def _get_initial_agent() -> Agent:
    global _initial_agent
    if _initial_agent is None:
        _initial_agent = Agent(INITIAL_MODEL, system_prompt=INITIAL_SYSTEM_PROMPT)
    return _initial_agent


def _get_final_agent() -> Agent:
    global _final_agent
    if _final_agent is None:
        _final_agent = Agent(FINAL_MODEL, system_prompt=FINAL_SYSTEM_PROMPT)
    return _final_agent


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Shared frame buffer — analyze() writes, pipeline reads
_window_frames: list[Frame] = []

# Completed initial guesses for the current cycle
_initial_guesses: list[str] = []

# In-flight o3 tasks
_o3_tasks: list[asyncio.Task] = []

# Final result from the pipeline (set when cycle is complete)
_final_result: str | None = None

# Whether the background pipeline is currently running
_pipeline_running: bool = False

# Cycle counter for display
_cycle_counter: int = 0

# History of all cycles: list of {"initial_guesses": [...], "final_guess": str}
_history: list[dict] = []


def _reset_cycle() -> None:
    """Reset all cycle state for the next iteration."""
    global _window_frames, _initial_guesses, _o3_tasks
    global _final_result, _pipeline_running
    _window_frames = []
    _initial_guesses = []
    _o3_tasks = []
    _final_result = None
    _pipeline_running = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_bytes(image) -> bytes:
    """Convert a PIL Image to JPEG bytes for sending to the LLM."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


async def _initial_guess(frames: list[Frame]) -> str:
    """Send a batch of frames (one window) to o3 (strong model)."""
    # Build message parts: all frames as images + text prompt
    parts: list = []
    for frame in frames:
        image_bytes = _image_to_bytes(frame.image)
        parts.append(BinaryContent(data=image_bytes, media_type="image/jpeg"))

    # Include history of wrong guesses so the model avoids repeating them
    history_note = ""
    if _history:
        wrong_guesses = [entry["final_guess"] for entry in _history]
        history_note = (
            "\n\nIMPORTANT: The following guesses have already been tried and "
            "were WRONG: " + ", ".join(wrong_guesses) + ". "
            "Do NOT suggest any of these. Try a completely different "
            "interpretation of the person's gestures."
        )

    parts.append(
        f"Here are {len(frames)} consecutive frames from a "
        f"{_WINDOW_DURATION_S}-second window of a charades game. The person is "
        f"acting out a word or phrase. "
        f"What is the person acting out? Give your best guess."
        f"{history_note}"
    )

    result = await _get_initial_agent().run(parts)
    return result.output.strip()


async def _o3_wrapper(frames: list[Frame], task_num: int) -> None:
    """Fire-and-forget wrapper: run o3 initial guess, collect result, print."""
    start = time.monotonic()
    try:
        guess = await _initial_guess(frames)
        elapsed = time.monotonic() - start
        _initial_guesses.append(guess)
        print(f"  [initial #{task_num}/{_WINDOWS_PER_CYCLE}] "
              f"{guess} ({len(frames)} frames, {elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"  [initial #{task_num}/{_WINDOWS_PER_CYCLE}] "
              f"ERROR: {e} ({elapsed:.1f}s)")


async def _final_guess(initial_guesses: list[str]) -> str:
    """Send only text guesses + history to Gemini 3.1 Pro (fast model) for consensus."""
    text_lines = [
        f"Here are {len(initial_guesses)} initial guesses from o3 (a vision "
        f"model) that watched a person playing charades (one guess per "
        f"{_WINDOW_DURATION_S}-second window):"
    ]
    text_lines.append("")
    for i, guess in enumerate(initial_guesses, 1):
        text_lines.append(f"  Window {i}: {guess}")

    if _history:
        text_lines.append("")
        text_lines.append("=== HISTORY OF PREVIOUS CYCLES ===")
        for cycle_num, entry in enumerate(_history, 1):
            text_lines.append(f"Cycle {cycle_num}:")
            text_lines.append(
                f"  Initial guesses: {', '.join(entry['initial_guesses'])}"
            )
            text_lines.append(
                f"  Final guess submitted: {entry['final_guess']} (WRONG)"
            )
        text_lines.append("")
        text_lines.append(
            "All previous final guesses were WRONG. Do NOT repeat them. "
            "Try a different, more specific or more creative answer."
        )

    text_lines.append("")
    text_lines.append(
        "Based on the initial guesses and the history above, "
        "what is your single best guess? Respond with ONLY the guess (1-5 words)."
    )

    result = await _get_final_agent().run("\n".join(text_lines))
    return result.output.strip()


# ---------------------------------------------------------------------------
# Background pipeline — runs on strict wall-clock timing
# ---------------------------------------------------------------------------

async def _run_pipeline() -> None:
    """Background coroutine: manages windows on a strict wall-clock schedule.

    - Every _WINDOW_DURATION_S seconds, grabs all buffered frames and launches
      a fire-and-forget strong model (o3) task
    - After _WINDOWS_PER_CYCLE windows, waits for all tasks, then runs the
      fast model (Gemini) for the final consensus guess
    """
    global _final_result, _cycle_counter

    _cycle_counter += 1
    cycle_duration = _WINDOW_DURATION_S * _WINDOWS_PER_CYCLE
    print(f"  [agent] Cycle {_cycle_counter} started "
          f"({_WINDOWS_PER_CYCLE} windows x {_WINDOW_DURATION_S:.0f}s "
          f"= {cycle_duration:.0f}s)")

    # Launch one fire-and-forget task per window on strict timing
    for window_num in range(1, _WINDOWS_PER_CYCLE + 1):
        # Wait exactly _WINDOW_DURATION_S seconds (real wall-clock)
        await asyncio.sleep(_WINDOW_DURATION_S)

        # Grab all frames buffered during this window
        frames = _window_frames[:]
        _window_frames.clear()

        if not frames:
            print(f"  [agent] Window {window_num}/{_WINDOWS_PER_CYCLE} — "
                  f"no frames collected, skipping")
            continue

        # Fire-and-forget o3 call
        print(f"  [agent] Window {window_num}/{_WINDOWS_PER_CYCLE} — "
              f"{len(frames)} frames → o3 (fire-and-forget)")
        task = asyncio.create_task(_o3_wrapper(frames, window_num))
        _o3_tasks.append(task)

    # All windows dispatched — wait for any pending o3 tasks
    if _o3_tasks:
        pending = [t for t in _o3_tasks if not t.done()]
        if pending:
            print(f"  [agent] Waiting for {len(pending)} remaining o3 task(s)...")
        await asyncio.gather(*_o3_tasks)

    # Check we have usable guesses
    if not _initial_guesses:
        print("  [agent] No initial guesses collected — skipping this cycle.")
        _reset_cycle()
        return

    non_skip = [g for g in _initial_guesses if g.upper() != "SKIP"]
    if not non_skip:
        print("  [agent] All initial guesses were SKIP — skipping this cycle.")
        _reset_cycle()
        return

    # Run fast model for final consensus (text-only)
    print(f"  [agent] Final guess (Gemini): {len(_initial_guesses)} initial "
          f"guesses → consensus...")
    final = await _final_guess(_initial_guesses)
    print(f"  [agent] Final guess: {final}")

    # Record in history
    _history.append({
        "initial_guesses": list(_initial_guesses),
        "final_guess": final,
    })

    # Signal result to analyze()
    if final.upper() != "SKIP":
        _final_result = final
    else:
        print("  [agent] Final model said SKIP — skipping.")
        _reset_cycle()


# ---------------------------------------------------------------------------
# Main entry point — called by __main__.py for every frame
# ---------------------------------------------------------------------------

async def analyze(frame: Frame) -> str | None:
    """Analyze a single frame and return a guess, or None to skip.

    This function is non-blocking. It:
    - Buffers every frame into a shared list
    - On first call, starts a background pipeline coroutine that runs on
      strict wall-clock timing (asyncio.sleep-based)
    - Checks if the background pipeline has produced a final result
    - Returns the result when ready, None otherwise
    """
    global _pipeline_running

    # Always buffer the frame (never blocks)
    _window_frames.append(frame)

    # Start background pipeline on first frame of a cycle
    if not _pipeline_running:
        _pipeline_running = True
        asyncio.create_task(_run_pipeline())
        return None

    # Check if pipeline produced a result
    if _final_result is not None:
        result = _final_result
        _reset_cycle()
        return result

    return None
