"""System prompt and analysis logic for the guessing game agent.

=== EDIT THIS FILE ===

Motion-focused strategy:
1. Buffer 12 frames over ~12 seconds to capture full movement cycles
2. Send all frames directly to the strong model with explicit motion analysis
3. Strong model analyzes temporal patterns across all frames
4. If wrong, carry forward history for next attempt
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
from collections import Counter

from openai import AsyncOpenAI

from core import Frame

# ---------------------------------------------------------------------------
# OpenRouter OpenAI-compatible client
# ---------------------------------------------------------------------------

_openrouter_client: AsyncOpenAI | None = None


def _get_openrouter_client() -> AsyncOpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _openrouter_client


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

VISION_MODEL_ID = "google/gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_frame_buffer: list[Frame] = []
_history: list[dict] = []
_FRAMES_PER_BATCH = 12


def _image_to_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _consensus_guess(guesses: list[str]) -> str | None:
    filtered = [g for g in guesses if g.upper() != "SKIP"]
    if not filtered:
        return None
    counts = Counter(filtered)
    return counts.most_common(1)[0][0]


async def analyze(frame: Frame) -> str | None:
    """Analyze frames to identify what is being acted out.

    Strategy:
    - Buffer 12 frames (~12 seconds) to capture full movement cycles
    - Send all frames to strong model for temporal analysis
    - The model sees the full motion sequence and determines the answer
    """
    global _frame_buffer

    _frame_buffer.append(frame)

    if len(_frame_buffer) < _FRAMES_PER_BATCH:
        return None

    frames = _frame_buffer[:]
    _frame_buffer.clear()

    print(f"\n=== ANALYZING {len(frames)} FRAMES ===")

    client = _get_openrouter_client()
    content: list = []

    for i, frame in enumerate(frames):
        image_bytes = _image_to_bytes(frame.image)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            }
        )

    text_parts = [
        "These 12 frames show a person acting out a word or phrase in charades.",
        "Frame 1 is the START, Frame 12 is the END of the movement sequence.",
        "",
        "TASK: Analyze the motion across ALL frames to determine what they are acting.",
        "",
        "COMPLEX MOVEMENT RECOGNITION:",
        "",
        "1. SEQUENTIAL MOVEMENTS (multi-step actions):",
        "   - Look for a START, MIDDLE, and END in the motion",
        "   - Driving: grip wheel → turn left/right → shift gear",
        "   - Phone call: hold phone → talk → hang up",
        "   - Eating: grab food → bring to mouth → chew",
        "   - Fishing: cast line → reel in → catch fish",
        "",
        "2. HAND SIGNATURES (subtle hand gestures):",
        "   - CRITICAL: Hands near chest/face with specific finger positions = communication/phone",
        "   - Two fingers up with thumb = peace sign / victory",
        "   - Circle with thumb and index = OK sign",
        "   - Fist with thumb tucked = gang sign (varies)",
        "   - Pinky and thumb = phone gesture",
        "   - Wrist rotation while gripping = steering wheel",
        "   - Two hands moving apart = opening/splitting something",
        "",
        "3. DIRECTIONAL PATTERNS:",
        "   - Circular (clockwise/counter): steering wheel, stirring, mixing",
        "   - Back-and-forth: sawing, swimming freestyle, ironing",
        "   - Up-and-down: jumping, hammering, nodding, bounce",
        "   - Side-to-side: dancing, swimming breaststroke, waving",
        "   - Diagonal: throwing, serving (tennis), salute",
        "",
        "4. BODY PART ISOLATION:",
        "   - Arms only: swimming, tennis, baseball swing",
        "   - Hands near head: phone, eating, glasses, headphones",
        "   - Full body: dancing, running, crawling, jumping jack",
        "   - Legs only: kicking, walking, running, bicycling",
        "   - Torso twist: hula hoop, dancing, golf swing follow-through",
        "",
        "5. REPETITION & CYCLES:",
        "   - Count the cycles in 12 frames: 2-3 cycles = rhythmic activity",
        "   - Swimming: 3-4 full arm cycles",
        "   - Running: multiple leg cycles",
        "   - Basketball dribble: 4-6 bounces",
        "   - Driving wheel: 2-3 full rotations",
        "",
        "6. SPECIFIC ACTIVITIES:",
        "   - SWIMMING: alternating windmill arms overhead, flutter kick",
        "   - DRIVING: both hands on imaginary wheel, turning motion",
        "   - BOXING: jab (quick forward punch), hook (curved punch)",
        "   - TENNIS: serving motion, racquet swing",
        "   - GUITAR: strumming motion, fingering chords",
        "   - TYPING: fingers moving near keyboard",
        "",
        "7. GANG SIGNS & HAND SIGNS:",
        "   - Look for distinct finger positions, not just any gesture",
        "   - Peace sign: index + middle up, others down",
        "   - I love you: pinky + thumb out, others down (ILU sign)",
        "   - Thumbs up: fist with thumb extended up",
        "   - Wave: hand moving side to side palm out",
        "   - C sign: hand forming letter C shape",
        "",
        "If motion is complex or unusual, think about what everyday activity it most resembles.",
        "Trust the most consistent interpretation across all frames.",
        "",
        "8. SYMBOLIC BODY SHAPES:",
        "   - Body parts forming geometric shapes (triangles, circles, lines) represent concepts",
        "   - Hand positions combined with arm/body placement form symbolic gestures",
        "   - Look for what everyday object or concept the shape resembles",
        "   - Consider common cultural symbols made with hands and body",
        "   - Interpret the gestalt of the pose, not just individual parts",
    ]

    if _history:
        text_parts.append("")
        text_parts.append("=== PREVIOUS WRONG GUESSES (do not repeat) ===")
        for entry in _history:
            text_parts.append(f"- {entry['final_guess']}")

    text_parts.extend(
        [
            "",
            "Based on the full motion sequence from start to end, give your answer.",
            'Answer with ONLY a short phrase (1-3 words). Examples: "driving car", "bird flying", "gang sign".',
        ]
    )

    content.append({"type": "text", "text": "\n".join(text_parts)})

    response = await client.chat.completions.create(
        model=VISION_MODEL_ID,
        messages=[{"role": "user", "content": content}],
        extra_body={
            "thinking": {"type": "enabled", "max_tokens": 4096},
            "include_reasoning": True,
        },
    )

    thinking = ""
    reasoning = response.choices[0].message.reasoning
    if reasoning:
        thinking = str(reasoning)

    answer = response.choices[0].message.content.strip() or ""

    print(f"\nAnswer: {answer}")
    if thinking:
        print(f"\nAnalysis:\n{thinking}")

    _history.append({"final_guess": answer})

    if answer.upper() == "SKIP":
        return None

    return answer
