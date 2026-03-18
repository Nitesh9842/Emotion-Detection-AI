"""
decision_engine.py — Weighted rule-based recommendation engine.

Takes predicted emotion, intensity, and optional context (time_of_day,
stress_level, energy_level, sleep_hours) to produce:
  1. A personalized, human-centric recommendation
  2. A timing suggestion (Immediate / Soon / Later / Scheduled)
"""

import random

random.seed(42)


# ── Recommendation Pools ────────────────────────────────────────────────────
# Each emotion maps to a list of (recommendation, base_urgency_weight) tuples.
# Higher weight → more likely to be flagged as urgent.

RECOMMENDATIONS = {
    "overwhelmed": [
        ("Take 5 slow, deep breaths — focus only on the air moving in and out.", 0.9),
        ("Write down the top 3 things weighing on you, then tackle just the first one.", 0.8),
        ("Step away from screens for 10 minutes. Look out a window or go outside.", 0.85),
        ("Text or call someone you trust and share what's on your mind.", 0.7),
        ("Put on calm music or nature sounds and close your eyes for 5 minutes.", 0.75),
        ("Splash cold water on your face — it activates your body's calming reflex.", 0.95),
        ("Organize one small area (desk, bag, phone home screen) to regain a sense of control.", 0.6),
    ],
    "restless": [
        ("Go for a brisk 10-minute walk — movement helps release nervous energy.", 0.8),
        ("Try the 5-4-3-2-1 grounding exercise: name 5 things you see, 4 you hear, 3 you feel, 2 you smell, 1 you taste.", 0.75),
        ("Write freely for 5 minutes — let your thoughts spill out without editing.", 0.65),
        ("Do a quick body scan: tense and release each muscle group from toes to head.", 0.7),
        ("Switch to a task that uses your hands (sketching, cooking, organizing).", 0.5),
        ("Set a 25-minute timer and commit to one single task (Pomodoro technique).", 0.55),
    ],
    "calm": [
        ("This is a great time for creative work or reflective journaling.", 0.2),
        ("Consider setting an intention or goal for the rest of your day.", 0.15),
        ("Share your calm energy — reach out to a friend or loved one.", 0.1),
        ("Use this clarity to plan tomorrow's priorities.", 0.15),
        ("Try a short gratitude exercise: list 3 things you appreciate right now.", 0.2),
        ("Read something inspiring or listen to a podcast you've been saving.", 0.1),
    ],
    "focused": [
        ("You're in the zone — protect this focus by silencing notifications.", 0.15),
        ("Channel this energy into your most important or challenging task.", 0.2),
        ("Set a clear mini-goal for the next 30 minutes to maximize this focus.", 0.2),
        ("After your focus session, reward yourself with a short break.", 0.1),
        ("Document your current progress or ideas while your thinking is sharp.", 0.25),
        ("Use this momentum to tackle something you've been procrastinating on.", 0.3),
    ],
    "neutral": [
        ("Check in with yourself: is there something you've been putting off?", 0.3),
        ("This is a good time for routine tasks or light planning.", 0.2),
        ("Try something new today — a different route, a new recipe, a new song.", 0.25),
        ("Do a quick energy boost: stretch, splash water, or grab a healthy snack.", 0.35),
        ("Reach out to someone you haven't talked to in a while.", 0.2),
        ("Reflect on one thing that went well today, no matter how small.", 0.15),
    ],
    "mixed": [
        ("Acknowledge the complexity of what you're feeling — mixed emotions are valid.", 0.5),
        ("Write down two columns: what feels good right now, and what feels hard.", 0.55),
        ("Try a 5-minute breathing exercise to create some mental space.", 0.6),
        ("Spend a moment separating your different feelings — name each one.", 0.5),
        ("Go for a walk and let your mind wander without trying to solve anything.", 0.45),
        ("Talk it out — call a friend or voice-memo your thoughts to yourself.", 0.5),
        ("Do one small thing that usually brings you joy (favorite song, snack, stretch).", 0.4),
    ],
}


# ── Timing Logic ────────────────────────────────────────────────────────────

def _compute_urgency_score(
    emotion: str,
    intensity: float,
    base_weight: float,
    stress_level: int = 3,
    energy_level: int = 3,
    sleep_hours: float = 7.0,
) -> float:
    """
    Compute a 0–1 urgency score from multiple signals.

    Components:
      - base_weight     (from recommendation pool)    × 0.30
      - intensity_score  (normalized 0–1)             × 0.30
      - stress_factor   (normalized 0–1)              × 0.20
      - fatigue_factor  (inverse of energy + sleep)   × 0.20
    """
    intensity_norm = (intensity - 1) / 4  # map 1–5 → 0–1
    stress_norm = (stress_level - 1) / 4  # map 1–5 → 0–1
    energy_norm = 1 - ((energy_level - 1) / 4)  # low energy → high urgency
    sleep_factor = max(0.0, 1 - sleep_hours / 8.0)  # poor sleep → high urgency
    fatigue = (energy_norm + sleep_factor) / 2

    score = (
        0.30 * base_weight
        + 0.30 * intensity_norm
        + 0.20 * stress_norm
        + 0.20 * fatigue
    )
    return min(max(score, 0.0), 1.0)


def _urgency_to_timing(score: float, time_of_day: str = "morning") -> dict:
    """Map urgency score to a human-readable timing recommendation."""
    if score >= 0.70:
        label = "Immediate"
        detail = "Right now — take a pause before doing anything else."
    elif score >= 0.50:
        label = "Soon"
        if time_of_day in ("morning", "early_morning"):
            detail = "Within the next hour, before your day gets busier."
        elif time_of_day == "afternoon":
            detail = "Within the next hour — take a mid-day reset."
        else:
            detail = "Before the evening ends — give yourself this time."
    elif score >= 0.30:
        label = "Later Today"
        if time_of_day in ("morning", "early_morning"):
            detail = "Schedule this for your lunch break or early afternoon."
        elif time_of_day == "afternoon":
            detail = "Set aside 15 minutes this evening for this."
        else:
            detail = "Try this before you wind down for bed tonight."
    else:
        label = "Whenever Ready"
        if time_of_day in ("evening", "night"):
            detail = "Tomorrow morning could be a lovely time for this."
        else:
            detail = "Sometime today when it feels natural — no rush."

    return {"label": label, "detail": detail}


# ── Main Engine ─────────────────────────────────────────────────────────────

def get_recommendation(
    emotion: str,
    intensity: float,
    time_of_day: str = "morning",
    stress_level: int = 3,
    energy_level: int = 3,
    sleep_hours: float = 7.0,
) -> dict:
    """
    Generate a personalized recommendation + timing.

    Parameters
    ----------
    emotion : str
        Predicted emotion class.
    intensity : float
        Predicted intensity (1–5).
    time_of_day : str
        Current time of day.
    stress_level : int (1–5)
    energy_level : int (1–5)
    sleep_hours : float

    Returns
    -------
    dict with keys: emotion, intensity, intensity_label, recommendation,
                    timing_label, timing_detail, urgency_score
    """
    # Normalize emotion
    emotion = emotion.lower().strip()
    if emotion not in RECOMMENDATIONS:
        emotion = "neutral"

    pool = RECOMMENDATIONS[emotion]

    # Score each recommendation and pick the best match
    scored = []
    for rec_text, base_w in pool:
        urgency = _compute_urgency_score(
            emotion, intensity, base_w, stress_level, energy_level, sleep_hours
        )
        scored.append((rec_text, base_w, urgency))

    # For high-intensity negative emotions → pick highest urgency
    # For low-intensity / positive emotions → pick moderate/lowest urgency
    if intensity >= 4 and emotion in ("overwhelmed", "restless", "mixed"):
        scored.sort(key=lambda x: x[2], reverse=True)
    elif intensity <= 2 and emotion in ("calm", "focused", "neutral"):
        scored.sort(key=lambda x: x[2])
    else:
        # Pick the one closest to middle urgency (balanced choice)
        mid = 0.5
        scored.sort(key=lambda x: abs(x[2] - mid))

    best_rec, _, best_urgency = scored[0]

    # Add a contextual note for very poor sleep or high stress
    contextual_note = ""
    if sleep_hours < 5:
        contextual_note = " Also, prioritize getting better rest tonight — sleep matters."
    elif stress_level >= 4 and energy_level <= 2:
        contextual_note = " You seem drained — be extra gentle with yourself today."

    # Intensity label
    if intensity <= 2:
        intensity_label = "Low"
    elif intensity <= 3.5:
        intensity_label = "Medium"
    else:
        intensity_label = "High"

    timing = _urgency_to_timing(best_urgency, time_of_day)

    return {
        "emotion": emotion.capitalize(),
        "intensity": round(intensity, 2),
        "intensity_label": intensity_label,
        "recommendation": best_rec + contextual_note,
        "timing_label": timing["label"],
        "timing_detail": timing["detail"],
        "urgency_score": round(best_urgency, 3),
    }
