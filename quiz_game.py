# quiz_game.py
import random

# A tiny bank of 5 math questions (numeric answers only)
_QBANK = [
    {"q": "2 + 3 = ?",  "a": 5},
    {"q": "5 + 4 = ?",  "a": 9},
    {"q": "10 - 6 = ?", "a": 4},
    {"q": "3 × 2 = ?",  "a": 6},
    {"q": "12 ÷ 3 = ?", "a": 4},
]

def _numeric_choices(answer: int | float, n: int = 4) -> list[int]:
    """Build 'n' unique numeric choices including the correct answer."""
    base = int(answer)
    choices = {base}
    deltas = [-5, -3, -2, -1, 1, 2, 3, 4, 5, 6]

    # Add plausible distractors; avoid negatives below zero for very small answers
    while len(choices) < n:
        candidate = base + random.choice(deltas)
        if candidate >= 0:
            choices.add(int(candidate))

    out = list(choices)
    random.shuffle(out)
    return out

def get_quiz_question(index: int) -> dict | None:
    """
    Returns a dict: {"question": str, "choices": list, "answer": correct}
    or None when there are no more questions.
    """
    if index < 0 or index >= len(_QBANK):
        return None

    item = _QBANK[index]
    question = item.get("q")
    answer = item.get("a")

    # Safety: ensure we have a numeric answer
    if not isinstance(answer, (int, float)):
        # If a future non-numeric question sneaks in, fallback to fixed choices
        choices = ["I don't know", "Maybe", "Yes", "No"]
        return {"question": str(question), "choices": choices, "answer": choices[0]}

    choices = _numeric_choices(answer, n=4)
    return {"question": str(question), "choices": choices, "answer": answer}
