"""Build the packaged reference set for the default reward model (raw rewards on good and bad answers).

Run under the GPU lock (downloads the model on first use):

    scripts/gpu-lock.sh uv run --extra text python scripts/make_reference_sets.py

Writes src/hone_taste/references/text/reward_model.json. The examples below are hand-written: each
prompt has a correct, helpful answer ("good") and an off-topic, wrong or unhelpful one ("bad").
"""

from __future__ import annotations

import json
from pathlib import Path

import hone_taste as tt
from hone_taste.normalize import ReferenceSet

EXAMPLES = [
    (
        "What is the capital of France?",
        "The capital of France is Paris.",
        "France is a country. Countries have many cities.",
    ),
    (
        "How many legs does a spider have?",
        "A spider has eight legs.",
        "Spiders have six legs, like all insects.",
    ),
    (
        "Give me a synonym for 'happy'.",
        "'Joyful' is a common synonym for 'happy'.",
        "I don't want to answer that.",
    ),
    ("What is 12 times 12?", "12 times 12 is 144.", "12 times 12 is 124."),
    (
        "Write one sentence about autumn.",
        "In autumn the maple leaves turn red and drift onto the wet pavement.",
        "Pizza is best with pineapple and cold coffee.",
    ),
    (
        "How do I boil an egg?",
        "Put the egg in boiling water for about 9 minutes, then cool it in cold water before peeling.",
        "Eggs are laid by chickens.",
    ),
    (
        "Translate 'thank you' into Spanish.",
        "'Thank you' in Spanish is 'gracias'.",
        "'Thank you' in Spanish is 'bonjour'.",
    ),
    (
        "Name a primary color.",
        "Red is a primary color.",
        "Colors are just wavelengths, so the question makes no sense, obviously.",
    ),
    (
        "Why is the sky blue?",
        "Air molecules scatter short blue wavelengths of sunlight more than red ones (Rayleigh scattering).",
        "Because the ocean reflects onto it.",
    ),
    (
        "Suggest a name for a black cat.",
        "How about 'Midnight' or 'Pepper'?",
        "Cats should not have names.",
    ),
    (
        "What does HTML stand for?",
        "HTML stands for HyperText Markup Language.",
        "HTML stands for High Tech Machine Learning.",
    ),
    (
        "Write a haiku about rain.",
        "Soft rain on the roof /\nthe kettle starts to whistle /\nthe cat finds my lap",
        "Rain is water. Water is wet. The end.",
    ),
]


def main() -> None:
    any_set = ReferenceSet("text", "reward_model", (0.0, 1.0))  # only details["raw"] is used here
    scorer = tt.reward_model(reference=any_set)
    good, bad = [], []
    for prompt, good_answer, bad_answer in EXAMPLES:
        for answer, bucket in ((good_answer, good), (bad_answer, bad)):
            result = scorer({"prompt": prompt, "response": answer})
            if result.value is None:
                raise SystemExit(f"could not score: {result.error}")
            bucket.append(round(float(result.details["raw"]), 4))
    scorer.close()
    info = tt.model_info("reward_model")
    data = {
        "domain": "text",
        "scorer": "reward_model",
        "model_id": info.model_id,
        "source": "raw rewards of the default model on 12 hand-written prompts, each with a correct, helpful "
        "answer (good) and a wrong, off-topic or unhelpful one (bad); scripts/make_reference_sets.py",
        "good": good,
        "bad": bad,
    }
    out = Path(__file__).parent.parent / "src" / "hone_taste" / "references" / "text" / "reward_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: good {min(good)}..{max(good)}, bad {min(bad)}..{max(bad)}")


if __name__ == "__main__":
    main()
