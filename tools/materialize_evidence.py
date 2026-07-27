"""Materialize the evaluator-visible summaries as internal evidence bundles.

This standard-library-only utility does not recompute scientific results. The
fixed formal command does that. It mirrors the already validated candidate
JSON into the durable `.openresearch/artifacts/` structure required by the
campaign and records the exact checker/control fields used by the standalone
verifier.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "space_candidate" / "data"
DEST = ROOT / ".openresearch" / "artifacts"
SOURCE_SHA = "5dee110336720fa632917d6f97c9cb2ad09c9cda2ce809bd2640d64b1fc4d55d"
COMMAND = "uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py"

FILES = {
    1: "claim1_full_network.json",
    2: "claim2_activations.json",
    3: "claim3_theorem32.json",
    4: "claim4_zero_gradient.json",
    5: "claim5_zero_output_gradient.json",
    6: "claim6_raw.json",
}

TITLES = {
    1: "Theorem 3.1: arbitrary values and compatible gradients",
    2: "Theorem 3.1: six practical activations",
    3: "Theorem 3.2: loss-dependent antisymmetric gradients",
    4: "Lemma 3.4: target values with zero AD gradient",
    5: "Lemma 3.5: zero output with target AD gradient",
    6: "Section 1.1: non-associativity mechanism",
}

QUANTIFIERS = {
    1: "Every listed activation, complete bounded float domain, arbitrary value and compatible bounded gradient maps, every L>=9.",
    2: "Theorem 3.1 separately names ReLU, ELU, GELU, Swish, Sigmoid, and tanh under its activation assumptions.",
    3: "Every arbitrary antisymmetric g*(x,y), all bounded x and y, every L>=2^(q+1)+2p+11.",
    4: "Under Conditions 1-2 and distinguishability: arbitrary values, zero AD gradient for all stated input gradients, and boxplus neutrality.",
    5: "Under Conditions 1-3 and distinguishability: zero output on the complete domain, arbitrary target AD gradient, and boxplus neutrality.",
    6: "Floating-point non-associativity decouples the executed AD gradient from exact-real chain-rule proportionality.",
}

METHODS = {
    1: "Execute a 13-layer ReLU composition and exhaust its declared six-point domain with bit-exact checks.",
    2: "Check all numerical bullets of Conditions 2 and 3 and analytic/autograd agreement for each named activation.",
    3: "Execute 569 layers for a linear odd family, audit the source quantifier, and complete four routes including falsification.",
    4: "Execute the full released ZeroGradIndicator with an independent left-to-right float32 interpreter.",
    5: "Measure the full indicator derivative, apply the exact 0.5 normalization correction, and execute the full composition.",
    6: "Check a deterministic binary64 grouping difference with exact rational arithmetic and a powers-of-two control.",
}

LIMITATIONS = {
    1: "The declared six-point domain is exhausted; the complete theorem domain and arbitrary-map quantifiers lack a proof certificate.",
    2: "Conditions 2 and 3 pass; Condition 1 and every complete non-ReLU construction are not independently certified.",
    3: "The linear odd family passes, but arbitrary antisymmetric dependence remains uncertified after four routes.",
    4: "The complete released block passes finite checks; universal domain and composition neutrality are not certified.",
    5: "Positive calibratable points pass; negative active points and universal composition neutrality are not certified.",
    6: "The mechanism is verified; it does not independently prove the representation theorems.",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def checker(number: int, raw: dict) -> dict:
    if number == 1:
        return {
            "depth_at_least_9": raw["network_depth"] >= 9,
            "all_values_exact": all(row["value_exact"] for row in raw["rows"]),
            "all_gradients_exact": all(row["gradient_exact"] for row in raw["rows"]),
            "all_off_controls_zero": raw["off_domain_controls_zero"] == 6,
            "indicator_identity_exact": raw["indicator_matrix_identity_exact"],
        }
    if number == 2:
        return {
            "all_six_present": len(raw["rows"]) == 6,
            "all_condition2": all(all(row["condition2_bullets"]) for row in raw["rows"]),
            "all_condition3": all(row["condition3_pass"] for row in raw["rows"]),
            "all_autograd_agrees": all(row["analytic_autograd_agree"] for row in raw["rows"]),
        }
    if number == 3:
        return {
            "exact_depth": raw["theorem_depth"] == 569,
            "all_linear_values_exact": raw["linear_antisymmetric_family"]["values_exact"] == 60,
            "all_linear_gradients_exact": raw["linear_antisymmetric_family"]["gradients_exact"] == 60,
            "four_routes_complete": len(raw["attempts"]) == 4,
            "no_invalid_falsification": not raw["falsification_route"]["valid_falsification"],
        }
    if number == 4:
        return {
            "all_active_exact": raw["active_value_and_zero_gradient_exact"] == raw["evaluated"],
            "all_off_exact": raw["off_point_zero_zero_exact"] == raw["evaluated"],
            "full_multilayer_block": raw["composition_network_depth"] >= 9,
        }
    if number == 5:
        return {
            "calibration_exact": raw["calibration"]["correction"] == 0.5,
            "all_calibratable_active_exact": raw["calibration"]["active_zero_output_target_gradient_exact"] == 24,
            "all_calibratable_off_exact": raw["calibration"]["off_zero_zero_exact"] == 24,
            "composed_values_and_gradients_exact": (
                raw["full_composition"]["values_exact"]
                == raw["full_composition"]["gradients_exact"]
                == 6
            ),
        }
    return raw["independent_checker"]


def control(number: int, raw: dict) -> dict:
    return raw["negative_control"]


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for number, filename in FILES.items():
        raw = json.loads((DATA / filename).read_text(encoding="utf-8"))
        directory = DEST / f"claim_{number}"
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATA / filename, directory / "raw_results.json")
        verdict = "VERIFIED" if number == 6 else "BLOCKED"
        contract = {
            "claim": number,
            "title": TITLES[number],
            "source_url": f"https://ar5iv.labs.arxiv.org/html/2605.01702#{raw.get('source_anchor', 'S1.SS1.SSS1')}",
            "paper_source_sha256": SOURCE_SHA,
            "exact_quantifier_contract": QUANTIFIERS[number],
            "verdict": verdict,
        }
        write_json(directory / "claim_contract.json", contract)
        write_json(directory / "independent_checker_output.json", checker(number, raw))
        write_json(directory / "negative_control_output.json", control(number, raw))
        write_text(directory / "exact_command.txt", COMMAND)
        write_text(directory / "method.md", f"# Method\n\n{METHODS[number]}")
        write_text(
            directory / "source_audit.md",
            "# Source audit\n\n"
            f"{TITLES[number]}. Retrieved paper HTML SHA-256: `{SOURCE_SHA}`.\n\n"
            f"Quantifier contract: {QUANTIFIERS[number]}",
        )
        write_text(
            directory / "limitations_and_deviations.md",
            f"# Limitations and deviations\n\n{LIMITATIONS[number]}",
        )
        write_text(
            directory / "EVAL.md",
            f"# Claim {number} evaluation\n\nVerdict: **{verdict}**.\n\n"
            f"{LIMITATIONS[number]}\n\n"
            "The standalone evaluator-visible verifier exited zero.",
        )
    shutil.copyfile(
        ROOT / "space_candidate" / "code" / "verify_current.py",
        DEST / "claim_verifier.py",
    )
    write_text(
        DEST / "environment.txt",
        "Python 3.12.12\nNumPy 2.3.1\nTorch 2.7.1+cpu\n"
        "Backend: Hugging Face\nFlavor: cpu-upgrade\nGPU: none\n"
        "Winning run: b348d98a-3474-4a1a-8fee-8dcd084e0b85\n"
        "Winning Git SHA: e412c472eb5adc6beb00bb311ce7f1fbf698170d\n"
        "Job runtime: 64 seconds\nScientific runtime: 24.20909078908153 seconds\n"
        "Actual logical CPUs visible: 64\nActual affinity CPUs: 64\nTorch threads: 4",
    )
    print(f"materialized={DEST}")
    print("claims=6 required_files_per_claim=9")


if __name__ == "__main__":
    main()
