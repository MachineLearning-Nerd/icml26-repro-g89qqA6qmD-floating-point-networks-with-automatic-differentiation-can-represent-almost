"""Evaluator-blind traversal and visibility audit for the candidate Space."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SPACE_PREFIX = "https://huggingface.co/spaces/DineshAI/g89qqA6qmD/blob/main/"
WINNING_RUN = "b348d98a-3474-4a1a-8fee-8dcd084e0b85"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--require-winning-run", action="store_true")
    args = parser.parse_args()
    root = args.candidate.resolve()
    opened: list[str] = []
    problems: list[str] = []

    def read(relative: str) -> str:
        path = root / relative
        if not path.is_file():
            problems.append(f"missing file: {relative}")
            return ""
        opened.append(relative)
        return path.read_text(encoding="utf-8")

    read("README.md")
    logbook_text = read("logbook.json")
    try:
        logbook = json.loads(logbook_text)
    except json.JSONDecodeError as exc:
        problems.append(f"invalid logbook.json: {exc}")
        logbook = {}

    nodes = {}
    root_node = logbook.get("root", {})
    if root_node:
        nodes[root_node.get("slug")] = root_node.get("file")
    for node in root_node.get("children", []):
        nodes[node.get("slug")] = node.get("file")

    queue = ["pages/index.md"]
    seen: set[str] = set()
    reachable_slugs: set[str] = set()
    while queue:
        relative = queue.pop(0)
        if relative in seen:
            continue
        seen.add(relative)
        text = read(relative)
        for slug in re.findall(r"\]\(#/([a-z0-9-]+)\)", text):
            reachable_slugs.add(slug)
            target = nodes.get(slug)
            if target is None:
                problems.append(f"unmapped slug: {slug}")
            else:
                queue.append(target)
        for target in re.findall(
            re.escape(SPACE_PREFIX) + r"([A-Za-z0-9_./-]+)", text
        ):
            target = target.rstrip(").,")
            if (root / target).is_file():
                read(target)
            else:
                problems.append(f"broken evaluator-visible file link: {target}")

    for number in range(1, 7):
        slug = f"claim-{number}-current"
        if slug not in reachable_slugs:
            problems.append(f"claim {number} canonical page not reachable")
            continue
        page = read(nodes[slug])
        required_tokens = [
            "Exact contract",
            "Raw JSON",
            "verifier",
            "control",
            "Fixed command",
            "Seed",
            "Git",
            "cpu-upgrade",
        ]
        for token in required_tokens:
            if token.lower() not in page.lower():
                problems.append(f"claim {number} page missing token: {token}")
        verdict = "VERIFIED" if number == 6 else "BLOCKED"
        if verdict not in page:
            problems.append(f"claim {number} page missing verdict {verdict}")

    matrix = read("pages/current-verification/page.md")
    if "missing" in matrix.lower():
        problems.append("visibility matrix contains a missing cell")
    if args.require_winning_run and WINNING_RUN not in matrix:
        problems.append("canonical cumulative page does not cite the winning run")

    for historical in (
        "pages/historical-index/page.md",
        "pages/overview/page.md",
        "pages/verify/page.md",
    ):
        if historical not in opened:
            problems.append(f"historical evidence not reachable: {historical}")

    result = {
        "canonical_entrypoints": ["README.md", "logbook.json", "pages/index.md"],
        "opened_files": sorted(set(opened)),
        "reachable_slugs": sorted(reachable_slugs),
        "problems": problems,
        "verdict": "PASS" if not problems else "FAIL",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
