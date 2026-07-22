from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from amie_self_play.config import Settings, load_model_catalog  # noqa: E402
from amie_self_play.llm import ModelAPIClient  # noqa: E402
from amie_self_play.simulation import SimulationSession  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real model API self-play acceptance")
    parser.add_argument("condition", nargs="?", default="腕管综合征")
    parser.add_argument("--refinements", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--max-utterances", type=int, default=30)
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    if args.config is not None:
        settings = replace(settings, config_path=args.config.expanduser())
    catalog = load_model_catalog(settings.config_path, settings)
    client = ModelAPIClient(settings, catalog=catalog)
    visible_events = {
        "vignette_completed",
        "message_completed",
        "moderator_result",
        "ddx_completed",
        "critique_completed",
        "evaluation_completed",
        "dialogue_completed",
        "round_review_completed",
    }

    async def report(event: dict) -> None:
        if event["type"] in visible_events:
            print(json.dumps(event, ensure_ascii=False), flush=True)

    session = SimulationSession(
        client, report, max_utterances=args.max_utterances
    )
    try:
        await session.start(args.condition, args.model or catalog.default_model)
        baseline_evaluation = session.rounds[0].evaluation
        if baseline_evaluation is None or baseline_evaluation.status != "complete":
            raise RuntimeError(
                "Baseline Round did not receive a complete evaluation_completed payload"
            )
        expected_counts = {
            "patient_actor": 26,
            "specialist": 32,
            "auto_paces": 4,
        }
        for field, expected_count in expected_counts.items():
            result = getattr(baseline_evaluation, field)
            if result is None or len(result.criteria) != expected_count:
                raise RuntimeError(
                    f"Baseline {field} evaluation did not contain {expected_count} axes"
                )
        if baseline_evaluation.accuracy is None:
            raise RuntimeError("Baseline accuracy evaluation is missing")
        for _ in range(args.refinements):
            if session.rounds[-1].status != "completed":
                break
            await session.refine()
    finally:
        await client.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
