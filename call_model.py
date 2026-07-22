from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from amie_self_play.config import Settings, load_model_catalog
from amie_self_play.llm import ModelAPIClient
from amie_self_play.prompt_config import load_prompt_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call one model from the configured AMIE model API catalog"
    )
    parser.add_argument("prompt", nargs="?", default="请用一句话介绍你自己。")
    parser.add_argument("--model", default=None, help="配置文件中的模型 name")
    parser.add_argument("--config", type=Path, default=None, help="模型 API JSON 配置路径")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    return parser.parse_args()


async def call_model(args: argparse.Namespace) -> str:
    settings = Settings.from_env()
    if args.config is not None:
        settings = replace(settings, config_path=args.config.expanduser())
    catalog = load_model_catalog(settings.config_path, settings)
    prompts = load_prompt_catalog(settings.prompt_config_path)
    client = ModelAPIClient(settings, catalog=catalog)
    try:
        return await client.complete(
            [
                {"role": "system", "content": prompts.cli_system},
                {"role": "user", "content": args.prompt},
            ],
            model_name=args.model or catalog.default_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    finally:
        await client.aclose()


def main() -> int:
    print(asyncio.run(call_model(parse_args())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
