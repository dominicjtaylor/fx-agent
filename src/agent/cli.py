"""
cli.py
FRONTIER CLI — terminal interface for the FX feature discovery system.

Commands:
  frontier run --cycles N [--hint TEXT] [--data-dir PATH]
  frontier status
  frontier inspect <name>
  frontier chat ["<question>"]
  frontier init
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ── ANSI colour helpers ───────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"

def green(s: str) -> str: return f"{GREEN}{s}{RESET}"
def red(s: str) -> str: return f"{RED}{s}{RESET}"
def yellow(s: str) -> str: return f"{YELLOW}{s}{RESET}"
def cyan(s: str) -> str: return f"{CYAN}{s}{RESET}"
def bold(s: str) -> str: return f"{BOLD}{s}{RESET}"
def dim(s: str) -> str: return f"{DIM}{s}{RESET}"

VERDICT_COLOUR = {
    "promoted": green,
    "rejected": red,
    "modified": yellow,
}

def colour_verdict(v: str) -> str:
    return VERDICT_COLOUR.get(v, lambda x: x)(v.upper())


# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "frontier_config.yaml"


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_PATH.exists():
        import yaml
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(cfg: dict) -> None:
    import yaml
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(data_dir: Optional[str] = None) -> dict:
    """
    Load time-series data for configured instruments.
    Returns: {instrument: DataFrame}
    """
    import pandas as pd

    _ensure_repo_on_path()
    from src.agent.context import load_context

    context = load_context()
    instruments = context["data"].get("pairs", [])

    cfg = load_config()
    base_dir = Path(data_dir or cfg.get("data_dir", REPO_ROOT / "data"))

    data = {}

    for inst in instruments:
        path = base_dir / f"{inst.lower()}.csv"

        if not path.exists():
            print(red(f"Missing data for {inst}: {path}"))
            print(dim("Run 'frontier init' to configure paths."))
            sys.exit(1)

        df = pd.read_csv(path, parse_dates=True, index_col=0)
        data[inst] = df
        print(dim(f"  Loaded {inst}: {len(df):,} rows"))

    return data


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_run(args):
    _ensure_repo_on_path()
    from src.agent.loop import run_cycle

    print(bold(f"\nFRONTIER — running {args.cycles} cycle(s)\n"))

    data = load_data(args.data_dir)

    for i in range(args.cycles):
        if args.cycles > 1:
            print(bold(f"── Cycle {i+1}/{args.cycles} ──"))

        entry = run_cycle(data, user_hint=args.hint)

        print()
        print(f"{bold(entry.get('name','?'))} [{colour_verdict(entry.get('verdict','?'))}]")
        print(entry.get("summary", ""))
        print()


def cmd_status(args):
    _ensure_repo_on_path()
    from src.agent.registry import load_registry
    from src.agent.active_features import load_active_features

    registry = load_registry()
    active = load_active_features()

    print(bold("\nFRONTIER Status"))
    print("─" * 40)

    print(f"Cycles:   {len(registry)}")
    print(f"Active:   {len(active)}")

    print()
    print(bold("Recent"))
    print("─" * 40)

    for e in reversed(registry[-5:]):
        print(f"{colour_verdict(e['verdict']):15s} {e['name']}")


def cmd_inspect(args):
    _ensure_repo_on_path()
    from src.agent.registry import load_registry

    registry = load_registry()
    matches = [e for e in registry if args.name.lower() in e["name"].lower()]

    if not matches:
        print(red("No match found"))
        return

    e = matches[0]

    print()
    print(bold(e["name"]), colour_verdict(e["verdict"]))
    print("─" * 50)

    for k in ["hypothesis", "construction", "summary"]:
        print(f"\n{bold(k.capitalize())}")
        print(e.get(k, ""))


def cmd_chat(args):
    _ensure_repo_on_path()
    import anthropic

    from src.agent.registry import load_registry
    from src.agent.context import load_context

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print(red("Missing ANTHROPIC_API_KEY"))
        return

    registry = load_registry()
    context = load_context()

    client = anthropic.Anthropic(api_key=key)

    prompt = f"""
You are a quantitative research assistant helping guide feature discovery.

Registry size: {len(registry)}
Instruments: {context['data'].get('pairs', [])}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=prompt,
        messages=[{"role": "user", "content": args.question or "What should I explore next?"}]
    )

    print("\n" + response.content[0].text + "\n")


def cmd_init(args):
    print(bold("\nFRONTIER Init\n"))

    cfg = load_config()

    data_dir = input("Data directory: ").strip()
    if data_dir:
        cfg["data_dir"] = data_dir

    save_config(cfg)
    print(green("Saved config\n"))


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="frontier")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("run")
    p.add_argument("-n", "--cycles", type=int, default=1)
    p.add_argument("--hint")

    sub.add_parser("status")

    p = sub.add_parser("inspect")
    p.add_argument("name")

    p = sub.add_parser("chat")
    p.add_argument("question", nargs="?")

    sub.add_parser("init")

    args = parser.parse_args()

    {
        "run": cmd_run,
        "status": cmd_status,
        "inspect": cmd_inspect,
        "chat": cmd_chat,
        "init": cmd_init,
    }.get(args.cmd, lambda x: parser.print_help())(args)


if __name__ == "__main__":
    main()