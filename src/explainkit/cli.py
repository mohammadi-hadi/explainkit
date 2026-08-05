"""Command line interface: run the built-in demo, refresh the README table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from explainkit import demo


def _cmd_demo(args: argparse.Namespace) -> int:
    stats = demo.write_demo(Path(args.out))
    print(
        f"demo report written to {Path(args.out) / 'report.md'} "
        f"({len(stats.selected)} tokens selected at T_c = 0.95)"
    )
    return 0


def _cmd_inject_readme(args: argparse.Namespace) -> int:
    demo.inject_readme(Path(args.readme), Path(args.results))
    print(f"updated {args.readme} from {args.results}/summary.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="explainkit", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    demo_cmd = commands.add_parser("demo", help="explain, select, replace and score")
    demo_cmd.add_argument("--out", default="results", help="output directory")
    demo_cmd.set_defaults(func=_cmd_demo)

    injector = commands.add_parser("inject-readme", help="refresh the README demo table")
    injector.add_argument("readme")
    injector.add_argument("--results", default="results", help="directory with summary.md")
    injector.set_defaults(func=_cmd_inject_readme)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
