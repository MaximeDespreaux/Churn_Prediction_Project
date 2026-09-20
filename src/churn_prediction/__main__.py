"""Single entry point for the pipeline: ``churn <command> [options]``.

Examples:
    churn build-features
    churn evaluate
    churn train
    churn predict --submission reports/submissions/submission_ensemble.csv
    churn tune --n-trials 50 --write

Equivalent to ``python -m churn_prediction <command>``. Every command accepts
``--help``.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "build-features": (
        "churn_prediction.build_features",
        "raw event logs -> feature tables",
    ),
    "evaluate": (
        "churn_prediction.evaluate",
        "hold-out evaluation of the saved parameters",
    ),
    "train": ("churn_prediction.train", "fit the ensemble with the saved parameters"),
    "predict": ("churn_prediction.predict", "score users with the fitted ensemble"),
    "tune": ("churn_prediction.tune", "Optuna hyper-parameter search"),
}


def usage() -> str:
    lines = ["usage: churn <command> [options]", "", "commands:"]
    lines += [f"  {name:<16} {help_}" for name, (_, help_) in COMMANDS.items()]
    lines += ["", "Run `churn <command> --help` for the options of a command."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(usage())
        return 0
    command, *rest = argv
    if command not in COMMANDS:
        print(f"churn: unknown command '{command}'\n\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[command][0])
    module.main(rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
