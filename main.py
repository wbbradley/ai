import argparse
import logging
import sys

from ai.config import Config, fetch_config
from ai.streaming import run_interactive_stream


def parse_args(config: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser("ai")
    parser.add_argument(
        "-p", "--provider", required=False, default=config.provider, help="Which provider to use"
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Open a previously existing chat or preemptively choose the name of the new chat. Must be a .json file.",
    )
    args = parser.parse_args()
    if args.filename is not None and (not args.filename.endswith(".json") and not args.filename == "-"):
        raise RuntimeError(f"filename must end with .json [filename={args.filename}]")
    return args


def main() -> None:
    try:
        config = fetch_config()
        if config.log_filename:
            logging.basicConfig(
                filename=config.log_filename,
                level=config.log_level_int,
                filemode="a",
            )
        args = parse_args(config)

        if args.filename == "-":
            for line in sys.stdin.readlines():
                print(f"you said: {line}")
        else:
            run_interactive_stream(config, args.filename, args.provider)
    except RuntimeError as e:
        sys.exit(f"ai: {e}")


if __name__ == "__main__":
    main()
