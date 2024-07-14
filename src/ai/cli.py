import argparse

from ai.config import Config, fetch_config
from ai.streaming import run_interactive_stream


def parse_args(config: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser("ai")
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument(
        "-p", "--provider", required=False, default=config.provider, help="Which provider to use"
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=False,
        default=None,
        help="Open a previously existing chat or preemptively choose the name of the new chat. Must be a .json file.",
    )
    args = parser.parse_args()
    if args.filename is not None and not args.filename.endswith(".json"):
        raise RuntimeError("filename must end with .json")
    return args


def main() -> None:
    config = fetch_config()
    args = parse_args(config)

    if args.filename is None:
        run_interactive_stream(config, args.filename, args.provider)


if __name__ == "__main__":
    main()
