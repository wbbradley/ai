import argparse

from ai.config import fetch_config
from ai.streaming import compressed_time, run_interactive_stream, run_single_query_stream


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
        help="Where to write the raw generated content",
    )
    parser.add_argument(
        "-r",
        "--report-filename",
        required=False,
        default=None,
        help="Where to write the report of the entire run.",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    config = fetch_config()
    args = parse_args(config)
    slug = str(compressed_time())

    report_filename = args.report_filename or f"ai-{slug}.json"

    if args.query is None:
        run_interactive_stream(config, report_filename, args.filename, args.model)
    else:
        run_single_query_stream(config, args.query, report_filename, args.filename, args.model)


if __name__ == "__main__":
    main()
