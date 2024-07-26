import argparse
import glob
import json
import logging
import os
import sys
import time
from typing import Optional

from ai.colors import colorize
from ai.config import Config, fetch_config
from ai.embedded import parse_embedded_buffer
from ai.output import set_quiet_mode
from ai.streaming import run_interactive_stream, stream_document_response


def parse_args(config: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser("ai")
    parser.add_argument(
        "-p", "--provider", required=False, default=config.provider, help="Which provider to use"
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        required=False,
        action="store_true",
        default=False,
        help="Do a dry run.",
    )
    parser.add_argument(
        "-e",
        "--embedded",
        required=False,
        action="store_true",
        default=False,
        help="Process a conversation from stdin, and writing a new response to stdout.",
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Open a previously existing chat or preemptively choose the name of the new chat. Must be a "
        ".json file. Use '-' to open the most recently modified file in your configured report_dir.",
    )
    args = parser.parse_args()
    if args.filename is not None and (
        not args.filename.endswith(".json") and not args.filename == "-" and not args.filename.endswith(".ai")
    ):
        raise RuntimeError(f"filename must end with .json [filename={args.filename}]")
    return args


def most_recent_json_file(directory: str) -> Optional[str]:
    files = glob.glob(os.path.join(directory, "*.json"))
    if not files:
        print(f"note: No prior files found in {colorize(directory)}", file=sys.stderr)
        return None
    return max(files, key=os.path.getmtime)


def main() -> None:
    if "--embedded" in sys.argv:
        set_quiet_mode()

    try:
        config = fetch_config()
        if config.log_filename:
            logging.basicConfig(
                filename=config.log_filename,
                level=config.log_level_int,
                filemode="a",
            )
        args = parse_args(config)

        if args.embedded:
            logging.info({"mode": "embedded", "filename": args.filename, "timestamp": time.time()})
            document = parse_embedded_buffer(config, open(args.filename, "r") if args.filename else sys.stdin)
            if args.dry_run:
                print(json.dumps({"messages": document.messages}, indent=2))
                return
            stream_document_response(config, document)
            return

        if args.filename == "-":
            # Reopen the most recent convo.
            filename = most_recent_json_file(config.report_dir)
            if filename is not None:
                print(f"\rRe-opening {colorize(filename)}...")
        else:
            filename = args.filename

        logging.info(
            {"mode": "interactive", "filename": filename, "provider": args.provider, "timestamp": time.time()}
        )
        run_interactive_stream(config, filename, args.provider)
    except RuntimeError as e:
        sys.exit(f"ai: {e}")


if __name__ == "__main__":
    main()
