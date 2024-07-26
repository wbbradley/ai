from ai.colors import colorize

quiet_mode = False


def set_quiet_mode() -> None:
    global quiet_mode
    quiet_mode = True


def get_quiet_mode() -> bool:
    return quiet_mode


def note(text: str, end: str = "\n") -> None:
    if quiet_mode:
        return
    print(f"ai: {colorize("note")}: {text}", end=end)
