import colorsys
import hashlib
from contextlib import contextmanager
from typing import Generator, Optional, Tuple


@contextmanager
def colored_output(r: int, g: int, b: int) -> Generator[None, None, None]:
    try:
        set_tty_color(r, g, b)
        yield
    finally:
        reset_tty()


def hash_to_color(input_string: str, saturation: float = 0.8, value: float = 0.8) -> Tuple[int, int, int]:
    # Generate a hash of the input string
    hash_object = hashlib.md5(input_string.encode())
    hash_hex: str = hash_object.hexdigest()

    # Convert the first 8 characters of the hash to an integer
    hash_int: int = int(hash_hex[:8], 16)

    # Map the integer to a hue value between 0 and 1
    hue: float = hash_int / 0xFFFFFFFF

    # Create HSV color with constant saturation and value
    hsv_color = (hue, saturation, value)

    # Convert HSV to RGB
    rgb_color = colorsys.hsv_to_rgb(*hsv_color)

    # Scale RGB values to 0-255 range and round to integers
    rgb_255: Tuple[int, int, int] = (
        round(rgb_color[0] * 255),
        round(rgb_color[1] * 255),
        round(rgb_color[2] * 255),
    )

    return rgb_255


def colorize(text: str, r: Optional[int] = None, g: Optional[int] = None, b: Optional[int] = None) -> str:
    if r is None or g is None or b is None:
        r, g, b = hash_to_color(text)
    return f"\001\033[38;2;{r};{g};{b}m\002{text}\001\033[0m\002"


def set_tty_color(r: int, g: int, b: int) -> None:
    print("\001\033[38;2;{};{};{}m\002".format(r, g, b))


def reset_tty() -> None:
    print("\001\033[0m\002")


def erase_line() -> None:
    from ai.output import get_quiet_mode

    if get_quiet_mode():
        return None
    print("\r\33[2K\r", end="")
