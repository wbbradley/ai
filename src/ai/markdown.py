from ai.colors import Color, color_code


def stress(x: int, max: int = 255) -> int:
    return max - int((max - x) / 2.5)


def markdown_to_ansi(
    text: str,
    normal_color: Color,
) -> str:
    r, g, b = normal_color.r, normal_color.g, normal_color.b
    output = color_code(r, g, b)
    state = "newline"
    star2_text = ""
    header_text = ""
    col = 0
    for ch in text:
        if state == "newline":
            if ch == "#":
                header_text += ch
                continue
            elif len(header_text) != 0:
                if ch == "\n":
                    output += color_code(stress(r), stress(g), stress(b), bold=True)
                    output += header_text
                    output += color_code(r, g, b, bold=False)
                    output += "\n"
                    header_text = ""
                else:
                    header_text += ch
                continue
            else:
                state = "normal"

        assert header_text == ""
        assert state != "newline"
        match state:
            case "normal":
                if ch == "*":
                    state = "1star"
                else:
                    output += ch
                    col += 1
            case "1star":
                if ch == "*":
                    state = "2star"
                    star2_text = ""
                else:
                    output += "*"
                    state = "normal"
            case "2star":
                if ch == "*":
                    state = "normal"
                    output += "**"
                else:
                    state = "eating-2star"
                    star2_text = ch
            case "eating-2star":
                if ch == "*":
                    state = "eating-2star-1"
                elif ch == "\n":
                    # Not in a bold run.
                    output += "**" + star2_text + ch
                    state = "normal"
                else:
                    star2_text += ch
            case "eating-2star-1":
                if ch == "*":
                    state = "normal"
                    output += color_code(stress(r), stress(g), stress(b), bold=True)
                    output += star2_text
                    output += color_code(r, g, b, bold=False)
                elif ch == "\n":
                    output += "**" + star2_text + "*" + ch
                    state = "normal"
                else:
                    star2_text += ch
    match state:
        case "1star":
            output += "*"
        case "2star":
            output += "**"
        case "eating-2star":
            output += "**" + star2_text
        case "eating-2star-1":
            output += "**" + star2_text + "*"

    output += "\001\033[0m\002"
    return output
