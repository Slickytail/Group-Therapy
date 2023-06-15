import json

def safe_json_parse(string: str):
    """
    Extracts and parses a JSON blob from a string, ignoring any extra input.
    """
    decoder = json.JSONDecoder()
    # so we split on { and then grab everything after that.
    # we also use decode_raw, which allows us to ignore extra input after the json
    (decoded, _) = decoder.raw_decode("{" + string.split("{", 1)[1])
    return decoded

# Thanks stackoverflow
def flush_input():
    """
    Flushes all pending input from stdin.
    Platform-agnostic.
    """
    try:
        import msvcrt # for windows
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        import sys, termios    # for linux/unix
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)

