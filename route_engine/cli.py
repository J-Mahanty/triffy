"""Terminal chat client.

This exists for a practical reason: a Discord bot needs a token, a registered
application and a working network, and demos happen in rooms where one of those
three things is missing. The CLI drives the identical ``ChatBrain``, so it is a
faithful stand-in for the bot rather than a reduced version of it.
"""
from __future__ import annotations

import argparse
import re
import sys

from .chat import ChatBrain
from .engine import TriffyEngine

# Minimal ANSI styling. Discord markdown does not survive in a terminal, so we
# translate the few markers the brain emits.
BOLD, DIM, CYAN, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[0m"

# Windows consoles still default to cp1252, which cannot encode the emoji and
# typographic characters the chat brain emits for Discord. Rather than stripping
# them everywhere and making Discord uglier, we transliterate at the terminal
# boundary only.
ASCII_MAP = {
    "\U0001f6a6": "[route]", "\U0001f552": "[time]", "\U0001f4a1": "[why]",
    "⚠": "!", "•": "-", "—": "-", "–": "-",
    "→": "->", "…": "...", "▸": ">", "·": "|",
    "‘": "'", "’": "'", "“": '"', "”": '"',
}


def _terminal_is_unicode_safe() -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "\U0001f6a6•".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def to_ascii(text: str) -> str:
    for k, v in ASCII_MAP.items():
        text = text.replace(k, v)
    return text.encode("ascii", "replace").decode("ascii")


def render(text: str, colour: bool = True, ascii_only: bool = False) -> str:
    if ascii_only:
        text = to_ascii(text)
    if not colour:
        return re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", BOLD + r"\1" + RESET, text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", DIM + r"\1" + RESET, text)
    return text


BANNER = r"""
  ______    _  __  __ _
 /_  __/___(_)/ _|/ _(_)___
  / / / __/ / |_ | |_| / _ \    live traffic intelligence
 /_/ /_/ /_/|__/ |_| |_\___/    type 'help' or 'quit'
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Triffy terminal client")
    ap.add_argument("--user", default="guest")
    ap.add_argument("--clock", default="18:30")
    ap.add_argument("--replay", default="", metavar="'YYYY-MM-DD HH:MM'",
                    help="with --live: replay recorded London traffic from this "
                         "moment (London time) instead of live readings")
    ap.add_argument("--live", action="store_true",
                    help="answer from live London cameras instead of simulated "
                         "Kolkata (needs the collector to have run)")
    ap.add_argument("--no-colour", action="store_true")
    ap.add_argument("--ascii", action="store_true",
                    help="force plain ASCII output (auto-detected otherwise)")
    ap.add_argument("message", nargs="*", help="run one message and exit")
    args = ap.parse_args()

    # Prefer real UTF-8 if the stream will take it; fall back to transliteration.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ascii_only = args.ascii or not _terminal_is_unicode_safe()

    colour = not args.no_colour
    if args.live:
        # Same brain, real data. LiveChatEngine gives LiveEngine the surface
        # ChatBrain expects, so there is no second parser to drift.
        from .live_chat import LiveChatEngine
        from .live_engine import LiveEngine
        from .live_engine import parse_replay
        live = LiveEngine(city="lon", replay_at=parse_replay(args.replay))
        eng = LiveChatEngine(live)
        banner = ("Network: **%s** | London time **%s** | %d of %d real cameras "
                  "reporting\n"
                  % (eng.net.meta.get("label", ""), eng.clock,
                     len(live.observations), len(live.mapped)))
    else:
        eng = TriffyEngine(start_clock=args.clock)
        banner = ("Network: **%s** | clock **%s** | %d cameras simulated\n"
                  % (eng.net.meta.get("label", ""), eng.clock,
                     len(eng.cams.cams)))
    brain = ChatBrain(eng)

    if args.message:
        print(render(brain.handle(" ".join(args.message), args.user),
                     colour, ascii_only))
        return

    print(CYAN + BANNER + RESET if colour else BANNER)
    # Built above, because live and simulated report different things: London
    # counts cameras that are actually reporting, Kolkata counts simulated ones.
    # The live engine has no .cams at all, so a shared line would crash it.
    print(render(banner, colour, ascii_only))

    while True:
        try:
            msg = input(CYAN + "you ▸ " + RESET if colour else "you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in ("quit", "exit", "q"):
            break
        if not msg:
            continue
        print()
        print(render(brain.handle(msg, args.user), colour, ascii_only))
        print()


if __name__ == "__main__":
    main()
