"""
tools/scan_encoding.py -- repo-wide scan for mojibake and stray BOMs.

Motivated by a real incident: using PowerShell's ``Get-Content -Raw`` /
``Set-Content`` to regex-edit a UTF-8 source file read it as ANSI and wrote it
back with 29 mojibake sequences and a BOM. The file still *ran* -- the damage
was confined to string literals -- which is exactly what made it dangerous.

This scans every text file for:
  * mojibake signatures (UTF-8 bytes that were decoded as cp1252 and re-encoded)
  * UTF-8 BOMs, which Python tolerates but which break naive readers and
    produce spurious diffs
  * bytes that are not valid UTF-8 at all

Run::

    python tools/scan_encoding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEXT_SUFFIXES = {".py", ".qs", ".md", ".json", ".txt", ".csv", ".toml", ".cfg",
                 ".ini", ".yml", ".yaml", ".gitignore"}
SKIP_DIRS = {".python", ".git", "__pycache__", "ms-playwright", ".venv", "venv"}

#: Signatures of the cp1252-read/UTF-8-write round trip. Each is what a common
#: non-ASCII character becomes after that mangling.
MOJIBAKE = (
    "âŸ¨", "âŸ©",   # angle brackets, from mathematical notation
    "Ã—",          # multiplication sign
    "â€”", "â€“",  # em dash, en dash
    "â€™", "â€œ", "â€", # curly quotes
    "Â°", "Â±", "Â»", "Â«", "Â ",
    "Îµ", "Î»", "Î¸", "Î£", "Î¼",  # Greek letters
    "â‰¥", "â‰¤", "â†’",
)


#: Emoji and pictograph ranges. This project uses no emoji anywhere -- not in
#: documents, code comments, figure titles, or commit messages -- and this gate
#: is what keeps it that way.
#:
#: The ranges are deliberately narrow. The project relies on a lot of non-ASCII
#: that is NOT emoji (arrows, Greek letters, angle brackets for <H>, proper
#: minus and multiplication signs, dagger footnotes), and a check that flagged
#: those would be muted within a week.
PICTOGRAPH_RANGES = (
    (0x1F000, 0x1FAFF, "pictographs / emoticons"),
    (0x2600, 0x26FF, "miscellaneous symbols"),
    (0x2700, 0x27BF, "dingbats"),
    (0x2B00, 0x2BFF, "misc symbols and arrows"),
    (0x1F1E6, 0x1F1FF, "regional indicator"),
    (0xFE0F, 0xFE0F, "variation selector-16"),
    (0x200D, 0x200D, "zero-width joiner"),
    (0x2049, 0x2049, "exclamation question"),
    (0x203C, 0x203C, "double exclamation"),
    (0x2122, 0x2122, "trade mark"),
    (0x2139, 0x2139, "information"),
)

#: Codepoints inside those ranges that are legitimate here. Keep this list
#: short and justified -- every entry is a hole in the gate.
PICTOGRAPH_ALLOWED = frozenset({
    0x2713,   # check mark, used in a couple of gate tables
})


def pictographs_in(text: str) -> list[tuple[int, int, str, str]]:
    """Return ``(line, codepoint, name, category)`` for each emoji found."""
    import unicodedata
    found = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for ch in line:
            cp = ord(ch)
            if cp in PICTOGRAPH_ALLOWED:
                continue
            for lo, hi, label in PICTOGRAPH_RANGES:
                if lo <= cp <= hi:
                    try:
                        name = unicodedata.name(ch)
                    except ValueError:
                        name = "<unnamed>"
                    found.append((lineno, cp, name, label))
                    break
    return found


def commit_messages() -> str | None:
    """Every commit message in the repository, or None if git is unavailable.

    Commit messages are scanned too: an emoji there is exactly as permanent as
    one in a file, and cannot be fixed without rewriting history.
    """
    import subprocess
    try:
        r = subprocess.run(
            ["git", "log", "--format=%H%n%B"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return None
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    problems: list[str] = []
    scanned = 0
    self_path = Path(__file__).resolve()

    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        # This file necessarily contains every signature it searches for.
        if path.resolve() == self_path:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue

        raw = path.read_bytes()
        rel = path.relative_to(PROJECT_ROOT)
        scanned += 1

        if raw.startswith(b"\xef\xbb\xbf"):
            problems.append(f"BOM         {rel}")

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            problems.append(f"not UTF-8   {rel}  ({exc})")
            continue

        hits = {m: text.count(m) for m in MOJIBAKE if m in text}
        if hits:
            detail = ", ".join(f"{m!r}x{n}" for m, n in sorted(hits.items()))
            problems.append(f"mojibake    {rel}  ({detail})")

        for lineno, cp, name, label in pictographs_in(text):
            problems.append(
                f"emoji       {rel}:{lineno}  U+{cp:04X} {name} ({label})"
            )

    msgs = commit_messages()
    n_commits = 0
    if msgs is None:
        problems.append("emoji       could not read git log; commit messages unscanned")
    else:
        n_commits = msgs.count("\n") and len([
            l for l in msgs.splitlines() if len(l) == 40 and all(
                c in "0123456789abcdef" for c in l)
        ])
        for _, cp, name, label in pictographs_in(msgs):
            problems.append(
                f"emoji       <commit message>  U+{cp:04X} {name} ({label})"
            )

    print("=" * 74)
    print(f"ENCODING SCAN -- {scanned} text files, {n_commits} commit messages")
    print("=" * 74)
    if problems:
        for p in problems:
            print(f"  {p}")
        print(f"\n{len(problems)} problem(s) found.")
        return 1
    print("  No mojibake, no stray BOMs, all files valid UTF-8.")
    print("  No emoji in files or commit messages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
