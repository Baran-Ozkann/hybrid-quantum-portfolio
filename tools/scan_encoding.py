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

    print("=" * 74)
    print(f"ENCODING SCAN -- {scanned} text files")
    print("=" * 74)
    if problems:
        for p in problems:
            print(f"  {p}")
        print(f"\n{len(problems)} problem(s) found.")
        return 1
    print("  No mojibake, no stray BOMs, all files valid UTF-8.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
