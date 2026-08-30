"""
tools/install_hooks.py -- install the tracked git hooks into this clone.

Git hooks live in .git/hooks/, which is not part of the repository and is
never populated by `git clone`. A hook checked in under tools/hooks/ has no
effect until something copies it there -- that is what this script does.

Why this exists: tools/scan_encoding.py (the encoding gate) has caught
mojibake in this repo before, but only when someone remembered to run it by
hand. The commit that introduced 87 lines of mojibake into README.md was not
checked against it. Wiring the gate into a pre-commit hook removes the
"remember to run it" step; this script wires it in.

Run once per clone::

    python tools/install_hooks.py
"""

from __future__ import annotations

import shutil
import stat
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOOKS_SRC = PROJECT_ROOT / "tools" / "hooks"
HOOKS_DST = PROJECT_ROOT / ".git" / "hooks"


def main() -> int:
    if not HOOKS_DST.parent.exists():
        print(f"error: {PROJECT_ROOT / '.git'} not found -- is this a git checkout?")
        return 1

    if not HOOKS_SRC.is_dir():
        print(f"error: {HOOKS_SRC} not found")
        return 1

    installed = []
    for src in sorted(HOOKS_SRC.iterdir()):
        if not src.is_file():
            continue
        dst = HOOKS_DST / src.name
        shutil.copyfile(src, dst)
        dst.chmod(dst.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        installed.append(src.name)

    if not installed:
        print(f"no hooks found in {HOOKS_SRC}")
        return 1

    print(f"installed hook(s): {', '.join(installed)}")
    print(f"  {HOOKS_SRC} -> {HOOKS_DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
