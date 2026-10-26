"""Extract metadata from an explicitly trusted installed Manim distribution."""

from __future__ import annotations

import argparse
import inspect
import pprint
import re
from pathlib import Path


def extract(module) -> dict:
    result = {}
    exported = getattr(
        module, "__all__", [name for name in vars(module) if not name.startswith("_")]
    )
    for name in sorted(exported):
        value = getattr(module, name)
        owner = getattr(value, "__module__", "")
        # Include constants and classes actually reexported by the installed package.
        if inspect.ismodule(value) or (callable(value) and not owner.startswith("manim")):
            continue
        try:
            signature = (
                re.sub(r" at 0x[0-9a-fA-F]+", "", str(inspect.signature(value)))
                if callable(value)
                else None
            )
        except (TypeError, ValueError):
            signature = None
        result[name] = {
            "kind": "class"
            if inspect.isclass(value)
            else "callable"
            if callable(value)
            else "constant",
            "module": owner,
            "signature": signature,
        }
        if inspect.isclass(value):
            result[name]["bases"] = [base.__name__ for base in value.__mro__[1:]]
    return {"schema_version": 1, "version": module.__version__, "symbols": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import manim

    payload = extract(manim)
    args.output.write_text(
        '"""Generated metadata from trusted Manim; contains no upstream source code."""\n\nCATALOG = '
        + pprint.pformat(payload, sort_dicts=True, width=100)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
