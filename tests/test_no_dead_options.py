"""Guard against accepted-but-ignored options.

A flag or config field that is accepted and then never read is worse than no flag at
all: someone sets it, believes they are running an ablation, and gets baseline
behaviour. This happened with muon_split_qkv / muon_split_gate_up, which existed on
the Muon constructor but had no implementation.

These checks are deliberately cheap and static (AST + source scan) so they run on
every test invocation.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _python_files():
    return sorted(SRC.glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))


def test_no_unused_function_parameters():
    """Every non-underscore, non-conventional parameter must be referenced in its body.

    Exceptions are the signatures Python itself imposes (signal handlers, autograd
    backwards' unused slots), which are named or underscore-prefixed by convention.
    """
    ignored = {
        "self", "cls", "ctx", "args", "kwargs",
        "sig", "frame",                    # signal.signal handler signature
        "b", "h",                          # flex_attention mask_mod(b, h, q_idx, kv_idx)
    }
    offenders = []
    for path in _python_files():
        text = path.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = node.args
            params = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
            if not params:
                continue
            body = ast.get_source_segment(text, node) or ""
            # drop the signature (everything before the first ':')
            body = body.split(":", 1)[1] if ":" in body else body
            for p in params:
                if p in ignored or p.startswith("_"):
                    continue
                if not re.search(rf"\b{re.escape(p)}\b", body):
                    offenders.append(f"{path.name}:{node.lineno} {node.name}() arg '{p}'")
    assert not offenders, "parameters accepted but never used:\n  " + "\n  ".join(offenders)


def test_every_cli_argument_is_read():
    """Every argparse option must be consumed as args.<dest>."""
    offenders = []
    for path in [ROOT / "src" / "train.py", ROOT / "monitor" / "web_server.py",
                 ROOT / "data" / "prepare_shards.py", *_python_files()]:
        if not path.exists():
            continue
        text = path.read_text()
        for option in re.findall(r'add_argument\(\s*"--([a-z0-9-]+)"', text):
            dest = option.replace("-", "_")
            if not re.search(rf"args\.{dest}\b", text):
                offenders.append(f"{path.name}: --{option} (args.{dest} never read)")
    assert not offenders, "CLI options accepted but never read:\n  " + "\n  ".join(offenders)


def test_every_config_field_is_read():
    """Every PulsarConfig field must be read somewhere outside the dataclass body."""
    import sys
    sys.path.insert(0, str(ROOT))
    from dataclasses import fields

    from src.model import PulsarConfig

    model_src = (SRC / "model.py").read_text()
    start = model_src.index("class PulsarConfig")
    end = model_src.index("    def to_dict", start)
    # source with the dataclass body removed, plus every other module
    outside = model_src[:start] + model_src[end:]
    outside += "\n" + "\n".join(
        f.read_text() for f in [*SRC.glob("*.py"), *pathlib.Path(ROOT, "scripts").glob("*.py"),
                                pathlib.Path(ROOT, "monitor", "web_server.py")]
        if f.exists() and f.name != "model.py"
    )

    unused = []
    for f in fields(PulsarConfig):
        if not re.search(rf"\b{f.name}\b", outside):
            unused.append(f.name)
    assert not unused, f"config fields never read: {unused}"


def test_no_known_dead_flags():
    """Regression guard for the specific flags that were accepted and ignored."""
    dead = ["split_qkv", "split_gate_up", "muon_split_qkv", "muon_split_gate_up"]
    found = []
    for path in _python_files():
        text = path.read_text()
        for name in dead:
            if name in text:
                found.append(f"{path.name}: {name}")
    assert not found, (
        "these flags were removed because they had no implementation; if you re-add "
        "them, implement and test the behaviour in the same change:\n  " + "\n  ".join(found)
    )
