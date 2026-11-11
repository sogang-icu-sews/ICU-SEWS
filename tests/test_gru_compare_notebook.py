"""Exercise the actual Colab runner without installing or mounting anything."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_notebook_runner_preserves_child_error(tmp_path, capsys):
    notebook = Path(__file__).resolve().parents[1] / "notebooks/ICU_SEWS_GRU_Compare.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    runner = None
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                runner = ast.Module(body=[node], type_ignores=[])
    assert runner is not None
    namespace = {"subprocess": subprocess, "Path": Path}
    exec(compile(runner, str(notebook), "exec"), namespace)
    log = tmp_path / "logs" / "failure.log"
    with pytest.raises(RuntimeError, match="ValueError: diagnostic sentinel"):
        namespace["run"](
            [
                sys.executable,
                "-u",
                "-c",
                "print('started'); raise ValueError('diagnostic sentinel')",
            ],
            log_path=log,
        )
    assert "ValueError: diagnostic sentinel" in capsys.readouterr().out
    assert "started" in log.read_text(encoding="utf-8")
    assert "ValueError: diagnostic sentinel" in log.read_text(encoding="utf-8")
    result = namespace["run"]([sys.executable, "-c", "print('success')"])
    assert result.returncode == 0
