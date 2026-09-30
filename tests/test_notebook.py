import json
import pathlib

import volsurf.data

NOTEBOOK = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "analysis.ipynb"


def test_stage_5c_skips_without_optionmetrics(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(volsurf.data, "_RAW", tmp_path)  # an empty data/raw/, as on a clean install
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    sources = [
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and cell.get("id", "").startswith("stage-5c")
    ]
    assert len(sources) >= 4

    # A fresh namespace without the Stage 5b names: any unguarded use of them raises NameError.
    namespace = {}
    for source in sources:
        exec(compile(source, "stage-5c", "exec"), namespace)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("Stage 5c skipped")
