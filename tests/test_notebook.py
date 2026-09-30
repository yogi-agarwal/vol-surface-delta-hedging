import json
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

import volsurf.data

NOTEBOOK = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "analysis.ipynb"
LONG_DECIMAL = re.compile(r"(?<![\w.])[-+]?\d*\.\d{5,}(?:[eE][-+]?\d+)?")  # 5 or more decimals
# Calls that would show OptionMetrics rows: the frame itself, or its first rows or full text.
ROW_DISPLAY = re.compile(r"\b(?:display|print)\(\s*om\b|\.head\(|\.to_string\(")


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _stage_5c_sources():
    """Sources of the Stage 5c code cells, keyed by cell id."""
    return {
        cell["id"]: "".join(cell["source"])
        for cell in _cells()
        if cell["cell_type"] == "code" and cell.get("id", "").startswith("stage-5c")
    }


def _output_texts(cell):
    """Text of a code cell's outputs: streams and text/* data (plain, HTML), never images."""
    for output in cell.get("outputs", []):
        if "text" in output:
            yield "".join(output["text"])
        for mime, data in output.get("data", {}).items():
            if mime.startswith("text/"):
                yield "".join(data)
        if output.get("output_type") == "error":
            yield "\n".join(output.get("traceback", []))


def test_stage_5c_skips_without_optionmetrics(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(volsurf.data, "_RAW", tmp_path)  # an empty data/raw/, as on a clean install
    sources = list(_stage_5c_sources().values())
    assert len(sources) >= 4

    # A fresh namespace without the Stage 5b names: any unguarded use of them raises NameError.
    namespace = {}
    for source in sources:
        exec(compile(source, "stage-5c", "exec"), namespace)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("Stage 5c skipped")


def test_stage_5c_cells_never_show_optionmetrics_rows():
    # Static licensing guard on the code itself, so it holds with or without the files.
    sources = _stage_5c_sources()
    assert len(sources) >= 4
    offending = [cell_id for cell_id, source in sources.items() if ROW_DISPLAY.search(source)]
    assert not offending, f"cells {offending} display OptionMetrics rows"
    # The pattern catches what it is meant to catch.
    for call in ("display(om)", "print(om.loc['2024'])", "x.head()", "om.to_string()"):
        assert ROW_DISPLAY.search(call), call
    assert not ROW_DISPLAY.search('print(f"OptionMetrics: {om.index.size} dates")')


def test_outputs_hold_no_optionmetrics_iv():
    # Licensing guard: no committed output may reproduce a daily OptionMetrics implied vol. Every
    # number printed with 5 or more decimals is compared, as printed and divided by 100 (vol
    # points), with every daily IV the study reads: the loader's four columns and the raw 30-day
    # call and put IVs. A failure names cells only, so it never prints a licensed value.
    om = volsurf.data.load_optionmetrics()
    if om is None:
        pytest.skip("the licensed OptionMetrics files are not in data/raw/")
    std = pd.read_csv(volsurf.data._RAW / volsurf.data._OM_STD, usecols=["days", "impl_volatility"])
    ivs = np.concatenate([om.to_numpy().ravel(), std.loc[std["days"] == 30, "impl_volatility"].to_numpy()])
    ivs = np.sort(ivs[~np.isnan(ivs)])

    hits = []
    for cell in _cells():
        numbers = np.array([float(m) for text in _output_texts(cell) for m in LONG_DECIMAL.findall(text)])
        if numbers.size == 0:
            continue
        numbers = np.concatenate([numbers, numbers / 100])
        right = np.clip(np.searchsorted(ivs, numbers), 1, ivs.size - 1)
        nearest = np.minimum(np.abs(ivs[right] - numbers), np.abs(ivs[right - 1] - numbers))
        if np.any(nearest <= 1e-6):
            hits.append(cell.get("id"))
    assert not hits, f"outputs of cells {hits} print a number equal to a daily OptionMetrics IV to 1e-6"
