from __future__ import annotations

import pytest

from scripts.benchmark_local import run_benchmarks


def test_local_benchmark_reports_all_operations_and_bounds(tmp_path) -> None:
    report = run_benchmarks(iterations=2, files=3)

    assert report["dataset"] == {"files": 3, "iterations_per_case": 2}
    assert set(report["results"]) == {
        "workspace_search",
        "workspace_read",
        "tool_schema_validation",
        "sqlite_append_and_bounded_context",
    }
    for measurement in report["results"].values():
        assert measurement["iterations"] == 2
        assert measurement["median_ms"] >= 0
        assert measurement["p95_ms"] >= measurement["median_ms"]
        assert measurement["single_operation_peak_traced_bytes"] >= 0
    assert "excludes inference" in report["scope"]


@pytest.mark.parametrize(
    ("iterations", "files"),
    [(0, 1), (1, 0)],
)
def test_local_benchmark_rejects_non_positive_sizes(
    iterations: int, files: int
) -> None:
    with pytest.raises(ValueError):
        run_benchmarks(iterations=iterations, files=files)
