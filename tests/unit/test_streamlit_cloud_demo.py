from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_cloud_demo_is_read_only_and_explains_local_limits() -> None:
    app_path = Path(__file__).resolve().parents[2] / "streamlit_cloud_demo.py"
    page = AppTest.from_file(str(app_path)).run()

    assert not page.exception
    assert any("read-only project showcase" in item.value for item in page.info)
    assert any("What runs locally" in item.value for item in page.subheader)
    assert any("Run the full assistant" in item.label for item in page.expander)
    assert any(
        "streamlit run app/ui/streamlit_app.py" in item.value
        for item in page.markdown
    )
