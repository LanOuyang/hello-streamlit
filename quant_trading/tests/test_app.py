import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


def test_app_runs_default():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert len(at.metric) == 5


@pytest.mark.parametrize("name", ["momentum", "bollinger", "rsi"])
def test_app_each_strategy(name):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox[0].select(name).run()
    assert not at.exception
