"""데이터 파트 테스트 fixture (tests/data_part 안에서만 쓰인다)."""

import pandas as pd
import pytest
from sews_testdata import load_cfg, make_patients


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_cfg()


@pytest.fixture(scope="session")
def raw(cfg) -> pd.DataFrame:
    return make_patients(cfg)
