"""모듈 확인(sews_check) 검사: 형식만 다르면 통과, 코드가 다르면 중단."""

import pytest
from sews_check import code_fingerprint, file_sha256, module_matches

CODE = '''"""docstring"""
import numpy as np


def f(x):
    """inner docstring"""
    return np.mean(x)  # comment
'''
SAME_CODE = "import numpy as np\ndef f(x):\n    return np.mean( x )\n"
OTHER_CODE = "import numpy as np\ndef f(x):\n    return np.median(x)\n"


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_same_file(tmp_path):
    p = _write(tmp_path / "m.py", CODE)
    assert module_matches(p, file_sha256(p)) == "same file"


def test_same_code_different_format(tmp_path):
    ref_dir = tmp_path / "drive"
    ref_dir.mkdir()
    ref = _write(ref_dir / "m.py", CODE)  # 전처리 때 쓴 원본
    new = _write(tmp_path / "m.py", SAME_CODE)  # 형식만 정리한 저장소 파일
    assert code_fingerprint(ref) == code_fingerprint(new)
    assert module_matches(new, file_sha256(ref), ref_dir=str(ref_dir)) == "same code"


def test_different_code_is_rejected(tmp_path):
    ref_dir = tmp_path / "drive"
    ref_dir.mkdir()
    ref = _write(ref_dir / "m.py", CODE)
    new = _write(tmp_path / "m.py", OTHER_CODE)
    with pytest.raises(RuntimeError):
        module_matches(new, file_sha256(ref), ref_dir=str(ref_dir))
    with pytest.raises(RuntimeError):  # 원본이 없으면 비교할 수 없으므로 중단
        module_matches(new, file_sha256(ref), ref_dir=str(tmp_path / "none"))
