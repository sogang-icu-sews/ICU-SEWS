"""모듈 버전 확인: 전처리 때 기록한 모듈과 지금 쓰는 모듈이 같은 코드인지 확인한다.

feature_registry.json에는 전처리 단계에서 쓴 모듈 파일의 sha256이 기록된다.
저장소 모듈은 코드 형식(Black, 줄바꿈, docstring)만 정리돼 바이트가 다를 수 있으므로,
sha256이 다르면 Drive의 원본 모듈(<root>/src)과 '실행되는 코드'가 같은지 비교한다.
실행되는 코드 = 주석·공백·docstring을 뺀 구문 트리(ast). 이것이 같으면 계산 결과도 같다.
"""

import ast
import hashlib
import os


def file_sha256(path: str) -> str:
    """SHA-256 hex digest of a file's bytes."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def code_fingerprint(path: str) -> str:
    """SHA-256 of the syntax tree without docstrings (ignores format, comments)."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def module_matches(path: str, expected_sha: str, ref_dir: str | None = None) -> str:
    """Return how `path` matches the recorded module, or raise RuntimeError.

    - 'same file': 바이트까지 같다.
    - 'same code': ref_dir의 원본(sha256 일치)과 실행되는 코드가 같다 (형식만 다름).
    """
    if file_sha256(path) == expected_sha:
        return "same file"
    if ref_dir:
        ref = os.path.join(ref_dir, os.path.basename(path))
        if (
            os.path.exists(ref)
            and file_sha256(ref) == expected_sha
            and code_fingerprint(ref) == code_fingerprint(path)
        ):
            return "same code"
    raise RuntimeError(
        f"{os.path.basename(path)}가 전처리 때 기록한 모듈과 다르다 "
        "(sha256 불일치, 원본과 실행 코드도 다름). 전처리 단계를 다시 실행하세요."
    )
