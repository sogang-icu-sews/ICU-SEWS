"""Streamlit dashboard for ICU SEWS.

Owner: 임경수
The dashboard calls FastAPI only; it never loads model artifacts directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from icu_sews.dashboard.api_client import get_health, predict

st.set_page_config(page_title="ICU SEWS", page_icon="🏥", layout="wide")
st.title("ICU 패혈증 조기 경보 시스템")
st.caption(
    "교육·연구용 프로토타입입니다. 실제 환자 진단이나 임상 의사결정에 사용하지 마세요. "
    "현재 데모 환자와 입력값은 합성 데이터입니다."
)

try:
    health = get_health()
except Exception as error:  # Show actionable API connection errors in the UI.
    st.error(f"API에 연결할 수 없습니다: {error}")
    st.info("API 서버 상태와 SEWS_API_URL 설정을 확인한 뒤 페이지를 새로고침하세요.")
    st.stop()

mock_mode = bool(health.get("mock_mode", True))
if mock_mode:
    st.warning("Mock 모델 사용 중: 위험도는 실제 모델 예측 결과가 아닙니다.")
elif not health.get("model_loaded", False):
    st.warning("API가 실행 중이지만 모델 로드 상태가 확인되지 않았습니다.")
else:
    st.success("API 연결됨 · 모델 로드 상태 확인됨")

# A small synthetic roster gives the patient-list interaction a usable starting point.
demo_patients = {
    "DEMO-001": {"label": "합성 환자 001", "resp": 22, "sbp": 105, "hr": 96, "map": 72, "lactate": 1.8},
    "DEMO-002": {"label": "합성 환자 002", "resp": 30, "sbp": 88, "hr": 118, "map": 61, "lactate": 3.2},
    "DEMO-003": {"label": "합성 환자 003", "resp": 16, "sbp": 124, "hr": 78, "map": 85, "lactate": 1.1},
}
st.subheader("환자 목록")
selected_id = st.selectbox(
    "환자 선택",
    options=list(demo_patients),
    format_func=lambda patient_id: f"{patient_id} · {demo_patients[patient_id]['label']}",
)
patient = demo_patients[selected_id]

st.subheader("최근 24시간 관찰값")
st.caption("슬라이더 값은 선택한 합성 환자의 24시간 동안 일정한 데모 값으로 사용됩니다.")
left, middle, right = st.columns(3)
with left:
    respiratory_rate = st.slider("호흡수 · Resp (/min)", 5, 45, patient["resp"])
    heart_rate = st.slider("심박수 · HR (/min)", 25, 220, patient["hr"])
with middle:
    systolic_pressure = st.slider("수축기혈압 · SBP (mmHg)", 60, 180, patient["sbp"])
    mean_pressure = st.slider("평균동맥압 · MAP (mmHg)", 35, 140, patient["map"])
with right:
    lactate = st.slider("젖산 · Lactate (mmol/L)", 0.0, 15.0, float(patient["lactate"]), step=0.1)

hours = list(range(1, 25))
demo_frame = pd.DataFrame(
    {
        "ICULOS": hours,
        "HR": [heart_rate] * 24,
        "Resp": [respiratory_rate] * 24,
        "SBP": [systolic_pressure] * 24,
        "MAP": [mean_pressure] * 24,
        "Lactate": [lactate] * 24,
    }
)
chart_tab, table_tab = st.tabs(["시계열 차트", "관찰값 표"])
with chart_tab:
    st.line_chart(demo_frame.set_index("ICULOS")[["HR", "Resp"]])
    st.line_chart(demo_frame.set_index("ICULOS")[["SBP", "MAP"]])
    st.line_chart(demo_frame.set_index("ICULOS")[["Lactate"]])
with table_tab:
    st.dataframe(demo_frame, use_container_width=True, hide_index=True)

st.subheader("위험도 평가")
if st.button("선택 환자 위험도 요청", type="primary"):
    payload = {
        "patient_id": selected_id,
        "observations": demo_frame.to_dict(orient="records"),
    }
    try:
        result: dict[str, Any] = predict(payload)
        st.session_state["last_prediction"] = result
        st.session_state["last_prediction_patient"] = selected_id
    except Exception as error:
        st.error(f"위험도 요청에 실패했습니다: {error}")
        st.info("API 로그와 요청 스키마를 확인한 뒤 다시 시도하세요.")

result = st.session_state.get("last_prediction")
result_patient = st.session_state.get("last_prediction_patient")
if result and result_patient == selected_id:
    if result.get("is_mock", mock_mode):
        st.warning("이 결과는 Mock 예측입니다. 임상적 의미가 없습니다.")
    score = result.get("risk_score")
    level = str(result.get("risk_level", "unknown")).lower()
    if score is not None:
        try:
            score_value = float(score)
            st.metric("패혈증 위험 점수", f"{score_value:.1%}")
        except (TypeError, ValueError):
            st.metric("패혈증 위험 점수", "응답 형식 확인 필요")
    level_labels = {
        "alert": ("경보 · Alert", "error"),
        "warning": ("주의 · Warning", "warning"),
        "normal": ("정상 범위 · Normal", "success"),
    }
    label, style = level_labels.get(level, (f"분류: {level}", "info"))
    getattr(st, style)(label)
    st.caption(
        f"환자 ID: {result.get('patient_id', selected_id)} · "
        f"모델 버전: {result.get('model_version', '미제공')} · "
        f"시각: {result.get('timestamp', datetime.now(timezone.utc).isoformat())}"
    )

    st.subheader("모델 설명")
    top_features = result.get("top_features") or []
    if top_features:
        feature_frame = pd.DataFrame(top_features)
        st.dataframe(feature_frame, use_container_width=True, hide_index=True)
        if {"feature", "shap_value"}.issubset(feature_frame.columns):
            st.bar_chart(feature_frame.set_index("feature")[["shap_value"]])
    else:
        st.info("API 응답에 top_features가 없어 SHAP 설명을 표시할 수 없습니다.")

    st.subheader("임상 점수 및 Calibration")
    sofa_components = result.get("sofa_components")
    qsofa_score = result.get("qsofa_score")
    if sofa_components:
        st.json(sofa_components)
    else:
        st.info("현재 API 계약에는 SOFA 구성요소가 포함되어 있지 않습니다. 점수를 추정하거나 결측값을 정상으로 간주하지 않습니다.")
    if qsofa_score is not None:
        st.metric("qSOFA", str(qsofa_score))
    else:
        st.caption("qSOFA: API 응답에 점수가 없어 표시하지 않습니다.")

    calibration = result.get("calibration")
    if isinstance(calibration, dict) and calibration:
        st.json(calibration)
    else:
        st.info("Calibration 지표가 API 응답에 없습니다. 보정 성능은 아직 확인할 수 없습니다.")
elif result and result_patient != selected_id:
    st.info("선택한 환자의 최신 결과가 없습니다. 위험도 요청을 눌러 평가하세요.")
