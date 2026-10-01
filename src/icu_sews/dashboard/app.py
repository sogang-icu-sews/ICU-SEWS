"""Initial Streamlit dashboard and API integration screen.

Owner: 임경수
Next work: patient list, real time-series charts, SOFA components, SHAP, and calibration.
"""

import pandas as pd
import streamlit as st

from icu_sews.dashboard.api_client import get_health, predict

st.set_page_config(page_title="ICU SEWS", page_icon="🏥", layout="wide")
st.title("ICU 패혈증 조기 경보 시스템")
st.caption("교육·연구용 프로토타입 — 실제 임상 의사결정에 사용할 수 없습니다.")

try:
    health = get_health()
    if health.get("mock_mode"):
        st.warning("현재 Mock 모델로 실행 중입니다. 표시된 위험도는 실제 예측이 아닙니다.")
    else:
        st.success("학습된 모델이 연결되었습니다.")
except Exception as error:  # Streamlit must display connection failures to the user.
    st.error(f"API에 연결할 수 없습니다: {error}")
    st.stop()

st.subheader("통합 테스트용 환자")
respiratory_rate = st.slider("호흡수(Resp)", 5, 45, 22)
systolic_pressure = st.slider("수축기혈압(SBP)", 60, 180, 105)

hours = list(range(1, 25))
demo_frame = pd.DataFrame(
    {
        "ICULOS": hours,
        "Resp": [respiratory_rate] * 24,
        "SBP": [systolic_pressure] * 24,
    }
)
st.line_chart(demo_frame.set_index("ICULOS"))

if st.button("위험도 요청", type="primary"):
    result = predict(
        {
            "patient_id": "DEMO_PATIENT",
            "observations": demo_frame.to_dict(orient="records"),
        }
    )
    risk_score = float(result["risk_score"])
    level = result["risk_level"]
    st.metric("패혈증 위험도", f"{risk_score:.1%}")
    if level == "alert":
        st.error("Alert")
    elif level == "warning":
        st.warning("Warning")
    else:
        st.success("Normal")
