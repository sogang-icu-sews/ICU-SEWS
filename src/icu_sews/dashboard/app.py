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

st.set_page_config(
    page_title="ICU SEWS | Patient Monitoring",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Consistent visual system: calm navy, restrained status colors, generous spacing.
st.markdown(
    """
    <style>
      :root {
        --sews-ink: #17243a;
        --sews-muted: #64748b;
        --sews-line: #e5eaf1;
        --sews-surface: #ffffff;
        --sews-canvas: #f5f7fb;
      }
      .stApp { background: var(--sews-canvas); color: var(--sews-ink); }
      [data-testid="stHeader"] { background: rgba(245,247,251,.92); }
      [data-testid="stMainBlockContainer"] {
        max-width: 1500px; padding-top: 1.6rem; padding-bottom: 3rem;
      }
      h1, h2, h3 { color: var(--sews-ink); letter-spacing: -.025em; }
      h2 { margin-top: 1.25rem; }
      [data-testid="stMetric"] {
        background: var(--sews-surface); border: 1px solid var(--sews-line);
        border-radius: 14px; padding: 16px 18px; box-shadow: 0 2px 8px rgba(20,35,60,.035);
      }
      [data-testid="stMetricLabel"] p { color: var(--sews-muted); font-size: .82rem; }
      [data-testid="stMetricValue"] { color: var(--sews-ink); font-weight: 700; }
      [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--sews-line); border-radius: 14px;
      }
      [data-testid="stFileUploader"] section {
        background: #fff; border: 1px dashed #b9c6d8; border-radius: 12px;
      }
      .stButton > button, .stDownloadButton > button {
        border-radius: 10px; min-height: 2.65rem; font-weight: 600;
        border: 1px solid #d7dfeb;
      }
      .stButton > button[kind="primary"] {
        background: #1d4ed8; border-color: #1d4ed8; color: #fff;
      }
      [data-testid="stTabs"] button { font-weight: 600; }
      [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--sews-line); border-radius: 10px; overflow: hidden;
      }
      [data-testid="stSidebar"] {
        background: #fff; border-right: 1px solid var(--sews-line);
      }
      .sews-hero {
        background: linear-gradient(115deg, #14243d 0%, #203e68 72%, #28558a 100%);
        border-radius: 18px; padding: 25px 30px; color: #fff; margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(20,40,70,.12);
      }
      .sews-eyebrow { color: #b9cce5; font-size: .76rem; font-weight: 700;
        letter-spacing: .13em; text-transform: uppercase; margin-bottom: 8px; }
      .sews-title { color: #fff; font-size: 1.8rem; font-weight: 750;
        letter-spacing: -.035em; line-height: 1.25; }
      .sews-subtitle { color: #d5e0ef; font-size: .92rem; margin-top: 8px; }
      .sews-note {
        color: #64748b; font-size: .78rem; line-height: 1.5;
        padding: 10px 13px; border-left: 3px solid #94a3b8;
        background: #eef2f7; border-radius: 0 8px 8px 0; margin: 12px 0 18px;
      }
      @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem .8rem 2rem; }
        .sews-hero { padding: 20px; }
        .sews-title { font-size: 1.4rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ICU SEWS")
st.caption("중환자실 환자 상태 모니터링 · 패혈증 조기 경보 연구 시스템")

try:
    health = get_health()
except Exception as error:  # Show actionable API connection errors to the user.
    st.error(f"API에 연결할 수 없습니다: {error}")
    st.info("API 서버 상태와 SEWS_API_URL 설정을 확인한 뒤 페이지를 새로고침하세요.")
    st.stop()

mock_mode = bool(health.get("mock_mode", True))
if mock_mode:
    st.info("개발용 모델 연결됨 · 예측 결과는 기능 확인용이며 임상 판단에 사용할 수 없습니다.")
elif not health.get("model_loaded", False):
    st.warning("API가 실행 중이지만 모델 로드 상태가 확인되지 않았습니다.")
else:
    st.success("API 연결됨 · 모델 로드 상태 확인됨")


with st.expander("환자 데이터 파일 업로드", expanded=False):
    st.caption("PSV 또는 CSV 파일을 업로드해 API 예측을 요청할 수 있습니다.")
    st.divider()
    st.header("환자 데이터 업로드")

    uploaded_file = st.file_uploader(
        "PSV 또는 CSV 파일 선택",
        type=["psv", "csv"],
        help="환자 한 명의 시간별 관찰 데이터 파일을 업로드하세요.",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".psv"):
                uploaded_df = pd.read_csv(uploaded_file, sep="|")
            else:
                uploaded_df = pd.read_csv(uploaded_file)

            if uploaded_df.empty:
                st.error("업로드한 파일에 데이터가 없습니다.")
            elif "ICULOS" not in uploaded_df.columns:
                st.error("필수 컬럼 ICULOS가 없습니다.")
            else:
                uploaded_df["ICULOS"] = pd.to_numeric(uploaded_df["ICULOS"], errors="coerce")

                if uploaded_df["ICULOS"].isna().any():
                    st.error("ICULOS에 숫자가 아닌 값이 포함되어 있습니다.")
                elif (uploaded_df["ICULOS"] < 1).any():
                    st.error("ICULOS는 1 이상의 값이어야 합니다.")
                else:
                    uploaded_df = uploaded_df.sort_values("ICULOS").reset_index(drop=True)

                    if uploaded_df["ICULOS"].duplicated().any():
                        st.error("중복된 ICULOS 값이 있습니다.")
                    else:
                        patient_id = st.text_input(
                            "환자 ID",
                            value=uploaded_file.name.rsplit(".", 1)[0],
                            key="uploaded_patient_id",
                        )

                        st.success("파일을 정상적으로 읽었습니다.")
                        st.write(f"전체 관찰 기록: {len(uploaded_df)}시간")
                        st.write(f"전체 컬럼 수: {len(uploaded_df.columns)}개")

                        # API contract allows at most 24 hourly observations.
                        recent_24h = uploaded_df.tail(24).copy()

                        # SepsisLabel is an outcome label, not a prediction input.
                        preview_df = recent_24h.drop(columns=["SepsisLabel"], errors="ignore")

                        st.subheader("업로드 환자 관찰 데이터")
                        st.caption(
                            "업로드한 실제 파일의 관찰값만 표시합니다. "
                            "누락된 값은 임의로 채우지 않습니다."
                        )

                        chart_columns = [
                            column
                            for column in (
                                "HR",
                                "O2Sat",
                                "Temp",
                                "SBP",
                                "MAP",
                                "DBP",
                                "Resp",
                                "EtCO2",
                                "Lactate",
                            )
                            if column in preview_df.columns
                            and pd.api.types.is_numeric_dtype(preview_df[column])
                            and preview_df[column].notna().any()
                        ]

                        chart_tab, table_tab = st.tabs(["시계열 차트", "관찰값 표"])
                        with chart_tab:
                            if chart_columns:
                                chart_df = preview_df.set_index("ICULOS")[chart_columns]
                                st.line_chart(chart_df)
                                st.caption(
                                    "가로축은 ICU 입실 후 경과 시간(ICULOS)이며, "
                                    "파일에 존재하는 생체 신호만 표시합니다."
                                )
                            else:
                                st.info(
                                    "차트로 표시할 수 있는 생체 신호가 없습니다. "
                                    "HR, O2Sat, Temp, SBP, MAP, DBP, Resp, "
                                    "EtCO2, Lactate 컬럼을 확인하세요."
                                )
                        with table_tab:
                            st.dataframe(
                                preview_df,
                                width="stretch",
                                hide_index=True,
                            )
                        # Predict using the uploaded patient's observations.
                        if st.button(
                            "업로드 환자 위험도 예측 요청",
                            type="primary",
                            key="uploaded_predict_button",
                        ):
                            if not patient_id.strip():
                                st.error("환자 ID를 입력하세요.")
                            else:
                                try:
                                    # Remove the outcome label to prevent label leakage.
                                    api_frame = recent_24h.drop(
                                        columns=["SepsisLabel"],
                                        errors="ignore",
                                    ).copy()

                                    # API requires integer ICULOS values.
                                    api_frame["ICULOS"] = api_frame["ICULOS"].astype(int)

                                    # Convert missing values to JSON null.
                                    records = (
                                        api_frame.astype(object)
                                        .where(pd.notna(api_frame), None)
                                        .to_dict(orient="records")
                                    )

                                    payload = {
                                        "patient_id": patient_id.strip(),
                                        "observations": records,
                                    }

                                    with st.spinner("위험도를 평가하고 있습니다..."):
                                        uploaded_result = predict(payload)

                                    st.session_state["uploaded_prediction"] = uploaded_result
                                    st.session_state["uploaded_prediction_id"] = patient_id.strip()

                                except Exception as error:
                                    st.error(f"예측 요청 실패: {error}")

                        # Display the uploaded patient's prediction separately.
                        uploaded_result = st.session_state.get("uploaded_prediction")

                        if (
                            uploaded_result
                            and st.session_state.get("uploaded_prediction_id") == patient_id.strip()
                        ):
                            st.subheader("업로드 환자 예측 결과")

                            if uploaded_result.get("is_mock", False):
                                st.warning("Mock 예측입니다. 실제 모델의 예측 결과가 아닙니다.")

                            score = uploaded_result.get("risk_score")
                            if score is not None:
                                st.metric(
                                    "패혈증 위험 점수",
                                    f"{float(score):.1%}",
                                )

                            level = uploaded_result.get("risk_level", "unknown")
                            st.write(f"위험 단계: {level}")

                            st.caption(
                                f"환자 ID: {uploaded_result.get('patient_id')}"
                                f" · 모델 버전: "
                                f"{uploaded_result.get('model_version', '미제공')}"
                            )

        except Exception as error:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {error}")


with st.sidebar:
    st.divider()
    st.subheader("개발 및 테스트")
    demo_mode = st.checkbox(
        "데모 모드 사용",
        value=False,
        help="합성 환자와 가상 생체 신호를 이용해 대시보드 기능을 테스트합니다.",
    )

if demo_mode:
    st.info(
        "데모 모드가 활성화되었습니다. 표시되는 환자와 관찰값은 모두 합성 데이터이며 "
        "실제 환자 정보가 아닙니다."
    )

    # A small synthetic roster gives the patient-list interaction a usable starting point.
    demo_patients = {
        "DEMO-001": {
            "label": "합성 환자 001",
            "resp": 22,
            "sbp": 105,
            "hr": 96,
            "map": 72,
            "lactate": 1.8,
        },
        "DEMO-002": {
            "label": "합성 환자 002",
            "resp": 30,
            "sbp": 88,
            "hr": 118,
            "map": 61,
            "lactate": 3.2,
        },
        "DEMO-003": {
            "label": "합성 환자 003",
            "resp": 16,
            "sbp": 124,
            "hr": 78,
            "map": 85,
            "lactate": 1.1,
        },
    }
    st.subheader("환자 목록")
    selected_id = st.selectbox(
        "환자 선택",
        options=list(demo_patients),
        format_func=lambda patient_id: f"{patient_id} · {demo_patients[patient_id]['label']}",
    )
    patient = demo_patients[selected_id]

    st.subheader("환자 상태 요약")
    st.caption("아래 값은 선택한 데모 환자의 예시 관찰값입니다.")
    summary_cols = st.columns(4)
    summary_items = [
        ("심박수", f"{patient['hr']} bpm"),
        ("호흡수", f"{patient['resp']} /min"),
        ("수축기혈압", f"{patient['sbp']} mmHg"),
        ("평균동맥압", f"{patient['map']} mmHg"),
    ]
    for col, (label, value) in zip(summary_cols, summary_items, strict=True):
        with col:
            st.metric(label, value)

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
        lactate = st.slider(
            "젖산 · Lactate (mmol/L)",
            0.0,
            15.0,
            float(patient["lactate"]),
            step=0.1,
        )

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
        st.dataframe(demo_frame, width="stretch", hide_index=True)

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
            st.dataframe(feature_frame, width="stretch", hide_index=True)
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
            st.info(
                "현재 API 계약에는 SOFA 구성요소가 포함되어 있지 않습니다. "
                "점수를 추정하거나 결측값을 정상으로 간주하지 않습니다."
            )
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
