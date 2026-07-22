"""
FFCell 대시보드 v3 — KPI 게이지 + 사이드바 내비게이션 + 전역 클래스 필터
실행 방법: streamlit run app.py
필요 데이터: ./data 폴더 (PART126 export 셀 결과물, 이전과 동일 구조)
"""

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FFCell 대시보드", layout="wide", page_icon="🤖")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LABEL_ORDER = ["Normal", "NoNose", "NoNose,NoBody2", "NoNose,NoBody2,NoBody1"]
LABEL_SHORT = {
    "Normal": "Normal", "NoNose": "NoNose",
    "NoNose,NoBody2": "+Body2", "NoNose,NoBody2,NoBody1": "+Body1",
}
COLOR_MAP = {
    "Normal": "#4C72B0", "NoNose": "#DD8452",
    "NoNose,NoBody2": "#C44E52", "NoNose,NoBody2,NoBody1": "#8172B2",
}
BASELINE_MACRO_RECALL = 0.28  # 39장 원래 GroupKFold(RF, 오염 피처 포함) 기준값


@st.cache_data
def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_json(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def missing(name):
    st.warning(f"`data/{name}` 없음 — PART126 export 셀 실행 여부를 확인해주세요.")


st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# 사이드바 — 내비게이션 + 전역 필터
# ============================================================================
st.sidebar.title("FFCell 대시보드")
page = st.sidebar.radio(
    "이동",
    ["개요 (KPI)", "EDA — 센서 패턴", "SHAP · 탐지 난이도", "알고리즘 비교", "이미지 CV (예정)"],
)

st.sidebar.divider()
st.sidebar.caption("전역 필터 (EDA 차트에 적용)")
selected_labels = st.sidebar.multiselect(
    "표시할 클래스", LABEL_ORDER, default=LABEL_ORDER,
    format_func=lambda x: LABEL_SHORT[x],
)
if not selected_labels:
    selected_labels = LABEL_ORDER

st.title("🤖 FFCell — 로봇 조립 결함 분류 대시보드")

# ============================================================================
# 페이지 1: 개요 (KPI 게이지)
# ============================================================================
if page == "개요 (KPI)":
    kpi = load_json("kpi_summary.json")
    if kpi is None:
        missing("kpi_summary.json")
    else:
        st.caption(f"전체 사이클 수: **{kpi['total_cycles']:,}개**  ·  주력 모델: **{kpi['main_method']}**")

        g1, g2, g3 = st.columns(3)

        with g1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=kpi["main_method_accuracy"] * 100,
                title={"text": "주력 모델 정확도"},
                number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4C72B0"}},
            ))
            fig.update_layout(height=260, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=kpi["main_method_macro_recall"] * 100,
                title={"text": "매크로 Recall (결함탐지 균형)"},
                number={"suffix": "%"},
                delta={"reference": BASELINE_MACRO_RECALL * 100, "suffix": "%p",
                       "increasing": {"color": "#2ca02c"}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#DD8452"}},
            ))
            fig.update_layout(height=260, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"기준선(39장 원래 GroupKFold RF): {BASELINE_MACRO_RECALL*100:.0f}%")

        with g3:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=kpi["best_binary_f1"] * 100,
                title={"text": f"최고 결함탐지 F1"},
                number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#C44E52"}},
            ))
            fig.update_layout(height=260, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"방법: {kpi['best_binary_f1_method']}")

        st.info(f"💡 탐지 난이도 참고: {kpi['hardest_class_note']}")

        per_class = load_csv("per_class_recall_rule.csv")
        if per_class is not None:
            st.subheader("클래스별 Recall — 한눈에 보기")
            cols = st.columns(len(LABEL_ORDER))
            for i, lbl in enumerate(LABEL_ORDER):
                row = per_class[per_class["label"] == lbl]
                val = row["recall"].values[0] if len(row) else None
                with cols[i]:
                    st.metric(LABEL_SHORT[lbl], f"{val*100:.0f}%" if pd.notna(val) else "N/A")

# ============================================================================
# 페이지 2: EDA — 센서 패턴 (전역 필터 적용)
# ============================================================================
elif page == "EDA — 센서 패턴":
    st.header("정상 vs 결함 — 센서 패턴 비교")

    class_dist = load_csv("eda_class_dist.csv")
    gripper = load_csv("eda_gripper_load.csv")
    timeline = load_csv("eda_timeline_sample.csv")

    col1, col2 = st.columns(2)
    if class_dist is not None:
        with col1:
            filt = class_dist[class_dist["label"].isin(selected_labels)]
            fig = px.bar(
                filt, x="label", y="cycle_count", color="label",
                color_discrete_map=COLOR_MAP, category_orders={"label": LABEL_ORDER},
                text="cycle_count",
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="사이클 수",
                               title="사이클 단위 클래스 분포")
            st.plotly_chart(fig, use_container_width=True)
    else:
        missing("eda_class_dist.csv")

    if gripper is not None:
        with col2:
            robots = sorted(gripper["robot"].unique())
            chosen_robot = st.selectbox("로봇 선택", robots, index=0)
            sub = gripper[(gripper["robot"] == chosen_robot) & (gripper["label"].isin(selected_labels))]
            fig = px.box(
                sub, x="label", y="gripper_load_mean", color="label",
                color_discrete_map=COLOR_MAP, category_orders={"label": LABEL_ORDER},
                points="outliers",
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Gripper Load (사이클 평균)",
                               title=f"{chosen_robot} 그리퍼 부하 — 클래스별 비교")
            st.plotly_chart(fig, use_container_width=True)
    else:
        missing("eda_gripper_load.csv")

    st.subheader("시간축 결함 발생 타임라인")
    if timeline is not None:
        timeline["_time"] = pd.to_datetime(timeline["_time"])
        filt = timeline[timeline["label"].isin(selected_labels)]
        fig = px.scatter(
            filt, x="_time", y="label", color="label",
            color_discrete_map=COLOR_MAP, category_orders={"label": LABEL_ORDER}, opacity=0.5,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(showlegend=False, xaxis_title="시간", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        missing("eda_timeline_sample.csv")

# ============================================================================
# 페이지 3: SHAP · 탐지 난이도
# ============================================================================
elif page == "SHAP · 탐지 난이도":
    st.header("SHAP 해석 · 결함 유형별 탐지 난이도")

    colA, colB = st.columns(2)
    with colA:
        shap_df = load_csv("shap_importance_top15.csv")
        if shap_df is not None:
            fig = px.bar(
                shap_df.sort_values("shap_mean_abs"), x="shap_mean_abs", y="feature", orientation="h",
                title="SHAP 기반 피처 중요도 Top 15",
            )
            fig.update_layout(xaxis_title="평균 |SHAP|", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            missing("shap_importance_top15.csv")

        shap_png = os.path.join(DATA_DIR, "shap_summary_blockA.png")
        if os.path.exists(shap_png):
            st.image(shap_png, caption="SHAP summary plot (Block A 공식 피처셋, 78개)")

    with colB:
        per_class = load_csv("per_class_recall_rule.csv")
        if per_class is not None:
            fig = px.bar(
                per_class, x="label", y="recall", color="label",
                color_discrete_map=COLOR_MAP, category_orders={"label": LABEL_ORDER},
                text=[f"{v:.0%}" if pd.notna(v) else "N/A" for v in per_class["recall"]],
                title="규칙기반 클래스별 Recall (탐지 난이도)",
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Recall", yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Nose 유무(NoNose 판정)는 R03 신호로 거의 완벽히 분리되어 recall이 높다. "
                "실제로 탐지가 어려운 쪽은 Body1(빠진 부품 수가 적을수록 신호가 약해지는 구조)이다."
            )
        else:
            missing("per_class_recall_rule.csv")

    with st.expander("규칙기반 판정 로직 / 피처 흐름도"):
        meta = load_json("rule_logic_meta.json")
        if meta is not None:
            for line in meta["hierarchy"]:
                st.markdown(f"- {line}")
            st.caption(f"Nose 판정 임계값(R03 그리퍼 부하 피크): {meta['R03_NOSE_THRESH']:.1f}")
        funnel = load_csv("feature_funnel.csv")
        if funnel is not None:
            st.dataframe(funnel, use_container_width=True, hide_index=True)

# ============================================================================
# 페이지 4: 알고리즘 비교 (탭으로 분리)
# ============================================================================
elif page == "알고리즘 비교":
    st.header("알고리즘별 성능 비교 — 세션1~4 학습 → 세션5 검증")

    tab_bin, tab_multi = st.tabs(["이진 (정상 vs 결함)", "4클래스 세부"])

    with tab_bin:
        binary_comp = load_csv("model_comparison_binary.csv")
        if binary_comp is not None:
            fig = px.bar(
                binary_comp, x="method", y="f1", color="method",
                text=[f"{v:.1%}" for v in binary_comp["f1"]],
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="F1", yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            display_bc = binary_comp.copy()
            for col in ["recall", "precision", "f1"]:
                display_bc[col] = (display_bc[col] * 100).round(1).astype(str) + "%"
            display_bc = display_bc.rename(columns={
                "method": "방법", "recall": "Recall", "precision": "Precision", "f1": "F1"
            })
            st.dataframe(display_bc, use_container_width=True, hide_index=True)
            st.caption(
                "Isolation Forest·LSTM-Autoencoder(있다면)는 '정상 아님'까지만 판정 가능하고 "
                "부품별 세부 판정은 못 한다. 정상 학습 샘플이 20개 미만이라 결과를 일반화 성능으로 "
                "과신하지 말 것."
            )
        else:
            missing("model_comparison_binary.csv")

    with tab_multi:
        multiclass_comp = load_csv("model_comparison_multiclass.csv")
        if multiclass_comp is not None:
            fig = px.bar(
                multiclass_comp, x="method", y="macro_recall", color="method",
                text=[f"{v:.1%}" for v in multiclass_comp["macro_recall"]],
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Macro Recall", yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            display_mc = multiclass_comp.copy()
            for col in ["accuracy", "macro_recall"]:
                display_mc[col] = (display_mc[col] * 100).round(1).astype(str) + "%"
            display_mc = display_mc.rename(columns={"method": "방법", "accuracy": "정확도", "macro_recall": "매크로 Recall"})
            st.dataframe(display_mc, use_container_width=True, hide_index=True)
        else:
            missing("model_comparison_multiclass.csv")

# ============================================================================
# 페이지 5: 이미지 CV (예정)
# ============================================================================
else:
    st.header("이미지 CV — 개발 중")
    st.info(
        "카메라 도메인 시프트 대응(Train: bbox_cam1 → Test: bbox_cam11) 파이프라인 구축 중. "
        "CV 베이스라인 확보 후 센서 판정과 나란히 비교하는 섹션을 추가할 예정입니다."
    )
