"""
FFCell 대시보드 v4 — 피드백 반영판
1페이지 개요: 조립 성공/실패 카운트 + 정확도/탐지균형 KPI, 전체 데이터 개요,
              센서별 실제 움직임(원신호), 정상 vs 결함 비교
2페이지: 모델 성능·통계 지표 (규칙기반/RF/하이브리드 비교, confusion matrix, 판정 로직)
3페이지: 결함탐지 지표 상단 + SHAP·클래스별 recall·피처중요도
4페이지: 이미지 CV (예정)
실행 방법: streamlit run app.py
필요 데이터: ./data 폴더 (PART126 + PART127 export 결과물)
"""

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FFCell 대시보드", layout="wide", page_icon="🤖")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
# 결함 유형별 대표 이미지 (n8n→Slack 알림 첨부용과 동일 소스, State9/Test 폴더 첫 파일)
# 판정 근거가 아니라 사람이 눈으로 참고하기 쉽게 붙이는 참고 사진일 뿐 — 실제 그 사이클 사진 아님
DEFECT_IMAGE_MAP = {
    "NoNose": "sample_nonose.png",
    "NoNose,NoBody2": "sample_nonose_nobody2.png",
    "NoNose,NoBody2,NoBody1": "sample_triple.png",
}
LABEL_ORDER = ["Normal", "NoNose", "NoNose,NoBody2", "NoNose,NoBody2,NoBody1"]
LABEL_SHORT = {
    "Normal": "Normal", "NoNose": "NoNose",
    "NoNose,NoBody2": "+Body2", "NoNose,NoBody2,NoBody1": "+Body1",
}
COLOR_MAP = {
    "Normal": "#4C72B0", "NoNose": "#DD8452",
    "NoNose,NoBody2": "#C44E52", "NoNose,NoBody2,NoBody1": "#8172B2",
}
ROBOT_SENSOR_COLS = {
    "R01": "I_R01_Gripper_Load", "R02": "I_R02_Gripper_Load",
    "R03": "I_R03_Gripper_Load", "R04": "I_R04_Gripper_Load",
}


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
    st.warning(f"`data/{name}` 없음 — export 셀 실행 여부를 확인해주세요.")


def binary_detection_metrics(cm_df):
    """confusion_matrix_rule.csv (실제 x 예측)로부터 정상 vs 결함 이진 탐지 지표를 계산."""
    idx_col = cm_df.columns[0]
    cm = cm_df.set_index(idx_col)
    defect_rows = [l for l in cm.index if l != "Normal"]
    defect_cols = [c for c in cm.columns if c != "Normal"]
    tp = cm.loc[defect_rows, defect_cols].values.sum()
    fn = cm.loc[defect_rows, "Normal"].values.sum() if "Normal" in cm.columns else 0
    fp = cm.loc["Normal", defect_cols].values.sum() if "Normal" in cm.index else 0
    tn = cm.loc["Normal", "Normal"] if ("Normal" in cm.index and "Normal" in cm.columns) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"recall": recall, "precision": precision, "f1": f1, "tp": tp, "fn": fn, "fp": fp, "tn": tn}


st.markdown(
    "<style>div[data-testid='stMetricValue']{font-size:1.5rem} .block-container{padding-top:2rem}</style>",
    unsafe_allow_html=True,
)

# ============================================================================
# 사이드바 — 내비게이션 + 전역 필터
# ============================================================================
st.sidebar.title("FFCell 대시보드")
page = st.sidebar.radio(
    "이동",
    ["① 개요", "② 모델 성능 · 통계 지표", "③ 결함탐지 · SHAP", "④ 이미지 CV (예정)", "⑤ 결함 대표 이미지 (Slack 알림용)"],
)
st.sidebar.divider()
st.sidebar.caption("전역 필터 (① 개요 페이지에 적용)")
selected_labels = st.sidebar.multiselect(
    "표시할 클래스", LABEL_ORDER, default=LABEL_ORDER, format_func=lambda x: LABEL_SHORT[x],
)
if not selected_labels:
    selected_labels = LABEL_ORDER

st.title("🤖 FFCell — 로봇 조립 결함 분류 대시보드")

class_dist = load_csv("eda_class_dist.csv")
kpi = load_json("kpi_summary.json")

# ============================================================================
# ① 개요
# ============================================================================
if page == "① 개요":
    # ── KPI: 조립 성공/실패 카운트 + 정확도/탐지균형 ──
    if class_dist is not None and kpi is not None:
        success_n = int(class_dist.loc[class_dist["label"] == "Normal", "cycle_count"].sum())
        total_n = int(class_dist["cycle_count"].sum())
        fail_n = total_n - success_n

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("조립 성공 (Normal)", f"{success_n:,}개", f"{success_n/total_n:.0%}")
        c2.metric("조립 실패 (결함)", f"{fail_n:,}개", f"{fail_n/total_n:.0%}", delta_color="inverse")
        c3.metric(f"정확도 ({kpi['main_method']})", f"{kpi['main_method_accuracy']*100:.1f}%")
        c4.metric("탐지 균형 (macro recall)", f"{kpi['main_method_macro_recall']*100:.1f}%")
    else:
        missing("eda_class_dist.csv / kpi_summary.json")

    st.divider()

    # ── 전체 데이터 개요 ──
    st.subheader("전체 데이터 개요")
    if class_dist is not None:
        filt = class_dist[class_dist["label"].isin(selected_labels)]
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(
                filt, x="label", y="cycle_count", color="label", color_discrete_map=COLOR_MAP,
                category_orders={"label": LABEL_ORDER}, text="cycle_count",
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="사이클 수")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(filt, names="label", values="cycle_count",
                         color="label", color_discrete_map=COLOR_MAP,
                         category_orders={"label": LABEL_ORDER})
            st.plotly_chart(fig, use_container_width=True)
    else:
        missing("eda_class_dist.csv")

    st.divider()

    # ── 센서별 실제 움직임 (원신호 파형) ──
    st.subheader("센서별 실제 움직임 — 로봇 원신호 파형")
    raw_sensor = load_csv("raw_sensor_example_cycles.csv")
    if raw_sensor is not None:
        robot = st.selectbox("로봇 선택", list(ROBOT_SENSOR_COLS.keys()), index=2)
        sensor_col = ROBOT_SENSOR_COLS[robot]
        if sensor_col in raw_sensor.columns:
            filt = raw_sensor[raw_sensor["label"].isin(selected_labels)]
            fig = px.line(
                filt, x="elapsed_s", y=sensor_col, color="label", color_discrete_map=COLOR_MAP,
                category_orders={"label": LABEL_ORDER},
            )
            fig.update_layout(xaxis_title="사이클 경과 시간(초)", yaxis_title=f"{robot} 그리퍼 부하")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "각 클래스 대표 사이클 1개씩의 실제 신호 파형입니다. "
                + ("※ R01·R04는 오염 이슈로 모델 학습에는 쓰지 않지만, 실제 신호 모양 확인용으로는 그대로 표시합니다."
                   if robot in ("R01", "R04") else "")
            )
        else:
            st.warning(f"{sensor_col} 컬럼이 데이터에 없습니다.")
    else:
        missing("raw_sensor_example_cycles.csv (PART127 export 필요)")

    st.divider()

    # ── 정상 vs 결함 비교 ──
    st.subheader("정상 vs 결함 — 로봇별 그리퍼 부하 비교")
    gripper = load_csv("eda_gripper_load.csv")
    if gripper is not None:
        robots = sorted(gripper["robot"].unique())
        chosen_robot = st.selectbox("비교할 로봇 선택", robots, index=0, key="cmp_robot")
        sub = gripper[(gripper["robot"] == chosen_robot) & (gripper["label"].isin(selected_labels))]
        fig = px.box(
            sub, x="label", y="gripper_load_mean", color="label", color_discrete_map=COLOR_MAP,
            category_orders={"label": LABEL_ORDER}, points="outliers",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Gripper Load (사이클 평균)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        missing("eda_gripper_load.csv")

# ============================================================================
# ② 모델 성능 · 통계 지표
# ============================================================================
elif page == "② 모델 성능 · 통계 지표":
    st.header("모델 성능 · 통계 지표")

    rule_summary = load_csv("rule_validation_summary.csv")
    if rule_summary is not None:
        st.subheader("성능 비교 — 세션1~4 학습 → 세션5 검증")
        display_df = rule_summary.copy()
        display_df["accuracy"] = (display_df["accuracy"] * 100).round(1).astype(str) + "%"
        display_df["macro_recall"] = (display_df["macro_recall"] * 100).round(1).astype(str) + "%"
        display_df = display_df.rename(columns={
            "method": "방법", "accuracy": "정확도", "macro_recall": "매크로 recall", "n_test": "테스트 사이클 수",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        fig = px.bar(rule_summary, x="method", y="accuracy", color="method",
                     text=[f"{v:.1%}" for v in rule_summary["accuracy"]])
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="정확도", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        missing("rule_validation_summary.csv")

    st.divider()
    st.subheader("Confusion Matrix — 규칙기반 단독")
    cm_rule = load_csv("confusion_matrix_rule.csv")
    if cm_rule is not None:
        idx_col = cm_rule.columns[0]
        cm = cm_rule.set_index(idx_col)
        fig = go.Figure(data=go.Heatmap(
            z=cm.values, x=cm.columns, y=cm.index, colorscale="Blues",
            text=cm.values, texttemplate="%{text}",
        ))
        fig.update_layout(xaxis_title="예측", yaxis_title="실제")
        st.plotly_chart(fig, use_container_width=True)
    else:
        missing("confusion_matrix_rule.csv")

    st.divider()
    with st.expander("규칙기반 판정 로직 · 피처 흐름도 보기"):
        meta = load_json("rule_logic_meta.json")
        if meta is not None:
            for line in meta["hierarchy"]:
                st.markdown(f"- {line}")
            st.caption(f"Nose 판정 임계값(R03 그리퍼 부하 피크): {meta['R03_NOSE_THRESH']:.1f}")
        funnel = load_csv("feature_funnel.csv")
        if funnel is not None:
            st.dataframe(funnel, use_container_width=True, hide_index=True)

# ============================================================================
# ③ 결함탐지 · SHAP
# ============================================================================
elif page == "③ 결함탐지 · SHAP":
    st.header("결함탐지 지표 · SHAP 해석 · 탐지 난이도")

    cm_rule = load_csv("confusion_matrix_rule.csv")
    if cm_rule is not None:
        m = binary_detection_metrics(cm_rule)
        c1, c2, c3 = st.columns(3)
        c1.metric("결함탐지 Recall", f"{m['recall']*100:.1f}%")
        c2.metric("결함탐지 Precision", f"{m['precision']*100:.1f}%")
        c3.metric("결함탐지 F1", f"{m['f1']*100:.1f}%")
        st.caption("정상(Normal) vs 결함(그 외 전체)으로 이진화했을 때의 규칙기반 탐지 성능입니다.")
    else:
        missing("confusion_matrix_rule.csv")

    st.divider()
    colA, colB = st.columns(2)

    with colA:
        st.subheader("SHAP 기반 피처 중요도 Top 15")
        shap_df = load_csv("shap_importance_top15.csv")
        if shap_df is not None:
            fig = px.bar(shap_df.sort_values("shap_mean_abs"), x="shap_mean_abs", y="feature", orientation="h")
            fig.update_layout(xaxis_title="평균 |SHAP|", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            missing("shap_importance_top15.csv")
        shap_png = os.path.join(DATA_DIR, "shap_summary_blockA.png")
        if os.path.exists(shap_png):
            st.image(shap_png, caption="SHAP summary plot (Block A 공식 피처셋, 78개)")

    with colB:
        st.subheader("클래스별 탐지 난이도 (recall)")
        per_class = load_csv("per_class_recall_rule.csv")
        if per_class is not None:
            fig = px.bar(
                per_class, x="label", y="recall", color="label", color_discrete_map=COLOR_MAP,
                category_orders={"label": LABEL_ORDER},
                text=[f"{v:.0%}" if pd.notna(v) else "N/A" for v in per_class["recall"]],
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Recall", yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("NoNose는 R03 신호로 거의 완벽히 분리되어 recall이 높다. 실제로 탐지가 어려운 쪽은 Body1이다.")
        else:
            missing("per_class_recall_rule.csv")

        if shap_df is not None and "rf_importance" in shap_df.columns:
            st.subheader("피처 중요도 (RandomForest 기준)")
            fig = px.bar(
                shap_df.sort_values("rf_importance"), x="rf_importance", y="feature", orientation="h",
            )
            fig.update_layout(xaxis_title="RF importance", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ④ 이미지 CV (예정)
# ============================================================================
elif page == "④ 이미지 CV (예정)":
    st.header("이미지 CV — 개발 중")
    st.info(
        "카메라 도메인 시프트 대응(Train: bbox_cam1 → Test: bbox_cam11) 파이프라인 구축 중. "
        "CV 베이스라인 확보 후 센서 판정과 나란히 비교하는 섹션을 추가할 예정입니다."
    )

# ============================================================================
# ⑤ 결함 대표 이미지 (Slack 알림용) — 판정 근거 아님, 알림 첨부용 참고 사진
# ============================================================================
else:
    st.header("결함 대표 이미지 — n8n → Slack 알림 첨부용")
    st.caption(
        "이미지를 판정(분류)에 쓰는 것이 아니라, 결함이 발생했을 때 n8n→Slack 알림에 "
        "참고 사진으로 첨부하는 용도로만 사용합니다. 실제 그 사이클을 촬영한 사진이 아니라, "
        "해당 결함 유형의 대표 예시 사진입니다 (Cycle_state_9 · Test 폴더 기준)."
    )
    st.divider()

    cols = st.columns(len(DEFECT_IMAGE_MAP))
    for col, (label, filename) in zip(cols, DEFECT_IMAGE_MAP.items()):
        img_path = os.path.join(IMAGES_DIR, filename)
        with col:
            if os.path.exists(img_path):
                st.image(img_path, caption=LABEL_SHORT.get(label, label), use_container_width=True)
            else:
                st.warning(f"`images/{filename}` 없음")
                st.caption(f"결함 유형: {label}")

    st.divider()
    st.info(
        "이 사진들은 판정 근거가 아니라 알림 첨부용 참고 자료입니다. "
        "센서(R03) 판정만으로 결함 유형이 확정되며, 이미지는 그 결과를 사람이 눈으로 "
        "참고하기 쉽게 붙여주는 역할만 합니다."
    )
