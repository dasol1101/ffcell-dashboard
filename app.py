"""
FFCell 대시보드 — HTML 목업(v3_10) 기반, Streamlit 안에 그대로 렌더링
디자인·페이지 구성은 dashboard_template.html(목업)을 100% 그대로 쓰고,
Streamlit은 data/ 폴더의 실제 CSV·JSON 값을 읽어 템플릿의 @@TOKEN@@ 자리에 채워
넣는 역할만 한다.

실행 방법: streamlit run app.py
필요 데이터: ./data 폴더
  - eda_class_dist.csv        (label, cycle_count)
  - kpi_summary.json          (main_method, main_method_accuracy, main_method_macro_recall)
  - confusion_matrix_rule.csv (실제 x 예측, 4x4)
  - per_class_recall_rule.csv (클래스별 recall — 참고용, 표는 confusion matrix에서 재계산)
  - rule_validation_summary.csv (참고용, 현재 템플릿에서는 미사용)
"""

import json
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="FFCell 대시보드", layout="wide", page_icon="🤖")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATE_PATH = os.path.join(BASE_DIR, "dashboard_template.html")

# 목업 표기 <-> 실제 라벨 매핑
LABEL_MAP = {
    "Normal": "Normal",
    "NoNose": "NoNose",
    "+NoBody2": "NoNose,NoBody2",
    "3중결손": "NoNose,NoBody2,NoBody1",
}
CM_ROW_ORDER = ["NoNose", "+NoBody2", "3중결손", "Normal"]  # 표의 행 순서 (목업과 동일)
CM_COL_ORDER = ["NoNose", "+NoBody2", "3중결손", "Normal"]  # 표의 열 순서 (목업과 동일)


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


def build_confusion_rows(cm_df):
    """confusion_matrix_rule.csv(실제 x 예측, 실제 라벨명 컬럼)로 목업과 동일한 <tr> HTML 생성."""
    idx_col = cm_df.columns[0]
    cm = cm_df.set_index(idx_col)

    rows_html = []
    for row_label_short in CM_ROW_ORDER:
        row_label_full = LABEL_MAP[row_label_short]
        cells = []
        row_total_defect_style = ""
        recall_val = None
        if row_label_full in cm.index:
            row = cm.loc[row_label_full]
            row_sum = 0
            correct = None
            for col_short in CM_COL_ORDER:
                col_full = LABEL_MAP[col_short]
                v = int(row[col_full]) if col_full in row.index else 0
                cells.append(f'<td class="mono">{v}</td>')
                row_sum += v
                if col_full == row_label_full:
                    correct = v
            recall_val = (correct / row_sum * 100) if row_sum and correct is not None else 0
        else:
            cells = ['<td class="mono">0</td>'] * len(CM_COL_ORDER)
            recall_val = 0

        recall_style = ' style="color:var(--normal);"' if recall_val < 99.9 else ""
        rows_html.append(
            f'<tr><td>{row_label_short}</td>{"".join(cells)}'
            f'<td class="mono"{recall_style}>{recall_val:.1f}%</td></tr>'
        )
    return "\n          ".join(rows_html)


def binary_detection_rate(cm_df):
    """정상 vs 결함(전체) 이진화 recall — '결함 탐지율' KPI에 사용."""
    idx_col = cm_df.columns[0]
    cm = cm_df.set_index(idx_col)
    defect_rows = [l for l in cm.index if l != "Normal"]
    defect_cols = [c for c in cm.columns if c != "Normal"]
    tp = cm.loc[defect_rows, defect_cols].values.sum()
    fn = cm.loc[defect_rows, "Normal"].values.sum() if "Normal" in cm.columns else 0
    return (tp / (tp + fn) * 100) if (tp + fn) else 0


def render_dashboard():
    if not os.path.exists(TEMPLATE_PATH):
        st.error(f"`{TEMPLATE_PATH}` 를 찾을 수 없습니다. dashboard_template.html을 앱과 같은 폴더에 두세요.")
        return

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        html = f.read()

    class_dist = load_csv("eda_class_dist.csv")
    kpi = load_json("kpi_summary.json")
    cm_rule = load_csv("confusion_matrix_rule.csv")

    missing = []
    if class_dist is None:
        missing.append("eda_class_dist.csv")
    if kpi is None:
        missing.append("kpi_summary.json")
    if cm_rule is None:
        missing.append("confusion_matrix_rule.csv")
    if missing:
        st.warning(f"다음 데이터가 없어 목업 기본값으로 표시됩니다: {', '.join(missing)}")
        components.html(html, height=4200, scrolling=True)
        return

    # ── 클래스별 개수 ──
    counts = {}
    for short, full in LABEL_MAP.items():
        row = class_dist[class_dist["label"] == full]
        counts[short] = int(row["cycle_count"].sum()) if len(row) else 0

    total = sum(counts.values())
    success_n = counts["Normal"]
    fail_n = total - success_n
    success_pct = round(success_n / total * 100, 1) if total else 0
    fail_pct = round(fail_n / total * 100, 1) if total else 0
    fail_rate = fail_pct

    # 도넛 차트 원둘레(339.3) 기준 arc 길이
    donut_normal_arc = round(339.3 * success_n / total, 1) if total else 0
    donut_fail_arc = round(339.3 * fail_n / total, 1) if total else 0

    tokens = {
        "TOTAL_CYCLES": str(total),
        "SUCCESS_N": str(success_n),
        "SUCCESS_PCT": f"{success_pct:.1f}",
        "FAIL_N": str(fail_n),
        "FAIL_PCT": f"{fail_pct:.1f}",
        "FAIL_RATE": f"{fail_rate:.1f}",
        "ACCURACY": f"{kpi['main_method_accuracy']:.3f}",
        "MACRO_RECALL": f"{kpi['main_method_macro_recall']:.3f}",
        "DETECTION_RATE": f"{binary_detection_rate(cm_rule):.1f}",
        "CLS_NORMAL_N": str(counts["Normal"]),
        "CLS_NORMAL_PCT": f"{counts['Normal']/total*100:.1f}" if total else "0",
        "CLS_NONOSE_N": str(counts["NoNose"]),
        "CLS_NONOSE_PCT": f"{counts['NoNose']/total*100:.1f}" if total else "0",
        "CLS_NOBODY2_N": str(counts["+NoBody2"]),
        "CLS_NOBODY2_PCT": f"{counts['+NoBody2']/total*100:.1f}" if total else "0",
        "CLS_TRIPLE_N": str(counts["3중결손"]),
        "CLS_TRIPLE_PCT": f"{counts['3중결손']/total*100:.1f}" if total else "0",
        "DONUT_NORMAL_ARC": f"{donut_normal_arc:.1f}",
        "DONUT_FAIL_ARC": f"{donut_fail_arc:.1f}",
        "CM_ROWS": build_confusion_rows(cm_rule),
    }

    for key, val in tokens.items():
        html = html.replace(f"@@{key}@@", val)

    components.html(html, height=4200, scrolling=True)


render_dashboard()
