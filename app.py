"""
FFCell 대시보드 — 최종 확정 HTML 목업을 그대로 렌더링
실행 방법: streamlit run app.py
필요 파일: ./dashboard.html, ./images/*.png (같은 폴더)
"""
import os
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="FFCell 대시보드", layout="wide", page_icon="🤖")

# Streamlit 기본 여백 제거 — HTML이 화면을 꽉 채우도록
st.markdown(
    "<style>.block-container{padding:0 !important; max-width:100% !important;}"
    "header{visibility:hidden;}</style>",
    unsafe_allow_html=True,
)

HTML_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")

with open(HTML_PATH, "r", encoding="utf-8") as f:
    html_content = f.read()

# 목업 안의 <img src="images/...">가 이 앱과 같은 폴더의 images/를 그대로 찾도록,
# 파일을 app.py와 같은 위치에 두면 별도 경로 처리 없이 그대로 작동합니다.
components.html(html_content, height=2400, scrolling=True)
