# ============================================================
# streamlit_demo.py
#
# Streamlit Demo
# 중앙 정렬 + 큰 질문 글씨 + 큰 Go 버튼
#
# 실행 방법:
# python -m streamlit run streamlit_demo.py
# ============================================================

import streamlit as st


# ============================================================
# 1. Streamlit 페이지 설정
# ============================================================

st.set_page_config(
    page_title="Streamlit Demo",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. CSS 디자인
#
# 주의:
# 아래 CSS 코드는 반드시
# st.markdown(""" ... """) 내부에 있어야 합니다.
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       전체 화면 배경
       ======================================================== */

    .stApp {
        background-color: #0e1117;
    }


    /* ========================================================
       전체 콘텐츠를 화면 가운데 배치
       ======================================================== */

    .block-container {
        max-width: 1000px !important;

        margin-left: auto !important;
        margin-right: auto !important;

        padding-top: 60px !important;
        padding-left: 40px !important;
        padding-right: 40px !important;
        padding-bottom: 50px !important;
    }


    /* ========================================================
       Streamlit Demo 제목
       ======================================================== */

    h1 {
        color: white !important;

        font-size: 72px !important;

        font-weight: 800 !important;

        margin-bottom: 45px !important;

        letter-spacing: -2px;
    }


    /* ========================================================
       What's your favorite color?
       
       질문 글씨를 Go 글씨와 동일한
       54px로 설정
       ======================================================== */

    div[data-testid="stTextInput"] label {
        color: white !important;

        font-size: 54px !important;

        font-weight: 700 !important;

        margin-bottom: 20px !important;
    }


    /* Streamlit 버전에 따라
       실제 질문 글씨가 p 태그에 들어가기 때문에
       p 태그에도 동일한 크기를 적용합니다. */

    div[data-testid="stTextInput"] label p {
        color: white !important;

        font-size: 54px !important;

        font-weight: 700 !important;

        line-height: 1.2 !important;
    }


    /* ========================================================
       색깔 입력창
       ======================================================== */

    div[data-testid="stTextInput"] input {
        background-color: #262730 !important;

        color: white !important;

        font-size: 32px !important;

        height: 82px !important;

        border-radius: 16px !important;

        border: 1px solid #262730 !important;

        padding-left: 22px !important;
    }


    /* ========================================================
       입력창의 안내 문구
       ======================================================== */

    div[data-testid="stTextInput"] input::placeholder {
        color: #9ca3af !important;

        font-size: 28px !important;
    }


    /* ========================================================
       입력창을 클릭했을 때
       ======================================================== */

    div[data-testid="stTextInput"] input:focus {
        border: 1px solid #ff4b4b !important;

        box-shadow: 0 0 0 1px #ff4b4b !important;
    }


    /* ========================================================
       Go 버튼 영역
       ======================================================== */

    div[data-testid="stButton"] {
        margin-top: 25px !important;

        margin-bottom: 30px !important;
    }


    /* ========================================================
       Go 버튼
       ======================================================== */

    div[data-testid="stButton"] button {
        background-color: #ff4b4b !important;

        color: white !important;

        font-size: 54px !important;

        font-weight: 700 !important;

        height: 105px !important;

        min-width: 170px !important;

        border-radius: 20px !important;

        border: none !important;

        padding-left: 35px !important;

        padding-right: 35px !important;
    }


    /* ========================================================
       Go 버튼 내부 글씨
       
       질문 글씨와 동일한 54px
       ======================================================== */

    div[data-testid="stButton"] button p {
        color: white !important;

        font-size: 54px !important;

        font-weight: 700 !important;

        line-height: 1 !important;

        margin: 0 !important;
    }


    /* ========================================================
       Go 버튼에 마우스를 올렸을 때
       ======================================================== */

    div[data-testid="stButton"] button:hover {
        background-color: #ff5c5c !important;

        color: white !important;

        border: none !important;
    }


    /* ========================================================
       Go 버튼 클릭 효과
       ======================================================== */

    div[data-testid="stButton"] button:active {
        background-color: #e63e3e !important;

        transform: scale(0.98);
    }


    /* ========================================================
       결과 글씨
       ======================================================== */

    .result-text {
        color: white;

        font-size: 36px;

        font-weight: 600;

        margin-top: 20px;
    }


    /* ========================================================
       Streamlit 기본 메뉴 숨기기
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }


    /* ========================================================
       Streamlit Footer 숨기기
       ======================================================== */

    footer {
        visibility: hidden;
    }


    /* ========================================================
       Streamlit Header
       ======================================================== */

    header[data-testid="stHeader"] {
        background-color: #0e1117;
    }


    /* ========================================================
       작은 화면 대응
       ======================================================== */

    @media (max-width: 768px) {

        .block-container {
            max-width: 95% !important;

            padding-left: 20px !important;
            padding-right: 20px !important;
        }


        h1 {
            font-size: 52px !important;
        }


        div[data-testid="stTextInput"] label p {
            font-size: 36px !important;
        }


        div[data-testid="stTextInput"] input {
            font-size: 26px !important;

            height: 70px !important;
        }


        div[data-testid="stButton"] button {
            height: 90px !important;

            min-width: 150px !important;
        }


        div[data-testid="stButton"] button p {
            font-size: 42px !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. 화면 제목
# ============================================================

st.title("Streamlit Demo")


# ============================================================
# 4. 사용자 색깔 입력
#
# value="green"을 사용하지 않았기 때문에
# 처음 화면에서는 입력값이 비어 있습니다.
#
# placeholder는 실제 입력값이 아니라
# 사용자에게 보여주는 안내 문구입니다.
# ============================================================

favorite_color = st.text_input(
    "What's your favorite color?",
    placeholder="Enter your favorite color"
)


# ============================================================
# 5. Go 버튼
# ============================================================

go_button = st.button("Go")


# ============================================================
# 6. Go 버튼을 클릭했는지 확인
# ============================================================

if go_button:

    # --------------------------------------------------------
    # 입력값 앞뒤의 공백 제거
    #
    # 예:
    # "   green   "
    #
    # 결과:
    # "green"
    # --------------------------------------------------------

    favorite_color = favorite_color.strip()


    # ========================================================
    # 7. 색깔을 입력한 경우
    # ========================================================

    if favorite_color:

        st.markdown(
            f"""
            <div class="result-text">
                I like {favorite_color} too!
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # 8. 아무것도 입력하지 않고 Go를 누른 경우
    # ========================================================

    else:

        st.warning(
            "Please enter your favorite color."
        )