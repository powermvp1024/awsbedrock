# ============================================================
# bedrock_chatbot.py
#
# Amazon Bedrock + Streamlit Chatbot
#
# 주요 기능
# ------------------------------------------------------------
# 1. Amazon Bedrock Converse API 사용
# 2. 사용자 질문 / Bedrock 답변 영역 구분
# 3. 새로운 질문을 하면 이전 질문/답변 제거
# ============================================================

import streamlit as st
import boto3

from botocore.exceptions import ClientError


# ============================================================
# STEP 1. Streamlit 페이지 설정
# ============================================================

st.set_page_config(
    page_title="Chatbot powered by Bedrock",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# STEP 2. AWS Bedrock 설정
# ============================================================

AWS_REGION = "us-east-1"

MODEL_ID = "global.anthropic.claude-sonnet-4-6"


# ============================================================
# STEP 3. Bedrock Runtime Client
# ============================================================

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION
)


# ============================================================
# STEP 4. Session State 초기화
#
# 가장 최근 질문/답변만 저장합니다.
# ============================================================

if "latest_question" not in st.session_state:
    st.session_state.latest_question = ""


if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = ""


# ============================================================
# STEP 5. CSS 디자인
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       전체 콘텐츠 영역
       ===================================================== */

    .block-container {

        max-width: 1100px !important;

        margin-left: auto !important;
        margin-right: auto !important;

        padding-top: 45px !important;
        padding-bottom: 50px !important;

    }


    /* =====================================================
       ★ 메인 제목

       Chatbot powered by Bedrock

       text-align: center
           → 화면 가운데 정렬

       font-size: 84px
           → 기존보다 더 크게 표시
       ===================================================== */

    .main-title {

        width: 100%;

        text-align: center;

        font-size: 84px;

        font-weight: 900;

        line-height: 1.15;

        letter-spacing: -3px;

        color: #272b3a;

        margin-top: 10px;

        margin-bottom: 70px;

    }


    /* =====================================================
       질문 입력 제목
       ===================================================== */

    div[data-testid="stTextInput"] label p {

        font-size: 32px !important;

        font-weight: 800 !important;

        color: #272b3a !important;

        margin-bottom: 12px !important;

    }


    /* =====================================================
       질문 입력창
       ===================================================== */

    div[data-testid="stTextInput"] input {

        background-color: #f4f6fa !important;

        color: #272b3a !important;

        font-size: 26px !important;

        height: 72px !important;

        border: 2px solid #e1e5ed !important;

        border-radius: 14px !important;

        padding-left: 22px !important;

    }


    /* =====================================================
       입력창 클릭 효과
       ===================================================== */

    div[data-testid="stTextInput"] input:focus {

        border: 2px solid #4f7cff !important;

        box-shadow:
            0 0 0 3px rgba(79,124,255,0.15) !important;

    }


    /* =====================================================
       입력창 Placeholder
       ===================================================== */

    div[data-testid="stTextInput"] input::placeholder {

        color: #9ca3af !important;

        font-size: 23px !important;

    }


    /* =====================================================
       질문하기 버튼
       ===================================================== */

    div[data-testid="stButton"] button {

        background-color: #4f7cff !important;

        color: white !important;

        height: 78px !important;

        border-radius: 14px !important;

        border: none !important;

        margin-top: 10px !important;

    }


    /* =====================================================
       질문하기 버튼 글씨
       ===================================================== */

    div[data-testid="stButton"] button p {

        color: white !important;

        font-size: 30px !important;

        font-weight: 800 !important;

    }


    /* =====================================================
       버튼 Hover
       ===================================================== */

    div[data-testid="stButton"] button:hover {

        background-color: #3f69e8 !important;

        color: white !important;

        border: none !important;

    }


    /* =====================================================
       질문 / 답변 Container
       ===================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {

        border-radius: 16px !important;

        border: 2px solid #e4e7ec !important;

        padding: 10px !important;

    }


    /* =====================================================
       일반 답변 글씨
       ===================================================== */

    div[data-testid="stMarkdownContainer"] p {

        font-size: 26px;

        line-height: 1.8;

    }


    /* =====================================================
       답변 목록
       ===================================================== */

    div[data-testid="stMarkdownContainer"] li {

        font-size: 25px;

        line-height: 1.8;

        margin-bottom: 7px;

    }


    /* =====================================================
       메인 제목에는 영향을 주지 않습니다.
       ===================================================== */

    div[data-testid="stMarkdownContainer"] h1 {

        font-size: 40px !important;

        margin-top: 20px !important;

        margin-bottom: 20px !important;

    }


    /* =====================================================
       답변 
       ===================================================== */

    div[data-testid="stMarkdownContainer"] h2 {

        font-size: 35px !important;

        margin-top: 25px !important;

        margin-bottom: 15px !important;

    }


    /* =====================================================
       답변 
       ===================================================== */

    div[data-testid="stMarkdownContainer"] h3 {

        font-size: 31px !important;

        margin-top: 22px !important;

        margin-bottom: 12px !important;

    }


    /* =====================================================
       구분선
       ===================================================== */

    hr {

        margin-top: 25px !important;

        margin-bottom: 25px !important;

    }


    /* =====================================================
       오류 메시지
       ===================================================== */

    div[data-testid="stAlert"] {

        border-radius: 14px !important;

    }


    /* =====================================================
       Streamlit 메뉴 숨기기
       ===================================================== */

    #MainMenu {

        visibility: hidden;

    }


    footer {

        visibility: hidden;

    }


    /* =====================================================
       작은 화면 대응
       ===================================================== */

    @media (max-width: 768px) {

        .main-title {

            font-size: 50px;

            letter-spacing: -1px;

            margin-bottom: 45px;

        }


        div[data-testid="stTextInput"] label p {

            font-size: 25px !important;

        }


        div[data-testid="stTextInput"] input {

            font-size: 22px !important;

            height: 64px !important;

        }


        div[data-testid="stButton"] button p {

            font-size: 25px !important;

        }

    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STEP 6. 메인 제목
#
# st.title() 대신 별도의 HTML 제목을 사용합니다.
#
# 이렇게 하면 화면 중앙 정렬과
# 글자 크기를 정확하게 조절할 수 있습니다.
# ============================================================

st.markdown(
    """
    <div class="main-title">
        Chatbot powered by Bedrock
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STEP 7. 질문 입력
# ============================================================

question = st.text_input(

    "질문을 입력하세요.",

    placeholder="예: LLM이 뭐야?",

    key="question_input"

)


# ============================================================
# STEP 8. 질문하기 버튼
# ============================================================

ask_button = st.button(

    "질문하기",

    use_container_width=True

)


# ============================================================
# STEP 9. 질문하기 버튼 클릭
# ============================================================

if ask_button:

    # 입력값 앞뒤 공백 제거
    question = question.strip()


    # ========================================================
    # 질문을 입력하지 않은 경우
    # ========================================================

    if not question:

        st.warning(
            "질문을 입력한 후 질문하기 버튼을 눌러주세요."
        )


    # ========================================================
    # 질문을 입력한 경우
    # ========================================================

    else:

        try:

            # =================================================
            # Bedrock 응답 생성
            # =================================================

            with st.spinner(
                "Amazon Bedrock이 답변을 생성하고 있습니다..."
            ):

                response = bedrock.converse(

                    modelId=MODEL_ID,

                    messages=[
                        {
                            "role": "user",

                            "content": [
                                {
                                    "text": question
                                }
                            ]
                        }
                    ],

                    inferenceConfig={
                        "maxTokens": 2000,
                        "temperature": 0.0
                    }

                )


            # =================================================
            # STEP 10. AI 답변 추출
            # =================================================

            answer = (

                response["output"]

                ["message"]

                ["content"][0]

                ["text"]

            )


            # =================================================
            # STEP 11. 최신 질문 저장
            #
            # 이전 질문은 새로운 질문으로 교체됩니다.
            # =================================================

            st.session_state.latest_question = question


            # =================================================
            # STEP 12. 최신 답변 저장
            #
            # 이전 답변도 새로운 답변으로 교체됩니다.
            # =================================================

            st.session_state.latest_answer = answer


        # =====================================================
        # AWS Bedrock 오류
        # =====================================================

        except ClientError as error:

            error_code = (
                error.response["Error"]["Code"]
            )

            error_message = (
                error.response["Error"]["Message"]
            )

            st.error(
                f"AWS Bedrock 오류\n\n"
                f"Error Code: {error_code}\n\n"
                f"{error_message}"
            )


        # =====================================================
        # 일반 Python 오류
        # =====================================================

        except Exception as error:

            st.error(
                f"Python 오류가 발생했습니다.\n\n{error}"
            )


# ============================================================
# STEP 13. 최신 사용자 질문 표시
# ============================================================

if st.session_state.latest_question:

    st.write("")
    st.write("")


    with st.container(border=True):

        st.markdown(
            "### 🔵 사용자 질문"
        )

        st.markdown(
            f"## {st.session_state.latest_question}"
        )


# ============================================================
# STEP 14. 최신 Bedrock 답변 표시
# ============================================================

if st.session_state.latest_answer:

    st.write("")


    with st.container(border=True):

        st.markdown(
            "### 🟠 Amazon Bedrock 답변"
        )


        st.divider()


        st.markdown(
            st.session_state.latest_answer
        )