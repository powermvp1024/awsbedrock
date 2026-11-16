# ============================================================
# [시스템 아키텍처 및 기술 명세서]
# 03_streaming_chatbot.py
# ============================================================
#
# 1. 프로그램 개요:
#    Amazon Bedrock의 최신 파운데이션 모델(Amazon Nova Lite, Stability AI Stable Image Core)을
#    Streamlit 웹 프레임워크와 연동하여 실시간 스트리밍 대화 및 프롬프트 기반 고화질 이미지 생성을
#    제공하는 교육/실습용 통합 웹 애플리케이션입니다.
#
# 2. 핵심 함수(Functions) 명세:
#    - create_text_bedrock_client():
#        * 역할: 텍스트 생성 전용 AWS Bedrock Runtime Boto3 클라이언트를 생성합니다.
#        * 특징: Streamlit의 @st.cache_resource 데코레이터를 적용하여 스크립트 재실행 시
#                중복 세션 연결을 방지하고 캐싱된 클라이언트를 재사용합니다.
#    - create_image_bedrock_client():
#        * 역할: 이미지 생성 모델(Stable Diffusion)이 호스팅된 리전 전용 Bedrock 클라이언트를 생성합니다.
#        * 특징: 동일하게 리소스 캐싱을 통해 네트워크 오버헤드를 최소화합니다.
#    - convert_to_image_prompt(user_prompt):
#        * 역할: 사용자가 한국어(또는 비정형 언어)로 입력한 텍스트를 Nova Lite를 통해
#                Stable Diffusion 규격에 최적화된 고품질 영문 프롬프트로 변환 및 확장합니다.
#        * 특징: 스트리밍이 필요 없는 단일 변환 작업이므로 converse API(동기)를 사용합니다.
#
# 3. 핵심 클래스(Classes) 및 예외 처리 구조:
#    - boto3.client:
#        * AWS Bedrock Runtime 서비스 엔드포인트와 통신하는 저수준 클라이언트 클래스입니다.
#    - ClientError (botocore.exceptions):
#        * AWS 서비스 측 HTTP 4xx/5xx 오류(잘못된 파라미터, 권한 부족, 모델 접근 비활성화 등)를
#          캡처하며 내부 error.response 구조를 통해 구체적 에러 코드와 메시지를 추출합니다.
#    - NoCredentialsError (botocore.exceptions):
#        * AWS CLI 인증 프로파일이나 환경변수(.env)의 자격 증명을 찾을 수 없을 때 발생하는 예외입니다.
#
# 4. 주요 상태 및 런타임 객체(Objects):
#    - text_bedrock / image_bedrock:
#        * 각각 TEXT_REGION과 IMAGE_REGION 엔드포인트로 바인딩된 BedrockRuntime.Client 객체입니다.
#    - st.session_state:
#        * Streamlit 세션 상태를 저장하는 딕셔너리형 래퍼 객체입니다. 사용자가 입력하거나 버튼을 클릭할 때
#          전체 스크립트가 위에서 아래로 재실행되어도 대화 기록(messages) 및 생성 이미지(generated_images)를
#          메모리에 유지합니다.
#    - placeholder (st.empty()):
#        * 화면의 특정 영역을 가변적으로 점유하는 컨테이너 객체입니다. 스트리밍 텍스트 청크(Delta)가
#          수신될 때마다 기존 텍스트를 실시간으로 덮어써서 타자기 효과를 구현합니다.
#
# ------------------------------------------------------------
# [동작 파이프라인]
#
# 기능 1 : AI 실시간 스트리밍 대화
#   사용자 입력 -> Streamlit st.chat_input -> Boto3 converse_stream API ->
#   Amazon Nova Lite -> EventStream 델타 수신 -> st.empty() 실시간 렌더링 -> session_state 저장
#
# 기능 2 : AI 고화질 이미지 생성
#   사용자 입력 -> Nova Lite (converse API) -> 영어 프롬프트 변환 ->
#   Stable Image Core (invoke_model API) -> Base64 페이로드 디코딩 ->
#   PNG 바이너리 바이트 변환 -> st.image 화면 출력 및 다운로드 버튼 제공
# ============================================================


# ============================================================
# STEP 01. 라이브러리 Import
# ============================================================

# 운영체제 환경변수 접근을 위한 Python 표준 라이브러리
import os

# Bedrock InvokeModel API의 JSON 페이로드 직렬화/역직렬화 라이브러리
import json

# Bedrock 이미지 모델이 반환하는 Base64 인코딩 문자열을 PNG 바이너리 바이트로 디코딩
import base64

# AWS 공식 Python SDK (Bedrock Runtime 통신용)
import boto3

# 반응형 대화형 웹 인터페이스 구축 프레임워크
import streamlit as st

# 로컬 .env 파일에 선언된 AWS 리전 및 모델 ID를 환경변수로 로드
from dotenv import load_dotenv

# AWS API 호출 과정에서 발생하는 인증 오류 및 클라이언트 예외 처리 클래스
from botocore.exceptions import (
    ClientError,
    NoCredentialsError
)


# ============================================================
# STEP 02. .env 파일 읽기
# ============================================================

# .env 파일로부터 환경변수를 주입받아 os.getenv로 호출 가능하게 설정
load_dotenv()


# ============================================================
# STEP 03. AWS Region 설정
# ============================================================

# Amazon Nova Lite 모델을 호출할 AWS 리전 (기본값: us-east-1 버지니아 북부)
TEXT_REGION = os.getenv(
    "TEXT_REGION",
    "us-east-1"
)

# Stability AI Stable Image Core 모델을 호출할 AWS 리전 (기본값: us-west-2 오리건)
IMAGE_REGION = os.getenv(
    "IMAGE_REGION",
    "us-west-2"
)


# ============================================================
# STEP 04. Bedrock Model ID 설정
# ============================================================

# 대화 및 프롬프트 변환을 담당할 Bedrock 텍스트 모델 식별자
TEXT_MODEL_ID = os.getenv(
    "TEXT_MODEL_ID",
    "amazon.nova-lite-v1:0"
)

# 이미지 생성을 담당할 Bedrock 이미지 파운데이션 모델 식별자
IMAGE_MODEL_ID = os.getenv(
    "IMAGE_MODEL_ID",
    "stability.stable-image-core-v1:1"
)


# ============================================================
# STEP 05. Streamlit 페이지 설정
# ============================================================

st.set_page_config(
    page_title="AWS Bedrock AI",
    page_icon="🐯",
    layout="centered",                 # 모바일 및 프로젝터 환경에서 가독성을 높이기 위해 중앙 정렬
    initial_sidebar_state="collapsed"   # 설정 사이드바는 초기 상태에서 접힘 처리
)


# ============================================================
# STEP 06. 학생 / 강의실용 대형 화면 CSS 주입
# ============================================================
# 빔프로젝터나 대형 모니터 환경에서 원거리 수강생도 코드와 글자를
# 명확하게 확인할 수 있도록 고대비/대형 폰트로 오버라이드합니다.
# ============================================================

st.markdown(
    """
<style>
/* 전체 웹 애플리케이션 배경 그라데이션 */
.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
}

/* 메인 블록 너비 및 여백 설정 */
.block-container {
    max-width: 1100px;
    padding-top: 2rem !important;
    padding-bottom: 7rem !important;
}

/* 최상단 메인 헤더 H1 */
h1 {
    text-align: center;
    font-size: 52px !important;
    line-height: 1.3 !important;
    font-weight: 800 !important;
    color: #1e293b !important;
    letter-spacing: -1.5px;
    margin-bottom: 15px !important;
}

/* 보조 헤더 H2 */
h2 {
    font-size: 40px !important;
    line-height: 1.4 !important;
    font-weight: 800 !important;
    color: #1e293b !important;
}

/* 섹션 타이틀 H3 */
h3 {
    font-size: 28px !important;
    line-height: 1.5 !important;
    font-weight: 750 !important;
    color: #1e293b !important;
}

/* 캡션(부연 설명) 텍스트 */
[data-testid="stCaptionContainer"] {
    text-align: center !important;
}
[data-testid="stCaptionContainer"] p {
    font-size: 23px !important;
    line-height: 1.6 !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
}

/* 일반 마크다운 본문 폰트 크기 및 줄간격 */
.stMarkdown p {
    font-size: 22px !important;
    line-height: 1.8 !important;
}

/* 모드 선택 라디오 버튼 영역 */
[data-testid="stRadio"] {
    margin-top: 18px;
    margin-bottom: 30px;
}
[data-testid="stRadio"] label p {
    font-size: 25px !important;
    font-weight: 750 !important;
    line-height: 1.5 !important;
    color: #1e293b !important;
}

/* 입력 폼 라벨 타이틀 */
[data-testid="stWidgetLabel"] p {
    font-size: 24px !important;
    line-height: 1.5 !important;
    font-weight: 800 !important;
    color: #1e293b !important;
}

/* 프롬프트 입력용 텍스트 영역(TextArea) */
[data-testid="stTextArea"] textarea {
    font-size: 25px !important;
    line-height: 1.7 !important;
    min-height: 180px !important;
    padding: 20px !important;
    color: #1e293b !important;
    border-radius: 14px !important;
}
[data-testid="stTextArea"] textarea::placeholder {
    font-size: 22px !important;
    color: #94a3b8 !important;
}

/* 종횡비 선택 드롭다운(Selectbox) */
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    min-height: 65px !important;
    font-size: 23px !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    font-size: 23px !important;
}

/* 실행 액션 버튼 */
.stButton button {
    min-height: 72px !important;
    border-radius: 15px !important;
    font-size: 25px !important;
    font-weight: 800 !important;
}
.stButton button p {
    font-size: 25px !important;
    font-weight: 800 !important;
}

/* 대화 메시지 컨테이너 카드 */
[data-testid="stChatMessage"] {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 22px 26px;
    margin-top: 14px;
    margin-bottom: 14px;
    box-shadow: 0 3px 12px rgba(15,23,42,0.05);
}
[data-testid="stChatMessage"] p {
    font-size: 22px !important;
    line-height: 1.9 !important;
}

/* 하단 대화 입력바 */
[data-testid="stChatInput"] {
    background: white !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 18px !important;
    min-height: 82px !important;
    box-shadow: 0 8px 30px rgba(15,23,42,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 24px !important;
    line-height: 1.6 !important;
    min-height: 64px !important;
    color: #1e293b !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    font-size: 21px !important;
    color: #94a3b8 !important;
}

/* 경고, 정보, 에러 알림 상자 */
[data-testid="stAlert"] {
    border-radius: 15px !important;
    padding: 18px !important;
}
[data-testid="stAlert"] p {
    font-size: 21px !important;
    line-height: 1.7 !important;
    font-weight: 500 !important;
}

/* 생성된 이미지 출력 카드 */
[data-testid="stImage"] img {
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(15,23,42,0.12);
}

/* 이미지 파일 다운로드 버튼 */
[data-testid="stDownloadButton"] button {
    min-height: 68px !important;
    border-radius: 15px !important;
}
[data-testid="stDownloadButton"] button p {
    font-size: 23px !important;
    font-weight: 800 !important;
}

/* 코드 스니펫 블록 */
code {
    font-size: 19px !important;
    line-height: 1.7 !important;
}

/* 사이드바 글자 크기 보정 */
[data-testid="stSidebar"] h1 { font-size: 34px !important; }
[data-testid="stSidebar"] h2 { font-size: 28px !important; }
[data-testid="stSidebar"] h3 { font-size: 24px !important; }
[data-testid="stSidebar"] p { font-size: 19px !important; line-height: 1.6 !important; }

/* Streamlit 기본 탑바 및 푸터 메뉴 제거 */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# STEP 07 & 08. Bedrock Client 생성 (캐싱 리소스 함수 정의)
# ============================================================

@st.cache_resource
def create_text_bedrock_client():
    """
    텍스트 모델(Nova Lite) 통신용 Bedrock Runtime 클라이언트 생성.
    st.cache_resource를 통해 앱 라이프사이클 동안 세션 객체를 재사용합니다.
    """
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=TEXT_REGION
    )


@st.cache_resource
def create_image_bedrock_client():
    """
    이미지 모델(Stable Image Core) 통신용 Bedrock Runtime 클라이언트 생성.
    이미지 모델이 서비스되는 특정 리전(us-west-2 등)으로 독립 연결합니다.
    """
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=IMAGE_REGION
    )


# ============================================================
# STEP 09. 실제 Bedrock Client 싱글톤 인스턴스 획득
# ============================================================

text_bedrock = create_text_bedrock_client()
image_bedrock = create_image_bedrock_client()


# ============================================================
# STEP 10. Session State 상태 저장소 초기화
# ============================================================
# Streamlit은 사용자 입력 시 전체 코드를 처음부터 재실행하므로,
# 휘발을 막기 위해 브라우저 세션에 대화 및 이미지 내역을 저장합니다.

if "messages" not in st.session_state:
    st.session_state.messages = []

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []


# ============================================================
# STEP 11. 이미지 Prompt 변환 함수 (Nova Lite 동기 호출)
# ============================================================

def convert_to_image_prompt(user_prompt):
    """
    사용자가 전달한 한국어/비정형 텍스트를 Stable Image Core에 적합한
    상세 영문 프롬프트로 재가공합니다. Nova Lite의 converse 동기 API를 사용합니다.
    """
    # 프롬프트 엔지니어링 가이드라인 설정
    instruction = f"""
You are an expert prompt engineer for an AI image generation model.

Convert the user's image request into one clear English
image-generation prompt.

Rules:
1. Preserve the user's requested main subject exactly.
2. Do not replace an animal with a human.
3. Do not add people unless the user explicitly asks for people.
4. Make the main subject very clear.
5. Add useful details about environment, lighting and composition.
6. Do not explain your answer.
7. Return only the final English image prompt.
8. Keep the prompt under 100 words.

User request:
{user_prompt}
"""

    # Amazon Nova Lite Converse API 동기 호출
    response = text_bedrock.converse(
        modelId=TEXT_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [{"text": instruction}]
            }
        ],
        inferenceConfig={
            "maxTokens": 250,
            "temperature": 0.1  # 프롬프트 변환의 일관성을 위해 낮은 창의성(온도) 부여
        }
    )

    # 응답 페이로드에서 완성된 텍스트 블록 추출
    content_blocks = response["output"]["message"]["content"]
    converted_text = ""
    for block in content_blocks:
        if "text" in block:
            converted_text += block["text"]

    return converted_text.strip()


# ============================================================
# STEP 12 ~ 14. 메인 제목 및 모드 전환 위젯
# ============================================================

st.title("🐯 AWS Bedrock AI")
st.caption("Amazon Nova Lite + Stable Image Core + Streamlit")

mode = st.radio(
    "사용할 AI 기능",
    [
        "🐯 AI 채팅",
        "🐯 이미지 생성"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


# ============================================================
# STEP 15. AI 채팅 모드 (실시간 스트리밍 대화)
# ============================================================

if mode == "🐯 AI 채팅":

    # 대화 기록이 없을 때 나타나는 웰컴 카드
    if len(st.session_state.messages) == 0:
        st.markdown(
            """
            <h2 style="text-align:center; margin-top:40px;">🐯 무엇을 도와드릴까요?</h2>
            <p style="text-align:center; color:#64748b; font-size:22px; line-height:1.9;">
            Amazon Nova Lite를 사용하는 AI 어시스턴트입니다.<br>
            AWS에 대해 자유롭게 질문해 주세요.
            </p>
            """,
            unsafe_allow_html=True
        )
        st.info("🐯 예: EC2와 Lambda의 차이점을 초보자도 이해할 수 있도록 설명해 주세요.")

    # 세션 상태에 저장되어 있던 기존 대화 메시지들을 순차적으로 화면에 렌더링
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 프롬프트 입력창
    prompt = st.chat_input("AWS에 대해 질문을 입력하세요...")

    if prompt:
        # 사용자 질문을 세션 상태에 추가하고 화면에 출력
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # ====================================================
        # STEP 16. Bedrock Converse 규격에 맞게 메시지 포맷 변환
        # ====================================================
        bedrock_messages = []
        for message in st.session_state.messages:
            bedrock_messages.append(
                {
                    "role": message["role"],
                    "content": [{"text": message["content"]}]
                }
            )

        # ====================================================
        # STEP 17. Amazon Nova Lite 호출 및 실시간 스트리밍 처리
        # ====================================================
        with st.chat_message("assistant"):
            try:
                # ConverseStream API 호출: 텍스트 생성 완료를 기다리지 않고 청크 단위로 스트림 수신
                response = text_bedrock.converse_stream(
                    modelId=TEXT_MODEL_ID,
                    messages=bedrock_messages,
                    inferenceConfig={
                        "maxTokens": 1000,
                        "temperature": 0.4
                    }
                )

                # 스트리밍 텍스트를 실시간으로 덮어쓸 동적 플레이스홀더 컨테이너
                placeholder = st.empty()
                full_response = ""

                # EventStream을 순회하며 실시간 청크(delta) 수신
                for event in response["stream"]:
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        if "text" in delta:
                            full_response += delta["text"]
                            # 생성 중 커서(▌) 효과 표시
                            placeholder.markdown(full_response + " ▌")

                # 완성된 최종 답변을 확정 렌더링 (커서 제거)
                placeholder.markdown(full_response)

                # 어시스턴트의 최종 답변을 세션 히스토리에 영구 저장
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response
                    }
                )

            # AWS 로컬 자격 증명(Access Key / IAM Role) 누락 예외
            except NoCredentialsError:
                st.error("🐯 AWS 인증정보를 찾을 수 없습니다.")
                st.code("aws sts get-caller-identity")

            # Bedrock API 수준의 에러(권한 부족, 모델 접근 비활성화 등) 핸들링
            except ClientError as error:
                error_code = error.response.get("Error", {}).get("Code", "Unknown")
                error_message = error.response.get("Error", {}).get("Message", str(error))

                st.error("🐯 AWS Bedrock 텍스트 생성 오류")
                st.write("**Error Code**")
                st.code(error_code)
                st.write("**Error Message**")
                st.code(error_message)

            # 기타 Python 런타임 일반 예외 처리
            except Exception as error:
                st.error("🐯 텍스트 생성 중 오류가 발생했습니다.")
                st.code(f"{type(error).__name__}: {error}")


# ============================================================
# STEP 18. AI 이미지 생성 모드
# ============================================================

else:
    # 이미지 생성 기능 소개 영역
    st.markdown(
        """
        <h2 style="text-align:center; margin-top:40px; margin-bottom:25px;">🐯 AI 이미지 생성</h2>
        <p style="text-align:center; color:#64748b; font-size:22px; line-height:1.9; margin-bottom:25px;">
        한국어 또는 영어로 원하는 이미지를 설명하세요.<br>
        Nova Lite가 이미지 생성용 Prompt를 만든 후 Stable Image Core가 이미지를 생성합니다.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.info("🐯 예: 잔디밭에 앉아 있는 귀여운 아기 호랑이")

    # STEP 19. 사용자 이미지 묘사 프롬프트 입력
    image_prompt = st.text_area(
        "만들고 싶은 이미지를 설명하세요",
        placeholder="예: 잔디밭에 앉아 있는 귀여운 아기 호랑이",
        height=190
    )

    # STEP 20. 이미지 출력 종횡비 선택 (정사각형, 가로, 세로)
    aspect_ratio = st.selectbox(
        "이미지 비율",
        ["1:1", "16:9", "9:16"],
        index=0
    )

    # STEP 21. 이미지 생성 실행 트리거 버튼
    generate_button = st.button("🐯 이미지 생성", type="primary", use_container_width=True)

    # STEP 22. 생성 버튼 클릭 시 파이프라인 시작
    if generate_button:
        # 빈 문자열 입력 유효성 검사
        if not image_prompt.strip():
            st.warning("🐯 만들고 싶은 이미지를 먼저 입력해 주세요.")
        else:
            try:
                # STEP 23. Nova Lite를 통해 한국어 프롬프트를 최적화된 영문 프롬프트로 변환
                with st.spinner("🐯 Nova Lite가 이미지 Prompt를 만들고 있습니다..."):
                    converted_prompt = convert_to_image_prompt(image_prompt.strip())

                if not converted_prompt:
                    raise RuntimeError("Nova Lite가 이미지 Prompt를 생성하지 못했습니다.")

                # STEP 24. 교육 목적으로 프롬프트 변환 전후의 결과를 시각적으로 대조 출력
                st.markdown("### 🐯 사용자가 입력한 Prompt")
                st.code(image_prompt.strip())

                st.markdown("### 🐯 Stable Image Core에 전달되는 Prompt")
                st.code(converted_prompt)

                # STEP 25 & 26. Stable Image Core 네이티브 규격의 JSON 요청 Body 구성
                request_body = {
                    "prompt": converted_prompt,
                    "aspect_ratio": aspect_ratio,
                    "output_format": "png"
                }
                request_json = json.dumps(request_body)

                # STEP 27. Bedrock InvokeModel API 동기 호출 (이미지 생성 작업 요청)
                with st.spinner("🐯 Stable Image Core가 이미지를 생성하고 있습니다..."):
                    response = image_bedrock.invoke_model(
                        modelId=IMAGE_MODEL_ID,
                        contentType="application/json",
                        accept="application/json",
                        body=request_json
                    )

                # STEP 28 & 29. 스트림 형태의 Response Body를 바이트로 읽고 JSON 역직렬화
                response_body = response["body"].read()
                result = json.loads(response_body)

                # STEP 30 & 31. 모델 측 오류 응답 및 이미지 배열 데이터 포함 여부 검증
                if result.get("errors"):
                    raise RuntimeError(str(result["errors"]))

                if "images" not in result or not result["images"]:
                    raise RuntimeError("Stable Image Core 응답에 이미지 데이터가 없습니다.")

                # STEP 32 & 33. Base64 텍스트 데이터를 디코딩하여 순수 PNG 바이너리 바이트로 변환
                base64_image = result["images"][0]
                image_bytes = base64.b64decode(base64_image)

                # STEP 34. 생성된 이미지 정보를 세션 히스토리에 기록
                st.session_state.generated_images.append(
                    {
                        "original_prompt": image_prompt.strip(),
                        "converted_prompt": converted_prompt,
                        "image": image_bytes
                    }
                )

                # STEP 35. 생성된 이미지를 화면에 렌더링
                st.image(
                    image_bytes,
                    caption=f"사용자 요청: {image_prompt.strip()}",
                    use_container_width=True
                )

                # STEP 36. 생성 완료 성공 알림 표시
                st.success("🐯 이미지 생성이 완료되었습니다.")

                # STEP 37. 사용자 로컬 저장을 위한 다운로드 버튼 제공
                st.download_button(
                    label="🐯 이미지 다운로드",
                    data=image_bytes,
                    file_name="stable_image_core.png",
                    mime="image/png",
                    use_container_width=True
                )

            # AWS 인증 오류 핸들링
            except NoCredentialsError:
                st.error("🐯 AWS 인증정보를 찾을 수 없습니다.")
                st.write("다음 명령으로 AWS 인증 상태를 확인하세요.")
                st.code("aws sts get-caller-identity")

            # Bedrock API 호출 예외 시 현재 환경변수 및 에러 코드 출력
            except ClientError as error:
                error_code = error.response.get("Error", {}).get("Code", "Unknown")
                error_message = error.response.get("Error", {}).get("Message", str(error))

                st.error("🐯 AWS Bedrock 호출 오류")
                st.write("**Error Code**")
                st.code(error_code)
                st.write("**Error Message**")
                st.code(error_message)

                # 디버깅을 위한 현재 환경변수 값 동시 노출
                st.write("**TEXT_REGION**")
                st.code(TEXT_REGION)
                st.write("**TEXT_MODEL_ID**")
                st.code(TEXT_MODEL_ID)
                st.write("**IMAGE_REGION**")
                st.code(IMAGE_REGION)
                st.write("**IMAGE_MODEL_ID**")
                st.code(IMAGE_MODEL_ID)

            # 기타 Python 실행 예외 핸들링
            except Exception as error:
                st.error("🐯 이미지 생성 중 오류가 발생했습니다.")
                st.code(f"{type(error).__name__}: {error}")


# ============================================================
# STEP 38. 사이드바 (실행 환경 정보 확인 및 세션 초기화)
# ============================================================

with st.sidebar:
    st.header("🐯 Bedrock 설정")

    # 텍스트 모델 정보
    st.subheader("🐯 텍스트 모델")
    st.write("**Region**")
    st.code(TEXT_REGION)
    st.write("**Model ID**")
    st.code(TEXT_MODEL_ID)
    st.write("**채팅 API**")
    st.code("ConverseStream")
    st.write("**Prompt 변환 API**")
    st.code("Converse")

    st.divider()

    # 이미지 모델 정보
    st.subheader("🐯 이미지 모델")
    st.write("**Region**")
    st.code(IMAGE_REGION)
    st.write("**Model ID**")
    st.code(IMAGE_MODEL_ID)
    st.write("**사용 API**")
    st.code("InvokeModel")

    st.divider()

    # 대화 기록 초기화 및 런타임 리로드
    if st.button("🐯 대화 내용 삭제", use_container_width=True):
        st.session_state.messages = []
        st.rerun()