# ============================================================
# AWS Bedrock + Streamlit 이미지 생성 애플리케이션
#
# 사용 기술
#   - Streamlit : 웹 화면 구성
#   - boto3     : AWS 서비스 호출
#   - Bedrock   : 생성형 AI 모델 실행
#   - Stability AI Stable Image Core : 이미지 생성
#   - Pillow    : 생성된 이미지 처리
#
# 실행:
#   streamlit run app.py
# ============================================================


# ------------------------------------------------------------
# 1. 필요한 라이브러리 가져오기
# ------------------------------------------------------------

# Streamlit 웹 애플리케이션을 만들기 위한 라이브러리
import streamlit as st

# Python에서 AWS 서비스를 사용하기 위한 AWS SDK
import boto3

# Python Dictionary와 JSON 데이터를 변환하기 위한 라이브러리
import json

# 이미지를 Python에서 처리하기 위한 Pillow 라이브러리
from PIL import Image

# 메모리에서 이미지 데이터를 파일처럼 처리하기 위한 라이브러리
import io

# Bedrock에서 반환된 Base64 이미지를 디코딩하기 위한 라이브러리
import base64

# 이미지 생성 시 매번 다른 seed를 사용하기 위한 라이브러리
import random

# AWS 오류를 처리하기 위한 라이브러리
from botocore.exceptions import ClientError


# ------------------------------------------------------------
# 2. Streamlit 웹 페이지 제목
# ------------------------------------------------------------

st.title("Building with Amazon Bedrock")

st.subheader("Stable Image Core Demo")

st.write(
    "텍스트 프롬프트를 입력하면 "
    "Amazon Bedrock의 Stable Image Core 모델이 이미지를 생성합니다."
)


# ------------------------------------------------------------
# 3. 이미지 스타일 목록
# ------------------------------------------------------------

# 사용자가 SelectBox에서 선택할 수 있는 스타일입니다.
#
# 주의:
# Stable Image Core API에 style_preset이라는 별도 파라미터를
# 전달하는 것이 아니라 프롬프트 뒤에 스타일 이름을 추가합니다.
#
# 예:
#
# 입력:
#   A cute cat
#
# cinematic 선택:
#   A cute cat, cinematic style

sd_presets = [

    "None",

    "3d-model",

    "analog-film",

    "anime",

    "cinematic",

    "comic-book",

    "digital-art",

    "enhance",

    "fantasy-art",

    "isometric",

    "line-art",

    "low-poly",

    "modeling-compound",

    "neon-punk",

    "origami",

    "photographic",

    "pixel-art",

    "tile-texture",
]


# ------------------------------------------------------------
# 4. Amazon Bedrock Runtime Client 생성
# ------------------------------------------------------------

# boto3는 Python에서 AWS 서비스를 사용할 수 있게 해주는 SDK입니다.
#
# 여기서는 Amazon Bedrock 모델을 실행해야 하기 때문에
#
# bedrock-runtime
#
# 서비스를 사용합니다.

bedrock_runtime = boto3.client(

    service_name="bedrock-runtime",

    # Stable Image Core를 사용할 AWS Region
    region_name="us-west-2"
)


# ------------------------------------------------------------
# 5. 이미지 생성 함수
# ------------------------------------------------------------

def generate_image(text, style):

    """
    사용자가 입력한 텍스트를 Bedrock으로 보내서
    이미지를 생성하는 함수입니다.

    Parameters
    ----------
    text : str
        사용자가 입력한 이미지 설명

    style : str
        사용자가 선택한 이미지 스타일

    Returns
    -------
    str
        Base64 형식의 이미지 문자열
    """


    # --------------------------------------------------------
    # Step 1
    # 선택한 스타일을 Prompt에 추가
    # --------------------------------------------------------

    if style != "None":

        text = f"{text}, {style} style"


    # --------------------------------------------------------
    # Step 2
    # Seed 생성
    # --------------------------------------------------------

    # seed는 이미지 생성 결과를 결정하는 숫자입니다.
    #
    # 같은 Prompt + 같은 Seed
    # → 비슷하거나 동일한 결과를 재현할 때 사용할 수 있습니다.
    #
    # 매번 다른 이미지를 만들기 위해 랜덤 seed를 사용합니다.

    seed = random.randint(
        0,
        4294967295
    )


    # --------------------------------------------------------
    # Step 3
    # Bedrock 모델에게 전달할 요청 데이터 생성
    # --------------------------------------------------------

    body = {

        # 사용자가 입력한 이미지 설명
        "prompt": text,

        # 생성할 이미지 비율
        "aspect_ratio": "1:1",

        # 이미지 생성 seed
        "seed": seed,

        # 출력 이미지 형식
        "output_format": "png"
    }


    # --------------------------------------------------------
    # Step 4
    # Python Dictionary → JSON 변환
    # --------------------------------------------------------

    body = json.dumps(body)


    # --------------------------------------------------------
    # Step 5
    # 사용할 Bedrock 모델 ID
    # --------------------------------------------------------

    model_id = "stability.stable-image-core-v1:1"


    # --------------------------------------------------------
    # Step 6
    # Amazon Bedrock 모델 호출
    # --------------------------------------------------------

    response = bedrock_runtime.invoke_model(

        # 사용할 AI 모델
        modelId=model_id,

        # 모델에게 전달할 데이터
        body=body,

        # 요청 데이터 형식
        contentType="application/json",

        # 응답 데이터 형식
        accept="application/json"
    )


    # --------------------------------------------------------
    # Step 7
    # Bedrock 응답 읽기
    # --------------------------------------------------------

    # response["body"]는 StreamingBody 객체입니다.
    #
    # .read()
    #
    # 를 사용하여 실제 데이터를 읽습니다.

    response_body = json.loads(
        response["body"].read()
    )


    # --------------------------------------------------------
    # Step 8
    # 이미지 생성 실패 여부 확인
    # --------------------------------------------------------

    finish_reasons = response_body.get(
        "finish_reasons",
        []
    )

    if finish_reasons:

        if finish_reasons[0] is not None:

            raise RuntimeError(
                f"이미지 생성 실패: {finish_reasons[0]}"
            )


    # --------------------------------------------------------
    # Step 9
    # 생성된 이미지 추출
    # --------------------------------------------------------

    images = response_body.get(
        "images",
        []
    )

    # 이미지가 없는 경우 오류 발생
    if not images:

        raise RuntimeError(
            "Bedrock 응답에 생성된 이미지가 없습니다."
        )


    # 첫 번째 이미지 반환
    return images[0]


# ------------------------------------------------------------
# 6. Base64 → PIL Image 변환 함수
# ------------------------------------------------------------

def base64_to_pil(base64_string):

    """
    Bedrock에서 반환된 Base64 문자열을
    Streamlit에서 출력할 수 있는 PIL Image로 변환합니다.
    """


    # --------------------------------------------------------
    # Step 1
    # Base64 → Binary
    # --------------------------------------------------------

    imgdata = base64.b64decode(
        base64_string
    )


    # --------------------------------------------------------
    # Step 2
    # Binary → PIL Image
    # --------------------------------------------------------

    image = Image.open(
        io.BytesIO(imgdata)
    )


    # 이미지 반환
    return image


# ------------------------------------------------------------
# 7. Streamlit UI - Style 선택
# ------------------------------------------------------------

style = st.selectbox(

    "Select Style",

    sd_presets
)


# ------------------------------------------------------------
# 8. Streamlit UI - Prompt 입력
# ------------------------------------------------------------

prompt = st.text_input(

    "Enter prompt",

    placeholder="예: A cute robot walking in Seoul at night"
)


# ------------------------------------------------------------
# 9. Generate Image 버튼
# ------------------------------------------------------------

if st.button("Generate Image"):


    # --------------------------------------------------------
    # Prompt 입력 여부 확인
    # --------------------------------------------------------

    if prompt:


        try:

            # ------------------------------------------------
            # 사용자에게 처리 중이라는 메시지 표시
            # ------------------------------------------------

            with st.spinner(
                "Amazon Bedrock에서 이미지를 생성하고 있습니다..."
            ):


                # --------------------------------------------
                # Bedrock 이미지 생성
                # --------------------------------------------

                base64_image = generate_image(

                    prompt,

                    style
                )


                # --------------------------------------------
                # Base64 → PIL Image
                # --------------------------------------------

                image = base64_to_pil(
                    base64_image
                )


            # ------------------------------------------------
            # 생성된 이미지 화면 출력
            # ------------------------------------------------

            st.success(
                "이미지 생성 완료!"
            )


            st.image(

                image,

                caption=f"Prompt: {prompt}",

                use_container_width=True
            )


        # ----------------------------------------------------
        # AWS Bedrock 관련 오류
        # ----------------------------------------------------

        except ClientError as e:

            error_message = e.response[
                "Error"
            ].get(
                "Message",
                str(e)
            )

            st.error(
                f"AWS Bedrock 오류: {error_message}"
            )


        # ----------------------------------------------------
        # 기타 Python 오류
        # ----------------------------------------------------

        except Exception as e:

            st.error(
                f"이미지 생성 중 오류가 발생했습니다: {e}"
            )


    else:

        # Prompt를 입력하지 않은 경우
        st.warning(
            "Prompt를 입력하세요."
        )