# ============================================================
# 01_invoke_model_example.py
#
# Amazon Bedrock - InvokeModel API 실습
# 모델: Amazon Nova Lite
#
# ============================================================
#
# [수업 목표]
#
# 1. InvokeModel API가 무엇인지 이해합니다.
#
# 2. InvokeModel에서는 모델이 요구하는 Native JSON 형식을
#    개발자가 직접 만들어 전달한다는 것
# 3. Python Dictionary를 json.dumps()를 이용하여
#    JSON 문자열로 변환
# 4. invoke_model()의 응답 Body를 직접 읽고
#    json.loads()로 변환
# 5. Converse API와 어떤 차이가 있는지 비교합니다.

# ============================================================
# 0. 필요한 라이브러리 Import
# ============================================================

import os
import json
import boto3

from dotenv import load_dotenv
from botocore.exceptions import ClientError, NoCredentialsError


# ============================================================
# boto3
# ------------------------------------------------------------
# Python 프로그램에서 AWS 서비스를 사용하기 위한
# AWS SDK입니다.
# Python
#   ↓
# boto3
#   ↓
# AWS API
# json
# ------------------------------------------------------------
# Python 데이터와 JSON 데이터를 변환하기 위해 사용합니다.
#
# InvokeModel에서는 이 json 라이브러리가 매우 중요합니다.
#
# Python Dictionary → JSON
#
#     json.dumps()
#
# JSON → Python Dictionary
#
#     json.loads()
#
#
# dotenv
# ------------------------------------------------------------
# .env 파일에 저장한 설정값을 Python으로 가져오기 위해
# 사용합니다.
# ============================================================
# 1. .env 파일 읽기
# ============================================================

load_dotenv()


# ============================================================
# .env를 사용하는 이유
# ------------------------------------------------------------
# Region이나 Model ID 등을 Python 코드에 직접 작성할 수도
# 있습니다.
# 하지만 별도의 .env 파일로 분리하면 설정 변경이 쉽습니다.
#
# 예:
#
# TEXT_REGION=us-east-1
# TEXT_MODEL_ID=amazon.nova-lite-v1:0
#
# Python에서는 os.getenv()로 읽습니다.
# ============================================================
# 2. AWS Region 설정
# ============================================================

region = os.getenv(
    "TEXT_REGION",
    "us-east-1"
)


# ============================================================
# 3. Foundation Model ID 설정
# ============================================================

model_id = os.getenv(
    "TEXT_MODEL_ID",
    "amazon.nova-lite-v1:0"
)

# ============================================================
# Region과 Model ID의 차이
# ------------------------------------------------------------
#
# Region
#     → 어느 AWS 리전의 Bedrock에 접속할 것인가?
#
# Model ID
#     → Bedrock에서 어떤 Foundation Model을 사용할 것인가?
#
#
# 예:
#
# us-east-1
#       ↓
# Amazon Bedrock
#       ↓
# amazon.nova-lite-v1:0
# ============================================================
# 4. 현재 설정 출력
# ============================================================

print("=" * 60)
print("01. Amazon Bedrock InvokeModel API")
print("=" * 60)

print(f"AWS Region : {region}")
print(f"Model ID   : {model_id}")

print("=" * 60)


# ============================================================
# 5. Bedrock Runtime Client 생성
# ============================================================

client = boto3.client(
    "bedrock-runtime",
    region_name=region
)


# ============================================================
# 왜 "bedrock-runtime"인가?
# ------------------------------------------------------------
# Bedrock의 Foundation Model을 실제로 호출할 때
# bedrock-runtime Client를 사용합니다.
#
# 쉽게 생각하면:
#
# boto3.client()
#       ↓
# "bedrock-runtime"
#       ↓
# Bedrock 모델 실행용 Client 생성
#
# 이후 이 client를 이용하여
#
# client.invoke_model()
#
# 또는
#
# client.converse()
#
# 등을 호출할 수 있습니다.

# ============================================================
# 6. 사용자 질문
# ============================================================

prompt = "AWS EC2 인스턴스 생성 전 확인할 항목을 3개로 알려줘."

print("\n[사용자 질문]")
print(prompt)


# ============================================================
# 7. InvokeModel Request Body 생성
# ============================================================

request_body = {

    "messages": [
        {
            "role": "user",

            "content": [
                {
                    "text": prompt
                }
            ]
        }
    ],

    "inferenceConfig": {

        "maxTokens": 300,

        "temperature": 0.3
    }
}


# ============================================================
# ★★★★★ 가장중요한 부분 5 ★★★★★
#
# InvokeModel의 핵심 특징
# ------------------------------------------------------------
#
# InvokeModel은 모델이 요구하는 Native Request 형식을
# 개발자가 직접 만들어야 합니다.
#
# 현재는 Amazon Nova Lite를 사용하므로
# Nova가 이해할 수 있는 구조로 Request Body를 만듭니다.
#
#
# request_body
#
# {
#     "messages": [...],
#     "inferenceConfig": {...}
# }
#
#
# 중요한 점:
#
# 다른 모델을 사용하면 Native Request 구조가
# 달라질 수 있습니다.
#
# InvokeModel은 모델의 Native API 형식을 이해하는 것이 중요합니다.
# [Converse와 가장 큰 차이]
#
# InvokeModel
#
# 모델별 Native Request 형식을 직접 처리
#
# Converse
#
# Bedrock에서 제공하는 공통 messages 인터페이스 사용
#
# ============================================================


# ============================================================
# 8. Python Dictionary → JSON 변환
# ============================================================

body = json.dumps(request_body)


# ============================================================
# ★★★★★ 가장중요 강조 포인트 6 ★★★★★
#
# 왜 json.dumps()가 필요한가?
# ------------------------------------------------------------
#
# request_body는 Python Dictionary입니다.
#
# type(request_body)
#
#     dict
#
#
# 하지만 invoke_model()의 body에는 JSON 형식의
# 데이터를 전달해야 합니다.
#
#
# 따라서:
#
# Python Dictionary
#
#     request_body
#
#          ↓
#
#     json.dumps()
#
#          ↓
#
# JSON 문자열
#
#
# dumps의 의미를 다음처럼 기억하면 편합니다.
#
# dumps
#     Python → JSON 문자열
#
# loads
#     JSON 문자열 → Python
#
# ============================================================


# ============================================================
# 9. InvokeModel API 호출
# ============================================================

try:

    print("\nInvokeModel API를 호출하고 있습니다...\n")

    response = client.invoke_model(

        modelId=model_id,

        contentType="application/json",

        accept="application/json",

        body=body
    )


    # ========================================================
    # ★★★★★ 가장중요 인트 7 ★★★★★
    #
    # 이것이 InvokeModel API 호출의 핵심 코드입니다.
    #
    # client.invoke_model()
    #
    #
    # modelId
    #     어떤 모델을 사용할 것인가?
    #
    # contentType
    #     모델에게 보내는 데이터 형식
    #
    # accept
    #     모델에서 받고 싶은 데이터 형식
    #
    # body
    #     모델에게 전달할 실제 JSON 데이터
    #
    #
    # 전체 구조:
    #
    # Python
    #   ↓
    # boto3
    #   ↓
    # invoke_model()
    #   ↓
    # Amazon Bedrock
    #   ↓
    # Amazon Nova Lite
    #
    # ========================================================


    # ========================================================
    # 10. 응답 Body 읽기
    # ========================================================

    response_body = response["body"].read()


    # ========================================================
    # ★★★★★ 가장중요 강조 포인트 8 ★★★★★
    #
    # InvokeModel의 응답은 바로 일반 Python dict 안의
    # 텍스트로 꺼내는 형태가 아니라
    # response["body"]를 읽어야 합니다.
    #
    #
    # response
    #     ↓
    # response["body"]
    #     ↓
    # .read()
    #     ↓
    # 응답 데이터
    #
    # ========================================================


    # ========================================================
    # 11. JSON → Python Dictionary 변환
    # ========================================================

    result = json.loads(response_body)


    # ========================================================
    # ★★★★★ 가장주요 강조 포인트 9 ★★★★★
    #
    # 모델에서 받은 JSON 응답을 Python에서 쉽게 사용하기
    # 위해 json.loads()를 사용합니다.
    #
    #
    # JSON
    #     ↓
    # json.loads()
    #     ↓
    # Python Dictionary
    #
    #
    # 앞에서 사용한 dumps와 반대입니다.
    #
    # json.dumps()
    #     Python → JSON
    #
    # json.loads()
    #     JSON → Python
    #
    # ========================================================


    # ========================================================
    # 12. AI 답변 추출
    # ========================================================

    answer = result[
        "output"
    ][
        "message"
    ][
        "content"
    ][0][
        "text"
    ]


    # ========================================================
    # 13. 결과 출력
    # ========================================================

    print("=" * 60)
    print("InvokeModel 응답")
    print("=" * 60)

    print(answer)

    print("=" * 60)


# ============================================================
# 14. AWS 인증 오류
# ============================================================

except NoCredentialsError:

    print("\n[AWS 인증 오류]")
    print("AWS 인증정보를 찾을 수 없습니다.")
    print("다음 명령으로 AWS 인증 상태를 확인하세요:")
    print("aws sts get-caller-identity")


# ============================================================
# 15. AWS Bedrock API 오류
# ============================================================

except ClientError as e:

    print("\n[AWS Bedrock 오류]")

    print(
        "Error Code    :",
        e.response["Error"]["Code"]
    )

    print(
        "Error Message :",
        e.response["Error"]["Message"]
    )


# ============================================================
# 16. 기타 오류
# ============================================================

except Exception as e:

    print("\n[프로그램 오류]")

    print("Error Type :", type(e).__name__)
    print("Message    :", str(e))