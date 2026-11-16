# ============================================================
# converse_basic_api.py
#
# Amazon Bedrock Converse API - 기본 테스트
#
# 목적:
# 사용자 질문 1개를 Claude에게 전달하고
# AI 답변을 출력합니다.
#
# Region : us-east-1
# API    : Converse API
# ============================================================

import boto3
from botocore.exceptions import ClientError


# ============================================================
# 1. AWS Region
# ============================================================

AWS_REGION = "us-east-1"


# ============================================================
# 2. Model ID
#
# 중요:
# 현재 AWS 계정에서 사용할 수 있는
# Inference Profile ID를 사용해야 합니다.
# ============================================================

MODEL_ID = "global.anthropic.claude-sonnet-4-6"


# ============================================================
# 3. Bedrock Runtime Client
# ============================================================

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION
)


# ============================================================
# 4. 사용자 질문
# ============================================================

prompt = "My name is Youngchul. What is Amazon Bedrock?"


# ============================================================
# 5. messages 생성
# ============================================================

messages = [
    {
        "role": "user",
        "content": [
            {
                "text": prompt
            }
        ]
    }
]


# ============================================================
# 6. 실행 정보
# ============================================================

print()
print("========================================")
print("Amazon Bedrock Converse API")
print("Basic Test")
print("========================================")

print("Region   :", AWS_REGION)
print("Model ID :", MODEL_ID)
print("Question :", prompt)

print("----------------------------------------")


# ============================================================
# 7. Converse API 호출
# ============================================================

try:

    response = bedrock.converse(

        modelId=MODEL_ID,

        messages=messages,

        inferenceConfig={
            "maxTokens": 1024,
            "temperature": 0.0
        }
    )


    # ========================================================
    # 8. AI 답변 추출
    # ========================================================

    response_message = response["output"]["message"]

    response_text = response_message["content"][0]["text"]


    # ========================================================
    # 9. 결과 출력
    # ========================================================

    print()
    print("AI Response")
    print("----------------------------------------")

    print(response_text)


    # ========================================================
    # 10. Token 사용량
    # ========================================================

    if "usage" in response:

        usage = response["usage"]

        print()
        print("----------------------------------------")
        print("Token Usage")
        print("----------------------------------------")

        print(
            "Input Tokens :",
            usage.get("inputTokens", 0)
        )

        print(
            "Output Tokens:",
            usage.get("outputTokens", 0)
        )

        print(
            "Total Tokens :",
            usage.get("totalTokens", 0)
        )


# ============================================================
# 11. AWS 오류
# ============================================================

except ClientError as error:

    print()
    print("========================================")
    print("AWS Bedrock Error")
    print("========================================")

    print(
        "Error Code   :",
        error.response["Error"]["Code"]
    )

    print(
        "Error Message:",
        error.response["Error"]["Message"]
    )


# ============================================================
# 12. 기타 오류
# ============================================================

except Exception as error:

    print()
    print("========================================")
    print("Python Error")
    print("========================================")

    print(error)