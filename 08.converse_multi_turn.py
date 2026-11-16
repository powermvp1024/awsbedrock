# ============================================================
# converse_multi_turn.py
#
# Amazon Bedrock Converse API
# Multi-turn Conversation 테스트
#
# 목적:
# 이전 대화를 messages에 저장하고
# 다음 질문을 할 때 이전 대화를 함께 전달합니다.
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
# 4. 대화 기록 저장용 List
#
# 매우 중요합니다.
#
# Converse API는 이전 대화를 자동으로 기억하는 것이 아니라
# 우리가 messages에 대화를 계속 추가해서 다시 보내줍니다.
# ============================================================

messages = []


# ============================================================
# 5. 첫 번째 질문
# ============================================================

first_question = (
    "My name is Youngchul. "
    "I am studying Amazon Bedrock. "
    "Please remember my name."
)


# 사용자 질문을 messages에 추가
messages.append(
    {
        "role": "user",
        "content": [
            {
                "text": first_question
            }
        ]
    }
)


print()
print("========================================")
print("Amazon Bedrock Converse API")
print("Multi-turn Conversation Test")
print("========================================")

print("Region   :", AWS_REGION)
print("Model ID :", MODEL_ID)

print()
print("USER 1:")
print(first_question)


# ============================================================
# 6. 첫 번째 Converse 호출
# ============================================================

try:

    response1 = bedrock.converse(

        modelId=MODEL_ID,

        messages=messages,

        inferenceConfig={
            "maxTokens": 1024,
            "temperature": 0.0
        }
    )


    # ========================================================
    # 7. 첫 번째 AI 답변
    # ========================================================

    assistant_message1 = response1["output"]["message"]

    assistant_text1 = (
        assistant_message1["content"][0]["text"]
    )


    print()
    print("ASSISTANT 1:")
    print(assistant_text1)


    # ========================================================
    # ★ 매우 중요
    #
    # AI의 첫 번째 답변도 messages에 추가합니다.
    #
    # 현재 messages:
    #
    # user
    # assistant
    #
    # 형태가 됩니다.
    # ========================================================

    messages.append(assistant_message1)


    # ========================================================
    # 8. 두 번째 질문
    #
    # 이번에는 이름을 다시 알려주지 않습니다.
    # ========================================================

    second_question = (
        "What is my name?"
    )


    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "text": second_question
                }
            ]
        }
    )


    print()
    print("----------------------------------------")

    print()
    print("USER 2:")
    print(second_question)


    # ========================================================
    # 9. 두 번째 Converse 호출
    #
    # messages 안에는 이미:
    #
    # user 질문1
    # assistant 답변1
    # user 질문2
    #
    # 가 들어 있습니다.
    # ========================================================

    response2 = bedrock.converse(

        modelId=MODEL_ID,

        messages=messages,

        inferenceConfig={
            "maxTokens": 1024,
            "temperature": 0.0
        }
    )


    # ========================================================
    # 10. 두 번째 AI 답변
    # ========================================================

    assistant_message2 = response2["output"]["message"]

    assistant_text2 = (
        assistant_message2["content"][0]["text"]
    )


    print()
    print("ASSISTANT 2:")
    print(assistant_text2)


    # ========================================================
    # 11. 두 번째 답변도 대화 기록에 추가
    # ========================================================

    messages.append(assistant_message2)


    # ========================================================
    # 12. 전체 대화 구조 출력
    # ========================================================

    print()
    print("========================================")
    print("Conversation History")
    print("========================================")


    for index, message in enumerate(messages, start=1):

        role = message["role"]

        text = message["content"][0]["text"]

        print()
        print(f"{index}. {role.upper()}")
        print(text)


# ============================================================
# 13. AWS 오류
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
# 14. 기타 오류
# ============================================================

except Exception as error:

    print()
    print("========================================")
    print("Python Error")
    print("========================================")

    print(error)