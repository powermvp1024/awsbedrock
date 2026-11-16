# ============================================================
# AWS Lambda + Amazon Bedrock Converse API
# Amazon Nova Micro를 이용한 텍스트 생성 예제
# ============================================================


# ------------------------------------------------------------
# 01. 필요한 라이브러리 import
# ------------------------------------------------------------

# JSON 데이터를 처리하기 위한 Python 기본 라이브러리
import json

# AWS 서비스를 Python에서 호출하기 위한 AWS SDK
import boto3

# AWS API 호출 오류를 처리하기 위한 예외 클래스
from botocore.exceptions import ClientError


# ------------------------------------------------------------
# 02. AWS Bedrock Runtime Client 생성
# ------------------------------------------------------------

# Amazon Bedrock Runtime 서비스를 호출하기 위한 Client 생성
#
# region_name:
#   Bedrock 모델을 사용하는 AWS 리전
#   현재 예제에서는 미국 동부 버지니아 북부(us-east-1)를 사용
#
# Lambda가 us-east-1이 아니더라도
# Bedrock 호출은 지정한 us-east-1로 연결할 수 있습니다.
# ------------------------------------------------------------

bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)


# ------------------------------------------------------------
# 03. 사용할 Amazon Bedrock 모델 지정
# ------------------------------------------------------------

# Amazon Nova Micro 모델 ID
#
# Nova Micro는 텍스트 생성에 빠르고 효율적인 모델입니다.
# 만약 해당 모델에 대한 접근 권한이 없다면
# Bedrock 콘솔에서 모델 액세스를 요청해야 합니다.
# ------------------------------------------------------------

MODEL_ID = "amazon.nova-micro-v1:0"


# ------------------------------------------------------------
# 04. HTTP 응답을 만들어주는 함수
# ------------------------------------------------------------

def generate_response(status_code, body):
    """
    API Gateway로 반환할 HTTP 응답을 생성하는 함수

    status_code:
        HTTP 상태 코드
        예:
            200 = 성공
            400 = 잘못된 요청
            500 = 서버 오류

    body:
        사용자에게 반환할 데이터
    """

    return {
        # HTTP 상태 코드
        "statusCode": status_code,

        # 응답 Header
        "headers": {
            # JSON 형태로 데이터를 반환한다는 의미
            "Content-Type": "application/json; charset=utf-8",

            # CORS 허용
            # 모든 웹사이트에서 이 API를 호출할 수 있도록 허용
            "Access-Control-Allow-Origin": "*",

            # 허용할 HTTP Method
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",

            # 허용할 Header
            "Access-Control-Allow-Headers": "Content-Type",
        },

        # API Gateway는 body를 문자열 형태로 전달받으므로
        # json.dumps()를 사용해서 JSON 문자열로 변환
        "body": json.dumps(
            body,
            ensure_ascii=False
        )
    }


# ------------------------------------------------------------
# 05. Lambda 실행 Handler 함수
# ------------------------------------------------------------

def lambda_handler(event, context):
    """
    AWS Lambda가 실행될 때 가장 먼저 호출되는 함수

    event:
        API Gateway 등에서 Lambda로 전달된 요청 데이터

    context:
        Lambda 실행 환경 정보
    """

    # --------------------------------------------------------
    # 05-1. Lambda에 전달된 event 확인
    # --------------------------------------------------------

    print("==========================================")
    print("Lambda 함수가 실행되었습니다.")
    print("==========================================")

    print(f"Received event: {json.dumps(event, ensure_ascii=False)}")


    # --------------------------------------------------------
    # 05-2. 현재 boto3 버전 확인
    # --------------------------------------------------------

    print(f"boto3 version: {boto3.__version__}")


    # --------------------------------------------------------
    # 05-3. HTTP Method 확인
    # --------------------------------------------------------

    # API Gateway HTTP API의 경우
    # event["requestContext"]["http"]["method"]
    # 형태로 HTTP Method를 확인할 수 있습니다.
    #
    # 테스트 이벤트나 직접 호출 시 해당 값이 없을 수 있으므로
    # get()을 사용하여 안전하게 처리
    # --------------------------------------------------------

    request_context = event.get("requestContext", {})

    http_info = request_context.get("http", {})

    http_method = http_info.get("method", "")

    print(f"HTTP Method: {http_method}")


    # --------------------------------------------------------
    # 05-4. OPTIONS 요청 처리 (CORS Preflight)
    # --------------------------------------------------------

    # 웹 브라우저에서 CORS를 사용할 때
    # OPTIONS 요청이 먼저 들어올 수 있습니다.
    # --------------------------------------------------------

    if http_method == "OPTIONS":

        return generate_response(
            200,
            {
                "message": "CORS preflight request"
            }
        )


    # --------------------------------------------------------
    # 05-5. GET 요청 처리
    # --------------------------------------------------------

    # 이 API는 POST 요청이 기본이므로
    # 사용자가 웹브라우저 등 GET으로 접근하면 안내 메시지 반환
    # --------------------------------------------------------

    if http_method == "GET":

        return generate_response(
            200,
            {
                "message": "GET 요청을 처리했습니다.",
                "info": "이 API는 POST 요청으로 prompt를 전달받습니다."
            }
        )


    # --------------------------------------------------------
    # 05-6. POST 이외의 요청 거절
    # --------------------------------------------------------

    if http_method and http_method != "POST":

        return generate_response(
            405,
            {
                "error": "POST 요청만 사용할 수 있습니다."
            }
        )


    # --------------------------------------------------------
    # 05-7. try 시작
    # --------------------------------------------------------

    try:

        # ----------------------------------------------------
        # 06. API Gateway의 body 가져오기
        # ----------------------------------------------------

        # API Gateway에서 전달되는 body는 비어있을 수 있으므로
        # 기본값 "{}" 설정
        # ----------------------------------------------------

        raw_body = event.get("body", "{}")

        print(f"Raw body: {raw_body}")


        # ----------------------------------------------------
        # 07. body가 Base64로 인코딩되어 있는 경우 처리
        # ----------------------------------------------------

        # 일반적인 API Gateway 요청에서는 False
        # 만약 Base64로 전달되면 에러 처리
        # ----------------------------------------------------

        if event.get("isBase64Encoded", False):

            return generate_response(
                400,
                {
                    "error": "Base64 encoded body는 현재 지원하지 않습니다."
                }
            )


        # ----------------------------------------------------
        # 08. JSON 문자열을 Python Dictionary로 변환
        # ----------------------------------------------------

        try:

            request_body = json.loads(raw_body)

        except json.JSONDecodeError:

            return generate_response(
                400,
                {
                    "error": "요청 body가 올바른 JSON 형식이 아닙니다."
                }
            )


        # ----------------------------------------------------
        # 09. prompt 추출
        # ----------------------------------------------------

        # 사용자가 다음과 같이 요청했다고 가정:
        # {
        #     "prompt": "Amazon Bedrock이란 무엇인가요?"
        # }
        # ----------------------------------------------------

        prompt = request_body.get("prompt")


        # ----------------------------------------------------
        # 10. prompt 누락 예외 처리
        # ----------------------------------------------------

        if not prompt:

            return generate_response(
                400,
                {
                    "error": "prompt 값이 필요합니다.",
                    "example": {
                        "prompt": "Amazon Bedrock이란 무엇인가요?"
                    }
                }
            )


        # ----------------------------------------------------
        # 11. prompt의 데이터 타입 검사
        # ----------------------------------------------------

        if not isinstance(prompt, str):

            return generate_response(
                400,
                {
                    "error": "prompt는 문자열이어야 합니다."
                }
            )


        # ----------------------------------------------------
        # 12. 불필요한 공백 제거
        # ----------------------------------------------------

        prompt = prompt.strip()


        # ----------------------------------------------------
        # 13. 공백 문자열 검사
        # ----------------------------------------------------

        if not prompt:

            return generate_response(
                400,
                {
                    "error": "prompt가 비어 있습니다."
                }
            )


        # ----------------------------------------------------
        # 14. 처리할 prompt CloudWatch Logs에 출력
        # ----------------------------------------------------

        print(f"Processing prompt: {prompt}")


        # ----------------------------------------------------
        # 15. Converse API에 전달할 메시지 구조 작성
        # ----------------------------------------------------

        # Converse API 규격:
        # role: user / assistant
        # content: [{'text': '...'}]
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # 16. 추론 옵션(Inference Config) 설정
        # ----------------------------------------------------

        inference_config = {

            # 최대 생성 토큰 수
            "maxTokens": 512,

            # 창의성 수치 (낮을수록 정밀하고 일관된 답변)
            "temperature": 0.2,

            # 상위 확률 샘플링 비율
            "topP": 0.9
        }


        # ----------------------------------------------------
        # 17. Amazon Bedrock Converse API 호출
        # ----------------------------------------------------

        print(f"Calling Bedrock model: {MODEL_ID}")


        response = bedrock_runtime.converse(

            # 사용할 모델 ID
            modelId=MODEL_ID,

            # 전달할 메시지
            messages=messages,

            # 추론 설정
            inferenceConfig=inference_config
        )


        # ----------------------------------------------------
        # 18. Bedrock 응답 수신 확인
        # ----------------------------------------------------

        print("Bedrock response received.")


        # ----------------------------------------------------
        # 19. 응답 메시지 추출
        # ----------------------------------------------------

        output_message = response["output"]["message"]


        # ----------------------------------------------------
        # 20. 생성된 텍스트 추출
        # ----------------------------------------------------

        content = output_message.get("content", [])


        # ----------------------------------------------------
        # 21. text 결합
        # ----------------------------------------------------

        model_response = ""

        for content_item in content:

            if "text" in content_item:

                model_response += content_item["text"]


        # ----------------------------------------------------
        # 22. 응답 내용이 비어있는지 확인
        # ----------------------------------------------------

        if not model_response:

            return generate_response(
                500,
                {
                    "error": "Bedrock 모델에서 텍스트 응답을 받지 못했습니다."
                }
            )


        # ----------------------------------------------------
        # 23. 생성된 응답 CloudWatch Logs 출력
        # ----------------------------------------------------

        print("==========================================")
        print("Model response:")
        print(model_response)
        print("==========================================")


        # ----------------------------------------------------
        # 24. 토큰 사용량 정보 추출
        # ----------------------------------------------------

        usage = response.get("usage", {})


        # ----------------------------------------------------
        # 25. 종료 이유 추출
        # ----------------------------------------------------

        stop_reason = response.get(
            "stopReason",
            "unknown"
        )


        # ----------------------------------------------------
        # 26. 최종 성공 응답 반환
        # ----------------------------------------------------

        return generate_response(

            # HTTP 200 = 성공
            200,

            {
                # 입력된 프롬프트
                "prompt": prompt,

                # AI 생성 답변
                "response": model_response,

                # 사용된 모델
                "model": MODEL_ID,

                # 종료 사유
                "stopReason": stop_reason,

                # 토큰 사용량
                "usage": usage
            }
        )


    # --------------------------------------------------------
    # 27. AWS ClientError 예외 처리
    # --------------------------------------------------------

    except ClientError as e:

        error_code = e.response.get(
            "Error",
            {}
        ).get(
            "Code",
            "Unknown"
        )


        error_message = e.response.get(
            "Error",
            {}
        ).get(
            "Message",
            str(e)
        )


        # CloudWatch Logs에 에러 출력
        print("==========================================")
        print("Bedrock ClientError")
        print(f"Error Code: {error_code}")
        print(f"Error Message: {error_message}")
        print("==========================================")


        # API 호출자에게 에러 반환
        return generate_response(

            500,

            {
                "error": "Bedrock API 호출 중 에러가 발생했습니다.",
                "errorCode": error_code,
                "message": error_message
            }
        )


    # --------------------------------------------------------
    # 28. 예상치 못한 기타 예외 처리
    # --------------------------------------------------------

    except Exception as e:

        # CloudWatch Logs에 에러 출력
        print("==========================================")
        print("Unexpected Error")
        print(str(e))
        print("==========================================")


        # API 호출자에게 에러 반환
        return generate_response(

            500,

            {
                "error": "Lambda 실행 중 예상치 못한 에러가 발생했습니다.",
                "message": str(e)
            }
        )