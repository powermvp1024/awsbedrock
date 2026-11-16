# -*- coding: utf-8 -*-

"""
================================================================================
[전체 코드 구조 및 함수 / 객체 / 클래스 구성 안내]
================================================================================

1. 클래스 (Classes):
   - Nominatim (geopy.geocoders): 도시명을 위도/경도로 변환하는 지오코딩 엔진
   - Session (boto3.Session): AWS 자격 증명 관리 세션
   - ClientError / ProfileNotFound (botocore.exceptions): AWS SDK 예외 처리기
   - Path (pathlib.Path): 파일 경로 핸들러

2. 주요 전역 객체 (Global Objects):
   - BASE_DIR: 현재 실행 경로 객체
   - TOOL_CONFIG: AWS Bedrock Converse API용 Function Calling 설정 딕셔너리
   - st.session_state: 화면용 기록과 Bedrock 요청용 기록을 분리해 저장

3. 주요 함수 (Functions):
   - create_bedrock_client(): Bedrock Runtime 클라이언트 생성 및 캐싱
   - get_coordinates(city_name): 도시명을 위도/경도로 변환
   - get_weather(city_name): Open-Meteo API 호출 날씨 조회
   - get_kbo_ranking(season): 2026 KBO 형식의 가상 샘플 데이터 반환
   - execute_tool(tool_name, tool_args): 도구 요청 라우터
   - run_conversation(messages): Bedrock Converse API Tool Use 반복 루프 제어
================================================================================
"""

import os
import re
import logging
import math
from pathlib import Path

import boto3
import httpx
import streamlit as st

from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
)
from geopy.exc import GeopyError


# ============================================================
# 01. 환경변수 및 AWS 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
AWS_PROFILE = os.getenv("AWS_PROFILE") or None
MODEL_ID = os.getenv("BEDROCK_MODEL_ID") or "amazon.nova-micro-v1:0"

# 한 질문에서 도구를 실행할 수 있는 라운드와 개별 요청 수의 상한입니다.
MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS = 8
LOGGER = logging.getLogger(__name__)


# ============================================================
# 02. Streamlit 화면 기본 구성 & 완벽한 화면 정중앙 레이아웃 CSS
# ============================================================

st.set_page_config(
    page_title="Weather & KBO AI Chatbot",
    page_icon="⛰️",
    layout="centered"
)

# 화면에는 질문과 최종 답변만, API 기록에는 원본 도구 블록도 보관합니다.
# 두 기록을 하나의 상태에 담아 성공 시 한 번에 교체합니다.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {"ui": [], "api": []}

# 장가계 고화질 산림/안개 배경
ZHANGJIAJIE_BG_URL = "https://images.unsplash.com/photo-1543097692-fa13c6cd8595?auto=format&fit=crop&w=1920&q=85"

# 기본 공통 스타일 (배경, 폰트, 대화창)
common_css = f"""
    .stApp {{
        background: linear-gradient(rgba(15, 23, 42, 0.4), rgba(20, 20, 25, 0.5)), url("{ZHANGJIAJIE_BG_URL}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    h1 {{
        color: #ffffff !important;
        text-shadow: 0 3px 14px rgba(0, 0, 0, 0.7) !important;
        font-weight: 700 !important;
        text-align: center !important;
    }}

    div[data-testid="stCaptionContainer"] p {{
        color: #fef08a !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6) !important;
        text-align: center !important;
    }}

    .stMarkdown p {{
        color: #f8fafc !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6) !important;
        text-align: center !important;
    }}

    /* 긴 순위표는 작은 화면에서 가로로 스크롤합니다. */
    [data-testid="stChatMessageContent"] {{
        min-width: 0;
        overflow-x: auto;
        color: #2b1d0c;
    }}

    [data-testid="stChatMessage"] [data-testid="stText"] {{
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }}

    [data-testid="stChatMessage"] table {{
        width: 100%;
        min-width: 34rem;
        background: #fffaf1;
    }}

    [data-testid="stChatMessage"] thead {{
        background: #ead8bf;
    }}

    /* 채팅 카드 스타일 */
    div[data-testid="stChatMessage"] {{
        background: rgba(253, 250, 243, 0.94) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 14px !important;
        border: 1px solid #d8c3a5 !important;
        box-shadow: 0 8px 24px rgba(20, 15, 10, 0.22) !important;
        margin-bottom: 0.9rem !important;
    }}

    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] td, div[data-testid="stChatMessage"] th {{
        color: #2b1d0c !important;
        text-shadow: none !important;
        text-align: left !important;
    }}
"""

def chat_layout_css(is_chat_empty: bool) -> str:
    if is_chat_empty:
        return f"""
        <style>
            {common_css}

            /* 1. 제목 및 안내문 컨테이너: 화면 높이 기준 중앙 상단에 배치 */
            .block-container {{
                max-width: 680px !important;
                min-height: 100vh !important;
                padding-top: max(2rem, calc(50vh - 150px)) !important; /* 화면 정중앙에 자리잡도록 동적 마진 계산 */
                padding-bottom: 0 !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: flex-start !important;
            }}

            h1 {{
                font-size: 2.3rem !important;
                margin: 0 !important;
                padding: 0 !important;
            }}

            div[data-testid="stCaptionContainer"] {{
                margin-top: 0.2rem !important;
                margin-bottom: 0.2rem !important;
            }}

            .stMarkdown p {{
                margin-top: 0.2rem !important;
                margin-bottom: 1.2rem !important;
                font-size: 1.05rem !important;
            }}

            /* 2. 검색창(stBottom): 하단 고정을 완전히 해제하고 화면 정중앙으로 강제 배치 */
            div[data-testid="stBottom"] {{
                position: fixed !important;
                top: max(13rem, calc(50vh + 35px)) !important; /* 안내 문구 바로 아래에 밀착 */
                bottom: auto !important;
                left: 50% !important;
                transform: translateX(-50%) !important;
                width: calc(100% - 2rem) !important;
                max-width: 650px !important;
                background: transparent !important;
                box-shadow: none !important;
                border: none !important;
                padding: 0 !important;
                z-index: 9999 !important;
            }}

            div[data-testid="stBottom"] > div {{
                background: transparent !important;
                padding: 0 !important;
            }}

            /* 3. 우드(Wood) 스타일 검색창 본체 */
            div[data-testid="stChatInput"] {{
                border-radius: 26px !important;
                border: 2px solid #855b32 !important;
                background: #fbf6ee !important;
                box-shadow: 0 10px 30px rgba(25, 15, 5, 0.45), inset 0 1px 2px rgba(255, 255, 255, 0.9) !important;
                padding: 6px 14px !important;
                transition: all 0.25s ease !important;
            }}

            div[data-testid="stChatInput"]:focus-within {{
                border-color: #533418 !important;
                background: #ffffff !important;
                box-shadow: 0 12px 36px rgba(40, 20, 5, 0.55) !important;
                transform: translateY(-2px);
            }}

            div[data-testid="stChatInput"] textarea {{
                font-size: 15px !important;
                font-weight: 500 !important;
                color: #382413 !important;
                background: transparent !important;
            }}

            div[data-testid="stChatInput"] textarea::placeholder {{
                color: #9c836d !important;
            }}

            div[data-testid="stChatInput"] button {{
                border-radius: 50% !important;
                background-color: #855b32 !important;
                color: #ffffff !important;
                transition: all 0.2s ease !important;
            }}
            div[data-testid="stChatInput"] button:hover {{
                background-color: #533418 !important;
                transform: scale(1.1) !important;
            }}
        </style>
        """
    else:
        return f"""
        <style>
            {common_css}

            .block-container {{
                max-width: 720px !important;
                padding-top: 2rem !important;
                padding-bottom: 6.5rem !important;
            }}

            div[data-testid="stBottom"] {{
                background: linear-gradient(180deg, rgba(15, 23, 42, 0) 0%, rgba(18, 14, 10, 0.85) 45%, rgba(12, 9, 6, 0.95) 100%) !important;
                padding-top: 8px !important;
                padding-bottom: 16px !important;
            }}

            div[data-testid="stChatInput"] {{
                border-radius: 24px !important;
                border: 2px solid #855b32 !important;
                background: #fbf6ee !important;
                box-shadow: 0 6px 20px rgba(45, 30, 15, 0.3) !important;
                padding: 6px 14px !important;
            }}

            div[data-testid="stChatInput"]:focus-within {{
                border-color: #533418 !important;
                background: #ffffff !important;
                box-shadow: 0 8px 26px rgba(60, 35, 15, 0.45) !important;
            }}

            div[data-testid="stChatInput"] textarea {{
                color: #382413 !important;
                background: transparent !important;
            }}

            div[data-testid="stChatInput"] button {{
                border-radius: 50% !important;
                background-color: #855b32 !important;
                color: #ffffff !important;
            }}
        </style>
        """


# 첫 질문이 들어오면 같은 실행에서 중앙 입력창을 하단으로 옮깁니다.
layout_slot = st.empty()
layout_slot.markdown(
    chat_layout_css(not st.session_state.chat_history["ui"]),
    unsafe_allow_html=True,
)

# 메인 헤더
st.title("⛰️ Weather & KBO AI Chatbot")
st.caption("AWS Bedrock · Amazon Nova · Streamlit")
st.write("장가계의 웅장한 풍경과 함께 한국 도시의 현재 날씨와 2026 KBO 가상 샘플 순위표를 안내해 드립니다.")


# ============================================================
# 03. AWS Bedrock 클라이언트 생성 함수
# ============================================================

@st.cache_resource
def create_bedrock_client():
    # 프로필을 지정하면 해당 프로필을 사용합니다. 비워 두면 boto3가
    # 환경변수, 기본 프로필, 실행 환경의 IAM 역할 등을 순서에 따라 확인합니다.
    # 지정된 프로필의 이름이 틀린 경우에는 오류를 화면에 안내합니다.
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.client(
        "bedrock-runtime",
        config=Config(
            connect_timeout=5,
            read_timeout=45,
            retries={"mode": "standard", "total_max_attempts": 2},
        ),
    )


# ============================================================
# 04. 위치 변환 유틸리티 (도시명 -> 위도/경도)
# ============================================================

@st.cache_data(ttl=86400, max_entries=256, show_spinner=False)
def get_coordinates(city_name: str) -> tuple[float, float]:
    # 좌표를 하루 동안 재사용합니다. 국가 코드를 제한해 동명 해외 도시를 제외합니다.
    geolocator = Nominatim(user_agent="bedrock_weather_kbo_chatbot", timeout=5)
    location = geolocator.geocode(city_name, country_codes="kr")
    if location is None:
        raise ValueError("국내 도시를 찾을 수 없습니다. 시·군·구 이름을 확인해 주세요.")
    return location.latitude, location.longitude


# ============================================================
# 05. 도구(Tool) 1: 현재 날씨 조회
# ============================================================

def tool_error(message: str) -> dict:
    """도구 오류는 모델에 전달할 짧은 안내문으로 정리합니다."""
    return {"status": "error", "message": message}


def get_weather(city_name: str) -> dict:
    if not isinstance(city_name, str) or not city_name.strip():
        return tool_error("조회할 국내 도시 이름을 입력해 주세요.")
    city_name = city_name.strip()
    if len(city_name) > 100:
        return tool_error("도시 이름은 100자 이하로 입력해 주세요.")

    try:
        latitude, longitude = get_coordinates(city_name)
    except GeopyError:
        return tool_error("위치 검색 서비스에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.")
    except ValueError as exc:
        return tool_error(str(exc))

    fields = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m")
    try:
        response = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": ",".join(fields),
                "timezone": "Asia/Seoul",
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
            },
            timeout=8.0,
        )
        response.raise_for_status()
        current = response.json()["current"]
        temperature, humidity, wind_speed = (current[field] for field in fields)
        # 누락된 값이나 NaN을 기온·습도·풍속으로 표시하지 않습니다.
        if not all(
            type(value) in (int, float) and math.isfinite(value)
            for value in (temperature, humidity, wind_speed)
        ):
            raise ValueError("Invalid weather measurements")
        observed_at = current["time"]
        if not isinstance(observed_at, str) or not observed_at:
            raise ValueError("Missing weather timestamp")
    except httpx.TimeoutException:
        return tool_error("날씨 서비스의 응답 시간이 초과되었습니다.")
    except httpx.HTTPError:
        return tool_error("날씨 서비스에서 데이터를 가져오지 못했습니다.")
    except (KeyError, TypeError, ValueError, OverflowError):
        return tool_error("날씨 서비스 응답에 필요한 값이 없습니다.")

    return {
        "status": "success",
        "city": city_name,
        "time_kst": observed_at,
        "temperature": f"{temperature}°C",
        "humidity": f"{humidity}%",
        "wind_speed": f"{wind_speed} km/h",
    }


# ============================================================
# 06. 도구(Tool) 2: 2026 KBO 가상 샘플 순위표
# ============================================================

def get_kbo_ranking(season: str = "2026") -> dict:
    if not isinstance(season, str) or season.strip() != "2026":
        return tool_error("이 앱에는 2026 KBO 형식의 가상 샘플만 있습니다.")

    # 원본의 예제 숫자입니다. 실제 리그 성적을 나타내지 않습니다.
    raw_data = [
        (1, "kt wiz", 144, 90, 5, 49, 0.647, "0.0"),
        (2, "삼성 라이온즈", 144, 83, 3, 58, 0.589, "8.0"),
        (3, "KIA 타이거즈", 144, 79, 2, 63, 0.556, "12.5"),
        (4, "LG 트윈스", 144, 78, 1, 65, 0.545, "14.0"),
        (5, "두산 베어스", 144, 73, 5, 66, 0.525, "17.0"),
        (6, "SSG 랜더스", 144, 65, 5, 74, 0.468, "25.0"),
        (7, "롯데 자이언츠", 144, 64, 3, 77, 0.454, "27.0"),
        (8, "NC 다이노스", 144, 64, 2, 78, 0.451, "27.5"),
        (9, "한화 이글스", 144, 57, 4, 83, 0.407, "33.5"),
        (10, "키움 히어로즈", 144, 50, 4, 90, 0.357, "40.5"),
    ]

    standings = []
    for rank, team, games, won, drawn, lost, wra, diff in raw_data:
        standings.append({
            "rank": rank,
            "team": team,
            "games": games,
            "wins": won,
            "draws": drawn,
            "losses": lost,
            "win_rate": wra,
            "game_diff": diff,
        })

    return {
        "status": "success",
        "season": "2026",
        "is_sample": True,
        "message": "화면 표시용 가상 샘플 데이터입니다. 실제 KBO 순위와 무관합니다.",
        "ranking": standings
    }


# ============================================================
# 07. Bedrock Tool 설정
# ============================================================

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "특정 한국 도시의 현재 날씨와 기온, 습도, 풍속을 조회합니다.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "city_name": {
                                "type": "string",
                                "description": "날씨를 조회할 한국 도시 이름 (예: 서울, 부산, 대전 등)"
                            }
                        },
                        "required": ["city_name"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "get_kbo_ranking",
                "description": "2026 KBO 형식의 가상 샘플 순위표를 반환합니다. 실제 순위는 제공하지 않습니다.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "season": {
                                "type": "string",
                                "description": "시즌 연도 (기본값: '2026')"
                            }
                        }
                    }
                }
            }
        }
    ]
}


def execute_tool(tool_name: str, tool_args: dict) -> dict:
    if not isinstance(tool_args, dict):
        return tool_error("도구 인수는 JSON 객체여야 합니다.")
    if tool_name == "get_weather":
        return get_weather(tool_args.get("city_name", ""))
    if tool_name == "get_kbo_ranking":
        return get_kbo_ranking(tool_args.get("season", "2026"))
    return tool_error("지원하지 않는 도구입니다.")


# ============================================================
# 08. Bedrock Converse 처리 루프
# ============================================================

class ConversationError(Exception):
    """API 호출을 중단하고 사용자에게 안내할 수 있는 오류입니다."""


# 대소문자, 속성, 중첩 태그, 종료 태그 없이 잘린 추론 블록을 처리합니다.
# 구조화된 reasoningContent는 UI용 텍스트를 추출할 때부터 제외합니다.
REASONING_TAG = re.compile(
    r"<\s*(/?)\s*(think(?:ing)?|reasoning|analysis|reflection)\b[^>]*(?:>|$)",
    re.IGNORECASE,
)


def strip_reasoning(text: str) -> str:
    visible = []
    active_tags = []
    cursor = 0
    for match in REASONING_TAG.finditer(text):
        if not active_tags:
            visible.append(text[cursor:match.start()])
        tag = match[2].lower()
        if match[1]:
            if active_tags and active_tags[-1] == tag:
                active_tags.pop()
        elif not match[0].endswith("/>"):
            active_tags.append(tag)
        cursor = match.end()
    if not active_tags:
        visible.append(text[cursor:])
    return "".join(visible).strip()


def kbo_sample_markdown(result: dict) -> str:
    # 고정 데이터로 표를 생성하므로 모델이 열을 빼거나 숫자를 바꿀 수 없습니다.
    # 제목과 표 사이에 빈 줄을 두고, 코드 펜스 없이 Markdown으로 렌더링합니다.
    lines = [
        "**2026 KBO 가상 샘플 순위표 · 실제 순위와 무관**",
        "",
        "| 순위 | 팀 | 경기 | 승 | 무 | 패 | 승률 | 게임차 |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["ranking"]:
        lines.append(
            f"| {row['rank']} | {row['team']} | {row['games']} | {row['wins']} | "
            f"{row['draws']} | {row['losses']} | {row['win_rate']:.3f} | {row['game_diff']} |"
        )
    return "\n".join(lines)


def run_conversation(api_history: list, prompt: str) -> tuple[list, str]:
    client = create_bedrock_client()
    system = [{
        "text": (
            "한국어로 친절하게 답변하세요. 날씨는 get_weather 결과로 안내하고 "
            "실패 시 원인을 설명하세요. KBO 실제 순위나 성적은 추측하거나 생성하지 마세요. "
            "2026 KBO 순위 요청에는 get_kbo_ranking을 호출하세요. 반환값은 가상 샘플이며 "
            "앱에서 표를 직접 붙이므로 답변에는 순위, 수치, 표를 반복하지 마세요. "
            "샘플을 실제 또는 최신 순위라고 표현하지 마세요. "
            "추론 과정이나 thinking 등의 태그 없이 최종 답변만 작성하세요."
        )
    }]
    # 리스트를 새로 만들고 기존 메시지 객체는 수정하지 않습니다. 모델이 서명한
    # reasoningContent도 API 기록에는 원형을 보존해야 다음 요청에서 재사용할 수 있습니다.
    messages = [*api_history, {"role": "user", "content": [{"text": prompt}]}]
    sample = None
    tool_calls = 0

    # 도구 라운드 4회 후 최종 답변을 받을 수 있도록 Converse는 최대 5번 호출합니다.
    for round_index in range(MAX_TOOL_ROUNDS + 1):
        response = client.converse(
            modelId=MODEL_ID,
            messages=messages,
            system=system,
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": 2048},
        )
        reason = response["stopReason"]
        if reason not in {"end_turn", "stop_sequence", "tool_use"}:
            if reason == "max_tokens":
                raise ConversationError("응답 길이 한도에 도달했습니다. 질문 범위를 좁혀 다시 보내 주세요.")
            raise ConversationError(f"모델이 답변을 완료하지 못했습니다 ({reason}).")

        output = response["output"]["message"]
        blocks = output["content"]
        requests = [block["toolUse"] for block in blocks if "toolUse" in block]
        if output["role"] != "assistant" or (reason == "tool_use") != bool(requests):
            raise ConversationError("Bedrock의 메시지 형식이 올바르지 않습니다.")
        messages.append(output)

        if reason != "tool_use":
            # 블록 사이에 임의 문자를 넣지 않아 여러 블록에 걸친 태그도 제거합니다.
            answer = strip_reasoning("".join(block["text"] for block in blocks if "text" in block))
            if sample is not None:
                answer = "\n\n".join(part for part in (kbo_sample_markdown(sample), answer) if part)
            if not answer:
                raise ConversationError("화면에 표시할 답변이 없습니다. 질문을 다시 보내 주세요.")
            return messages, answer

        tool_calls += len(requests)
        if round_index == MAX_TOOL_ROUNDS or tool_calls > MAX_TOOL_CALLS:
            raise ConversationError("도구 호출 한도에 도달했습니다. 질문을 나누어 다시 보내 주세요.")

        results = []
        # 한 assistant 메시지의 요청을 모두 처리해 동일한 toolUseId로 한 번에 돌려줍니다.
        for request in requests:
            name = request["name"]
            result = execute_tool(name, request.get("input"))
            if name == "get_kbo_ranking" and result.get("is_sample"):
                sample = result
            results.append({
                "toolResult": {
                    "toolUseId": request["toolUseId"],
                    "content": [{"json": result}],
                    "status": result["status"],
                }
            })
        messages.append({"role": "user", "content": results})

    raise ConversationError("도구 호출 한도에 도달했습니다.")


def conversation_error_message(exc: Exception) -> str:
    # AWS의 원본 예외에는 계정·리소스 정보가 들어갈 수 있어 안내문만 화면에 표시합니다.
    if isinstance(exc, ConversationError):
        return str(exc)
    if isinstance(exc, ProfileNotFound):
        return "AWS 프로필을 찾을 수 없습니다. .env의 AWS_PROFILE을 확인해 주세요."
    if isinstance(exc, (NoCredentialsError, PartialCredentialsError)):
        return "AWS 자격 증명을 읽을 수 없습니다. AWS 프로필 또는 환경변수를 확인해 주세요."
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        hints = {
            "AccessDeniedException": "Bedrock 호출 권한과 모델 접근 권한을 확인해 주세요.",
            "ValidationException": "AWS 리전과 BEDROCK_MODEL_ID의 조합을 확인해 주세요.",
            "ResourceNotFoundException": "모델 또는 추론 프로파일 ID를 확인해 주세요.",
            "ThrottlingException": "요청 한도에 도달했습니다. 잠시 후 다시 시도해 주세요.",
            "ExpiredTokenException": "AWS 로그인 또는 임시 자격 증명을 갱신해 주세요.",
        }
        return f"Bedrock 오류 ({code}). {hints.get(code, '잠시 후 다시 시도해 주세요.')}"
    if isinstance(exc, BotoCoreError):
        return "AWS 연결 또는 인증 설정을 확인한 뒤 다시 시도해 주세요."
    LOGGER.exception("챗봇 요청 처리 중 예상하지 못한 오류")
    return "질문을 처리하는 중 오류가 발생했습니다. 질문을 다시 보내 주세요."


# ============================================================
# 09. Streamlit 대화 인터페이스
# ============================================================

history = st.session_state.chat_history
for message in history["ui"]:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.text(message["text"])
        else:
            st.markdown(message["text"])

user_prompt = st.chat_input(
    "질문을 입력해 보세요 (예: 서울 날씨 어때?, 2026 KBO 샘플 순위 알려줘)",
    max_chars=4000,
)
if user_prompt and (prompt := user_prompt.strip()):
    if not history["ui"]:
        layout_slot.markdown(chat_layout_css(False), unsafe_allow_html=True)

    with st.chat_message("user"):
        st.text(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("답변을 준비하고 있습니다..."):
                updated_api, answer = run_conversation(history["api"], prompt)
        except Exception as exc:
            # 실패한 질문은 이번 화면에 표시하되, 저장하지 않아 재시도 시 중복되지 않습니다.
            st.error(conversation_error_message(exc))
        else:
            # 성공한 질문·답변과 도구 이력을 같은 시점에 저장합니다.
            st.session_state.chat_history = {
                "api": updated_api,
                "ui": [
                    *history["ui"],
                    {"role": "user", "text": prompt},
                    {"role": "assistant", "text": answer},
                ],
            }
            st.markdown(answer)
