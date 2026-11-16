
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Python 파일이 위치한 폴더 확인
BASE_DIR = Path(__file__).resolve().parent

# 2. .env 파일 경로
ENV_FILE = BASE_DIR / ".env"

print("현재 실행 파일:", __file__)
print(".env 파일 경로:", ENV_FILE)
print(".env 파일 존재:", ENV_FILE.is_file())

# 3. 환경변수 로드
if not ENV_FILE.is_file():
    raise FileNotFoundError(
        f".env 파일이 없습니다: {ENV_FILE}"
    )

loaded = load_dotenv(
    dotenv_path=ENV_FILE,
    override=True
)

print(".env 로드 결과:", loaded)

# 4. 환경변수 출력
print("텍스트 리전:", os.getenv("TEXT_REGION"))
print("이미지 리전:", os.getenv("IMAGE_REGION"))
print("텍스트 모델:", os.getenv("TEXT_MODEL_ID"))
print("이미지 모델:", os.getenv("IMAGE_MODEL_ID"))

# 5. 설정 검증
names = [
    "TEXT_REGION",
    "IMAGE_REGION",
    "TEXT_MODEL_ID",
    "IMAGE_MODEL_ID"
]

missing = [name for name in names if not os.getenv(name)]

if missing:
    print("설정 실패! 누락된 변수:", missing)
else:
    print("설정 완료!")
