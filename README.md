# 🏖️ AI Travel Planner - Gangneung

한국관광공사 API와 LangChain RAG를 활용한 **강릉 지역 특화 AI 여행 계획 생성 시스템**

## 🎯 프로젝트 개요

AI Travel Planner는 FastAPI와 최신 AI 기술을 결합하여 강릉 지역의 맞춤형 여행 일정을 자동으로 생성하는 인텔리전트 시스템입니다. 한국관광공사 공식 API 데이터를 기반으로 한 벡터 데이터베이스와 OpenAI GPT-4o 모델을 활용하여, 사용자의 선호도와 여행 스타일에 최적화된 일정을 제공합니다.

## ✨ 주요 기능

### 🤖 AI 기반 여행 일정 생성
- **GPT-4o 모델**: 자연어 처리를 통한 지능적인 일정 계획
- **RAG (검색 증강 생성)**: ChromaDB 벡터 데이터베이스를 활용한 정확한 장소 정보 검색
- **한국어 최적화**: 한국어 특화 임베딩 모델 (`jhgan/ko-sroberta-multitask`) 사용

### 🗺️ 스마트 경로 최적화
- **지리적 클러스터링**: 위도/경도 좌표 기반 인접 지역 그룹화
- **최적 동선 계산**: 전체 이동 거리 최소화 알고리즘
- **실시간 시간 계산**: 체류시간과 이동시간을 고려한 정확한 일정 수립

### 🅿️ 통합 주차장 정보 시스템
- **실시간 주차장 매칭**: 각 여행지 주변 최적 주차장 추천 (상위 3개)
- **거리 기반 정렬**: 목적지로부터의 거리순 정렬
- **상세 정보 제공**: 주차요금, 수용대수, 주차장 구분 정보

### 📊 다양한 카테고리 지원
- **관광지**: 해변, 명소, 공원, 역사유적
- **음식점**: 맛집, 카페, 전통음식점, 디저트
- **숙박시설**: 호텔, 펜션, 게스트하우스
- **레포츠**: 해양스포츠, 레저활동, 체험프로그램
- **문화시설**: 박물관, 미술관, 공연장

## 🛠 기술 스택

### Backend Framework

- **FastAPI** 0.104+ # 고성능 비동기 웹 API 프레임워크
- **Pydantic** 2.0+ # 데이터 검증 및 시리얼화
- **Uvicorn** # ASGI 서버

### AI & Machine Learning

- **LangChain** 0.1+ # LLM 애플리케이션 구축 프레임워크
- **OpenAI GPT-4o** # 대화형 AI 모델
- **HuggingFace(`jhgan/ko-sroberta-multitask`)** # 한국어 임베딩 모델
- **Transformers** # 트랜스포머 모델 라이브러리

### Vector Database & Search

- **ChromaDB** # 벡터 데이터베이스
- **Geopy** # 지리적 거리 계산
- **Pandas** # 데이터 처리 및 분석

### External APIs

- **한국관광공사 API** # 관광지 정보 수집
- **강릉시 공공데이터** # 주차장 정보


## 환경 설정
### 1. GCP -> VM 생성(스토리지 20GB 이상) or 로컬 
- **GCP를 이용할 경우** : 방화벽 설정 필요
  
### 2. 레포지토리 클론

```
git clone https://github.com/DDukSoon/GangneungPlanner.git
```

### 3. Conda 사용(선택)

```
conda create -n planner python=3.11
conda activate planner
```

### 3. 필수 패키지 설치

```
pip install requirements.txt
```

### 4. api 키 설정

.env 파일에 api 키 설정 
```
# OpenAI API 키 (필수)
OPENAI_API_KEY=<your-openai-api-key-here>

# 한국관광공사 API 키 (필수)
KNTO_KEY=your-<knto-service-key-here>
```
