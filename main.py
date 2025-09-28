import os
import json
import traceback
from dotenv import load_dotenv
from typing import List, Optional

import pandas as pd
from geopy.distance import geodesic

# --- FastAPI 관련 라이브러리 ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- LangChain 관련 라이브러리 ---
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser

# --- 0. 환경 변수 로드 및 FastAPI 앱 초기화 ---
print(">>> [시작] 0. 환경 변수 로드 및 앱 초기화...")
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 만들어 키를 저장해주세요.")

app = FastAPI(title="AI 여행 플래너 API")


# --- Pydantic 모델 정의 (API 입/출력 데이터 형식) ---

class PlannerRequest(BaseModel):
    query: str

class ParkingInfo(BaseModel):
    name: str = Field(description="주차장 이름")
    category: str = Field(description="주차장 구분 (공영/민영 등)")
    capacity: int = Field(description="총 주차 가능 대수")
    fee_info: str = Field(description="요금 정보")
    distance_km: float = Field(description="장소로부터의 거리 (km)")
    lat: float = Field(description="주차장 위도")
    lng: float = Field(description="주차장 경도")

class LocationInfo(BaseModel):
    address: str = Field(description="장소의 전체 주소")
    lat: float = Field(description="장소의 위도")
    lng: float = Field(description="장소의 경도")

class ScheduleItem(BaseModel):
    name: str = Field(description="일정 장소의 이름")
    location_info: LocationInfo = Field(description="장소의 위치 정보")
    visit_time: str = Field(description="추천 방문 시간 (예: 13:00)")
    stay_duration_minutes: int = Field(description="해당 장소에서의 예상 체류 시간 (분 단위)")
    place_type: str = Field(description="장소의 유형 ('아침', '점심', '저녁', '활동') 중 하나")
    content_id: str = Field(description="장소의 contentid")
    category: str = Field(description="장소의 카테고리 (예: 음식점, 관광지)")
    nearby_parking: List[ParkingInfo] = Field(description="장소 근처 추천 주차장 목록", default=[])

class DailySchedule(BaseModel):
    date: str = Field(description="여행 날짜 (예: 1일차, 2일차)")
    items: List[ScheduleItem] = Field(description="해당 날짜의 일정 목록")

class TravelPlan(BaseModel):
    title: str = Field(description="여행 전체를 아우르는 멋진 제목")
    schedule: List[DailySchedule] = Field(description="날짜별 일정 목록")


# --- 1. DB 및 RAG Chain 준비 ---
print(">>> [시작] 1. AI 모델 및 RAG Chain 로딩 중...")

def get_db():
    print(">>> 임베딩 모델과 Vector DB를 불러옵니다...")
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    persist_directory = "./chroma_db_planner"
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"'{persist_directory}' 폴더를 찾을 수 없습니다.")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("✅ Vector DB 준비 완료.")
    return db

def combine_documents(docs_dict):
    combined_docs = []
    for docs in docs_dict.values():
        combined_docs.extend(docs)
    unique_docs = {doc.page_content: doc for doc in combined_docs}.values()
    return list(unique_docs)

def get_rag_chain(db: Chroma):
    print(">>> LLM과 Prompt, JSON Parser를 설정합니다...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    parser = JsonOutputParser(pydantic_object=TravelPlan)

    template = """
    당신은 매우 논리적이고 현실적인 여행 플래너입니다. 주어진 규칙을 반드시 준수하여, 사용자의 요청과 'Context'에 기반한 완벽한 여행 일정을 JSON 형식으로 생성해야 합니다.

    [지시사항]
    1.  **후보군 선별:**
        - 'Context'에 제시된 장소만을 사용합니다.
        - 관광에 부적합한 장소는 제외하고, '사용자 요청'의 핵심 취지에 맞는 장소들로 최종 후보군을 엄선합니다.

    2.  **경로 최적화 (가장 중요한 단계):**
        - 후보군의 `lat`, `lng` 좌표를 기준으로, 인접한 지역의 활동들을 함께 묶어 그룹(클러스터)을 만듭니다.
        - 그룹 간 이동과 그룹 내 이동을 모두 고려하여, 전체 이동 거리가 최소화되는 최적의 동선을 확정합니다.

    3.  **일정 구성:**
        - **식사:** 위에서 확정한 동선에 맞춰, 각 날짜에 '아침', '점심', '저녁' 식사를 반드시 하나씩 포함해야 합니다. '음식점' 카테고리에서 디저트/기념품 가게가 아닌, 실제 식사를 제공하는 식당을 선택해야 합니다.

    4.  **[매우 중요] 시간 계산 규칙:**
        - **각 날짜의 첫 일정은 '아침' 식사이며, 방문 시간은 09:00으로 고정합니다.**
        - '아침' 식사 이후의 모든 장소는, 2번 단계에서 결정된 최적의 경로 순서에 따라 아래 공식을 적용하여 `visit_time`을 순차적으로 계산합니다.
        - `다음 visit_time` = `이전 visit_time` + `이전 stay_duration_minutes` + `두 장소 간 이동 시간` + `여유 시간`
        - 이동 시간은 `lat`, `lng` 좌표를 기반으로 10km당 약 20~30분으로 계산합니다.
        - '여유 시간'은 주요 식사나 체류 시간이 긴 활동(90분 이상) 후에는 20-30분을, 간단한 활동 후에는 10-15분을 추가하여 계획의 강약을 조절합니다.

    5.  **필드 값 할당:**
        - **`stay_duration_minutes`:** 장소의 종류와 규모에 따라 아래 기준에 맞춰 구체적으로 조절합니다.
            - `간단한 식사/카페/작은 명소`: 30-60분
            - `여유있는 식사 (점심/저녁)`: 60-90분
            - `중규모 관광지 (박물관, 해변 산책 등)`: 90-120분
        - **`place_type`:** 모든 장소에 대해 '아침', '점심', '저녁', '활동' 중 하나로 명확히 지정합니다.

    6.  **예외 처리:**
        - 'Context' 정보가 부족하여 위 규칙을 따르는 일정 생성이 불가능할 경우, `title`에 "정보 부족"을, `schedule`은 빈 리스트 `[]`로 반환합니다.
    {format_instructions}
    -----------------
    [Context]
    {context}
    -----------------
    [사용자 요청]
    {question}
    -----------------
    """
    prompt = ChatPromptTemplate.from_template(
        template,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # --- [수정] 기존 카테고리 기반 리트리버 구성으로 복원 ---
    CATEGORIES = ["음식점", "관광지", "숙박", "레포츠", "문화시설"]
    
    category_retrievers = {
        category: db.as_retriever(search_kwargs={'k': 10, 'filter': {'category': category}})
        for category in CATEGORIES
    }
    
    retrieval_chain = RunnableParallel(category_retrievers)

    rag_chain = (
        {"context": retrieval_chain | RunnableLambda(combine_documents), "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
    print("✅ RAG Chain 준비 완료.")
    return rag_chain


# --- 2. 주차장 정보 함수 ---
def parking_info_from_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")
    required_cols = ['주차장명', '주차장구분', '주차구획수', '요금정보', '위도', '경도']
    cols_to_use = [col for col in required_cols if col in df.columns]
    return df[cols_to_use] if cols_to_use else None

def find_nearby_parking(df: pd.DataFrame, lat: float, lng: float, top_n: int = 3) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    df_distance = df.copy()
    current_location = (lat, lng)
    df_distance[['위도', '경도']] = df_distance[['위도', '경도']].apply(pd.to_numeric, errors='coerce')
    df_distance.dropna(subset=['위도', '경도'], inplace=True)
    df_distance['거리(km)'] = df_distance.apply(
        lambda row: geodesic(current_location, (row['위도'], row['경도'])).km, axis=1
    )
    return df_distance.sort_values(by='거리(km)').head(top_n)


# --- 3. 전역 변수 로드 및 서버 실행 준비 ---
chroma_db = get_db()
travel_planner_chain = get_rag_chain(chroma_db)
parking_df = parking_info_from_csv("./강원특별자치도_강릉시_주차장정보_20230828.csv")
print(">>> [완료] AI 모델 및 모든 설정이 준비되었습니다. API 서버가 실행됩니다.")


# --- 4. API 엔드포인트 정의 ---
@app.get("/", summary="서버 상태 확인")
def read_root():
    return {"status": "AI 여행 플래너 API가 정상적으로 실행 중입니다."}

@app.post("/plan", summary="여행 계획 생성", response_model=TravelPlan)
async def create_travel_plan(request: PlannerRequest):
    """
    LLM이 동선, 체류시간, 방문시간까지 모두 포함된 계획을 생성하고, 주차 정보를 추가합니다.
    """
    try:
        # 1. RAG Chain 실행 (LLM이 동선까지 최적화한 최종 계획 생성)
        response_dict = await travel_planner_chain.ainvoke(request.query)
        travel_plan = TravelPlan.model_validate(response_dict)

        # 2. 주차장 정보 추가
        if parking_df is not None and not parking_df.empty:
            for daily_schedule in travel_plan.schedule:
                for item in daily_schedule.items:
                    lat, lng = item.location_info.lat, item.location_info.lng
                    nearby_parking_df = find_nearby_parking(df=parking_df, lat=lat, lng=lng, top_n=3)
                    if nearby_parking_df is not None and not nearby_parking_df.empty:
                        renamed_df = nearby_parking_df.rename(
                            columns={'주차장명': 'name', '주차장구분': 'category', '주차구획수': 'capacity',
                                     '요금정보': 'fee_info', '거리(km)': 'distance_km', '위도': 'lat', '경도': 'lng'}
                        )
                        parking_list = [ParkingInfo(**row) for row in renamed_df.to_dict('records')]
                        item.nearby_parking = parking_list
        # 3. 최종 결과 반환
        return travel_plan

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
