# create_db.py (오픈소스 임베딩 버전)

import os
import time
import requests
from typing import Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# -----------------------------

type_dict = {"관광지" : 12, "문화시설" : 14, "레포츠" : 28, "숙박" : 32, "쇼핑" : 38, "음식점" : 39}
key_mapping = {'addr1' : 'address', 'mapy' : 'lat', 'mapx' : 'lng', 'title': 'name'}

def get_info_from_knto(area_code: int = 32, sigungu_code: int = 1, num_of_rows: int = 100) -> Optional[dict]:
    """
    한국관광공사 API를 호출하여 원본 JSON 응답을 반환합니다.
    """
    load_dotenv()
    if not os.getenv("KNTO_KEY"):
        raise ValueError("KNTO_KEY가 .env 파일에 설정되지 않았습니다.")
    URL = "http://apis.data.go.kr/B551011/KorService2/areaBasedList2"
    
    info_list = []
    for key, value in type_dict.items():                 
        params = {
            "numOfRows": num_of_rows,
            "pageNo": 1,
            "MobileOS": "ETC",
            "MobileApp": "AppTest",
            "serviceKey":os.getenv("KNTO_KEY"),
            "_type": "json",
            "areaCode": area_code,
            "sigunguCode": sigungu_code,
            "contentTypeId" : value,
        }

        try:
            response = requests.get(URL, params=params, timeout=10)
            response.raise_for_status()
            json_list = refactoring_json(response.json())
            preprocess_json_list = preprocess_json(json_list, key)
            info_list.extend(preprocess_json_list)
            print(info_list)
        except requests.exceptions.RequestException as e:
            print(f"KNTO API 요청 중 오류가 발생했습니다: {e}")
            return None
    
    return info_list 

def refactoring_json(api_response: dict) -> list:
    if not api_response:
        return []
    try:
        items = api_response['response']['body']['items']['item']
        print("✅ JSON 데이터 정제 성공")
        return items if isinstance(items, list) else []
    except (KeyError, TypeError):
        print("JSON 데이터에서 item 리스트를 찾을 수 없거나 형식이 다릅니다.")
        return []

def preprocess_json(json_list : list, type : str) -> list:
    preprocess_json_list = []
    for json in json_list:
        temp_json = {}
        
        for key, value in key_mapping.items():
            temp_json[value] = json[key]
        
        temp_json['contentid'] = json['contentid']
        temp_json['category'] = type
        preprocess_json_list.append(temp_json)
    
    return preprocess_json_list
    
def get_content_overview(contentid : str) -> str:
    load_dotenv()
    if not os.getenv("KNTO_KEY"):
        raise ValueError("KNTO_KEY가 .env 파일에 설정되지 않았습니다.")
    URL = "http://apis.data.go.kr/B551011/KorService2/detailCommon2"
    params = {
        "MobileOS": "ETC",
        "MobileApp": "AppTest",
        "serviceKey":os.getenv("KNTO_KEY"),
        "_type": "json",
        "contentId" : contentid
    }
    try:
        response = requests.get(URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json()['response']['body']['items']['item'][0]['overview']
    except requests.exceptions.RequestException as e:
        print(f"KNTO API 요청 중 오류가 발생했습니다: {e}")
        return "-"
    

def create_vector_db():
    """
    여행지 데이터를 읽어 Chroma Vector DB를 생성하고 파일로 저장합니다.
    """
    print(">>> 1. 한국관광공사 API에서 데이터 가져오는 중...")
    info_list = get_info_from_knto()
    
    if not info_list:
        print("처리할 데이터가 없습니다. 스크립트를 종료합니다.")
        return
        
    documents = []
    for info in info_list:
        # API 응답에 따라 키 이름이 다를 수 있으므로 확인이 필요합니다.
        # 예시 키: title, addr1, cat1, cat2, mapx, mapy, contentid
        page_content = get_content_overview(info['contentid'])
        metadata = info
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    # --- 2. Vector DB 생성 (오픈소스 모델 사용) ---
    persist_directory = "./chroma_db_planner"
    
    # --- 이 부분이 변경되었습니다 ---
    print(">>> 2. 오픈소스 임베딩 모델 로딩 중... (최초 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)")
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'} # GPU 사용 시 'cuda'로 변경
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # -----------------------------

    if os.path.exists(persist_directory):
        print(f">>> '{persist_directory}' 폴더가 이미 존재합니다. DB를 새로 만들지 않습니다.")
        return

    print(">>> 3. 새로운 Vector DB를 구축합니다...")
    db = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    print(f"\n✅ Vector DB 생성이 완료되었습니다. '{persist_directory}' 폴더를 확인하세요.")
    
    # 생성한 DB에서 첫 항목 출력
    all_data = db.get()
    if all_data and all_data.get("ids"):
        print("=== 새로 생성한 Chroma DB 첫 문서 미리보기 ===")
        print("id:", all_data["ids"])
        print("document (앞 300자):", (all_data["documents"] or "")[:300])
        print("metadata:", all_data["metadatas"])

if __name__ == "__main__":
    create_vector_db()
