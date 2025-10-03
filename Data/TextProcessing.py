import os
import re
import json
import glob
import asyncio
import dotenv
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import aiofiles

# pymilvus 최신 API에 필요한 모듈만 가져옵니다.
from pymilvus import FieldSchema, CollectionSchema, DataType
from pymilvus.milvus_client import MilvusClient
from sentence_transformers import SentenceTransformer

# --- 1. 기본 설정 ---
dotenv.load_dotenv(dotenv_path="../.env")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"

DATA_DIR = "SamInKorean"
COLLECTION_NAME = "semantic_db"
MODEL_NAME = "jhgan/ko-sbert-nli"

BATCH_SIZE = 256
CONCURRENCY_LIMIT = 10
# 💡 소비자(데이터 삽입) 태스크의 개수 설정
CONSUMER_COUNT = 2


# --- 2. 비동기 처리 함수들 ---
def get_json_file_list(origin_path=DATA_DIR) -> list:
    """지정된 경로와 모든 하위 디렉토리에서 .json 파일의 전체 경로 리스트를 반환합니다."""
    #search_pattern = os.path.join(origin_path, '**', '*.json')
    #return glob.glob(search_pattern, recursive=True)
    test_path = ['SamInKorean/1476785_50000.json']
    return test_path

def _clean_text(t: str) -> str:
    t = re.sub(r"<[^>]+>", " ", t)      # 태그 제거
    t = re.sub(r"\{[^}]*\}", " ", t)    # {…} 제거
    t = re.sub(r"\s+", " ", t)
    return t.strip()

async def producer(file_path: str, queue: asyncio.Queue, semaphore: asyncio.Semaphore):
    async with semaphore:
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                data = json.loads(await f.read())

            ch = (data.get("channel") or {})
            items = ch.get("item") or []
            if isinstance(items, dict):
                items = [items]

            enq_local = 0
            for it in items:
                wi = it.get("wordinfo") or {}
                word = wi.get("word") or ""
                s = it.get("senseinfo")
                senses = s if isinstance(s, list) else ([s] if s else [])
                for sense in senses:
                    definition = sense.get("definition") or sense.get("definition_original") or ""
                    if word and isinstance(definition, str) and definition.strip():
                        await queue.put({"word": word, "definition": _clean_text(definition)[:2000]})
                        enq_local += 1

            if enq_local == 0:
                print(f"⚠️ {os.path.basename(file_path)}: 파싱 결과 0건")
        except Exception as e:
            print(f"파일 처리 오류 {file_path}: {e}")


def vectorize_batch(model, definitions):
    """CPU-bound 작업인 벡터 변환을 수행하는 동기 함수입니다."""
    return model.encode(definitions, show_progress_bar=False, batch_size=BATCH_SIZE).tolist()


async def consumer(name: str, queue: asyncio.Queue, milvus_client: MilvusClient, model: SentenceTransformer,
                   executor: ThreadPoolExecutor):
    """큐에서 데이터를 가져와 배치 단위로 벡터화하고 Milvus에 삽입합니다."""
    processed_count = 0
    while True:
        try:
            # 💡 [핵심 수정] 첫 아이템은 큐에 데이터가 들어올 때까지 기다립니다.
            item = await queue.get()
            if item is None:  # 종료 신호(None)를 받으면 루프를 탈출합니다.
                break

            batch_data = [item]
            queue.task_done()

            # 💡 [핵심 수정] 큐에 남은 데이터를 최대한 BATCH_SIZE까지 채웁니다.
            while len(batch_data) < BATCH_SIZE:
                try:
                    # get_nowait()은 큐가 비어있으면 즉시 예외를 발생시킵니다.
                    item = queue.get_nowait()
                    if item is None:
                        # 만약 배치 중간에 종료 신호를 만나면, 다른 consumer를 위해 다시 큐에 넣습니다.
                        await queue.put(None)
                        break
                    batch_data.append(item)
                    queue.task_done()
                except asyncio.QueueEmpty:
                    # 큐가 비었으면 현재까지 모인 데이터로 처리를 진행합니다.
                    break

            # --- 데이터 처리 로직 (이전과 동일) ---
            words = [d['word'] for d in batch_data]
            definitions = [d['definition'] for d in batch_data]

            vectors = await asyncio.get_running_loop().run_in_executor(
                executor, vectorize_batch, model, definitions
            )

            entities = [
                {"word": w, "definition": d, "semantic_vector": v}
                for w, d, v in zip(words, definitions, vectors)
            ]

            insert_task = partial(milvus_client.insert, collection_name=COLLECTION_NAME, data=entities)
            await asyncio.get_running_loop().run_in_executor(executor, insert_task)
            processed_count += len(batch_data)

        except Exception as e:
            print(f"[{name}] 처리 중 오류 발생: {e}")
            break

    if processed_count > 0:
        print(f"[{name}]이 총 {processed_count}개의 데이터를 처리하고 종료합니다.")


# --- 3. 메인 실행 로직 ---

async def main():
    # 0. Milvus 클라이언트 및 모델 로드
    print("Milvus 서버에 연결 시도 중...")
    milvus_client = MilvusClient(uri=MILVUS_URI)
    print("Milvus 서버에 성공적으로 연결되었습니다!")

    print(f"의미 분석용 언어 모델 '{MODEL_NAME}'을(를) 로드합니다...")
    semantic_model = SentenceTransformer(MODEL_NAME)
    VECTOR_DIMENSION = semantic_model.get_sentence_embedding_dimension()
    print(f"모델 로드 완료! 벡터 차원: {VECTOR_DIMENSION}")

    # 1. Milvus 컬렉션 준비
    if milvus_client.has_collection(collection_name=COLLECTION_NAME):
        milvus_client.drop_collection(collection_name=COLLECTION_NAME)
        print(f"기존 컬렉션 '{COLLECTION_NAME}'을(를) 삭제했습니다.")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="word", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="definition", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="semantic_vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
    ]
    schema = CollectionSchema(fields, description="비동기 처리된 의미 벡터 DB")
    milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    print(f"컬렉션 '{COLLECTION_NAME}' 생성 완료.")

    # 2. 비동기 작업 환경 설정
    queue = asyncio.Queue(maxsize=BATCH_SIZE * 5)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    executor = ThreadPoolExecutor()

    # 3. 생산자(Producer)와 소비자(Consumer) 태스크 생성
    file_list = get_json_file_list()
    if not file_list:
        print(f"'{DATA_DIR}' 디렉토리에서 JSON 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    producer_tasks = [producer(file_path, queue, semaphore) for file_path in file_list]
    # 💡 [핵심 수정] 각 소비자에게 고유한 이름을 부여하여 로그 추적 용이
    consumer_tasks = [consumer(f"Consumer-{i + 1}", queue, milvus_client, semantic_model, executor) for i in
                      range(CONSUMER_COUNT)]

    # 4. 태스크 실행 및 대기
    print(f"총 {len(file_list)}개의 파일을 처리합니다...")
    # 💡 [핵심 수정] 생산자와 소비자를 동시에 실행
    producers_future = asyncio.gather(*producer_tasks)
    consumers_future = asyncio.gather(*consumer_tasks)

    # 모든 생산자 작업이 끝날 때까지 대기
    await producers_future
    print("모든 파일 읽기 및 파싱 완료.")

    # 💡 [핵심 수정] 생산자가 모두 끝나면, 소비자에게 종료 신호(None)를 보냄
    for _ in range(CONSUMER_COUNT):
        await queue.put(None)

    # 모든 소비자가 종료 신호를 받고 작업을 마칠 때까지 대기
    await consumers_future
    print("모든 데이터 삽입 완료.")

    # 5. 인덱스 생성 및 연결 종료
    print("데이터 플러시 및 인덱스 생성을 시작합니다...")
    milvus_client.flush(COLLECTION_NAME)

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="semantic_vector",
        index_type="HNSW",
        metric_type="COSINE",  # SBERT 계열은 COSINE 권장
        params={"M": 16, "efConstruction": 200}
    )
    milvus_client.create_index(COLLECTION_NAME, index_params=index_params)

    # 검색/쿼리 전에 로드 (이제 index not found 안 납니다)
    milvus_client.load_collection(COLLECTION_NAME)

    # 헬스체크(닫기 전)
    print("\n[Health Check]")
    # print("describe_index:", milvus_client.describe_index
    # (COLLECTION_NAME))
    # print("stats:", milvus_client.get_collection_stats(COLLECTION_NAME))
    # print("sample:", milvus_client.query(
    #     collection_name=COLLECTION_NAME,
    #     filter="",
    #     limit=5,
    #     output_fields=["id", "word", "definition"]
    # ))

    print("\n모든 작업이 완료되었습니다!")
    # print("\n[Health Check] ----------")
    # print("collections:", milvus_client.list_collections())  # 컬렉션 존재 확인
    # print("describe:", milvus_client.describe_collection(COLLECTION_NAME))  # 스키마/인덱스 확인
    # print("stats (row_count may be 0 while streaming):", milvus_client.get_collection_stats(COLLECTION_NAME))
    #
    # # 실제 엔티티 샘플 조회: filter="" + limit 사용
    # try:
    #     sample = milvus_client.query(
    #         collection_name=COLLECTION_NAME,
    #         filter="",
    #         limit=5,
    #         output_fields=["id", "word", "definition"]
    #     )
    #     print("query sample:", sample)
    # except Exception as e:
    #     print("query error:", e)

    milvus_client.close()

    #----------------
    check = MilvusClient(uri=MILVUS_URI)  # 필요하면 db_name="default" 도 명시
    print("\n[Health Check - new client]")
    print("collections:", check.list_collections())
    print("describe:", check.describe_collection(COLLECTION_NAME))
    print("stats:", check.get_collection_stats(COLLECTION_NAME))
    print("sample:", check.query(
        collection_name=COLLECTION_NAME,
        filter="",
        limit=5,
        output_fields=["id", "word", "definition"]
    ))
    check.close()
    #---------------
    executor.shutdown()


if __name__ == "__main__":
    import time

    start = time.time()
    asyncio.run(main())
    end = time.time()

    #5만개의 데이터를 저장할때 166초가 걸림 즉 1초당 만 번 가동,
    # 시간 복잡도: 임베딩 생성 O(N·L²), 인덱스 빌드 O(N·M·logN) 총: O(N·L²) + O(N·M·logN)
    # 나중에 발음 벡터(phn_vec) 추가 시도해도 총합은 O(N·(L² + M·logN)) 스케일 유지됨.

    print(end - start)
