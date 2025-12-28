"""
Data/VectorBuilder.py

메모리 최적화 VectorDB 구축 및 관리

메모리 최적화:
- 실시간 메모리 모니터링
- 점진적 가비지 컬렉션
- 배치 크기 동적 조정
- 프로세스 수 최소화
- 메모리 임계값 체크
"""
import os
import time
import asyncio
import tracemalloc
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
import numpy as np
import psutil  # 메모리 모니터링
import gc
import collections

from pymilvus import (
    MilvusClient,
    FieldSchema,
    CollectionSchema,
    DataType
)
from sentence_transformers import SentenceTransformer

from .OptimizedParser import (
    OptimizedParser,
    batch_generator
)

# 보안 강화된 설정 관리
from .Config import (
    get_milvus_connection_params,
    COLLECTION_NAME,
    VECTOR_MODEL,
    DATA_DIR,
    PROCESS_COUNT,
    BULK_INSERT_SIZE,
    INDEX_TYPE,
    METRIC_TYPE,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    MAX_MEMORY_GB,
    MEMORY_PROFILE
)


# ============================================
# 메모리 모니터링
# ============================================



def show_growth():
    print("\n[객체 개수 통계]")
    # 현재 메모리에 있는 모든 객체 수집
    objects = gc.get_objects()
    counts = collections.defaultdict(int)

    for o in objects:
        counts[type(o)] += 1

    # 가장 많은 객체 Top 5 출력
    for t, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{t}: {c}개")


def get_memory_usage() -> Dict[str, float]:
    """
    현재 메모리 사용량 (GB)

    Returns:
        {
            'used': 사용 중 메모리,
            'available': 사용 가능 메모리,
            'percent': 사용률 (%),
            'total': 총 메모리
        }
    """
    mem = psutil.virtual_memory()
    return {
        'used': mem.used / (1024 ** 3),
        'available': mem.available / (1024 ** 3),
        'percent': mem.percent,
        'total': mem.total / (1024 ** 3)
    }


def check_memory_threshold(threshold_gb: float = None) -> bool:
    """
    메모리 임계값 체크

    Args:
        threshold_gb: 임계값 (GB), None이면 설정값 사용

    Returns:
        True: 여유 있음
        False: 부족함
    """
    if threshold_gb is None:
        threshold_gb = MAX_MEMORY_GB

    mem = get_memory_usage()
    return mem['available'] >= threshold_gb


# ============================================
# 멀티프로세싱 워커 함수
# ============================================

_vectorizer = None

def init_vectorizer_worker():
    """프로세스 초기화: 벡터 모델 로드"""
    global _vectorizer
    _vectorizer = SentenceTransformer(VECTOR_MODEL)
    print(f"  ✓ [PID {os.getpid()}] 벡터 모델 로드")

def vectorize_texts_worker(texts: List[str]) -> np.ndarray:
    """멀티프로세싱 벡터화"""
    global _vectorizer
    result = _vectorizer.encode(
        texts,
        show_progress_bar=False,
        batch_size=len(texts),
        convert_to_numpy=True
    )

    # 명시적 메모리 해제
    del texts
    gc.collect()

    return result


# ============================================
# 메모리 최적화 VectorDB 구축
# ============================================

class VectorDBBuilder:
    """
    메모리 최적화 VectorDB 구축 및 관리

    메모리 최적화 기능:
    - 실시간 메모리 모니터링
    - 점진적 가비지 컬렉션
    - 동적 배치 크기 조정
    - 메모리 임계값 체크
    """

    def __init__(self):
        self._print_header()

        self.milvus_client = None
        self.process_pool = None
        self.vector_dim = None

        # 통계
        self.stats = {
            'total_files': 0,
            'total_entries': 0,
            'total_batches': 0,
            'vectorize_time': 0,
            'insert_time': 0,
            'index_time': 0,
            'peak_memory_gb': 0,
            'gc_count': 0
        }

    def _print_header(self):
        """헤더 출력"""
        print("\n" + "="*70)
        print("🔨 VectorDB 구축 및 관리 (메모리 최적화)")
        print("="*70)
        print(f"메모리 프로파일: {MEMORY_PROFILE.upper()}")
        print(f"  • 프로세스: {PROCESS_COUNT}개")
        print(f"  • 배치 크기: {BULK_INSERT_SIZE:,}개")
        print(f"  • 최대 메모리: ~{MAX_MEMORY_GB}GB")
        print(f"  • 메모리 모니터링: 활성화")
        print("="*70 + "\n")

    # ============================================
    # 초기화 및 연결
    # ============================================

    def connect(self):
        """Milvus 연결 (보안 강화)"""
        if self.milvus_client is None:
            conn_params = get_milvus_connection_params()

            print(f"🔌 Milvus 연결: {conn_params['uri']}")
            if 'user' in conn_params:
                print(f"   인증: 사용자 {conn_params['user'][:3]}***")

            self.milvus_client = MilvusClient(**conn_params)
            print("✅ 연결 성공\n")
        return self.milvus_client

    def initialize_model(self):
        """벡터 모델 초기화"""
        if self.vector_dim is None:
            print(f"🔧 벡터 모델 정보: {VECTOR_MODEL}")
            temp_model = SentenceTransformer(VECTOR_MODEL)
            self.vector_dim = temp_model.get_sentence_embedding_dimension()
            print(f"✅ 벡터 차원: {self.vector_dim}D\n")
            del temp_model
            gc.collect()
        return self.vector_dim

    # ============================================
    # 스키마 관리
    # ============================================

    def create_schema(self) -> CollectionSchema:
        """Milvus 스키마 정의"""
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(name="target_code", dtype=DataType.INT64),
            FieldSchema(name="word", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="definition", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="pos", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="word_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="entry_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="pronunciation", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="similar_words", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="parent_words", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(
                name="semantic_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.vector_dim
            )
        ]

        return CollectionSchema(
            fields=fields,
            description="한국어 사전 의미 검색 DB"
        )

    def create_collection(self, drop_existing: bool = False):
        """컬렉션 생성"""
        client = self.connect()

        if client.has_collection(COLLECTION_NAME):
            if drop_existing:
                client.drop_collection(COLLECTION_NAME)
                print(f"🗑️  기존 컬렉션 '{COLLECTION_NAME}' 삭제")
            else:
                print(f"⚠️  컬렉션 '{COLLECTION_NAME}' 이미 존재")
                return

        dim = self.initialize_model()
        schema = self.create_schema()

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema
        )
        print(f"✅ 컬렉션 '{COLLECTION_NAME}' 생성\n")

    # ============================================
    # 초기 구축 (메모리 최적화)
    # ============================================

    async def build(self, data_dir: str = None):
        """VectorDB 초기 구축 (메모리 최적화)"""
        if data_dir is None:
            data_dir = DATA_DIR

        total_start = time.time()

        # 메모리 체크
        mem = get_memory_usage()
        print(f"💾 초기 메모리:")
        print(f"   사용 중: {mem['used']:.2f}GB / {mem['total']:.2f}GB ({mem['percent']:.1f}%)")
        print(f"   사용 가능: {mem['available']:.2f}GB\n")

        if not check_memory_threshold():
            print(f"⚠️  메모리 부족: 최소 {MAX_MEMORY_GB}GB 필요")
            print(f"   현재 사용 가능: {mem['available']:.2f}GB")
            print("\n💡 해결 방법:")
            print("   1. 메모리 프로파일 변경: MEMORY_PROFILE=conservative")
            print("   2. 프로세스 수 감소: PROCESS_COUNT=2")
            print("   3. 배치 크기 감소: BULK_INSERT_SIZE=500\n")
            return

        # 초기화
        self.connect()
        self.initialize_model()
        self.create_collection(drop_existing=True)

        # 파일 목록
        file_list = self._get_json_files(data_dir)
        if not file_list:
            print(f"❌ '{data_dir}'에서 JSON 파일을 찾을 수 없습니다.")
            return

        self.stats['total_files'] = len(file_list)
        print(f"📁 JSON 파일: {len(file_list)}개\n")

        # 프로세스 풀 초기화
        print(f"⚙️  {PROCESS_COUNT}개 프로세스 초기화...\n")
        self.process_pool = ProcessPoolExecutor(
            max_workers=PROCESS_COUNT,
            initializer=init_vectorizer_worker
        )
        print("✅ 프로세스 풀 준비\n")

        # 메모리 최적화 스트리밍 처리
        print("="*70)
        print("🔄 메모리 최적화 스트리밍 처리")
        print("="*70 + "\n")

        await self._memory_optimized_insert(file_list)

        print(f"\n✅ 데이터 삽입 완료")
        print(f"   항목: {self.stats['total_entries']:,}개")
        print(f"   배치: {self.stats['total_batches']:,}개")
        print(f"   GC 실행: {self.stats['gc_count']:,}회")

        # 인덱스 생성
        print("\n" + "="*70)
        print("🔄 인덱스 생성")
        print("="*70)
        self.create_index()

        # 정리
        if self.process_pool:
            self.process_pool.shutdown()

        # 최종 가비지 컬렉션
        gc.collect()

        # 최종 통계
        total_time = time.time() - total_start
        self._print_build_stats(total_time)

    async def _memory_optimized_insert(self, file_list: List[str]):
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        """메모리 최적화 삽입"""
        loop = asyncio.get_running_loop()

        entry_gen = OptimizedParser.parse_multiple_files_generator(file_list)
        batch_gen = batch_generator(entry_gen, BULK_INSERT_SIZE)

        batch_count = 0
        vectorize_start = time.time()

        for batch in batch_gen:
            batch_count += 1
            batch_size = len(batch)
            self.stats['total_entries'] += batch_size

            # 메모리 체크
            mem = get_memory_usage()
            self.stats['peak_memory_gb'] = max(
                self.stats['peak_memory_gb'],
                mem['used']
            )

            print(f"  배치 {batch_count:3d}: {batch_size:,}개 | "
                  f"메모리: {mem['used']:.2f}GB ({mem['percent']:.1f}%)")

            # 메모리 부족 시 가비지 컬렉션
            if mem['available'] < MAX_MEMORY_GB * 0.5:
                gc.collect()
                self.stats['gc_count'] += 1
                mem_after = get_memory_usage()
                print(f"     → GC 실행: {mem['used']:.2f}GB → {mem_after['used']:.2f}GB")

            # 배치 처리
            await self._vectorize_and_insert_batch(batch, loop)

            # 명시적 메모리 해제
            del batch

            # 10배치마다 강제 GC
            if batch_count % 10 == 0:
                show_growth()
                gc.collect()
                self.stats['gc_count'] += 1

                elapsed = time.time() - vectorize_start
                rate = self.stats['total_entries'] / elapsed
                mem = get_memory_usage()

                print(f"     → 누적: {self.stats['total_entries']:,}개 | "
                      f"{rate:.0f}개/초 | "
                      f"메모리: {mem['used']:.2f}GB\n")

                # [추가] 2. 일정 배치마다 메모리 증가량 분석 (예: 10번째 배치마다)
                print(f"\n--- 배치 {batch_count} 메모리 분석 ---")
                snapshot2 = tracemalloc.take_snapshot()

                # 이전 스냅샷과 비교하여 증가한 메모리 상위 10개 출력
                top_stats = snapshot2.compare_to(snapshot1, 'lineno')

                print("[Top 10 메모리 증가 원인]")
                for stat in top_stats[:10]:
                    print(stat)

                # 기준점 갱신 (선택 사항: 계속 누적되는걸 보려면 갱신 X)
                # snapshot1 = snapshot2

        self.stats['total_batches'] = batch_count
        self.stats['vectorize_time'] = time.time() - vectorize_start

    async def _vectorize_and_insert_batch(
        self,
        batch: List[Dict],
        loop: asyncio.AbstractEventLoop
    ):
        """배치 벡터화 및 삽입 (메모리 최적화)"""
        texts = [entry['vector_text'] for entry in batch]

        # 청크 크기 동적 조정 (메모리 고려)
        chunk_size = max(1, len(texts) // PROCESS_COUNT)
        text_chunks = [
            texts[i:i+chunk_size]
            for i in range(0, len(texts), chunk_size)
        ]

        tasks = [
            loop.run_in_executor(self.process_pool, vectorize_texts_worker, chunk)
            for chunk in text_chunks if chunk
        ]
        vector_chunks = await asyncio.gather(*tasks)
        vectors = np.vstack(vector_chunks).tolist()

        # 명시적 메모리 해제
        del texts, text_chunks, vector_chunks

        entities = self._create_entities(batch, vectors)

        insert_start = time.time()
        await loop.run_in_executor(
            None,
            partial(
                self.milvus_client.insert,
                collection_name=COLLECTION_NAME,
                data=entities
            )
        )
        self.stats['insert_time'] += time.time() - insert_start

        # 명시적 메모리 해제
        del vectors, entities

    def _create_entities(
        self,
        batch: List[Dict],
        vectors: List[List[float]]
    ) -> List[Dict]:
        """엔티티 생성"""
        entities = []
        for entry, vector in zip(batch, vectors):
            entity = {
                "target_code": entry['target_code'],
                "word": entry['word'],
                "definition": entry['definition'],
                "pos": entry['pos'],
                "word_type": entry['word_type'],
                "entry_type": entry['entry_type'],
                "pronunciation": entry['pronunciation'],
                "similar_words": ",".join(entry['similar_words']),
                "parent_words": ",".join(entry['parent_words']),
                "semantic_vector": vector
            }
            entities.append(entity)
        return entities

    # ============================================
    # 증분 업데이트
    # ============================================

    async def incremental_update(self, new_files: List[str]):
        """증분 업데이트"""
        print(f"🔄 증분 업데이트 시작 ({len(new_files)}개 파일)\n")

        self.connect()
        self.initialize_model()

        print(f"⚙️  {PROCESS_COUNT}개 프로세스 초기화...\n")
        self.process_pool = ProcessPoolExecutor(
            max_workers=PROCESS_COUNT,
            initializer=init_vectorizer_worker
        )

        await self._memory_optimized_insert(new_files)

        print(f"\n✅ 증분 업데이트 완료: {self.stats['total_entries']:,}개 추가")

        if self.process_pool:
            self.process_pool.shutdown()

    # ============================================
    # 인덱스 관리
    # ============================================

    def create_index(self):
        """인덱스 생성"""
        start = time.time()

        print("🔨 데이터 플러시...")
        self.milvus_client.flush(COLLECTION_NAME)

        print(f"🔨 {INDEX_TYPE} 인덱스 생성...")
        index_params = self.milvus_client.prepare_index_params()

        if INDEX_TYPE == "HNSW":
            index_params.add_index(
                field_name="semantic_vector",
                index_type="HNSW",
                metric_type=METRIC_TYPE,
                params={
                    "M": HNSW_M,
                    "efConstruction": HNSW_EF_CONSTRUCTION
                }
            )

        self.milvus_client.create_index(
            collection_name=COLLECTION_NAME,
            index_params=index_params
        )

        print("🔨 컬렉션 로드...")
        self.milvus_client.load_collection(COLLECTION_NAME)

        self.stats['index_time'] = time.time() - start
        print("✅ 완료\n")

    def rebuild_index(self):
        """인덱스 재구축"""
        print("🔄 인덱스 재구축 중...")

        self.milvus_client.drop_index(
            collection_name=COLLECTION_NAME,
            field_name="semantic_vector"
        )

        self.create_index()
        print("✅ 인덱스 재구축 완료")

    # ============================================
    # 컬렉션 관리
    # ============================================

    def get_collection_info(self) -> Dict:
        """컬렉션 정보 조회"""
        client = self.connect()
        stats = client.get_collection_stats(COLLECTION_NAME)

        return {
            "collection_name": COLLECTION_NAME,
            "total_entries": stats.get('row_count', 0),
            "vector_dim": self.vector_dim or self.initialize_model(),
            "index_type": INDEX_TYPE
        }

    def delete_collection(self):
        """컬렉션 삭제"""
        client = self.connect()
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
            print(f"🗑️  컬렉션 '{COLLECTION_NAME}' 삭제 완료")
        else:
            print(f"⚠️  컬렉션 '{COLLECTION_NAME}' 없음")

    # ============================================
    # 유틸리티
    # ============================================

    def _get_json_files(self, data_dir: str) -> List[str]:
        """JSON 파일 목록"""
        data_path = Path(data_dir)
        if not data_path.exists():
            return []

        json_files = list(data_path.rglob("*.json"))
        json_files.sort()
        return [str(f) for f in json_files]

    def _print_build_stats(self, total_time: float):
        """구축 통계 출력"""
        stats = self.milvus_client.get_collection_stats(COLLECTION_NAME)
        row_count = stats.get('row_count', 0)

        print("\n" + "="*70)
        print("📊 VectorDB 구축 통계")
        print("="*70)
        print(f"컬렉션: {COLLECTION_NAME}")
        print(f"총 파일: {self.stats['total_files']}개")
        print(f"총 항목: {row_count:,}개")
        print(f"\n💾 메모리 프로파일: {MEMORY_PROFILE.upper()}")
        print(f"   프로세스: {PROCESS_COUNT}개")
        print(f"   배치 크기: {BULK_INSERT_SIZE:,}개")
        print(f"   피크 메모리: {self.stats['peak_memory_gb']:.2f}GB")
        print(f"   GC 실행: {self.stats['gc_count']:,}회")
        print(f"\n⏱️  시간 분석:")
        print(f"   벡터화: {self.stats['vectorize_time']:.1f}초")
        print(f"   삽입: {self.stats['insert_time']:.1f}초")
        print(f"   인덱스: {self.stats['index_time']:.1f}초")
        print(f"   총 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print(f"\n⚡ 처리 속도:")
        print(f"   평균: {row_count/total_time:,.0f}개/초")
        print("="*70 + "\n")


async def main():
    """메인 실행"""
    builder = VectorDBBuilder()
    await builder.build()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    asyncio.run(main())
