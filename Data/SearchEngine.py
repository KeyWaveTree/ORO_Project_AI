"""
Data/SearchEngine.py

메모리 관리 최적화 검색 엔진

주요 개선:
- 벡터 모델 싱글톤 재사용
- Milvus 연결 재사용
- 명시적 메모리 해제
- 컨텍스트 매니저 지원
"""
import gc
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

# 보안 강화된 설정 관리
from .Config import (
    get_milvus_connection_params,
    COLLECTION_NAME,
    VECTOR_MODEL,
    SEARCH_EF,
    DEFAULT_TOP_K
)


# ============================================
# 벡터 모델 싱글톤 (TextProcessing과 공유)
# ============================================

class VectorModelManager:
    """벡터 모델 싱글톤 관리자"""
    _instance = None
    _model = None
    _model_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, model_name: str) -> SentenceTransformer:
        """벡터 모델 가져오기 (싱글톤)"""
        if self._model is not None and self._model_name == model_name:
            return self._model

        if self._model is not None:
            del self._model
            gc.collect()

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

        return self._model


# ============================================
# Milvus 연결 싱글톤
# ============================================

class MilvusConnectionManager:
    """
    Milvus 연결 싱글톤 관리자

    메모리 누수 방지:
    - Milvus 연결을 1번만 생성
    - 프로세스 전체에서 재사용
    """
    _instance = None
    _client = None
    _conn_params = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_client(self) -> MilvusClient:
        """Milvus 클라이언트 가져오기 (싱글톤)"""
        current_params = get_milvus_connection_params()

        # 연결 파라미터가 변경되었거나 연결이 없으면 새로 생성
        if self._client is None or self._conn_params != current_params:
            if self._client is not None:
                # 기존 연결 종료 (필요 시)
                pass

            self._client = MilvusClient(**current_params)
            self._conn_params = current_params

        return self._client


# ============================================
# 메모리 관리 최적화 검색 엔진
# ============================================

class DictionarySearch:
    """
    한국어 사전 의미 검색 엔진 (메모리 관리 최적화)

    메모리 최적화:
    - 벡터 모델 싱글톤 재사용
    - Milvus 연결 싱글톤 재사용
    - 명시적 메모리 해제
    - 컨텍스트 매니저 지원
    """

    def __init__(self, verbose: bool = False):
        """
        초기화

        Args:
            verbose: 상세 로그 출력 여부
        """
        self.verbose = verbose
        self.collection_loaded = False

        # 싱글톤 관리자
        self.model_manager = VectorModelManager()
        self.conn_manager = MilvusConnectionManager()

        # 초기화
        self._connect()
        self._init_vectorizer()
        self._ensure_collection_loaded()

    def _connect(self):
        """Milvus 연결 (싱글톤 재사용)"""
        self.client = self.conn_manager.get_client()

        if self.verbose:
            conn_params = get_milvus_connection_params()
            print(f"🔌 Milvus 연결: {conn_params['uri']}")
            if 'user' in conn_params:
                print(f"   인증: 사용자 {conn_params['user'][:3]}***")
            print("✅ 연결 성공 (재사용)\n")

        return self.client

    def _init_vectorizer(self):
        """벡터 모델 초기화 (싱글톤 재사용)"""
        self.vectorizer = self.model_manager.get_model(VECTOR_MODEL)

        if self.verbose:
            print(f"🔧 벡터 모델 로드: {VECTOR_MODEL}")
            print("✅ 모델 로드 완료 (재사용)\n")

        return self.vectorizer

    def _ensure_collection_loaded(self):
        """컬렉션 로드 확인 및 에러 처리"""
        if self.collection_loaded:
            return True

        try:
            if not self.client.has_collection(COLLECTION_NAME):
                raise RuntimeError(
                    f"VectorDB 컬렉션 '{COLLECTION_NAME}'이 존재하지 않습니다.\n"
                    f"💡 해결방법:\n"
                    f"   1. VectorDB 구축: python -m Data.VectorBuilder\n"
                    f"   2. TextProcessing에서 자동 구축: auto_build=True"
                )

            self.client.load_collection(COLLECTION_NAME)
            self.collection_loaded = True

            if self.verbose:
                stats = self.client.get_collection_stats(COLLECTION_NAME)
                row_count = stats.get('row_count', 0)
                print(f"✅ 컬렉션 로드: {COLLECTION_NAME} ({row_count:,}개 항목)\n")

            return True

        except MilvusException as e:
            error_msg = str(e).lower()

            if "index not found" in error_msg:
                raise RuntimeError(
                    f"VectorDB 인덱스가 생성되지 않았습니다.\n"
                    f"데이터는 있지만 인덱스가 없어 검색할 수 없습니다.\n"
                    f"💡 해결방법:\n"
                    f"   1. VectorDB 재구축: python -m Data.VectorBuilder\n"
                    f"   2. TextProcessing에서 자동 구축: auto_build=True\n"
                    f"   3. 수동 인덱스 생성: builder.create_index()"
                ) from e

            elif "collection not loaded" in error_msg:
                raise RuntimeError(
                    f"컬렉션 '{COLLECTION_NAME}'을 로드할 수 없습니다.\n"
                    f"💡 해결방법:\n"
                    f"   1. Milvus 서버 재시작: docker restart milvus-standalone\n"
                    f"   2. VectorDB 재구축: python -m Data.VectorBuilder"
                ) from e

            else:
                raise RuntimeError(
                    f"Milvus 에러: {e}\n"
                    f"💡 해결방법:\n"
                    f"   1. Milvus 서버 확인: docker ps | grep milvus\n"
                    f"   2. VectorDB 재구축: python -m Data.VectorBuilder"
                ) from e

        except Exception as e:
            raise RuntimeError(
                f"컬렉션 로드 중 에러: {e}\n"
                f"💡 해결방법:\n"
                f"   1. VectorDB 재구축: python -m Data.VectorBuilder\n"
                f"   2. Milvus 서버 재시작: docker restart milvus-standalone"
            ) from e

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료 (메모리 정리)"""
        self.cleanup()
        return False

    def cleanup(self):
        """메모리 정리"""
        # 벡터 모델과 Milvus 연결은 싱글톤이므로 해제하지 않음
        # (프로세스 전체에서 재사용)

        # 명시적 가비지 컬렉션
        gc.collect()

    def search(
        self,
        query: str,
        top_k: int = None,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        의미 기반 검색 (메모리 최적화)

        Args:
            query: 검색 쿼리
            top_k: 결과 개수 (기본: DEFAULT_TOP_K)
            search_params: 검색 파라미터

        Returns:
            검색 결과 리스트
        """
        if top_k is None:
            top_k = DEFAULT_TOP_K

        # 컬렉션 로드 확인
        self._ensure_collection_loaded()

        # 쿼리 벡터화
        query_vector = self.vectorizer.encode(query).tolist()

        # 검색 파라미터
        if search_params is None:
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": SEARCH_EF}
            }

        try:
            # 검색 실행
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vector],
                limit=top_k,
                search_params=search_params,
                output_fields=[
                    "word", "definition", "pos", "pronunciation",
                    "similar_words", "parent_words"
                ]
            )

            # 결과 포맷팅
            formatted_results = []

            for hits in results:
                for hit in hits:
                    result = {
                        'word': hit.get('entity', {}).get('word', ''),
                        'definition': hit.get('entity', {}).get('definition', ''),
                        'pos': hit.get('entity', {}).get('pos', ''),
                        'pronunciation': hit.get('entity', {}).get('pronunciation', ''),
                        'similarity': hit.get('distance', 0.0),
                        'similar_words': hit.get('entity', {}).get('similar_words', '').split(','),
                        'parent_words': hit.get('entity', {}).get('parent_words', '').split(',')
                    }
                    formatted_results.append(result)

            # 메모리 정리
            del query_vector
            del results
            gc.collect()

            return formatted_results

        except MilvusException as e:
            error_msg = str(e).lower()

            if "collection not loaded" in error_msg:
                # 재로드 시도
                try:
                    self.collection_loaded = False
                    self._ensure_collection_loaded()
                    # 재시도
                    return self.search(query, top_k, search_params)
                except:
                    pass

            raise RuntimeError(
                f"검색 중 에러: {e}\n"
                f"💡 해결방법:\n"
                f"   1. VectorDB 재구축: python -m Data.VectorBuilder\n"
                f"   2. Milvus 재시작: docker restart milvus-standalone"
            ) from e

        except Exception as e:
            raise RuntimeError(
                f"검색 중 예상치 못한 에러: {e}\n"
                f"💡 해결방법:\n"
                f"   1. 쿼리 확인: '{query}'\n"
                f"   2. VectorDB 상태 확인"
            ) from e

    def batch_search(
        self,
        queries: List[str],
        top_k: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        배치 검색 (메모리 최적화)

        Args:
            queries: 검색 쿼리 리스트
            top_k: 각 쿼리당 결과 개수

        Returns:
            {쿼리: 결과} 딕셔너리
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, top_k)

            # 각 검색 후 메모리 정리
            gc.collect()

        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            stats = self.client.get_collection_stats(COLLECTION_NAME)
            return {
                'collection_name': COLLECTION_NAME,
                'row_count': stats.get('row_count', 0),
                'loaded': self.collection_loaded
            }
        except Exception as e:
            return {
                'collection_name': COLLECTION_NAME,
                'error': str(e),
                'loaded': False
            }


# ============================================
# 편의 함수
# ============================================

def search_korean_dictionary(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    간단한 검색 함수 (메모리 관리)

    Args:
        query: 검색 쿼리
        top_k: 결과 개수

    Returns:
        검색 결과
    """
    with DictionarySearch() as engine:
        return engine.search(query, top_k)


def main():
    """메모리 관리 예시"""
    print("="*70)
    print("🔍 SearchEngine 메모리 관리 예시")
    print("="*70 + "\n")

    # 방법 1: 컨텍스트 매니저 (권장)
    print("방법 1: 컨텍스트 매니저 (자동 정리)\n")
    print("-" * 70)

    with DictionarySearch(verbose=True) as engine:
        # 여러 검색 수행
        for query in ["한글", "컴퓨터", "사랑"]:
            print(f"\n검색: '{query}'")
            results = engine.search(query, top_k=3)

            print(f"결과: {len(results)}개")
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['word']} ({r['pos']}) - {r['similarity']:.3f}")

    print("\n" + "="*70)
    print("✅ 컨텍스트 매니저 종료 (자동 정리 완료)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
