"""
Data/TextProcessing.py

완전 통합 버전: VectorDB 자동 구축 + 메모리 관리

주요 기능:
1. VectorDB 자동 감지 및 복구 (인덱스 생성/재구축)
2. 메모리 관리 최적화 (싱글톤, GC, 컨텍스트 매니저)
3. 메모리 모니터링
"""
import gc
import weakref
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException
import psutil
import os

from .SearchEngine import DictionarySearch

from .Config import (get_milvus_connection_params, COLLECTION_NAME,
                    MIN_CONFIDENCE, PRONUNCIATION_WEIGHT)

# ============================================
# 싱글톤 벡터 모델 관리자
# ============================================

class VectorModelManager:
    """벡터 모델 싱글톤 관리자 (메모리 절약)"""
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

    def clear(self):
        """모델 메모리 해제"""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_name = None
            gc.collect()


# ============================================
# 메모리 모니터링
# ============================================

def get_process_memory_mb() -> float:
    """현재 프로세스 메모리 사용량 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def print_memory_usage(label: str = ""):
    """메모리 사용량 출력"""
    mem_mb = get_process_memory_mb()
    print(f"💾 메모리 [{label}]: {mem_mb:.1f}MB")


# ============================================
# 데이터 클래스
# ============================================

@dataclass
class CleanedText:
    """정제된 텍스트 결과"""
    original_text: str
    cleaned_text: str
    confidence: float
    changes: List[Tuple[str, str]]


@dataclass
class SearchResult:
    """검색 결과"""
    word: str
    definition: str
    pos: str
    pronunciation: str
    semantic_score: float
    pronunciation_score: float
    combined_score: float
    confidence: str
    similar_words: List[str]
    parent_words: List[str]


# ============================================
# 완전 통합 TextProcessing
# ============================================

class TextProcessing:
    """
    Dysarthric Speech 텍스트 처리 파이프라인

    통합 기능:
    1. VectorDB 자동 감지 및 복구
    2. 메모리 관리 최적화
    3. 메모리 모니터링
    """

    # 클래스 레벨 검색 엔진 캐시 (약한 참조)
    _search_engine_cache = weakref.WeakValueDictionary()

    def __init__(
        self,
        verbose: bool = False,
        auto_build: Optional[bool] = None,
        monitor_memory: bool = False
    ):
        """
        초기화

        Args:
            verbose: 상세 로그 출력
            auto_build: VectorDB 자동 구축 (True/False/None)
            monitor_memory: 메모리 사용량 모니터링
        """
        self.verbose = verbose
        self.auto_build = auto_build
        self.monitor_memory = monitor_memory

        if self.monitor_memory:
            print_memory_usage("초기화 전")

        # 헤더 출력
        if self.verbose:
            self._print_header()

        # VectorDB 확인 및 준비
        self._ensure_vectordb_ready()

        # 검색 엔진 초기화 (재사용)
        self._init_search_engine()

        # 벡터 모델 관리자
        self.model_manager = VectorModelManager()

        if self.monitor_memory:
            print_memory_usage("초기화 후")

    def _init_search_engine(self):
        """검색 엔진 초기화 (재사용)"""
        cache_key = "default"

        if cache_key in self._search_engine_cache:
            self.search_engine = self._search_engine_cache[cache_key]
            if self.verbose:
                print("✅ 검색 엔진 재사용 (메모리 절약)\n")
        else:
            self.search_engine = DictionarySearch(verbose=self.verbose)
            self._search_engine_cache[cache_key] = self.search_engine
            if self.verbose:
                print("✅ 검색 엔진 초기화 완료\n")

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료 (메모리 정리)"""
        self.cleanup()
        return False

    def cleanup(self):
        """메모리 정리"""
        if self.monitor_memory:
            print_memory_usage("정리 전")

        gc.collect()

        if self.monitor_memory:
            print_memory_usage("정리 후")

    # ============================================
    # VectorDB 관리
    # ============================================

    def _print_header(self):
        """헤더 출력"""
        print("\n" + "="*70)
        print("🎙️  Dysarthric Speech 텍스트 처리 파이프라인")
        print("="*70 + "\n")

    def _check_vectordb_exists(self) -> tuple[bool, str]:
        """
        VectorDB 컬렉션 및 인덱스 존재 여부 확인

        Returns:
            (상태, 메시지)
        """
        try:


            conn_params = get_milvus_connection_params()
            client = MilvusClient(**conn_params)

            if not client.has_collection(COLLECTION_NAME):
                if self.verbose:
                    print(f"⚠️  VectorDB 컬렉션 없음: {COLLECTION_NAME}")
                return False, "컬렉션없음"

            stats = client.get_collection_stats(COLLECTION_NAME)
            row_count = stats.get('row_count', 0)

            if row_count == 0:
                if self.verbose:
                    print(f"⚠️  컬렉션은 있지만 비어있음: {COLLECTION_NAME}")
                return False, "데이터없음"

            try:
                client.load_collection(COLLECTION_NAME)

                if self.verbose:
                    print(f"✅ VectorDB 확인: {COLLECTION_NAME} ({row_count:,}개 항목)")
                return True, "완전"

            except MilvusException as e:
                error_msg = str(e).lower()

                if "index not found" in error_msg:
                    if self.verbose:
                        print(f"⚠️  인덱스 없음: {COLLECTION_NAME}")
                        print(f"   데이터는 있지만 ({row_count:,}개) 인덱스가 생성되지 않음")
                    return False, "인덱스없음"
                else:
                    if self.verbose:
                        print(f"✅ VectorDB 확인: {COLLECTION_NAME} ({row_count:,}개 항목)")
                    return True, "완전"

        except MilvusException as e:
            error_msg = str(e).lower()
            if "collection not found" in error_msg:
                if self.verbose:
                    print(f"⚠️  VectorDB 컬렉션 없음")
                return False, "컬렉션없음"
            else:
                if self.verbose:
                    print(f"⚠️  Milvus 연결 실패: {e}")
                return False, "에러"

        except Exception as e:
            if self.verbose:
                print(f"⚠️  VectorDB 확인 중 에러: {e}")
            return False, "에러"

    def _ask_user_for_index_fix(self) -> str:
        """인덱스 없는 경우 사용자에게 물어보기"""
        print("\n" + "="*70)
        print("🔧 VectorDB 복구 필요")
        print("="*70)
        print("\n📊 현재 상태:")
        print("   • 컬렉션: 존재 ✅")
        print("   • 데이터: 존재 ✅")
        print("   • 인덱스: 없음 ❌")
        print("\n💡 해결 방법:")
        print("   1. 인덱스만 생성 (빠름, 5-10분)")
        print("   2. 삭제 후 재구축 (느림, 80-100분)")
        print("\n" + "="*70)

        while True:
            response = input("\n🤔 어떻게 하시겠습니까? (1=인덱스생성/2=재구축/n=건너뜀): ").strip().lower()

            if response in ['1', 'index', 'i']:
                return "index"
            elif response in ['2', 'rebuild', 'r']:
                return "rebuild"
            elif response in ['n', 'no', 'skip', 's', '아니오', 'ㄴ']:
                return "skip"
            else:
                print("   ⚠️  '1', '2', 또는 'n'을 입력해주세요.")

    def _create_index_only(self) -> bool:
        """기존 데이터에 인덱스만 생성"""
        print("\n" + "="*70)
        print("🔨 인덱스 생성 시작")
        print("="*70 + "\n")

        try:
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            try:
                from VectorBuilder import VectorDBBuilder
            except ImportError:
                try:
                    from VectorBuilder_MEMORY_OPT import VectorDBBuilder
                except ImportError:
                    print("❌ VectorBuilder를 찾을 수 없습니다")
                    return False

            builder = VectorDBBuilder()
            builder.connect()
            builder.initialize_model()

            print("🔨 인덱스 생성 중...")
            builder.create_index()

            print("\n" + "="*70)
            print("✅ 인덱스 생성 완료!")
            print("="*70 + "\n")

            return True

        except Exception as e:
            print("\n" + "="*70)
            print("❌ 인덱스 생성 실패")
            print("="*70)
            print(f"\n에러: {e}")
            print("\n💡 대안:")
            print("   삭제 후 재구축을 시도하세요")
            print("\n" + "="*70 + "\n")

            import traceback
            if self.verbose:
                print("\n상세 에러:")
                traceback.print_exc()

            return False

    def _delete_and_rebuild(self) -> bool:
        """기존 컬렉션 삭제 후 재구축"""
        print("\n" + "="*70)
        print("🗑️  기존 VectorDB 삭제 후 재구축")
        print("="*70 + "\n")

        try:
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            try:
                from VectorBuilder import VectorDBBuilder
            except ImportError:
                try:
                    from VectorBuilder_MEMORY_OPT import VectorDBBuilder
                except ImportError:
                    print("❌ VectorBuilder를 찾을 수 없습니다")
                    return False

            import multiprocessing as mp
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass

            builder = VectorDBBuilder()

            print("🗑️  기존 컬렉션 삭제 중...")
            builder.delete_collection()
            print("✅ 삭제 완료\n")

            print("🔨 VectorDB 재구축 시작...\n")

            import asyncio

            try:
                loop = asyncio.get_running_loop()
                import threading
                result = [None]
                error = [None]

                def run_in_thread():
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        new_loop.run_until_complete(builder.build())
                        new_loop.close()
                        result[0] = True
                    except Exception as e:
                        error[0] = e

                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()

                if error[0]:
                    raise error[0]

            except RuntimeError:
                asyncio.run(builder.build())

            print("\n" + "="*70)
            print("✅ VectorDB 재구축 완료!")
            print("="*70 + "\n")

            return True

        except Exception as e:
            print("\n" + "="*70)
            print("❌ VectorDB 재구축 실패")
            print("="*70)
            print(f"\n에러: {e}")
            print("\n💡 수동 구축:")
            print("   python -m Data.VectorBuilder")
            print("\n" + "="*70 + "\n")

            import traceback
            if self.verbose:
                print("\n상세 에러:")
                traceback.print_exc()

            return False

    def _build_vectordb(self) -> bool:
        """VectorDB 새로 구축"""
        print("\n" + "="*70)
        print("🔨 VectorDB 자동 구축 시작")
        print("="*70 + "\n")

        try:
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            try:
                from VectorBuilder import VectorDBBuilder
            except ImportError:
                try:
                    from VectorBuilder_MEMORY_OPT import VectorDBBuilder
                except ImportError:
                    print("❌ VectorBuilder를 찾을 수 없습니다")
                    return False

            import multiprocessing as mp
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass

            builder = VectorDBBuilder()

            import asyncio

            try:
                loop = asyncio.get_running_loop()
                import threading
                result = [None]
                error = [None]

                def run_in_thread():
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        new_loop.run_until_complete(builder.build())
                        new_loop.close()
                        result[0] = True
                    except Exception as e:
                        error[0] = e

                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()

                if error[0]:
                    raise error[0]

            except RuntimeError:
                asyncio.run(builder.build())

            print("\n" + "="*70)
            print("✅ VectorDB 구축 완료!")
            print("="*70 + "\n")

            return True

        except Exception as e:
            print("\n" + "="*70)
            print("❌ VectorDB 구축 실패")
            print("="*70)
            print(f"\n에러: {e}")
            print("\n💡 수동 구축:")
            print("   python -m Data.VectorBuilder")
            print("\n" + "="*70 + "\n")

            import traceback
            if self.verbose:
                print("\n상세 에러:")
                traceback.print_exc()

            return False

    def _ask_user_to_build(self) -> bool:
        """사용자에게 VectorDB 구축 여부 물어보기"""
        print("\n" + "="*70)
        print("🔨 VectorDB 구축 필요")
        print("="*70)
        print("\n📊 구축 정보:")
        print("   • 데이터: 110만개 한국어 사전 항목")
        print("   • 소요 시간: 80-100분 (ultra_conservative)")
        print("   • 메모리: ~4GB 피크")
        print("   • 1회만 구축하면 재사용 가능")
        print("\n" + "="*70)

        while True:
            response = input("\n🤔 지금 VectorDB를 구축하시겠습니까? (y/n): ").strip().lower()

            if response in ['y', 'yes', '예', 'ㅇ']:
                return True
            elif response in ['n', 'no', '아니오', 'ㄴ']:
                return False
            else:
                print("   ⚠️  'y' 또는 'n'을 입력해주세요.")

    def _ensure_vectordb_ready(self):
        """VectorDB 준비 상태 확인 및 필요 시 구축/복구"""
        exists, status = self._check_vectordb_exists()

        if exists and status == "완전":
            return

        if status == "인덱스없음":
            if self.auto_build is False:
                print("\n" + "="*70)
                print("❌ VectorDB 인덱스가 생성되지 않았습니다")
                print("="*70)
                print("\n데이터는 있지만 인덱스가 없어 검색할 수 없습니다.")
                print("\n💡 해결방법:")
                print("   1. 인덱스 생성: auto_build=True")
                print("   2. 수동 구축: python -m Data.VectorBuilder")
                print("\n" + "="*70 + "\n")
                raise RuntimeError("VectorDB index not found")

            elif self.auto_build is True:
                if self.verbose:
                    print("\n🔨 VectorDB 인덱스 자동 생성 모드")
                success = self._create_index_only()
                if not success:
                    print("\n⚠️  인덱스 생성 실패, 재구축을 시도합니다...")
                    success = self._delete_and_rebuild()
                    if not success:
                        raise RuntimeError("VectorDB rebuild failed")

            else:  # auto_build is None
                choice = self._ask_user_for_index_fix()

                if choice == "index":
                    success = self._create_index_only()
                    if not success:
                        print("\n💡 인덱스 생성에 실패했습니다.")
                        print("   재구축을 시도하시겠습니까?")
                        retry = input("   (y/n): ").strip().lower()

                        if retry in ['y', 'yes']:
                            success = self._delete_and_rebuild()
                            if not success:
                                raise RuntimeError("VectorDB rebuild failed")
                        else:
                            raise RuntimeError("VectorDB index creation failed")

                elif choice == "rebuild":
                    success = self._delete_and_rebuild()
                    if not success:
                        raise RuntimeError("VectorDB rebuild failed")

                else:
                    print("\n" + "="*70)
                    print("⏸️  VectorDB 복구 건너뜀")
                    print("="*70)
                    print("\n💡 나중에 복구:")
                    print("   python -m Data.VectorBuilder")
                    print("\n⚠️  검색 기능은 VectorDB 복구 후 사용 가능합니다.")
                    print("="*70 + "\n")
                    raise RuntimeError("VectorDB fix skipped by user")

        else:
            if self.auto_build is False:
                print("\n" + "="*70)
                print("❌ VectorDB가 구축되지 않았습니다")
                print("="*70)
                print(f"\n상태: {status}")
                print("\n💡 구축 방법:")
                print("   1. 수동 구축: python -m Data.VectorBuilder")
                print("   2. 자동 구축: TextProcessing(auto_build=True)")
                print("\n" + "="*70 + "\n")
                raise RuntimeError("VectorDB not found")

            elif self.auto_build is True:
                if self.verbose:
                    print("\n🔨 VectorDB 자동 구축 모드")
                success = self._build_vectordb()
                if not success:
                    raise RuntimeError("VectorDB build failed")

            else:  # auto_build is None
                should_build = self._ask_user_to_build()

                if should_build:
                    success = self._build_vectordb()
                    if not success:
                        raise RuntimeError("VectorDB build failed")
                else:
                    print("\n" + "="*70)
                    print("⏸️  VectorDB 구축 건너뜀")
                    print("="*70)
                    print("\n💡 나중에 구축:")
                    print("   python -m Data.VectorBuilder")
                    print("\n⚠️  검색 기능은 VectorDB 구축 후 사용 가능합니다.")
                    print("="*70 + "\n")
                    raise RuntimeError("VectorDB build skipped by user")

    # ============================================
    # 발음 정제
    # ============================================

    def clean_dysarthric_input(
        self,
        text: str,
        confidence_threshold: float = 0.7
    ) -> CleanedText:
        """불명확한 발음 입력 정제"""
        original = text
        cleaned = text
        changes = []
        confidence = 1.0

        if 'ㄴ' in text and len(text) > 1:
            cleaned = text.replace('ㄴ', 'n')
            changes.append((text, cleaned))
            confidence = 0.85

        return CleanedText(
            original_text=original,
            cleaned_text=cleaned,
            confidence=confidence,
            changes=changes
        )

    # ============================================
    # Dysarthric 검색 (메모리 최적화)
    # ============================================

    def dysarthric_search_pipeline(
        self,
        query: str,
        top_k: int = 5,
        use_pronunciation: bool = True,
        rerank: bool = True
    ) -> List[SearchResult]:
        """Dysarthric speech 검색 파이프라인"""
        if self.monitor_memory:
            print_memory_usage("검색 시작")

        cleaned = self.clean_dysarthric_input(query)
        search_query = cleaned.cleaned_text

        results = self.search_engine.search(search_query, top_k=top_k * 2)

        if use_pronunciation and rerank:
            results = self._rerank_by_pronunciation(
                results,
                search_query,
                cleaned.confidence
            )

        final_results = results[:top_k]

        search_results = []
        for r in final_results:
            search_results.append(SearchResult(
                word=r['word'],
                definition=r['definition'],
                pos=r['pos'],
                pronunciation=r['pronunciation'],
                semantic_score=r['similarity'],
                pronunciation_score=r.get('pronunciation_score', 0.0),
                combined_score=r.get('combined_score', r['similarity']),
                confidence=self._get_confidence_level(
                    r.get('combined_score', r['similarity'])
                ),
                similar_words=r.get('similar_words', []),
                parent_words=r.get('parent_words', [])
            ))

        del results
        del final_results

        if self.monitor_memory:
            print_memory_usage("검색 후 (정리 전)")

        gc.collect()

        if self.monitor_memory:
            print_memory_usage("검색 후 (정리 완료)")

        return search_results

    def _rerank_by_pronunciation(
        self,
        results: List[Dict],
        query: str,
        input_confidence: float
    ) -> List[Dict]:
        """발음 기반 재순위화"""
        for r in results:
            pron_score = self._calculate_pronunciation_similarity(
                query,
                r['pronunciation']
            )
            r['pronunciation_score'] = pron_score


            r['combined_score'] = (
                r['similarity'] * (1 - PRONUNCIATION_WEIGHT) +
                pron_score * PRONUNCIATION_WEIGHT
            )

        results.sort(key=lambda x: x['combined_score'], reverse=True)

        return results

    def _calculate_pronunciation_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """발음 유사도 계산"""
        if not text1 or not text2:
            return 0.0

        common = set(text1) & set(text2)
        total = set(text1) | set(text2)

        return len(common) / len(total) if total else 0.0

    def _get_confidence_level(self, score: float) -> str:
        """신뢰도 레벨"""


        if MIN_CONFIDENCE == "high":
            threshold_high = 0.8
            threshold_medium = 0.6
        elif MIN_CONFIDENCE == "medium":
            threshold_high = 0.7
            threshold_medium = 0.5
        else:
            threshold_high = 0.6
            threshold_medium = 0.4

        if score >= threshold_high:
            return "high"
        elif score >= threshold_medium:
            return "medium"
        else:
            return "low"


# ============================================
# 메인 실행
# ============================================

def main():
    """예시 실행"""
    print("="*70)
    print("🎙️  TextProcessing 예시 실행")
    print("="*70 + "\n")

    try:
        with TextProcessing(verbose=True, auto_build=None, monitor_memory=True) as processor:
            print("\n" + "="*70)
            print("예시: 기본 기능 테스트")
            print("="*70 + "\n")

            print("1. 불명확한 발음 정제")
            print("-" * 70)

            test_inputs = ["하ㄴ글", "컴퓨터", "사ㄹ랑"]

            for text in test_inputs:
                cleaned = processor.clean_dysarthric_input(text)
                print(f"입력: '{cleaned.original_text}'")
                print(f"정제: '{cleaned.cleaned_text}'")
                print(f"신뢰도: {cleaned.confidence:.2f}")
                print()

            print("\n2. Dysarthric 검색 파이프라인")
            print("-" * 70 + "\n")

            results = processor.dysarthric_search_pipeline(
                "한글",
                top_k=5,
                use_pronunciation=True,
                rerank=True
            )

            for i, r in enumerate(results, 1):
                print(f"{i}. {r.word} ({r.pos})")
                print(f"   의미: {r.semantic_score:.3f} | "
                      f"발음: {r.pronunciation_score:.3f} | "
                      f"통합: {r.combined_score:.3f}")
                print(f"   신뢰도: {r.confidence}")
                print()

            print("="*70)
            print("✅ 테스트 완료!")
            print("="*70 + "\n")

    except RuntimeError as e:
        error_msg = str(e)

        if "skipped by user" in error_msg:
            print("\n💡 Tip:")
            print("   VectorDB 없이도 기본 기능(발음 정제)은 테스트 가능합니다:")
            print()

        elif "not found" in error_msg or "index" in error_msg:
            print("\n💡 해결방법:")
            print("   1. 자동 구축: processor = TextProcessing(auto_build=True)")
            print("   2. 수동 구축: python -m Data.VectorBuilder")
            print()

        else:
            print(f"\n❌ 에러: {e}\n")

    except Exception as e:
        print(f"\n❌ 예상치 못한 에러: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
