"""
Data/config.py

보안 강화된 통합 설정 관리 모듈

보안 원칙:
1. 민감 정보는 절대 하드코딩하지 않음
2. 환경 변수를 런타임에만 읽음
3. 출력 시 민감 정보 자동 마스킹
4. 로그에 민감 정보 노출 방지
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import dotenv

# .env 파일 로드 (런타임에만)
env_path = Path(__file__).parent.parent / ".env"
dotenv.load_dotenv(dotenv_path=env_path)


# ============================================
# 민감 정보 정의
# ============================================

SENSITIVE_KEYS = {
    'MILVUS_PASSWORD',
    'MILVUS_USER',
    'MILVUS_TOKEN',
    'API_KEY',
    'SECRET_KEY',
    'AWS_SECRET_ACCESS_KEY',
    'OPENAI_API_KEY'
}


# ============================================
# 안전한 환경 변수 읽기
# ============================================

def get_env(key: str, default: Any = None, sensitive: bool = False) -> Any:
    """
    환경 변수 안전하게 읽기

    Args:
        key: 환경 변수 키
        default: 기본값
        sensitive: 민감 정보 여부

    Returns:
        환경 변수 값 (없으면 기본값)
    """
    value = os.getenv(key, default)

    # 민감 정보는 로그에 기록하지 않음
    if not sensitive and value != default:
        pass  # 일반 정보만 로그 가능

    return value


def mask_sensitive(value: str, show_chars: int = 3) -> str:
    """
    민감 정보 마스킹

    Args:
        value: 마스킹할 값
        show_chars: 표시할 문자 수

    Returns:
        마스킹된 문자열

    Example:
        >>> mask_sensitive("my_secret_password")
        "my_***"
    """
    if not value or len(value) <= show_chars:
        return "***"
    return value[:show_chars] + "***"


# ============================================
# Milvus 연결 설정
# ============================================

MILVUS_HOST = get_env("MILVUS_HOST", "localhost")
MILVUS_PORT = int(get_env("MILVUS_PORT", "19530"))
MILVUS_USER = get_env("MILVUS_USER", None, sensitive=True)
MILVUS_PASSWORD = get_env("MILVUS_PASSWORD", None, sensitive=True)
MILVUS_TOKEN = get_env("MILVUS_TOKEN", None, sensitive=True)
MILVUS_DB_NAME = get_env("MILVUS_DB_NAME", "default")

# URI 생성 (비밀번호 제외)
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"


# ============================================
# VectorDB 컬렉션 설정
# ============================================

COLLECTION_NAME = get_env("COLLECTION_NAME", "korean_dictionary")
VECTOR_MODEL = get_env("VECTOR_MODEL", "jhgan/ko-sbert-nli")


# ============================================
# 데이터 경로 설정
# ============================================

DATA_DIR = get_env("DATA_DIR", "../")


# ============================================
# 메모리 프로파일 설정
# ============================================

# 메모리 프로파일 선택
# - conservative: 최소 메모리 (느림, 안전)
# - balanced: 균형 (기본)
# - performance: 빠름 (메모리 많이 사용)
MEMORY_PROFILE = get_env("MEMORY_PROFILE", "conservative")

def get_memory_settings():
    """
    메모리 프로파일에 따른 설정

    Returns:
        {
            'process_count': 프로세스 수,
            'batch_size': 배치 크기,
            'max_memory_gb': 최대 메모리 (GB)
        }
    """
    import multiprocessing as mp
    cpu_count = mp.cpu_count()

    profiles = {
        'minimal': {
            'process_count': 1,
            'batch_size': 200,
            'max_memory_gb': 2,
            'description': '최소 메모리 (16GB 시스템용, 매우 느림)'
        },
        'ultra_conservative': {
            'process_count': 2,
            'batch_size': 500,
            'max_memory_gb': 4,
            'description': '극도로 낮은 메모리 (32GB 시스템용, 느림)'
        },
        'conservative': {
            'process_count': min(4, cpu_count),
            'batch_size': 1000,
            'max_memory_gb': 8,
            'description': '최소 메모리 사용 (64GB 시스템용)'
        },
        'balanced': {
            'process_count': min(6, max(4, cpu_count // 2)),
            'batch_size': 2000,
            'max_memory_gb': 16,
            'description': '균형 잡힌 설정 (128GB+ 시스템용)'
        },
        'performance': {
            'process_count': max(8, cpu_count - 2),
            'batch_size': 5000,
            'max_memory_gb': 32,
            'description': '빠른 처리 (256GB+ 시스템용)'
        }
    }

    return profiles.get(MEMORY_PROFILE, profiles['ultra_conservative'])


# 메모리 설정 적용
_MEMORY_SETTINGS = get_memory_settings()

def get_process_count() -> int:
    """
    프로세스 수 결정

    우선순위:
    1. PROCESS_COUNT 환경 변수 (수동 설정)
    2. 메모리 프로파일 설정
    """
    configured = int(get_env("PROCESS_COUNT", "0"))

    if configured > 0:
        return configured
    else:
        return _MEMORY_SETTINGS['process_count']


PROCESS_COUNT = get_process_count()
BULK_INSERT_SIZE = int(get_env("BULK_INSERT_SIZE", str(_MEMORY_SETTINGS['batch_size'])))
MAX_MEMORY_GB = _MEMORY_SETTINGS['max_memory_gb']


# ============================================
# 성능 최적화 설정
# ============================================


# ============================================
# Milvus 인덱스 설정
# ============================================

INDEX_TYPE = get_env("INDEX_TYPE", "HNSW")
METRIC_TYPE = get_env("METRIC_TYPE", "COSINE")

# HNSW 파라미터
HNSW_M = int(get_env("HNSW_M", "16"))
HNSW_EF_CONSTRUCTION = int(get_env("HNSW_EF_CONSTRUCTION", "200"))


# ============================================
# 검색 설정
# ============================================

SEARCH_EF = int(get_env("SEARCH_EF", "100"))
DEFAULT_TOP_K = int(get_env("DEFAULT_TOP_K", "10"))


# ============================================
# Dysarthric Speech 처리 설정
# ============================================

PRONUNCIATION_WEIGHT = float(get_env("PRONUNCIATION_WEIGHT", "0.3"))
MAX_SIMILAR_WORDS = int(get_env("MAX_SIMILAR_WORDS", "3"))
MIN_CONFIDENCE = get_env("MIN_CONFIDENCE", "low")


# ============================================
# 로깅 설정
# ============================================

LOG_LEVEL = get_env("LOG_LEVEL", "INFO")
VERBOSE = get_env("VERBOSE", "True").lower() in ("true", "1", "yes")


# ============================================
# 개발/테스트 설정
# ============================================

DEV_MODE = get_env("DEV_MODE", "False").lower() in ("true", "1", "yes")
TEST_DATA_LIMIT = int(get_env("TEST_DATA_LIMIT", "0"))


# ============================================
# 설정 검증
# ============================================

def validate_config():
    """설정값 검증 (값은 노출하지 않음)"""
    errors = []

    # Milvus 포트 검증
    if not (1 <= MILVUS_PORT <= 65535):
        errors.append(f"Invalid MILVUS_PORT: must be 1-65535")

    # 프로세스 수 검증
    if not (1 <= PROCESS_COUNT <= 128):
        errors.append(f"Invalid PROCESS_COUNT: must be 1-128")

    # Bulk 크기 검증
    if not (100 <= BULK_INSERT_SIZE <= 50000):
        errors.append(f"Invalid BULK_INSERT_SIZE: must be 100-50000")

    # HNSW 파라미터 검증
    if not (4 <= HNSW_M <= 64):
        errors.append(f"Invalid HNSW_M: must be 4-64")

    if not (8 <= HNSW_EF_CONSTRUCTION <= 512):
        errors.append(f"Invalid HNSW_EF_CONSTRUCTION: must be 8-512")

    # 발음 가중치 검증
    if not (0.0 <= PRONUNCIATION_WEIGHT <= 1.0):
        errors.append(f"Invalid PRONUNCIATION_WEIGHT: must be 0.0-1.0")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))


# ============================================
# 안전한 설정 정보 출력
# ============================================

def get_safe_config() -> Dict[str, Any]:
    """
    민감 정보가 마스킹된 설정 딕셔너리

    Returns:
        마스킹된 설정 정보
    """
    return {
        'milvus': {
            'host': MILVUS_HOST,
            'port': MILVUS_PORT,
            'user': mask_sensitive(MILVUS_USER) if MILVUS_USER else None,
            'password': '***' if MILVUS_PASSWORD else None,
            'token': '***' if MILVUS_TOKEN else None,
            'database': MILVUS_DB_NAME,
            'uri': MILVUS_URI
        },
        'vectordb': {
            'collection': COLLECTION_NAME,
            'model': VECTOR_MODEL
        },
        'data': {
            'directory': DATA_DIR
        },
        'performance': {
            'processes': PROCESS_COUNT,
            'batch_size': BULK_INSERT_SIZE
        },
        'index': {
            'type': INDEX_TYPE,
            'metric': METRIC_TYPE,
            'hnsw_m': HNSW_M,
            'hnsw_ef_construction': HNSW_EF_CONSTRUCTION
        },
        'search': {
            'ef': SEARCH_EF,
            'default_top_k': DEFAULT_TOP_K
        },
        'dysarthric': {
            'pronunciation_weight': PRONUNCIATION_WEIGHT,
            'max_similar_words': MAX_SIMILAR_WORDS,
            'min_confidence': MIN_CONFIDENCE
        }
    }


def print_config(verbose: bool = True, show_sensitive: bool = False):
    """
    설정 정보 출력 (민감 정보 자동 마스킹)

    Args:
        verbose: 출력 여부
        show_sensitive: 민감 정보 표시 여부 (기본 False)
    """
    if not verbose:
        return

    config = get_safe_config()

    print("\n" + "="*70)
    print("⚙️  설정 정보 (민감 정보 마스킹됨)")
    print("="*70)

    print(f"\n📡 Milvus:")
    print(f"   URI: {config['milvus']['uri']}")
    print(f"   Database: {config['milvus']['database']}")
    if config['milvus']['user']:
        print(f"   User: {config['milvus']['user']}")
    if config['milvus']['password']:
        print(f"   Password: {config['milvus']['password']}")

    print(f"\n📚 VectorDB:")
    print(f"   Collection: {config['vectordb']['collection']}")
    print(f"   Model: {config['vectordb']['model']}")

    print(f"\n📁 Data:")
    print(f"   Directory: {config['data']['directory']}")

    print(f"\n💾 메모리 프로파일: {MEMORY_PROFILE.upper()}")
    print(f"   설명: {_MEMORY_SETTINGS['description']}")
    print(f"   최대 메모리: ~{_MEMORY_SETTINGS['max_memory_gb']}GB")
    print(f"   프로세스: {config['performance']['processes']}개")
    print(f"   배치 크기: {config['performance']['batch_size']:,}개")

    print(f"\n🔍 인덱스:")
    print(f"   Type: {config['index']['type']}")
    print(f"   Metric: {config['index']['metric']}")
    print(f"   M: {config['index']['hnsw_m']}")
    print(f"   efConstruction: {config['index']['hnsw_ef_construction']}")

    print(f"\n🔎 검색:")
    print(f"   ef: {config['search']['ef']}")
    print(f"   Default Top-K: {config['search']['default_top_k']}")

    print(f"\n🎙️  Dysarthric:")
    print(f"   Pronunciation Weight: {config['dysarthric']['pronunciation_weight']}")
    print(f"   Max Similar Words: {config['dysarthric']['max_similar_words']}")
    print(f"   Min Confidence: {config['dysarthric']['min_confidence']}")

    print("\n🔒 보안:")
    print("   민감 정보는 자동으로 마스킹됩니다")
    print("   .env 파일은 절대 git에 커밋하지 마세요")

    print("="*70 + "\n")


# ============================================
# Milvus 연결 설정 가져오기 (인증 포함)
# ============================================

def get_milvus_connection_params() -> Dict[str, Any]:
    """
    Milvus 연결 파라미터 (인증 정보 포함)

    이 함수는 런타임에만 호출되며, 반환값은 절대 로그에 기록하지 않음

    Returns:
        Milvus 연결 파라미터
    """
    params = {
        'uri': MILVUS_URI
    }

    # 인증 정보 추가 (있는 경우만)
    if MILVUS_USER:
        params['user'] = MILVUS_USER
    if MILVUS_PASSWORD:
        params['password'] = MILVUS_PASSWORD
    if MILVUS_TOKEN:
        params['token'] = MILVUS_TOKEN
    if MILVUS_DB_NAME != 'default':
        params['db_name'] = MILVUS_DB_NAME

    return params


# 초기화 시 검증
validate_config()


if __name__ == "__main__":
    """설정 확인용 (민감 정보 마스킹됨)"""
    print_config(verbose=True, show_sensitive=False)

    # 안전한 설정 딕셔너리 출력
    import json
    print("\n" + "="*70)
    print("📋 설정 딕셔너리 (JSON)")
    print("="*70)
    print(json.dumps(get_safe_config(), indent=2, ensure_ascii=False))
    print()

