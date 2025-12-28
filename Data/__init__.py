# 1. 설정 정보 노출
from .Config import (
    COLLECTION_NAME,
    VECTOR_MODEL,
    MEMORY_PROFILE,
    get_safe_config
)

# 2. 핵심 클래스 노출 (Facade 패턴 효과)
# 이제 외부에서는 'from Data import VectorDBBuilder' 로 바로 접근 가능
from .VectorBuilder import VectorDBBuilder
from .SearchEngine import DictionarySearch, search_korean_dictionary
from .TextProcessing import TextProcessing

# 3. 모듈 레벨에서 공개할 목록 정의
__all__ = [
    "VectorDBBuilder",
    "DictionarySearch",
    "TextProcessing",
    "search_korean_dictionary",
    "COLLECTION_NAME",
    "VECTOR_MODEL",
    "MEMORY_PROFILE",
    "get_safe_config"
]