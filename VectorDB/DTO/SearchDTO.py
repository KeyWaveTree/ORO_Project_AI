"""
VectorDB SDK - Search DTO Module
=================================

검색 관련 데이터 전송 객체

역할:
- 검색 요청 정의
- 검색 응답 정의
- 필터, 정렬 등 검색 옵션
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from .BaseDTO import RequestDTO, ResponseDTO


class MetricType(Enum):
    """유사도 측정 방식"""
    COSINE = "COSINE"
    L2 = "L2"
    IP = "IP"  # Inner Product


class SortOrder(Enum):
    """정렬 순서"""
    ASC = "asc"
    DESC = "desc"


@dataclass
class SearchRequest(RequestDTO):
    """
    벡터 검색 요청

    Attributes:
        collection_name: 컬렉션 이름
        query_vector: 쿼리 벡터
        top_k: 반환할 결과 수
        metric_type: 유사도 측정 방식
        filter_expr: 필터 표현식 (예: "age > 18 AND city == 'Seoul'")
        output_fields: 반환할 필드 목록
        offset: 오프셋 (페이지네이션)
    """

    collection_name: str
    query_vector: List[float]
    top_k: int = 10
    metric_type: MetricType = MetricType.COSINE
    filter_expr: Optional[str] = None
    output_fields: Optional[List[str]] = None
    offset: int = 0

    def validate(self) -> bool:
        """검색 요청 검증"""
        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")

        if not self.query_vector:
            raise ValueError("query_vector cannot be empty")

        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.top_k > 1000:
            raise ValueError("top_k cannot exceed 1000")

        if self.offset < 0:
            raise ValueError("offset must be non-negative")

        return True


@dataclass
class BatchSearchRequest(RequestDTO):
    """
    배치 벡터 검색 요청

    여러 벡터를 한 번에 검색
    """

    collection_name: str
    query_vectors: List[List[float]]
    top_k: int = 10
    metric_type: MetricType = MetricType.COSINE
    filter_expr: Optional[str] = None
    output_fields: Optional[List[str]] = None

    def validate(self) -> bool:
        """배치 검색 요청 검증"""
        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")

        if not self.query_vectors:
            raise ValueError("query_vectors cannot be empty")

        if len(self.query_vectors) > 100:
            raise ValueError("batch size cannot exceed 100")

        # 모든 벡터 차원 확인
        if self.query_vectors:
            first_dim = len(self.query_vectors[0])
            for i, vec in enumerate(self.query_vectors):
                if len(vec) != first_dim:
                    raise ValueError(f"Vector {i} dimension mismatch: {len(vec)} != {first_dim}")

        return True


@dataclass
class SearchResultItem:
    """
    검색 결과 항목

    Attributes:
        id: 벡터 ID
        distance: 유사도 거리
        score: 유사도 점수 (1 - distance for COSINE)
        fields: 반환된 필드 데이터
    """

    id: Any
    distance: float
    score: Optional[float] = None
    fields: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """점수 자동 계산"""
        if self.score is None and self.distance is not None:
            # COSINE의 경우: score = 1 - distance
            self.score = 1.0 - self.distance


@dataclass
class SearchResponse(ResponseDTO):
    """
    검색 응답

    Attributes:
        results: 검색 결과 목록
        total_count: 전체 결과 수
        latency_ms: 검색 소요 시간 (밀리초)
    """

    results: List[SearchResultItem] = field(default_factory=list)
    total_count: int = 0

    def get_top_result(self) -> Optional[SearchResultItem]:
        """최상위 결과 반환"""
        return self.results[0] if self.results else None

    def get_ids(self) -> List[Any]:
        """결과 ID 목록 반환"""
        return [item.id for item in self.results]

    def get_scores(self) -> List[float]:
        """점수 목록 반환"""
        return [item.score for item in self.results if item.score is not None]

    def filter_by_score(self, min_score: float) -> List[SearchResultItem]:
        """최소 점수 이상 결과 필터링"""
        return [
            item for item in self.results
            if item.score is not None and item.score >= min_score
        ]


@dataclass
class BatchSearchResponse(ResponseDTO):
    """
    배치 검색 응답

    여러 쿼리에 대한 결과 목록
    """

    results: List[List[SearchResultItem]] = field(default_factory=list)

    def get_result(self, index: int) -> Optional[List[SearchResultItem]]:
        """특정 인덱스 쿼리 결과 반환"""
        if 0 <= index < len(self.results):
            return self.results[index]
        return None

    def get_all_ids(self) -> List[List[Any]]:
        """모든 쿼리의 ID 목록 반환"""
        return [[item.id for item in result] for result in self.results]


if __name__ == "__main__":
    print("=" * 60)
    print("SearchDTO 테스트")
    print("=" * 60)

    # 검색 요청 생성
    search_req = SearchRequest(
        collection_name="my_collection",
        query_vector=[1.0, 2.0, 3.0],
        top_k=5,
        filter_expr="age > 18"
    )

    print(f"✅ SearchRequest: {search_req.collection_name}, k={search_req.top_k}")

    # 검증
    try:
        search_req.validate()
        print("✅ Validation passed")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")

    # 검색 응답 생성
    results = [
        SearchResultItem(id="doc1", distance=0.1, fields={"title": "Document 1"}),
        SearchResultItem(id="doc2", distance=0.2, fields={"title": "Document 2"}),
        SearchResultItem(id="doc3", distance=0.3, fields={"title": "Document 3"})
    ]

    search_resp = SearchResponse(
        success=True,
        results=results,
        total_count=3,
        latency_ms=15.5
    )

    print(f"\n✅ SearchResponse: {len(search_resp.results)} results")
    print(f"📊 Top result: {search_resp.get_top_result().id}")
    print(f"📊 Scores: {search_resp.get_scores()}")

    # 점수 필터링
    filtered = search_resp.filter_by_score(0.75)
    print(f"📊 Filtered (>0.75): {len(filtered)} results")

    # 배치 검색
    batch_req = BatchSearchRequest(
        collection_name="my_collection",
        query_vectors=[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ],
        top_k=3
    )

    print(f"\n✅ BatchSearchRequest: {len(batch_req.query_vectors)} queries")

    try:
        batch_req.validate()
        print("✅ Batch validation passed")
    except ValueError as e:
        print(f"❌ Batch validation failed: {e}")
