"""
VectorDB SDK - Collection DTO Module
=====================================

컬렉션(테이블) 관리 DTO

역할:
- 컬렉션 생성/삭제
- 컬렉션 정보 조회
- 스키마 정의
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from .base_dto import RequestDTO, ResponseDTO


class FieldType(Enum):
    """필드 데이터 타입"""
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    STRING = "STRING"
    VARCHAR = "VARCHAR"
    BOOL = "BOOL"
    ARRAY = "ARRAY"
    JSON = "JSON"
    FLOAT_VECTOR = "FLOAT_VECTOR"
    BINARY_VECTOR = "BINARY_VECTOR"


class IndexType(Enum):
    """인덱스 타입"""
    FLAT = "FLAT"
    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    IVF_PQ = "IVF_PQ"
    HNSW = "HNSW"
    ANNOY = "ANNOY"


@dataclass
class FieldSchema:
    """
    필드 스키마

    Attributes:
        name: 필드 이름
        field_type: 필드 타입
        is_primary: 기본 키 여부
        auto_id: 자동 ID 생성 여부
        max_length: 최대 길이 (VARCHAR)
        dimension: 벡터 차원 (VECTOR)
        description: 설명
    """

    name: str
    field_type: FieldType
    is_primary: bool = False
    auto_id: bool = False
    max_length: Optional[int] = None
    dimension: Optional[int] = None
    description: Optional[str] = None

    def validate(self) -> bool:
        """필드 스키마 검증"""
        if not self.name:
            raise ValueError("Field name cannot be empty")

        if self.field_type == FieldType.VARCHAR and self.max_length is None:
            raise ValueError("VARCHAR field must have max_length")

        if self.field_type == FieldType.FLOAT_VECTOR and self.dimension is None:
            raise ValueError("FLOAT_VECTOR field must have dimension")

        if self.dimension is not None and self.dimension <= 0:
            raise ValueError("dimension must be positive")

        return True


@dataclass
class IndexParams:
    """
    인덱스 파라미터

    Attributes:
        index_type: 인덱스 타입
        metric_type: 유사도 측정 방식
        params: 추가 파라미터 (예: {"M": 16, "efConstruction": 200})
    """

    index_type: IndexType
    metric_type: str = "COSINE"
    params: Optional[Dict[str, Any]] = None

    def get_param(self, key: str, default: Any = None) -> Any:
        """파라미터 조회"""
        if self.params is None:
            return default
        return self.params.get(key, default)


@dataclass
class CreateCollectionRequest(RequestDTO):
    """
    컬렉션 생성 요청

    Attributes:
        collection_name: 컬렉션 이름
        fields: 필드 스키마 목록
        description: 컬렉션 설명
        auto_id: 자동 ID 생성
    """

    collection_name: str
    fields: List[FieldSchema]
    description: Optional[str] = None
    auto_id: bool = True

    def validate(self) -> bool:
        """컬렉션 생성 요청 검증"""
        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")

        if not self.fields:
            raise ValueError("fields cannot be empty")

        # 모든 필드 검증
        for field_schema in self.fields:
            field_schema.validate()

        # 기본 키 확인
        primary_fields = [f for f in self.fields if f.is_primary]
        if not primary_fields:
            raise ValueError("At least one primary field is required")

        if len(primary_fields) > 1:
            raise ValueError("Only one primary field is allowed")

        # 벡터 필드 확인
        vector_fields = [
            f for f in self.fields
            if f.field_type in [FieldType.FLOAT_VECTOR, FieldType.BINARY_VECTOR]
        ]
        if not vector_fields:
            raise ValueError("At least one vector field is required")

        return True

    def get_primary_field(self) -> Optional[FieldSchema]:
        """기본 키 필드 반환"""
        for field in self.fields:
            if field.is_primary:
                return field
        return None

    def get_vector_fields(self) -> List[FieldSchema]:
        """벡터 필드 목록 반환"""
        return [
            f for f in self.fields
            if f.field_type in [FieldType.FLOAT_VECTOR, FieldType.BINARY_VECTOR]
        ]


@dataclass
class CreateCollectionResponse(ResponseDTO):
    """컬렉션 생성 응답"""

    collection_name: Optional[str] = None


@dataclass
class DropCollectionRequest(RequestDTO):
    """컬렉션 삭제 요청"""

    collection_name: str

    def validate(self) -> bool:
        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")
        return True


@dataclass
class DropCollectionResponse(ResponseDTO):
    """컬렉션 삭제 응답"""
    pass


@dataclass
class DescribeCollectionRequest(RequestDTO):
    """컬렉션 정보 조회 요청"""

    collection_name: str


@dataclass
class CollectionInfo:
    """
    컬렉션 정보

    Attributes:
        name: 컬렉션 이름
        description: 설명
        fields: 필드 스키마 목록
        num_entities: 엔티티 수
        created_timestamp: 생성 시간
    """

    name: str
    description: Optional[str] = None
    fields: List[FieldSchema] = field(default_factory=list)
    num_entities: int = 0
    created_timestamp: Optional[int] = None


@dataclass
class DescribeCollectionResponse(ResponseDTO):
    """컬렉션 정보 조회 응답"""

    collection_info: Optional[CollectionInfo] = None


@dataclass
class ListCollectionsRequest(RequestDTO):
    """컬렉션 목록 조회 요청"""
    pass


@dataclass
class ListCollectionsResponse(ResponseDTO):
    """컬렉션 목록 조회 응답"""

    collections: List[str] = field(default_factory=list)

    def get_count(self) -> int:
        """컬렉션 개수 반환"""
        return len(self.collections)


@dataclass
class CreateIndexRequest(RequestDTO):
    """
    인덱스 생성 요청

    Attributes:
        collection_name: 컬렉션 이름
        field_name: 필드 이름
        index_params: 인덱스 파라미터
    """

    collection_name: str
    field_name: str
    index_params: IndexParams

    def validate(self) -> bool:
        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")
        if not self.field_name:
            raise ValueError("field_name cannot be empty")
        return True


@dataclass
class CreateIndexResponse(ResponseDTO):
    """인덱스 생성 응답"""
    pass


if __name__ == "__main__":
    print("=" * 60)
    print("CollectionDTO 테스트")
    print("=" * 60)

    # 필드 스키마 정의
    fields = [
        FieldSchema(
            name="id",
            field_type=FieldType.INT64,
            is_primary=True,
            auto_id=True
        ),
        FieldSchema(
            name="title",
            field_type=FieldType.VARCHAR,
            max_length=500
        ),
        FieldSchema(
            name="embedding",
            field_type=FieldType.FLOAT_VECTOR,
            dimension=768
        )
    ]

    # 컬렉션 생성 요청
    create_req = CreateCollectionRequest(
        collection_name="my_collection",
        fields=fields,
        description="Test collection"
    )

    print(f"✅ CreateCollectionRequest: {create_req.collection_name}")
    print(f"📊 Fields: {len(create_req.fields)}")
    print(f"🔑 Primary field: {create_req.get_primary_field().name}")
    print(f"📐 Vector fields: {[f.name for f in create_req.get_vector_fields()]}")

    # 검증
    try:
        create_req.validate()
        print("✅ Validation passed")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")

    # 인덱스 파라미터
    index_params = IndexParams(
        index_type=IndexType.HNSW,
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )

    print(f"\n✅ IndexParams: {index_params.index_type.value}")
    print(f"📊 M: {index_params.get_param('M')}")

    # 인덱스 생성 요청
    index_req = CreateIndexRequest(
        collection_name="my_collection",
        field_name="embedding",
        index_params=index_params
    )

    print(f"\n✅ CreateIndexRequest: {index_req.field_name}")

    try:
        index_req.validate()
        print("✅ Index validation passed")
    except ValueError as e:
        print(f"❌ Index validation failed: {e}")
