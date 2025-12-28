"""
VectorDB SDK - Insert DTO Module
=================================

데이터 삽입 관련 DTO

역할:
- 벡터 삽입 요청
- 배치 삽입 요청
- 삽입 응답
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .BaseDTO import RequestDTO, ResponseDTO


@dataclass
class InsertRequest(RequestDTO):
    """
    벡터 삽입 요청

    Attributes:
        collection_name: 컬렉션 이름
        vectors: 삽입할 벡터 목록
        ids: 벡터 ID 목록 (선택, None이면 자동 생성)
        fields: 추가 필드 데이터 목록
    """

    collection_name: str
    vectors: List[List[float]]
    ids: Optional[List[str]] = None
    fields: Optional[List[Dict[str, Any]]] = None

    def validate(self) -> bool:
        """삽입 요청 검증"""
        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")

        if not self.vectors:
            raise ValueError("vectors cannot be empty")

        # 벡터 차원 일관성 확인
        if self.vectors:
            first_dim = len(self.vectors[0])
            for i, vec in enumerate(self.vectors):
                if len(vec) != first_dim:
                    raise ValueError(f"Vector {i} dimension mismatch: {len(vec)} != {first_dim}")

        # ID 개수 확인
        if self.ids is not None and len(self.ids) != len(self.vectors):
            raise ValueError(f"ids count ({len(self.ids)}) != vectors count ({len(self.vectors)})")

        # 필드 개수 확인
        if self.fields is not None and len(self.fields) != len(self.vectors):
            raise ValueError(f"fields count ({len(self.fields)}) != vectors count ({len(self.vectors)})")

        return True

    def get_vector_count(self) -> int:
        """삽입할 벡터 개수 반환"""
        return len(self.vectors)

    def get_dimension(self) -> int:
        """벡터 차원 반환"""
        return len(self.vectors[0]) if self.vectors else 0


@dataclass
class UpsertRequest(InsertRequest):
    """
    벡터 Upsert 요청 (Insert or Update)

    이미 존재하는 ID는 업데이트, 없으면 삽입
    """
    pass


@dataclass
class InsertResponse(ResponseDTO):
    """
    삽입 응답

    Attributes:
        inserted_ids: 삽입된 벡터 ID 목록
        inserted_count: 삽입된 벡터 수
    """

    inserted_ids: List[Any] = field(default_factory=list)
    inserted_count: int = 0

    def __post_init__(self):
        """삽입 개수 자동 계산"""
        if self.inserted_count == 0:
            self.inserted_count = len(self.inserted_ids)


@dataclass
class UpdateRequest(RequestDTO):
    """
    벡터 업데이트 요청

    Attributes:
        collection_name: 컬렉션 이름
        id: 업데이트할 벡터 ID
        vector: 새로운 벡터 (선택)
        fields: 업데이트할 필드 (선택)
    """

    collection_name: str
    id: str
    vector: Optional[List[float]] = None
    fields: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        """업데이트 요청 검증"""
        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")

        if not self.id:
            raise ValueError("id cannot be empty")

        if self.vector is None and self.fields is None:
            raise ValueError("Either vector or fields must be provided")

        return True


@dataclass
class UpdateResponse(ResponseDTO):
    """
    업데이트 응답

    Attributes:
        updated_id: 업데이트된 벡터 ID
        updated: 업데이트 성공 여부
    """

    updated_id: Optional[str] = None
    updated: bool = False


@dataclass
class DeleteRequest(RequestDTO):
    """
    벡터 삭제 요청

    Attributes:
        collection_name: 컬렉션 이름
        ids: 삭제할 벡터 ID 목록 (선택)
        filter_expr: 필터 표현식 (선택)
    """

    collection_name: str
    ids: Optional[List[str]] = None
    filter_expr: Optional[str] = None

    def validate(self) -> bool:
        """삭제 요청 검증"""
        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")

        if self.ids is None and self.filter_expr is None:
            raise ValueError("Either ids or filter_expr must be provided")

        return True


@dataclass
class DeleteResponse(ResponseDTO):
    """
    삭제 응답

    Attributes:
        deleted_count: 삭제된 벡터 수
        deleted_ids: 삭제된 벡터 ID 목록 (선택)
    """

    deleted_count: int = 0
    deleted_ids: Optional[List[str]] = None


if __name__ == "__main__":
    print("=" * 60)
    print("InsertDTO 테스트")
    print("=" * 60)

    # 삽입 요청 생성
    insert_req = InsertRequest(
        collection_name="my_collection",
        vectors=[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ],
        ids=["vec1", "vec2"],
        fields=[
            {"title": "Document 1", "age": 25},
            {"title": "Document 2", "age": 30}
        ]
    )

    print(f"✅ InsertRequest: {insert_req.get_vector_count()} vectors, dim={insert_req.get_dimension()}")

    # 검증
    try:
        insert_req.validate()
        print("✅ Validation passed")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")

    # 삽입 응답
    insert_resp = InsertResponse(
        success=True,
        inserted_ids=["vec1", "vec2"],
        latency_ms=12.3
    )

    print(f"\n✅ InsertResponse: {insert_resp.inserted_count} inserted")

    # 업데이트 요청
    update_req = UpdateRequest(
        collection_name="my_collection",
        id="vec1",
        fields={"age": 26}
    )

    print(f"\n✅ UpdateRequest: id={update_req.id}")

    try:
        update_req.validate()
        print("✅ Update validation passed")
    except ValueError as e:
        print(f"❌ Update validation failed: {e}")

    # 삭제 요청
    delete_req = DeleteRequest(
        collection_name="my_collection",
        ids=["vec1", "vec2"]
    )

    print(f"\n✅ DeleteRequest: {len(delete_req.ids)} ids to delete")

    try:
        delete_req.validate()
        print("✅ Delete validation passed")
    except ValueError as e:
        print(f"❌ Delete validation failed: {e}")
