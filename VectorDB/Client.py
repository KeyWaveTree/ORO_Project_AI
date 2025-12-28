"""
VectorDB SDK - Client Module
=============================

클라이언트 파사드

역할:
- 사용자 친화적 API 제공
- DTO 변환 자동화
- 비즈니스 로직
"""

from typing import List
from VectorDB.DTO.CollectionDTO import CreateCollectionResponse, CreateCollectionRequest, DropCollectionResponse, \
    DropCollectionRequest, DescribeCollectionResponse, DescribeCollectionRequest, ListCollectionsResponse, \
    ListCollectionsRequest, IndexParams, CreateIndexResponse, CreateIndexRequest
from VectorDB.DTO.InsertDTO import DeleteResponse, DeleteRequest
from VectorDB.DTO.SearchDTO import BatchSearchResponse, BatchSearchRequest
from .ConnectionHandler import Connection, ConnectionConfig
from .DTO import *
from .Exceptions import *


class VectorDBClient:
    """
    VectorDB 클라이언트 (파사드)

    사용자가 사용하는 메인 클래스

    예제:
        >>> client = VectorDBClient("localhost", 19530)
        >>> result = client.search("my_collection", [1.0, 2.0, 3.0], top_k=5)
    """

    def __init__(
            self,
            host: str = "localhost",
            port: int = 19530,
            timeout: int = 30,
            max_retries: int = 3,
            auth_token: Optional[str] = None,
            auto_connect: bool = True
    ):
        """
        Args:
            host: 호스트
            port: 포트
            timeout: 타임아웃 (초)
            max_retries: 최대 재시도 횟수
            auth_token: 인증 토큰
            auto_connect: 자동 연결 여부
        """
        # 설정 생성
        self.config = ConnectionConfig(
            host=host,
            port=port,
            timeout=timeout,
            max_retries=max_retries,
            auth_token=auth_token
        )

        # 연결 생성
        self.connection = Connection(self.config)

        if auto_connect:
            self.connection.connect()

    # ========== 검색 API ==========

    def search(
            self,
            collection_name: str,
            query_vector: List[float],
            top_k: int = 10,
            metric_type: MetricType = MetricType.COSINE,
            filter_expr: Optional[str] = None,
            output_fields: Optional[List[str]] = None
    ) -> SearchResponse:
        """
        벡터 검색

        Args:
            collection_name: 컬렉션 이름
            query_vector: 쿼리 벡터
            top_k: 반환할 결과 수
            metric_type: 유사도 측정 방식
            filter_expr: 필터 표현식
            output_fields: 반환할 필드

        Returns:
            SearchResponse

        예제:
            >>> result = client.search("docs", [1.0, 2.0], top_k=5)
            >>> print(f"Found {len(result.results)} results")
        """
        # Request DTO 생성
        request = SearchRequest(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            metric_type=metric_type,
            filter_expr=filter_expr,
            output_fields=output_fields
        )

        # 검증
        request.validate()

        # 전송
        response_data = self.connection.send(
            method="POST",
            endpoint="/v1/search",
            payload=request.to_dict()
        )

        # Response DTO 변환
        return SearchResponse.from_dict(response_data)

    def batch_search(
            self,
            collection_name: str,
            query_vectors: List[List[float]],
            top_k: int = 10,
            metric_type: MetricType = MetricType.COSINE
    ) -> BatchSearchResponse:
        """배치 검색"""
        request = BatchSearchRequest(
            collection_name=collection_name,
            query_vectors=query_vectors,
            top_k=top_k,
            metric_type=metric_type
        )

        request.validate()

        response_data = self.connection.send(
            method="POST",
            endpoint="/v1/batch_search",
            payload=request.to_dict()
        )

        return BatchSearchResponse.from_dict(response_data)

    # ========== 삽입 API ==========

    def insert(
            self,
            collection_name: str,
            vectors: List[List[float]],
            ids: Optional[List[str]] = None,
            fields: Optional[List[Dict[str, Any]]] = None
    ) -> InsertResponse:
        """
        벡터 삽입

        Args:
            collection_name: 컬렉션 이름
            vectors: 벡터 목록
            ids: ID 목록 (선택)
            fields: 필드 데이터 (선택)

        Returns:
            InsertResponse

        예제:
            >>> vectors = [[1.0, 2.0], [3.0, 4.0]]
            >>> result = client.insert("docs", vectors)
            >>> print(f"Inserted {result.inserted_count} vectors")
        """
        request = InsertRequest(
            collection_name=collection_name,
            vectors=vectors,
            ids=ids,
            fields=fields
        )

        request.validate()

        response_data = self.connection.send(
            method="POST",
            endpoint="/v1/insert",
            payload=request.to_dict()
        )

        return InsertResponse.from_dict(response_data)

    def upsert(
            self,
            collection_name: str,
            vectors: List[List[float]],
            ids: List[str],
            fields: Optional[List[Dict[str, Any]]] = None
    ) -> InsertResponse:
        """Upsert (Insert or Update)"""
        from VectorDB.DTO.InsertDTO import UpsertRequest
        request = UpsertRequest(
            collection_name=collection_name,
            vectors=vectors,
            ids=ids,
            fields=fields
        )

        request.validate()

        response_data = self.connection.send(
            method="POST",
            endpoint="/v1/upsert",
            payload=request.to_dict()
        )

        return InsertResponse.from_dict(response_data)

    def delete(
            self,
            collection_name: str,
            ids: Optional[List[str]] = None,
            filter_expr: Optional[str] = None
    ) -> DeleteResponse:
        """
        벡터 삭제

        Args:
            collection_name: 컬렉션 이름
            ids: 삭제할 ID 목록
            filter_expr: 필터 표현식

        Returns:
            DeleteResponse
        """
        request = DeleteRequest(
            collection_name=collection_name,
            ids=ids,
            filter_expr=filter_expr
        )

        request.validate()

        response_data = self.connection.send(
            method="POST",
            endpoint="/v1/delete",
            payload=request.to_dict()
        )

        return DeleteResponse.from_dict(response_data)

    # ========== 컬렉션 API ==========

    def create_collection(
            self,
            collection_name: str,
            fields: List[FieldSchema],
            description: Optional[str] = None
    ) -> CreateCollectionResponse:
        """
        컬렉션 생성

        Args:
            collection_name: 컬렉션 이름
            fields: 필드 스키마 목록
            description: 설명

        Returns:
            CreateCollectionResponse

        예제:
            >>> fields = [
            ...     FieldSchema("id", FieldType.INT64, is_primary=True),
            ...     FieldSchema("embedding", FieldType.FLOAT_VECTOR, dimension=768)
            ... ]
            >>> client.create_collection("docs", fields)
        """
        request = CreateCollectionRequest(
            collection_name=collection_name,
            fields=fields,
            description=description
        )

        request.validate()

        response_data = self.connection.send(
            method="POST",
            endpoint="/v1/collection/create",
            payload=request.to_dict()
        )

        return CreateCollectionResponse.from_dict(response_data)

    def drop_collection(self, collection_name: str) -> DropCollectionResponse:
        """컬렉션 삭제"""
        request = DropCollectionRequest(collection_name=collection_name)
        request.validate()

        response_data = self.connection.send(
            method="POST",
            endpoint="/v1/collection/drop",
            payload=request.to_dict()
        )

        return DropCollectionResponse.from_dict(response_data)

    def describe_collection(self, collection_name: str) -> DescribeCollectionResponse:
        """컬렉션 정보 조회"""
        request = DescribeCollectionRequest(collection_name=collection_name)

        response_data = self.connection.send(
            method="GET",
            endpoint=f"/v1/collection/{collection_name}",
            payload=request.to_dict()
        )

        return DescribeCollectionResponse.from_dict(response_data)

    def list_collections(self) -> ListCollectionsResponse:
        """컬렉션 목록 조회"""
        request = ListCollectionsRequest()

        response_data = self.connection.send(
            method="GET",
            endpoint="/v1/collections",
            payload=request.to_dict()
        )

        return ListCollectionsResponse.from_dict(response_data)

    def create_index(
            self,
            collection_name: str,
            field_name: str,
            index_params: IndexParams
    ) -> CreateIndexResponse:
        """
        인덱스 생성

        Args:
            collection_name: 컬렉션 이름
            field_name: 필드 이름
            index_params: 인덱스 파라미터

        Returns:
            CreateIndexResponse
        """
        request = CreateIndexRequest(
            collection_name=collection_name,
            field_name=field_name,
            index_params=index_params
        )

        request.validate()

        response_data = self.connection.send(
            method="POST",
            endpoint="/v1/index/create",
            payload=request.to_dict()
        )

        return CreateIndexResponse.from_dict(response_data)

    # ========== 연결 관리 ==========

    def close(self):
        """연결 종료"""
        self.connection.disconnect()

    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self.connection.is_connected()

    def __enter__(self):
        """Context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager"""
        self.close()


if __name__ == "__main__":
    print("=" * 60)
    print("VectorDBClient 테스트")
    print("=" * 60)

    # 클라이언트 생성
    client = VectorDBClient(
        host="localhost",
        port=19530,
        auto_connect=False  # 실제 서버 없음
    )

    print(f"✅ Client created: {client.config.host}:{client.config.port}")

    # Context manager
    # with VectorDBClient("localhost", 19530) as client:
    #     result = client.search("docs", [1.0, 2.0], top_k=5)
