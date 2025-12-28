"""
VectorDB SDK
============

클라이언트 개발자를 위한 Vector DB 연결 모듈

역할:
- 쉽고 빠른 Vector DB 연결
- 유연한 API 제공
- 자동 DTO 변환
- 에러 처리

사용 예제:
    >> from vectordb_sdk import VectorDBClient
    >>
    >> client = VectorDBClient("localhost", 19530)
    >> result = client.search("docs", [1.0, 2.0, 3.0], top_k=5)
    >> print(f"Found {len(result.results)} results")
"""

__version__ = "1.0.0"
__author__ = "VectorDB SDK"

# 메인 클라이언트
from .Client import VectorDBClient

# DTOs
from .DTO import *

# Exceptions
from .Exceptions import *

# Connection
from .ConnectionHandler import ConnectionConfig, ConnectionState

# Transport
from .Transport import TransportProtocol

__all__ = [
    # 메인 클라이언트
    "VectorDBClient",

    # 설정
    "ConnectionConfig",
    "ConnectionState",
    "TransportProtocol",

    # DTOs - Search
    "MetricType",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "BatchSearchRequest",
    "BatchSearchResponse",

    # DTOs - Insert
    "InsertRequest",
    "InsertResponse",
    "UpsertRequest",
    "UpdateRequest",
    "UpdateResponse",
    "DeleteRequest",
    "DeleteResponse",

    # DTOs - Collection
    "FieldType",
    "IndexType",
    "FieldSchema",
    "IndexParams",
    "CreateCollectionRequest",
    "CreateCollectionResponse",
    "DescribeCollectionResponse",
    "ListCollectionsResponse",
    "CreateIndexRequest",
    "CreateIndexResponse",

    # Exceptions
    "VectorDBException",
    "ConnectionError",
    "ConnectionTimeoutError",
    "AuthenticationError",
    "InvalidRequestError",
    "ValidationError",
    "CollectionNotFoundError",
    "CollectionAlreadyExistsError",
    "ServerError",
]
