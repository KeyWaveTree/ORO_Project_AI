from typing import List

from VectorDB import ConnectionHandler
from VectorDB.DTO.InsertDTO import InsertResponse, InsertRequest
from VectorDB.DTO.SearchDTO import SearchResponse, SearchRequest
from dataclasses import asdict


class VectorDBClient:
    """Vector DB와 상호작용하기 위한 사용하기 쉬운 클라이언트 (파사드)"""

    def __init__(self, host: str, port: int):
        """
        클라이언트 객체를 생성합니다.
        내부적으로 통신을 담당할 핸들러 객체를 생성하여 소유합니다.
        """
        self._handler = ConnectionHandler(host, port)
        print("🚀 VectorDB 클라이언트가 성공적으로 생성되었습니다!")

    def search(self, query_vector: List[float], top_k: int = 10) -> SearchResponse:
        """
        벡터 검색을 수행합니다. 복잡한 과정은 모두 이 메소드 안에 숨겨져 있습니다.
        """
        # 1. 요청 데이터를 담을 DTO 객체 생성
        request_dto = SearchRequest(queryVector=query_vector, topK=top_k)

        # 2. 내부 핸들러를 통해 서버에 요청 전송 (DTO를 dict로 변환)
        raw_response = self._handler.post("/search", payload=asdict(request_dto))

        # 3. 서버로부터 받은 응답(dict)을 응답 DTO 객체로 변환하여 반환
        return InsertResponse(**raw_response)

    def insert(self, vectors: List[List[float]], ids: List[str]) -> InsertResponse:
        """
        새로운 벡터 데이터를 데이터베이스에 삽입합
        """
        # 1. 요청 DTO 생성
        request_dto = InsertRequest(vectors=vectors, ids=ids)

        # 2. 내부 핸들러를 통해 요청 전송
        raw_response = self._handler.post("/insert", payload=asdict(request_dto))

        # 3. 응답 DTO로 변환하여 반환
        return InsertResponse(**raw_response)
