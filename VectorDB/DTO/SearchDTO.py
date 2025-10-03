from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class SearchResultItem:
    """개별 검색 결과를 나타내는 DTO.

    Attributes:
        id (Any): 검색된 데이터의 고유 ID.
        distance (float): 쿼리 벡터와의 거리. 값이 작을수록 유사도가 높다는 의미
    """
    id: Any
    distance: float

@dataclass
class SearchRequest:
    '''
    검색 요청에 필요한 데이터를 담는 클래스

    queryVector: (List[float]) 요청할려는 음성 데이터
    (음성을 스펙트로그램이나 MFCC같은 형태로 변환하여 임베딩 벡터를 생성한 것을 전달하면 된다.)
    topK : int 요청한 검색결과중 가장 유사한 k개의 결과만 가져오도록 지정하는 변수
    벡터DB는 search 요청을 받으면 모든 벡터 사이의 유사도를 계산하고 계산된 유사도를 기준으로
    모든 결과를 정렬하여 상위 N개만큼의 결과만 return 해준다. 성능적으로 수백만개의 검색 결과를 모두 반환하는 것은 매우 느리고
    네트워크 자원을 엄청나게 낭비하기 때문
    '''
    queryVector:List[float]
    topK:int

@dataclass
class SearchResponse:
    '''
    검색 결과 응답을 나타내는 DTO.

    results: (List[SearchResultItem]): 검색 결과의 목록. 각 항목은 SearchResultItem 객체

    latencyMs:float 검색 요청을 보낸 후 응답받기까지 걸린 시간을 저장하기 위한 변수
    모듈의 성능을 측정하고 모니터링할 때 중요하기에 추가함.
    '''
    results: List[SearchResultItem]
    latencyMs: float


