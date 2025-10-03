from dataclasses import dataclass
from typing import List

@dataclass
class InsertRequest:
    '''
    데이터 삽입 요청에 필요한 데이터를 담는 클래스

    vectors: (List[List[float]]) 한번에 요청으로 여러개의 벡터 데이터를 동시에 삽입하기 위한 2차원 리스트
    성능과 효율성때문에 일일이 요청 들어올때마다 처리할 수 없으니 한번의 요청으로 요청한 데이터를 처리

    ids: List[str] 요청받은 음성 리스트 벡터의 id
    '''
    vectors: List[List[float]]
    ids:List[str]

@dataclass
class InsertResponse:
    '''
    데이터 삽입 결과를 담는 클래스

    success:bool 삽입 성공 여부를 저장하는 변수
    insertedCount 요청된 데이터중 삽입한 데이터의 개수
    '''
    success:bool
    insertedCount:int