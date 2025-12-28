"""
VectorDB SDK - DTO Base Module
================================

모든 데이터 전송 객체(DTO)의 기본 클래스

역할:
- DTO 공통 기능 제공
- 직렬화/역직렬화
- 검증 인터페이스
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Type, TypeVar
from abc import ABC, abstractmethod
import json

T = TypeVar('T', bound='BaseDTO')


@dataclass
class BaseDTO(ABC):
    """
    모든 DTO의 기본 클래스

    기능:
    - 딕셔너리 변환 (to_dict)
    - JSON 변환 (to_json)
    - 역직렬화 (from_dict, from_json)
    - 검증 (validate)
    """

    def to_dict(self) -> Dict[str, Any]:
        """DTO를 딕셔너리로 변환"""
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        """DTO를 JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """딕셔너리에서 DTO 생성"""
        return cls(**data)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """JSON 문자열에서 DTO 생성"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate(self) -> bool:
        """
        DTO 유효성 검증

        서브클래스에서 오버라이드하여 구현

        Returns:
            bool: 유효하면 True

        Raises:
            ValueError: 유효하지 않으면 예외
        """
        return True

    def __repr__(self) -> str:
        """읽기 쉬운 문자열 표현"""
        return f"{self.__class__.__name__}({self.to_dict()})"


@dataclass
class RequestDTO(BaseDTO):
    """요청 DTO 기본 클래스"""

    request_id: Optional[str] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        """요청 ID 자동 생성"""
        if self.request_id is None:
            import uuid
            self.request_id = str(uuid.uuid4())

        if self.timestamp is None:
            import time
            self.timestamp = time.time()


@dataclass
class ResponseDTO(BaseDTO):
    """응답 DTO 기본 클래스"""

    success: bool
    request_id: Optional[str] = None
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None

    def is_success(self) -> bool:
        """성공 여부 확인"""
        return self.success and self.error_message is None

    def get_error(self) -> Optional[str]:
        """에러 메시지 반환"""
        return self.error_message


@dataclass
class PaginatedRequestDTO(RequestDTO):
    """페이지네이션 지원 요청 DTO"""

    page: int = 1
    page_size: int = 10

    def validate(self) -> bool:
        """페이지네이션 파라미터 검증"""
        if self.page < 1:
            raise ValueError("page must be >= 1")
        if self.page_size < 1 or self.page_size > 1000:
            raise ValueError("page_size must be between 1 and 1000")
        return True

    def get_offset(self) -> int:
        """오프셋 계산"""
        return (self.page - 1) * self.page_size


@dataclass
class PaginatedResponseDTO(ResponseDTO):
    """페이지네이션 지원 응답 DTO"""

    total_count: int = 0
    page: int = 1
    page_size: int = 10

    def get_total_pages(self) -> int:
        """전체 페이지 수 계산"""
        return (self.total_count + self.page_size - 1) // self.page_size

    def has_next_page(self) -> bool:
        """다음 페이지 존재 여부"""
        return self.page < self.get_total_pages()

    def has_prev_page(self) -> bool:
        """이전 페이지 존재 여부"""
        return self.page > 1


if __name__ == "__main__":
    # 테스트
    print("=" * 60)
    print("BaseDTO 테스트")
    print("=" * 60)


    @dataclass
    class TestRequest(RequestDTO):
        query: str
        limit: int = 10

        def validate(self) -> bool:
            if not self.query:
                raise ValueError("query cannot be empty")
            if self.limit <= 0:
                raise ValueError("limit must be positive")
            return True


    # 생성
    req = TestRequest(query="test query", limit=5)
    print(f"✅ Request created: {req}")

    # 검증
    try:
        req.validate()
        print("✅ Validation passed")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")

    # 직렬화
    print(f"\n📤 to_dict(): {req.to_dict()}")
    print(f"📤 to_json(): {req.to_json(indent=2)}")

    # 역직렬화
    json_str = req.to_json()
    restored = TestRequest.from_json(json_str)
    print(f"\n📥 from_json(): {restored}")


    # 페이지네이션
    @dataclass
    class TestPaginatedRequest(PaginatedRequestDTO):
        query: str = ""


    page_req = TestPaginatedRequest(query="test", page=2, page_size=20)
    print(f"\n📄 Paginated request: page={page_req.page}, offset={page_req.get_offset()}")
