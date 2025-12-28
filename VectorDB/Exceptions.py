from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(Enum):
    """에러 코드"""

    # 연결 에러 (1xxx)
    CONNECTION_ERROR = 1000
    CONNECTION_TIMEOUT = 1001
    CONNECTION_REFUSED = 1002
    CONNECTION_CLOSED = 1003

    # 인증 에러 (2xxx)
    AUTHENTICATION_ERROR = 2000
    INVALID_TOKEN = 2001
    TOKEN_EXPIRED = 2002
    PERMISSION_DENIED = 2003

    # 요청 에러 (3xxx)
    INVALID_REQUEST = 3000
    INVALID_PARAMETER = 3001
    MISSING_PARAMETER = 3002
    VALIDATION_ERROR = 3003

    # 리소스 에러 (4xxx)
    COLLECTION_NOT_FOUND = 4000
    COLLECTION_ALREADY_EXISTS = 4001
    INDEX_NOT_FOUND = 4002
    VECTOR_NOT_FOUND = 4003

    # 서버 에러 (5xxx)
    SERVER_ERROR = 5000
    SERVICE_UNAVAILABLE = 5001
    TIMEOUT = 5002

    # 알 수 없는 에러 (9xxx)
    UNKNOWN_ERROR = 9000


class VectorDBException(Exception):
    """
    VectorDB SDK 기본 예외

    모든 SDK 예외의 부모 클래스
    """

    def __init__(
            self,
            message: str,
            error_code: Optional[ErrorCode] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or ErrorCode.UNKNOWN_ERROR
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.error_code.name}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """예외를 딕셔너리로 변환"""
        return {
            "error_code": self.error_code.value,
            "error_name": self.error_code.name,
            "message": self.message,
            "details": self.details
        }


# ============================================
# 연결 관련 예외
# ============================================

class ConnectionError(VectorDBException):
    """연결 에러"""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.CONNECTION_ERROR, details)


class ConnectionTimeoutError(VectorDBException):
    """연결 타임아웃"""

    def __init__(self, message: str = "Connection timeout", details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.CONNECTION_TIMEOUT, details)


class ConnectionRefusedError(VectorDBException):
    """연결 거부"""

    def __init__(self, message: str = "Connection refused", details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.CONNECTION_REFUSED, details)


class ConnectionClosedError(VectorDBException):
    """연결 종료"""

    def __init__(self, message: str = "Connection closed", details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.CONNECTION_CLOSED, details)


# ============================================
# 인증 관련 예외
# ============================================

class AuthenticationError(VectorDBException):
    """인증 에러"""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.AUTHENTICATION_ERROR, details)


class InvalidTokenError(VectorDBException):
    """유효하지 않은 토큰"""

    def __init__(self, message: str = "Invalid authentication token", details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.INVALID_TOKEN, details)


class PermissionDeniedError(VectorDBException):
    """권한 거부"""

    def __init__(self, message: str = "Permission denied", details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.PERMISSION_DENIED, details)


# ============================================
# 요청 관련 예외
# ============================================

class InvalidRequestError(VectorDBException):
    """유효하지 않은 요청"""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.INVALID_REQUEST, details)


class InvalidParameterError(VectorDBException):
    """유효하지 않은 파라미터"""

    def __init__(self, parameter_name: str, message: str, details: Optional[Dict] = None):
        full_message = f"Invalid parameter '{parameter_name}': {message}"
        super().__init__(full_message, ErrorCode.INVALID_PARAMETER, details)


class ValidationError(VectorDBException):
    """검증 에러"""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.VALIDATION_ERROR, details)


# ============================================
# 리소스 관련 예외
# ============================================

class CollectionNotFoundError(VectorDBException):
    """컬렉션을 찾을 수 없음"""

    def __init__(self, collection_name: str, details: Optional[Dict] = None):
        message = f"Collection '{collection_name}' not found"
        super().__init__(message, ErrorCode.COLLECTION_NOT_FOUND, details)


class CollectionAlreadyExistsError(VectorDBException):
    """컬렉션이 이미 존재"""

    def __init__(self, collection_name: str, details: Optional[Dict] = None):
        message = f"Collection '{collection_name}' already exists"
        super().__init__(message, ErrorCode.COLLECTION_ALREADY_EXISTS, details)


class VectorNotFoundError(VectorDBException):
    """벡터를 찾을 수 없음"""

    def __init__(self, vector_id: Any, details: Optional[Dict] = None):
        message = f"Vector '{vector_id}' not found"
        super().__init__(message, ErrorCode.VECTOR_NOT_FOUND, details)


# ============================================
# 서버 관련 예외
# ============================================

class ServerError(VectorDBException):
    """서버 에러"""

    def __init__(self, message: str = "Internal server error", details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.SERVER_ERROR, details)


class ServiceUnavailableError(VectorDBException):
    """서비스 사용 불가"""

    def __init__(self, message: str = "Service unavailable", details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.SERVICE_UNAVAILABLE, details)


class TimeoutError(VectorDBException):
    """타임아웃"""

    def __init__(self, message: str = "Request timeout", details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.TIMEOUT, details)


# ============================================
# 유틸리티 함수
# ============================================

def create_exception_from_response(
        status_code: int,
        response_data: Dict[str, Any]
) -> VectorDBException:
    """
    HTTP 응답에서 예외 생성

    Args:
        status_code: HTTP 상태 코드
        response_data: 응답 데이터

    Returns:
        적절한 예외 객체
    """
    message = response_data.get("message", "Unknown error")
    error_code = response_data.get("error_code")
    details = response_data.get("details", {})

    # 상태 코드에 따라 예외 선택
    if status_code == 400:
        return InvalidRequestError(message, details)
    elif status_code == 401:
        return AuthenticationError(message, details)
    elif status_code == 403:
        return PermissionDeniedError(message, details)
    elif status_code == 404:
        return CollectionNotFoundError(message, details)
    elif status_code == 408:
        return TimeoutError(message, details)
    elif status_code == 409:
        return CollectionAlreadyExistsError(message, details)
    elif status_code == 500:
        return ServerError(message, details)
    elif status_code == 503:
        return ServiceUnavailableError(message, details)
    else:
        return VectorDBException(message, ErrorCode.UNKNOWN_ERROR, details)


if __name__ == "__main__":
    print("=" * 60)
    print("Exceptions 테스트")
    print("=" * 60)

    # 연결 에러
    try:
        raise ConnectionTimeoutError(details={"host": "localhost", "port": 19530})
    except VectorDBException as e:
        print(f"❌ {e}")
        print(f"📊 Details: {e.to_dict()}")

    # 컬렉션 에러
    try:
        raise CollectionNotFoundError("my_collection")
    except VectorDBException as e:
        print(f"\n❌ {e}")
        print(f"📊 Code: {e.error_code.value}")

    # 파라미터 에러
    try:
        raise InvalidParameterError("top_k", "must be positive")
    except VectorDBException as e:
        print(f"\n❌ {e}")

    # HTTP 응답에서 예외 생성
    response_data = {
        "message": "Collection not found",
        "error_code": 4000,
        "details": {"collection_name": "test"}
    }

    exception = create_exception_from_response(404, response_data)
    print(f"\n❌ Created from response: {exception}")
