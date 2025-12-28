"""
VectorDB SDK - Transport Module
================================

통신 계층 구현

역할:
- HTTP/gRPC 등 프로토콜 지원
- 요청/응답 전송
- 재시도 로직
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum
import time

from VectorDB import VectorDBException


class TransportProtocol(Enum):
    """지원하는 통신 프로토콜"""
    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"


class Transport(ABC):
    """
    통신 인터페이스

    모든 Transport 구현체의 부모 클래스
    """

    @abstractmethod
    def send(
            self,
            method: str,
            endpoint: str,
            payload: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        요청 전송

        Args:
            method: HTTP 메서드 (GET, POST, etc.)
            endpoint: API 엔드포인트
            payload: 요청 데이터
            headers: 요청 헤더
            timeout: 타임아웃 (초)

        Returns:
            응답 데이터
        """
        pass

    @abstractmethod
    def close(self):
        """연결 종료"""
        pass


class HTTPTransport(Transport):
    """
    HTTP/HTTPS 통신 구현

    requests 라이브러리 사용
    """

    def __init__(
            self,
            base_url: str,
            default_timeout: int = 30,
            default_headers: Optional[Dict[str, str]] = None,
            verify_ssl: bool = True
    ):
        """
        Args:
            base_url: 기본 URL (예: http://localhost:19530)
            default_timeout: 기본 타임아웃
            default_headers: 기본 헤더
            verify_ssl: SSL 인증서 검증 여부
        """
        self.base_url = base_url.rstrip('/')
        self.default_timeout = default_timeout
        self.default_headers = default_headers or {"Content-Type": "application/json"}
        self.verify_ssl = verify_ssl
        self.session = None

    def _get_session(self):
        """세션 가져오기 (지연 초기화)"""
        if self.session is None:
            import requests
            self.session = requests.Session()
            self.session.headers.update(self.default_headers)
        return self.session

    def send(
            self,
            method: str,
            endpoint: str,
            payload: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """HTTP 요청 전송"""
        import requests
        from Exceptions import (
            ConnectionTimeoutError,
            ConnectionRefusedError,
            ServerError,
            create_exception_from_response
        )

        session = self._get_session()
        url = f"{self.base_url}{endpoint}"
        timeout = timeout or self.default_timeout

        # 헤더 병합
        request_headers = self.default_headers.copy()
        if headers:
            request_headers.update(headers)

        try:
            # 요청 전송
            response = session.request(
                method=method.upper(),
                url=url,
                json=payload,
                headers=request_headers,
                timeout=timeout,
                verify=self.verify_ssl
            )

            # 성공 응답
            if response.status_code < 400:
                return response.json() if response.text else {}

            # 에러 응답
            try:
                error_data = response.json()
            except:
                error_data = {"message": response.text or "Unknown error"}

            raise create_exception_from_response(response.status_code, error_data)

        except requests.exceptions.Timeout:
            raise ConnectionTimeoutError(f"Request timeout after {timeout}s")

        except requests.exceptions.ConnectionError as e:
            raise ConnectionRefusedError(f"Connection failed: {str(e)}")

        except Exception as e:
            if isinstance(e, VectorDBException):
                raise
            raise ServerError(f"Unexpected error: {str(e)}")

    def close(self):
        """세션 종료"""
        if self.session:
            self.session.close()
            self.session = None


class RetryableTransport(Transport):
    """
    재시도 로직을 가진 Transport 래퍼

    기존 Transport를 래핑하여 재시도 기능 추가
    """

    def __init__(
            self,
            transport: Transport,
            max_retries: int = 3,
            retry_delay: float = 1.0,
            backoff_factor: float = 2.0
    ):
        """
        Args:
            transport: 래핑할 Transport
            max_retries: 최대 재시도 횟수
            retry_delay: 초기 재시도 대기 시간 (초)
            backoff_factor: 지수 백오프 팩터
        """
        self.transport = transport
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

    def send(
            self,
            method: str,
            endpoint: str,
            payload: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """재시도 로직과 함께 요청 전송"""
        from Exceptions import (
            ConnectionTimeoutError,
            ConnectionRefusedError,
            ServerError
        )

        last_exception = None
        delay = self.retry_delay

        for attempt in range(self.max_retries + 1):
            try:
                return self.transport.send(method, endpoint, payload, headers, timeout)

            except (ConnectionTimeoutError, ConnectionRefusedError, ServerError) as e:
                last_exception = e

                if attempt < self.max_retries:
                    print(f"⚠️  Attempt {attempt + 1} failed, retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= self.backoff_factor
                else:
                    raise

            except Exception as e:
                # 재시도 불가능한 예외는 즉시 발생
                raise

        # 모든 재시도 실패
        if last_exception:
            raise last_exception

    def close(self):
        """내부 Transport 종료"""
        self.transport.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Transport 테스트")
    print("=" * 60)

    # HTTP Transport 생성
    transport = HTTPTransport(
        base_url="http://localhost:19530",
        default_timeout=5
    )

    print(f"✅ HTTPTransport created: {transport.base_url}")

    # 재시도 Transport
    retryable = RetryableTransport(
        transport=transport,
        max_retries=3,
        retry_delay=1.0
    )

    print(f"✅ RetryableTransport: max_retries={retryable.max_retries}")

    # 실제 요청은 서버가 없어서 테스트 불가
    # transport.send("POST", "/search", {"query": "test"})
