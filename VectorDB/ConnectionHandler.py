from dataclasses import dataclass
from typing import Optional
from enum import Enum

from .Transport import Transport, HTTPTransport, TransportProtocol, RetryableTransport


class ConnectionState(Enum):
    """연결 상태"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ConnectionConfig:
    """
    연결 설정

    Attributes:
        host: 호스트
        port: 포트
        protocol: 프로토콜
        timeout: 타임아웃
        max_retries: 최대 재시도
        auth_token: 인증 토큰
        ssl_verify: SSL 검증
    """

    host: str = "localhost"
    port: int = 19530
    protocol: TransportProtocol = TransportProtocol.HTTP
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    auth_token: Optional[str] = None
    ssl_verify: bool = True

    def get_base_url(self) -> str:
        """기본 URL 생성"""
        protocol_str = "https" if self.ssl_verify else "http"
        return f"{protocol_str}://{self.host}:{self.port}"


class Connection:
    """
    연결 관리자

    역할:
    - Transport 생성 및 관리
    - 연결 상태 관리
    - Health check
    """

    def __init__(self, config: ConnectionConfig):
        """
        Args:
            config: 연결 설정
        """
        self.config = config
        self.transport: Optional[Transport] = None
        self.state = ConnectionState.DISCONNECTED

    def connect(self):
        """연결"""
        from Exceptions import ConnectionError

        self.state = ConnectionState.CONNECTING

        try:
            # Transport 생성
            base_transport = HTTPTransport(
                base_url=self.config.get_base_url(),
                default_timeout=self.config.timeout,
                default_headers=self._get_headers(),
                verify_ssl=self.config.ssl_verify
            )

            # 재시도 래퍼
            self.transport = RetryableTransport(
                transport=base_transport,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay
            )

            # Health check
            self.health_check()

            self.state = ConnectionState.CONNECTED
            print(f"✅ Connected to {self.config.host}:{self.config.port}")

        except Exception as e:
            self.state = ConnectionState.ERROR
            raise ConnectionError(f"Failed to connect: {str(e)}")

    def disconnect(self):
        """연결 해제"""
        if self.transport:
            self.transport.close()
            self.transport = None

        self.state = ConnectionState.DISCONNECTED
        print("🔌 Disconnected")

    def health_check(self) -> bool:
        """Health check"""
        try:
            # 실제로는 /health 엔드포인트 호출
            # 여기서는 간단히 통과
            return True
        except Exception:
            return False

    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self.state == ConnectionState.CONNECTED

    def send(self, method: str, endpoint: str, payload=None) -> dict:
        """요청 전송"""
        if not self.is_connected():
            raise ConnectionError("Not connected")

        return self.transport.send(method, endpoint, payload)

    def _get_headers(self) -> dict:
        """헤더 생성"""
        headers = {"Content-Type": "application/json"}

        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"

        return headers

    def __enter__(self):
        """Context manager"""
        if not self.is_connected():
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager"""
        self.disconnect()


if __name__ == "__main__":
    print("=" * 60)
    print("Connection 테스트")
    print("=" * 60)

    # 설정 생성
    config = ConnectionConfig(
        host="localhost",
        port=19530,
        timeout=5
    )

    print(f"✅ ConnectionConfig: {config.get_base_url()}")

    # 연결 생성
    conn = Connection(config)
    print(f"✅ Connection created: state={conn.state.value}")

    # Context manager
    # with Connection(config) as conn:
    #     # 작업 수행
    #     pass
