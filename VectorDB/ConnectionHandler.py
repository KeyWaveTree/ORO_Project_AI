from typing import Dict
import requests

class ConnectionHandler:
    """서버와의 저수준(low-level) 통신을 시뮬레이션하는 내부 클래스"""
    def __init__(self, host: str, port: int):
        self._base_url = f"http://{host}:{port}"
        print(f"🔩 내부 핸들러: 서버 주소 {self._base_url} 로 요청을 보낼 준비 완료.")

    def post(self, endpoint: str, payload: Dict) -> Dict:
        full_url = self._base_url + endpoint
        try:
            print(f"  -> 🌐 {full_url} 로 실제 네트워크 POST 요청 전송...")
            # requests.post를 사용해 JSON 페이로드와 함께 실제 HTTP 요청을 보냄
            response = requests.post(full_url, json=payload, timeout=5)  # 5초 타임아웃
            response.raise_for_status()  # 200번대 성공 코드가 아니면 에러 발생

            print("  <- ✅실제 네트워크 응답 수신!")
            return response.json()  # 응답 본문을 JSON(dict)으로 변환하여 반환

        except requests.exceptions.RequestException as e:
            print(f"  <-❌네트워크 오류 발생: {e}")
            return {"error": str(e)}


