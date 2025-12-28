import sys
import os
import argparse
import asyncio
import traceback

# 현재 위치를 참조 경로에 추가 (Data 패키지 인식용)
sys.path.append(os.getcwd())


# Data/__init__.py 덕분에 임포트가 깔끔해짐
from Data import (
    VectorDBBuilder,
    DictionarySearch,
    COLLECTION_NAME,
    get_safe_config
    )



# ==========================================
# 관리 기능 구현
# ==========================================

def action_status():
    """서버 및 DB 상태 점검"""
    print(f"\n📊 [Server Status] '{COLLECTION_NAME}' 상태 점검")
    print("=" * 60)

    try:
        # 설정 정보 출력
        config = get_safe_config()
        print(f"⚙️  설정 프로파일: {config['performance']['description']}")
        print(f"💾 할당 메모리: ~{config['performance']['max_memory_gb']}GB")

        # DB 연결 및 정보 조회
        builder = VectorDBBuilder()
        builder.connect()  # 연결 테스트

        info = builder.get_collection_info()
        print(f"\n📡 Milvus 연결: 성공")
        print(f"   - 컬렉션: {info.get('collection_name')}")
        print(f"   - 데이터: {info.get('total_entries', 0):,} Rows")
        print(f"   - 차원수: {info.get('vector_dim')} dim")
        print(f"   - 인덱스: {info.get('index_type')}")

        print("\n✅ 상태: 정상 (Ready)")

    except Exception as e:
        print(f"\n❌ 상태: 비정상 (Error)")
        print(f"   이유: {e}")


def action_fix_index():
    """인덱스 누락 시 복구 (동기 실행)"""
    print(f"\n🔧 [Fix Index] 인덱스 복구 및 로드 작업을 시작합니다...")
    print("=" * 60)

    try:
        builder = VectorDBBuilder()
        builder.connect()

        # VectorBuilder.py의 create_index는 내부적으로 load_collection까지 수행함
        builder.create_index()

        print("\n✅ 인덱스 생성 완료.")
        print("✅ 컬렉션 메모리 로드 완료.")
        print("▶️  이제 검색이 가능합니다.")

    except Exception as e:
        print(f"\n❌ 작업 실패: {e}")
        traceback.print_exc()


def action_build_db():
    """DB 전체 구축 (비동기 실행)"""
    print(f"\n🏗️  [Build DB] 벡터 DB 전체 구축을 시작합니다...")
    print("⚠️  주의: 기존 데이터가 있다면 삭제되고 재생성됩니다.")
    print("=" * 60)

    confirm = input("정말로 진행하시겠습니까? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("🚫 취소되었습니다.")
        return

    try:
        builder = VectorDBBuilder()
        # build()는 async 함수이므로 asyncio로 실행
        asyncio.run(builder.build())
        print("\n✅ DB 구축이 완료되었습니다.")

    except Exception as e:
        print(f"\n❌ 구축 실패: {e}")
        traceback.print_exc()


def action_search_test(query):
    """검색 테스트"""
    print(f"\n🔎 [Test Search] 검색어: '{query}'")
    print("=" * 60)

    try:
        # 컨텍스트 매니저 사용 (자동 메모리 정리)
        with DictionarySearch(verbose=False) as engine:
            results = engine.search(query, top_k=3)

            if not results:
                print("   결과 없음")

            for i, res in enumerate(results, 1):
                print(f"   {i}. [{res['word']}] 유사도: {res['similarity']:.4f}")
                print(f"      뜻: {res['definition'][:60]}...")

    except Exception as e:
        print(f"\n❌ 검색 에러: {e}")
        print("💡 팁: 'python manage_server.py fix-index'를 먼저 실행해보세요.")


# ==========================================
# 메인 실행기 (CLI 파서)
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Vector DB Server Management Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="관리 명령어")

    # 1. status 명령
    subparsers.add_parser("status", help="현재 DB 및 서버 설정 상태 확인")

    # 2. fix-index 명령
    subparsers.add_parser("fix-index", help="인덱스 생성 및 컬렉션 로드 (복구용)")

    # 3. build 명령
    subparsers.add_parser("build", help="DB 전체 새로 구축 (주의: 기존 데이터 삭제)")

    # 4. test 명령
    test_parser = subparsers.add_parser("test", help="검색 기능 테스트")
    test_parser.add_argument("-q", "--query", type=str, default="한글", help="테스트할 검색어")

    args = parser.parse_args()

    # 명령어 분기
    if args.command == "status":
        action_status()
    elif args.command == "fix-index":
        action_fix_index()
    elif args.command == "build":
        action_build_db()
    elif args.command == "test":
        action_search_test(args.query)
    else:
        parser.print_help()