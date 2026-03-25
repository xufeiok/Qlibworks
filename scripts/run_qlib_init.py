import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qlworks.config import QLIB_DATA_DIR  # noqa: E402


def main() -> int:
    try:
        if str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))
        import qlib  # type: ignore
    except Exception as e:
        print("qlib not available")
        print(str(e))
        return 1
    if not QLIB_DATA_DIR.exists():
        print(str(QLIB_DATA_DIR))
        return 2
    qlib.init(provider_uri=str(QLIB_DATA_DIR))
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
