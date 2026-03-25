import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qlworks.config import CSV_DIR  # noqa: E402
from qlworks.duckdb_adapter import DuckDBOHLCV  # noqa: E402


def main() -> None:
    adapter = DuckDBOHLCV()
    out = adapter.export_per_symbol_csv(CSV_DIR)
    print(str(out))


if __name__ == "__main__":
    main()
