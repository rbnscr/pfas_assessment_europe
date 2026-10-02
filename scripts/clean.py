import shutil
from pathlib import (
    Path,
)

RESULTS_DIR = Path("results")
LOGS_DIR = Path("logs")

CLEAR_LIST = [RESULTS_DIR]


def clean_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    for item in directory.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


if __name__ == "__main__":
    for i in CLEAR_LIST:
        clean_directory(i)
        print(f"Cleaned contents of {i}")
