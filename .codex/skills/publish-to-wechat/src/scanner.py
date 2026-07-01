import os
from typing import List

class FileScanner:
    def __init__(self):
        self.supported_extensions = ['.md', '.markdown']

    def scan(self, path: str) -> List[str]:
        """
        Scan for markdown files in the given path.
        If path is a file, return it if valid.
        If path is a directory, return all markdown files recursively.
        """
        files_found = []
        path = os.path.abspath(path)

        if not os.path.exists(path):
            print(f"Error: Path does not exist: {path}")
            return []

        if os.path.isfile(path):
            if self._is_markdown(path):
                files_found.append(path)
            else:
                print(f"Warning: File {path} is not a markdown file.")
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if self._is_markdown(file):
                        files_found.append(os.path.join(root, file))
        
        return files_found

    def _is_markdown(self, filename: str) -> bool:
        return any(filename.lower().endswith(ext) for ext in self.supported_extensions)

if __name__ == "__main__":
    # Test
    scanner = FileScanner()
    # print(scanner.scan("."))
