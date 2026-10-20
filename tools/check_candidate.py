"""CPU-only syntax, sensitive-pattern and packaging checks. Not a legal audit."""
import ast
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
patterns = {
    "private_key": r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY",
    "github_token": r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}",
    "credential_url": r"https?://[^/\s:]+:[^/\s@]+@",
    "local_absolute_path": r"(?<![A-Za-z])[A-Za-z]:[\\/]|/home/|/Users/",
}
findings = []
files = [p for folder in ("src", "examples", "tests") for p in (root / folder).rglob("*.py")]
for path in files:
    content = path.read_text(encoding="utf-8")
    ast.parse(content, filename=str(path))
    for category, pattern in patterns.items():
        for match in re.finditer(pattern, content):
            findings.append((str(path.relative_to(root)), content[:match.start()].count("\n") + 1, category))
print(f"Parsed {len(files)} Python files.")
print("Sensitive-pattern findings (file, line, category only):", findings)
print("No automated scan can guarantee absence of sensitive information.")
if findings:
    raise SystemExit(1)
