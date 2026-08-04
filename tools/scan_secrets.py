#!/usr/bin/env python
"""仓库 Secret 扫描：敏感变量字面量、set_token、高熵长字符串。

规则按 design：只扫描 Git 受跟踪 .py/.ipynb/.yaml/.yml/.toml/.md；
豁免由解析规则判定（Git SHA、SHA256、URL、绝对路径、.invalid 域名、${SECRET_KEY}），
不允许按整文件跳过。
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys

SENSITIVE_KEYWORDS = {
    "password", "passwd", "pwd", "secret", "token", "api", "apikey",
    "access", "auth", "private", "corp", "agent",
}


def _is_sensitive_var(var: str) -> bool:
    """变量名按 _ 分词，任一 token 命中敏感词即视为敏感变量。"""
    parts = var.lower().split("_")
    return any(p in SENSITIVE_KEYWORDS for p in parts)


ASSIGN_LITERAL = re.compile(r"""(?P<var>\w+)\s*[:=]\s*["'](?P<value>[^"']{4,})["']""")
SET_TOKEN = re.compile(r"""set_token\s*\(\s*["'](?P<value>[^"']{4,})["']""")
SECRET_KEY_REF = re.compile(r"^[A-Z][A-Z0-9_]{3,}$")
# 高熵长字符串：长度>=20 且 Shannon entropy>4.0
HEX_PATTERN = re.compile(r"\b[0-9a-fA-F]{40}\b|\b[0-9a-fA-F]{64}\b")
URL_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^\s]+$")
ABS_PATH_PATTERN = re.compile(r"^([/][^\s]*|([A-Za-z]:[\\/][^\s]*))$")
INVALID_DOMAIN = re.compile(r"\.invalid\b")
ENV_REF = re.compile(r"^\$\{[A-Z_][A-Z0-9_]*\}$")

EXTENSIONS = (".py", ".ipynb", ".yaml", ".yml", ".toml", ".md")

REDACTED_MARKERS = ("[REDACTED]", "[redacted]", "<redacted>", "xxxx", "XXX")


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    from collections import Counter

    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _exempt(value: str) -> bool:
    """解析规则豁免：SHA/URL/路径/.invalid/${REF}。"""
    if value in ("null", "None", "none", "true", "false"):
        return True
    if HEX_PATTERN.fullmatch(value):
        return True
    if URL_PATTERN.match(value):
        return True
    if ABS_PATH_PATTERN.match(value):
        return True
    if INVALID_DOMAIN.search(value):
        return True
    if ENV_REF.match(value):
        return True
    if SECRET_KEY_REF.fullmatch(value):
        return True
    return False


def _scan_line(line: str, path: str) -> list[str]:
    violations: list[str] = []
    # 敏感变量绑定非空字面量
    for match in ASSIGN_LITERAL.finditer(line):
        var = match.group("var")
        value = match.group("value")
        if _is_sensitive_var(var) and value and not _exempt(value):
            violations.append(f"{path}: 敏感变量 {var} 绑定非空字面量")
    # set_token(<literal>)
    for match in SET_TOKEN.finditer(line):
        value = match.group("value")
        if value and not _exempt(value):
            violations.append(f"{path}: set_token 字面量")
    # 只检查看起来像随机 token 的 ASCII 字符串字面量；普通文案不是 Secret。
    for token in re.findall(r'''["']([A-Za-z0-9][A-Za-z0-9_+=-]{19,})["']''', line):
        if _exempt(token):
            continue
        if REDACTED_MARKERS and any(m in token.lower() for m in REDACTED_MARKERS):
            continue
        if len(token) >= 20 and shannon_entropy(token) > 4.0:
            violations.append(f"{path}: 高熵长字符串 {token[:30]}...")
    return violations


def tracked_files(repo_root: str) -> list[str]:
    proc = subprocess.run(["git", "ls-files"], cwd=repo_root, capture_output=True,
                          text=True, encoding="utf-8", check=False)
    return [f for f in proc.stdout.splitlines() if f.endswith(EXTENSIONS)]


def scan_repo(repo_root: str) -> list[str]:
    violations: list[str] = []
    for rel in tracked_files(repo_root):
        full = os.path.join(repo_root, rel)
        try:
            with open(full, "r", encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        for line in content.splitlines():
            violations.extend(_scan_line(line, rel))
    return violations


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    violations = scan_repo(repo_root)
    if violations:
        print("Secret 扫描发现违规：")
        for v in violations:
            print(f"  {v}")
        return 1
    print("Secret 扫描通过：无明文凭证")
    return 0


if __name__ == "__main__":
    sys.exit(main())
