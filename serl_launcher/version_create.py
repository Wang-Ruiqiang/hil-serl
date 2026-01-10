import re, sys, subprocess

req_in = "requirements.txt"
# 读入 requirements.txt，提取“包名”（忽略注释、空行、复杂 spec）
lines = []
pkgs = []
for raw in open(req_in, "r", encoding="utf-8"):
    s = raw.strip()
    if not s or s.startswith("#"):
        continue
    # 跳过 -r / -c / --find-links / git+ 等复杂行
    if s.startswith(("-", "--")) or "git+" in s or "@" in s:
        continue
    # 只取行首包名（直到遇到比较符号或空格）
    m = re.match(r"^([A-Za-z0-9_.-]+)", s)
    if m:
        pkgs.append(m.group(1))

# 查询已安装版本
out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
installed = {}
for l in out.splitlines():
    if "==" in l and not l.startswith("#"):
        name, ver = l.split("==", 1)
        installed[name.lower()] = ver

missing = []
result = []
for p in pkgs:
    ver = installed.get(p.lower())
    if ver is None:
        missing.append(p)
        continue
    result.append(f"{p}>={ver}")

# 输出
print("\n".join(result))
if missing:
    print("\n# ---- not found in environment (check manually) ----")
    for m in missing:
        print(f"# {m}")
