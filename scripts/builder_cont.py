Q = chr(34)
N = chr(10)
BASE = "E:\\Quant\\Qlibworks\\src\\qlworks\\evaluation"
DRAFT = "E:\\Quant\\Qlibworks_draft\\factor_evaluation"

def write_file(name, content):
    path = os.path.join(BASE, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    sz = len(content)
    print(f"  wrote: {name} ({sz} bytes)")

def patch_file(name, old, new):
    path = os.path.join(BASE, name)
    with open(path, "r", encoding="utf-8") as f:
        t = f.read()
    t = t.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(t)
    print(f"  patched: {name}")

print("Creating evaluation module files...")
print("---")