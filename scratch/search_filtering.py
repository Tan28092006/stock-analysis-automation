import os
from pathlib import Path

root = Path("d:/Chungkhoan")
for dirpath, _, filenames in os.walk(root):
    for f in filenames:
        if f.endswith(".py") or f.endswith(".html"):
            p = Path(dirpath) / f
            try:
                content = p.read_text(encoding="utf-8")
                if "to_plain_dict" in content:
                    print("Found to_plain_dict in:", p)
                if "candidates" in content and "filter" in content:
                    # Let's see if there's any candidate filtering
                    if "lambda" in content or "filter(" in content or ".candidates" in content:
                        print("Found candidate references in:", p)
            except Exception:
                pass
