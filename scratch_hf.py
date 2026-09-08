import subprocess, os
from huggingface_hub import create_repo, HfApi

HF_TOKEN  = os.environ.get("HF_TOKEN", "")   # set via environment variable
HF_USER   = "gortonoyan"
SPACE     = "steel-defect-detection"
BASE      = os.path.dirname(os.path.abspath(__file__))

def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
    if r.stdout.strip(): print(r.stdout.strip())
    if r.stderr.strip(): print(r.stderr.strip())
    return r.returncode == 0

# 1. Create Gradio Space (free)
print("[1/3] Creating free Gradio Space...")
create_repo(
    repo_id=f"{HF_USER}/{SPACE}",
    repo_type="space",
    space_sdk="gradio",   # <-- free tier!
    exist_ok=True,
    token=HF_TOKEN,
    private=False,
)
print(f"  Space ready -> https://huggingface.co/spaces/{HF_USER}/{SPACE}")

# 2. Upload app.py and requirements_space.txt as requirements.txt
print("\n[2/3] Uploading files to Space...")
api = HfApi()
api.upload_file(
    path_or_fileobj=os.path.join(BASE, "app.py"),
    path_in_repo="app.py",
    repo_id=f"{HF_USER}/{SPACE}",
    repo_type="space",
    token=HF_TOKEN,
)
api.upload_file(
    path_or_fileobj=os.path.join(BASE, "requirements_space.txt"),
    path_in_repo="requirements.txt",
    repo_id=f"{HF_USER}/{SPACE}",
    repo_type="space",
    token=HF_TOKEN,
)
# Upload entire src/ package
import os as _os
src_dir = _os.path.join(BASE, "src")
for fname in _os.listdir(src_dir):
    fpath = _os.path.join(src_dir, fname)
    if _os.path.isfile(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f"src/{fname}",
            repo_id=f"{HF_USER}/{SPACE}",
            repo_type="space",
            token=HF_TOKEN,
        )
        print(f"  Uploaded src/{fname}")

print("\n[3/3] Done! Space is building...")
print(f"""
============================================================
LIVE DEMO (free, public):
  https://huggingface.co/spaces/{HF_USER}/{SPACE}

It will take ~5 minutes to install dependencies and start.
============================================================
""")
