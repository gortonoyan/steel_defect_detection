#!/usr/bin/env python3
"""
deploy.py  — One-shot deployment script.
Pushes the project to GitHub and uploads model weights to Hugging Face Hub.

Usage:
    python deploy.py

You will be prompted for:
  1. Your GitHub username
  2. Your GitHub Personal Access Token (repo scope)
  3. Your Hugging Face username
  4. Your Hugging Face token (Write role)
"""

import os
import sys
import subprocess
import getpass

GITHUB_REPO_NAME = "severstal-steel-defect-detection"
HF_REPO_NAME     = "severstal-steel-defect-detection"
HF_SPACE_NAME    = "severstal-steel-defect-detection"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=BASE_DIR, **kwargs)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}")
        sys.exit(1)
    return result


def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)


# ── Collect credentials ────────────────────────────────────────────
print("\n🔑  Credential Collection")
print("    (nothing is stored — tokens are used only for this session)\n")

gh_user  = input("GitHub username: ").strip()
gh_token = getpass.getpass("GitHub PAT (repo scope): ").strip()
hf_user  = input("Hugging Face username: ").strip()
hf_token = getpass.getpass("Hugging Face token (Write): ").strip()


# ── Step 1: Create GitHub repo via API ────────────────────────────
step("1/4  Creating GitHub repository (if it doesn't exist)")
import urllib.request, json, urllib.error

api_url = "https://api.github.com/user/repos"
payload = json.dumps({
    "name": GITHUB_REPO_NAME,
    "description": "Production-ready ensemble segmentation for steel defect detection. U-Net + FPN + DeepLabV3+ served via FastAPI.",
    "private": False,
    "auto_init": False,
}).encode()

req = urllib.request.Request(api_url, data=payload, method="POST")
req.add_header("Authorization", f"token {gh_token}")
req.add_header("Content-Type", "application/json")
req.add_header("Accept", "application/vnd.github+json")

try:
    with urllib.request.urlopen(req) as resp:
        repo_data = json.loads(resp.read())
        print(f"✅  Repository created: {repo_data['html_url']}")
except urllib.error.HTTPError as e:
    body = json.loads(e.read())
    if "already exists" in body.get("errors", [{}])[0].get("message", ""):
        print("✅  Repository already exists — continuing.")
    else:
        print(f"GitHub API error: {body}")
        sys.exit(1)


# ── Step 2: Push code to GitHub ───────────────────────────────────
step("2/4  Pushing code to GitHub")

remote_url = f"https://{gh_user}:{gh_token}@github.com/{gh_user}/{GITHUB_REPO_NAME}.git"

# Remove existing remote if any, then re-add with token
subprocess.run(["git", "remote", "remove", "origin"], cwd=BASE_DIR, capture_output=True)
run(["git", "remote", "add", "origin", remote_url])
run(["git", "branch", "-M", "main"])
run(["git", "push", "-u", "origin", "main"])
print(f"✅  Code pushed → https://github.com/{gh_user}/{GITHUB_REPO_NAME}")


# ── Step 3: Upload model weights to HF Hub ────────────────────────
step("3/4  Uploading model weights to Hugging Face Hub")

from huggingface_hub import HfApi, create_repo

api = HfApi()

# Create model repo
try:
    create_repo(
        repo_id=f"{hf_user}/{HF_REPO_NAME}",
        repo_type="model",
        exist_ok=True,
        token=hf_token,
        private=False,
    )
    print(f"✅  HF model repo ready: https://huggingface.co/{hf_user}/{HF_REPO_NAME}")
except Exception as e:
    print(f"HF model repo error: {e}")

# Upload all .pth files
ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
for fname in os.listdir(ckpt_dir):
    if fname.endswith(".pth"):
        local_path = os.path.join(ckpt_dir, fname)
        # Normalise filename (remove spaces/parens)
        clean_name = fname.replace(" ", "_").replace("(", "").replace(")", "").replace("__", "_")
        print(f"  Uploading {fname} → checkpoints/{clean_name}  ({os.path.getsize(local_path)//1_000_000} MB) ...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"checkpoints/{clean_name}",
            repo_id=f"{hf_user}/{HF_REPO_NAME}",
            repo_type="model",
            token=hf_token,
        )
        print(f"  ✅  {clean_name} uploaded.")

print(f"\n✅  All weights uploaded → https://huggingface.co/{hf_user}/{HF_REPO_NAME}")


# ── Step 4: Create HF Space and push code ─────────────────────────
step("4/4  Creating Hugging Face Space and deploying app")

# Update HF_REPO_ID in api/app.py with the correct username
app_py = os.path.join(BASE_DIR, "api", "app.py")
with open(app_py, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(
    'HF_REPO_ID = "narekgabrielyan/severstal-steel-defect-detection"',
    f'HF_REPO_ID = "{hf_user}/{HF_REPO_NAME}"'
)
with open(app_py, "w", encoding="utf-8") as f:
    f.write(content)

# Also patch the README badge links
readme_path = os.path.join(BASE_DIR, "README.md")
with open(readme_path, "r", encoding="utf-8") as f:
    readme = f.read()
readme = readme.replace("narekgabrielyan", gh_user)
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)

# Commit the username patch
run(["git", "add", "api/app.py", "README.md"])
subprocess.run(["git", "commit", "-m", f"chore: set HF repo ID and README links to {gh_user}"], cwd=BASE_DIR)
run(["git", "push", "origin", "main"])

# Create the HF Space
try:
    create_repo(
        repo_id=f"{hf_user}/{HF_SPACE_NAME}",
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        token=hf_token,
        private=False,
    )
    print(f"✅  HF Space created: https://huggingface.co/spaces/{hf_user}/{HF_SPACE_NAME}")
except Exception as e:
    print(f"HF Space creation error: {e}")

# Push code to Space
space_remote = f"https://{hf_user}:{hf_token}@huggingface.co/spaces/{hf_user}/{HF_SPACE_NAME}"
subprocess.run(["git", "remote", "remove", "space"], cwd=BASE_DIR, capture_output=True)
run(["git", "remote", "add", "space", space_remote])
run(["git", "push", "space", "main"])

# ── Done ──────────────────────────────────────────────────────────
print(f"""
{'='*60}
🎉  DEPLOYMENT COMPLETE

  GitHub repo  : https://github.com/{gh_user}/{GITHUB_REPO_NAME}
  HF Models    : https://huggingface.co/{hf_user}/{HF_REPO_NAME}
  Live Demo    : https://huggingface.co/spaces/{hf_user}/{HF_SPACE_NAME}

The Space will take a few minutes to build Docker and start up.
Open the Live Demo URL above to share with anyone in the world!
{'='*60}
""")
