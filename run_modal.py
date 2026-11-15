"""
Run a script from this folder on a Modal H200. output/ (rollouts, TensorBoard logs, checkpoints) lives on a Volume.

    uvx modal run run_modal.py --args "zero_shot_eval.py --max_tokens 512"
    uvx modal run run_modal.py --args "train_logged.py --loss_type grpo --max_tokens 256"
    STARTER=../my/starter.py uvx modal run run_modal.py --args "..."    # upload another starter.py in place of this one

Copy the results here to browse them with serve.py:

    uvx modal volume get --force grpo-output rollouts output/
    uvx modal volume get --force grpo-output tb output/
"""
import os
import shlex
import subprocess
from pathlib import Path

import modal

HERE: Path = Path(__file__).parent
STARTER: str | None = os.environ.get("STARTER")
REMOTE: str = "/root/grpo"

image: modal.Image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")  # Triton compiles its launcher with a C compiler
    .pip_install_from_requirements(str(HERE / "requirements.txt"), extra_index_url="https://download.pytorch.org/whl/cu128")
    .add_local_dir(HERE, REMOTE, ignore=["output", ".venv", ".git", "__pycache__", "*.pdf", *(["starter.py"] if STARTER else [])])
)
if STARTER:
    image = image.add_local_file(STARTER, f"{REMOTE}/starter.py")

output_volume: modal.Volume = modal.Volume.from_name("grpo-output", create_if_missing=True)
hf_volume: modal.Volume = modal.Volume.from_name("grpo-hf-cache", create_if_missing=True)
app: modal.App = modal.App("grpo-tutorial", image=image)


@app.function(gpu="H200", timeout=6 * 3600,
              volumes={f"{REMOTE}/output": output_volume, "/root/.cache/huggingface": hf_volume})
def run(args: str) -> int:
    returncode: int = subprocess.run(["python", "-u", *shlex.split(args)], cwd=REMOTE).returncode
    output_volume.commit()
    hf_volume.commit()
    return returncode


@app.local_entrypoint()
def main(args: str) -> None:
    returncode: int = run.remote(args)
    if returncode != 0:
        raise SystemExit(returncode)
