import os
import subprocess

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def check_repo_clean():
    # Ensure working tree is clean for reproducibility
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT_DIR,
    ).stdout.strip()
    if git_status:
        msg = (
            "Uncommitted changes detected. Commit or stash them before running "
            "experiments to ensure reproducibility.\n"
            f"Dirty files:\n{git_status}"
        )
        raise RuntimeError(msg)


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
