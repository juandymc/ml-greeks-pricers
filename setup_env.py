#!/usr/bin/env python3
"""env_panel.py
A minimal interactive panel to manage a Python virtual environment (KISS-style).

Features
--------
1. Create a new environment at any path (deletes an existing one after confirmation).
2. Add libraries (from a *requirements.txt* file **or** a comma-separated list).
3. Remove libraries.

All prompts are in English and actions are logged to the console.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def sh(cmd: str) -> None:
    """Run *cmd* in the shell, printing it first."""
    print(f"[RUN] {cmd}")
    subprocess.check_call(cmd, shell=True)


def bin_path(env: Path, exe: str) -> Path:
    """Return the path to *exe* inside *env* (cross-platform)."""
    return env / ("Scripts" if os.name == "nt" else "bin") / exe


# ──────────────────────────────────────────────────────────────────────────
# Core actions
# ──────────────────────────────────────────────────────────────────────────

def create_env() -> None:
    env_path = Path(input("Full path for the new environment: ").strip()).expanduser()
    if env_path.exists():
        if input("Path exists. Overwrite? [y/N]: ").lower() != "y":
            print("[INFO] Creation aborted.")
            return
        shutil.rmtree(env_path)

    sh(f"python3 -m venv \"{env_path}\"")

    req = input("Path to requirements.txt (leave blank to skip): ").strip()
    if req:
        sh(f"\"{bin_path(env_path, 'pip')}\" install -r \"{req}\"")

    print("[SUCCESS] Environment ready.")


def add_libs() -> None:
    env_path = Path(input("Environment path: ").strip()).expanduser()
    if not env_path.exists():
        print("[ERROR] Environment not found.")
        return

    target = input("Libraries to add (comma separated) **or** requirements file path: ").strip()
    pip = bin_path(env_path, "pip")

    if target.endswith(".txt"):
        sh(f"\"{pip}\" install -r \"{target}\"")
    else:
        pkgs = " ".join(p.strip() for p in target.split(",") if p.strip())
        if pkgs:
            sh(f"\"{pip}\" install {pkgs}")
        else:
            print("[WARN] No valid packages supplied.")


def remove_libs() -> None:
    env_path = Path(input("Environment path: ").strip()).expanduser()
    if not env_path.exists():
        print("[ERROR] Environment not found.")
        return

    pkgs = " ".join(p.strip() for p in input("Libraries to remove (comma separated): ").split(",") if p.strip())
    if pkgs:
        sh(f"\"{bin_path(env_path, 'pip')}\" uninstall -y {pkgs}")
    else:
        print("[WARN] No valid packages supplied.")


# ──────────────────────────────────────────────────────────────────────────
# Menu loop
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    menu = textwrap.dedent(
        """
        === Simple Env Panel ===
        1) Create environment
        2) Add libraries
        3) Remove libraries
        4) Exit
        """
    )

    actions = {
        "1": create_env,
        "2": add_libs,
        "3": remove_libs,
        "4": lambda: sys.exit(0),
    }

    while True:
        print(menu)
        choice = input("Choose an option: ").strip()
        action = actions.get(choice)
        if action:
            try:
                action()
            except subprocess.CalledProcessError as err:
                print(f"[ERROR] Command failed with exit code {err.returncode}.")
        else:
            print("[WARN] Invalid choice. Try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Exiting…")
