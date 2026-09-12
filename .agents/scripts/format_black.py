#!/usr/bin/env python3
"""Hook script: Auto-formats modified Python files with Black."""
import sys
import json
import os
import subprocess
import shutil

def main():
    try:
        input_data = json.load(sys.stdin)
    except Exception:
        print(json.dumps({}))
        return

    tool_call = input_data.get("toolCall", {})
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {})

    target_file = None
    if tool_name in ["replace_file_content", "write_to_file"]:
        target_file = args.get("TargetFile")

    if target_file and target_file.endswith(".py") and os.path.exists(target_file):
        # Locate black in venv or local path
        black_exec = "/home/jiggra/talk_to_me/venv/bin/black"
        if not os.path.exists(black_exec):
            black_exec = shutil.which("black") or "/home/jiggra/.local/bin/black"

        if os.path.exists(black_exec) or shutil.which("black"):
            try:
                subprocess.run([black_exec, target_file], capture_output=True, text=True, check=False)
            except Exception:
                pass

    print(json.dumps({}))

if __name__ == "__main__":
    main()
