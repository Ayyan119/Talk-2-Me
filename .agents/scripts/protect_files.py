#!/usr/bin/env python3
"""Hook script: Protects environment files and critical database assets from accidental destruction."""
import sys
import json
import re

def main():
    try:
        input_data = json.load(sys.stdin)
    except Exception:
        print(json.dumps({"decision": "allow"}))
        return

    tool_call = input_data.get("toolCall", {})
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {})

    # Protected file patterns (regex)
    # DB files: .db, .sqlite, .sqlite3, .db3, .s3db, .sl3
    # ENV files: .env, .env.local, .env.production, .env.development, etc.
    protected_file_pattern = r'(\.env(\.[\w-]+)?|\.(db|sqlite|sqlite3|db3|s3db|sl3))\b'

    # Deletion / destruction command indicators
    deletion_cmd_pattern = r'\b(rm|unlink|shred|git\s+rm)\b'

    if tool_name == "run_command":
        cmd = args.get("CommandLine", "")

        is_deletion = bool(re.search(deletion_cmd_pattern, cmd, re.IGNORECASE))
        has_protected_file = bool(re.search(protected_file_pattern, cmd, re.IGNORECASE))

        if is_deletion and has_protected_file:
            print(json.dumps({
                "decision": "deny",
                "reason": f"SECURITY ENFORCEMENT HOOK: Deleting database (.db/.sqlite) or environment (.env) files is strictly prohibited. Command blocked: '{cmd}'"
            }))
            return

    elif tool_name in ["replace_file_content", "write_to_file"]:
        target_file = args.get("TargetFile", "")
        if re.search(protected_file_pattern, target_file, re.IGNORECASE):
            code_content = args.get("CodeContent", "")
            if tool_name == "write_to_file" and len(code_content.strip()) == 0:
                print(json.dumps({
                    "decision": "deny",
                    "reason": f"SECURITY ENFORCEMENT HOOK: Truncating/emptying protected file '{target_file}' is prohibited."
                }))
                return

    print(json.dumps({"decision": "allow"}))

if __name__ == "__main__":
    main()
