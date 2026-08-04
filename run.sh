#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    if ! command -v powershell.exe >/dev/null 2>&1; then
        echo "Error: Windows interoperability is disabled in this WSL installation." >&2
        exit 1
    fi

    windows_launcher="$(wslpath -w "$project_dir/run.ps1")"
    exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$windows_launcher" "$@"
fi

exec python3 "$project_dir/main.py" "$@"
