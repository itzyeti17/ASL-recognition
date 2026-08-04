#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    echo "This setup script is intended for WSL." >&2
    echo "On native Linux, install with: python3 -m pip install -r requirements.txt" >&2
    exit 1
fi

windows_setup="$(wslpath -w "$project_dir/setup.ps1")"
exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$windows_setup"
