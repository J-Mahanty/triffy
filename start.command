#!/bin/sh
# Double-click to start Triffy on a Mac (also runs on Linux: sh start.command).
# Everything it does is in start.py.
cd "$(dirname "$0")" || exit 1
for py in .venv/bin/python venv/bin/python python3; do
    if command -v "$py" >/dev/null 2>&1; then
        "$py" start.py
        exit $?
    fi
done
echo "Python 3 is not installed. Get it from https://www.python.org/downloads/"
printf "Press Return to close. "
read -r _
