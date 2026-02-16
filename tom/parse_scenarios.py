#!/usr/bin/env python3
"""Parse scenario descriptions from log files."""

import re
import os
from pathlib import Path


def get_most_recent_log(log_dir: str = "tom_llm_logs") -> str:
    log_path = Path(log_dir)
    log_files = list(log_path.glob("*.log"))
    if not log_files:
        raise FileNotFoundError(f"No log files found in {log_dir}")
    return str(max(log_files, key=os.path.getmtime))


def main():
    log_file = get_most_recent_log()
    with open(log_file, 'r') as f:
        content = f.read()

    pattern = re.compile(
        r"-{47}\n(.*?)\n-{46}",
        re.DOTALL
    )

    descriptions = [m.group(1).strip() for m in pattern.finditer(content)]

    output_file = Path(log_file).stem + '_scenarios.txt'
    with open(output_file, 'w') as f:
        f.write('\n\n'.join(descriptions))

    print(f"Wrote {len(descriptions)} scenarios to {output_file}")


if __name__ == '__main__':
    main()
