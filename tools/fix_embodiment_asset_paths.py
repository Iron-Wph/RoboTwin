#!/usr/bin/env python3
import re
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    embodiments_dir = repo_root / "assets" / "embodiments"

    if not embodiments_dir.is_dir():
        raise SystemExit(f"Missing {embodiments_dir}. Download assets first.")

    repo_root_posix = repo_root.as_posix()
    current_prefix = f"{repo_root_posix}/assets/embodiments/"
    absolute_embodiment_prefix = re.compile(r"(?:/[^\s'\"{}]+)+/assets/embodiments/")

    yml_files = sorted(embodiments_dir.rglob("*.yml"))
    updated = 0

    for yml_file in yml_files:
        original = yml_file.read_text(encoding="utf-8")
        content = original.replace("${ASSETS_PATH}", repo_root_posix)
        content = content.replace("$ASSETS_PATH", repo_root_posix)
        content = absolute_embodiment_prefix.sub(current_prefix, content)

        if content != original:
            yml_file.write_text(content, encoding="utf-8")
            updated += 1

    bad_refs = []
    for yml_file in yml_files:
        content = yml_file.read_text(encoding="utf-8")
        for match in absolute_embodiment_prefix.finditer(content):
            if match.group(0) != current_prefix:
                bad_refs.append((yml_file, match.group(0)))

    if bad_refs:
        print("Found stale embodiment asset paths:")
        for yml_file, stale_path in bad_refs[:20]:
            print(f"  {yml_file}: {stale_path}")
        raise SystemExit(1)

    print(f"Embodiment asset paths fixed: {updated} files updated")


if __name__ == "__main__":
    main()
