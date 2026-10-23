#!/usr/bin/env bash
# Publish (or update) one documentation variant on the gh-pages branch.
#
# Usage:
#   _tools/publish_variant.sh <variant-id> <source-dir> [title] [description] [--no-push]
#
# Copies <source-dir> into <variant-id>/, updates variants.json, then commits
# and pushes to origin/gh-pages. Run from anywhere; the script operates on this
# worktree via `git -C`.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

push=1
args=()
for argument in "$@"; do
  if [[ "$argument" == "--no-push" ]]; then
    push=0
  else
    args+=("$argument")
  fi
done
set -- "${args[@]}"

variant="${1:-}"
source_dir="${2:-}"
title="${3:-$variant}"
description="${4:-}"

if [[ -z "$variant" || -z "$source_dir" ]]; then
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
if [[ ! "$variant" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "error: variant id must match [a-z0-9][a-z0-9._-]*" >&2
  exit 2
fi
if [[ ! -d "$source_dir" ]]; then
  echo "error: source directory not found: $source_dir" >&2
  exit 2
fi

dest="$root/$variant"
mkdir -p "$dest"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "$source_dir"/ "$dest"/
else
  rm -rf "$dest"
  mkdir -p "$dest"
  cp -a "$source_dir"/. "$dest"/
fi

python3 - "$root/variants.json" "$variant" "$title" "$description" <<'PY'
import datetime
import json
import sys

path, variant, title, description = sys.argv[1:5]
try:
    with open(path, encoding="utf8") as handle:
        data = json.load(handle)
except (OSError, json.JSONDecodeError):
    data = {"variants": []}

variants = [item for item in data.get("variants", []) if item.get("id") != variant]
variants.append(
    {
        "id": variant,
        "title": title,
        "description": description,
        "url": f"{variant}/",
        "updated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
    }
)
data["variants"] = sorted(
    variants, key=lambda item: item.get("updated", ""), reverse=True
)
with open(path, "w", encoding="utf8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
print(f"updated {path}")
PY

git -C "$root" add -A
if git -C "$root" diff --cached --quiet; then
  echo "no changes to publish"
else
  # The orphan gh-pages branch has no pre-commit config of its own.
  PRE_COMMIT_ALLOW_NO_CONFIG=1 git -C "$root" commit -m "docs: publish variant '$variant'"
fi

if [[ "$push" == "1" ]]; then
  git -C "$root" push origin gh-pages
fi

echo "published: $variant/ -> https://epsilonforge.github.io/gsim/$variant/"
