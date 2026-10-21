# gsim documentation variants (EpsilonForge)

This orphan `gh-pages` branch hosts **variants** of the gsim documentation and
results, independent of any pull request against `gdsfactory/gsim`.

- Landing page: `https://epsilonforge.github.io/gsim/`
- Each variant lives in its own subdirectory and is listed in `variants.json`.
- Nothing here is part of PR #406; the PR branch stays code-only.

## Layout

```
.
├── index.html                 # landing page (renders variants.json)
├── variants.json              # manifest of published variants
├── _tools/publish_variant.sh  # publish/update one variant
└── <variant-id>/              # one directory per variant
    └── index.html, data/, ...
```

## One-time setup

Enable Pages on `EpsilonForge/gsim`:

> Settings → Pages → **Deploy from a branch** → Branch: `gh-pages` / `(root)`

`.nojekyll` is present so files/directories starting with `_` are served as-is.

## Publish a variant

From any checkout, point the script at a **built** directory:

```sh
_tools/publish_variant.sh <variant-id> <source-dir> "Title" "One-line description"
```

It copies `<source-dir>` into `<variant-id>/`, updates `variants.json`, commits
and pushes `gh-pages`. Use `--no-push` to inspect the commit first.

### Results / report variant

```sh
_tools/publish_variant.sh results path/to/results "PDK validation results" \
  "Meep FDTD vs PDK reference data"
```

### Full documentation variant

Build the Zensical site first, then publish the build output:

```sh
just docs                                     # -> docs/site
_tools/publish_variant.sh pdk-validation docs/site \
  "gsim docs (pdk-validation)" "Docs for the PDK validation branch"
```

The docs nav references `nbs/*.md`, which CI normally generates from executed
notebooks. Build locally after populating `docs/nbs/` (e.g. `just nbrun ...`),
or dispatch the `Docs` workflow on the branch and copy its artifact.

## Update or remove

```sh
# update: run publish_variant.sh again with the same id
git rm -r <variant-id>
python3 - <<'PY'
import json
p="variants.json"; d=json.load(open(p))
d["variants"]=[v for v in d["variants"] if v["id"]!="<variant-id>"]
json.dump(d, open(p,"w"), indent=2)
PY
git add -A && git commit -m "docs: remove <variant-id>" && git push origin gh-pages
```
