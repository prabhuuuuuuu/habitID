"""Consistency checks for the paper without a TeX toolchain. Read-only on paper/ and paper_assets/.

Checks: \\input targets exist; every \\ref/\\eqref resolves to a \\label (including labels inside inputs);
every \\cite key has a \\bibitem; \\begin/\\end balance per environment; \\TBD occurrences (line numbers);
every decimal number in the text occurs in a generated asset or in results/ (otherwise flagged as possibly
hand-typed). Writes results/tables/latex_checks.json."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from habitid.utils.config import PAPER, PAPER_ASSETS, RESULTS  # noqa: E402
from habitid.utils.runtime import write_json  # noqa: E402

MAIN = PAPER / "main.tex"


def read_with_inputs(path: Path, seen=None):
    """Return the text of a file with \\input contents expanded (for label and number checks)."""
    seen = seen or set()
    text = path.read_text(encoding="utf-8")
    missing = []
    def sub(m):
        target = (path.parent / m.group(1)).resolve()
        if not target.exists():
            missing.append(m.group(1))
            return ""
        if target in seen:
            return ""
        return read_with_inputs(target, seen | {target})[0]
    expanded = re.sub(r"\\input\{([^}]+)\}", sub, text)
    return expanded, missing


def main():
    raw = MAIN.read_text(encoding="utf-8")
    expanded, missing_inputs = read_with_inputs(MAIN)
    # strip comments (unescaped %)
    body = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in expanded.splitlines())

    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = re.findall(r"\\(?:eq)?ref\{([^}]+)\}", body)
    unresolved_refs = sorted({r for r in refs if r not in labels})

    bib_keys = set(re.findall(r"\\bibitem\{([^}]+)\}", raw))
    cite_keys = set()
    for c in re.findall(r"\\cite[a-z]*\{([^}]+)\}", body):
        cite_keys |= {k.strip() for k in c.split(",")}
    missing_cites = sorted(cite_keys - bib_keys)
    uncited_bib = sorted(bib_keys - cite_keys)

    balance = {}
    for env in set(re.findall(r"\\begin\{([A-Za-z*]+)\}", body)) | set(re.findall(r"\\end\{([A-Za-z*]+)\}", body)):
        b = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", body))
        e = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", body))
        if b != e:
            balance[env] = {"begin": b, "end": e}

    tbd_lines = []
    for i, ln in enumerate(raw.splitlines(), 1):
        if "\\TBD" in ln and "newcommand" not in ln and not ln.lstrip().startswith("%"):
            tbd_lines.append({"line": i, "text": ln.strip()[:120]})

    # every decimal number must come from a generated asset or from results/
    generated = ""
    for p in PAPER_ASSETS.rglob("*.tex"):
        generated += p.read_text(encoding="utf-8")
    for p in RESULTS.rglob("*.json"):
        generated += p.read_text(encoding="utf-8")
    decimals = sorted(set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", body)))
    not_found = []
    for d in decimals:
        if d in generated:
            continue
        # formatted results use 2-4 decimals; accept a rounded match
        if any(s.startswith(d) for s in re.findall(r"\d+\.\d+", generated) if len(s) <= len(d) + 2):
            continue
        not_found.append(d)

    report = {
        "inputs_missing": missing_inputs,
        "unresolved_refs": unresolved_refs,
        "missing_bibitems_for_cites": missing_cites,
        "bibitems_never_cited": uncited_bib,
        "unbalanced_environments": balance,
        "tbd_occurrences_outside_macro_definition": tbd_lines,
        "decimal_numbers_not_found_in_assets_or_results": not_found,
        "labels_defined": len(labels), "refs_used": len(refs), "cite_keys": len(cite_keys),
    }
    write_json(RESULTS / "tables" / "latex_checks.json", report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
