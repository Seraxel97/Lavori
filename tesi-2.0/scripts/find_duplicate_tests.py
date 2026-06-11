"""Analisi duplicate/redundant test functions cross-files nella suite tests/.

Rileva:
  1. Nomi di funzione test identici tra file diversi (duplicati esatti per nome).
  2. Nomi di funzione con prefissi/suffissi numerici o varianti minori (quasi-duplicati).
  3. File test con zero assertions (placeholder/skeleton).
  4. Funzioni test duplicate DENTRO lo stesso file.

Output: JSON su stdout + report Markdown opzionale (--report).

Usage:
    python scripts/find_duplicate_tests.py [--tests-dir tests/] [--report reports/TESTS_DEDUPE.md]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path


def collect_test_functions(tests_dir: Path) -> dict[str, list[str]]:
    """Ritorna {filepath: [func_name, ...]} per tutti i test_*.py."""
    result: dict[str, list[str]] = {}
    for fpath in sorted(tests_dir.glob("test_*.py")):
        try:
            tree = ast.parse(fpath.read_text(encoding="utf-8"))
        except SyntaxError:
            result[str(fpath)] = []
            continue
        funcs = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
        result[str(fpath)] = funcs
    return result


def find_cross_file_duplicates(
    func_map: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Funzioni con lo stesso nome in file diversi → {func_name: [file1, file2, ...]}."""
    name_to_files: dict[str, list[str]] = defaultdict(list)
    for fpath, funcs in func_map.items():
        for fn in set(funcs):
            name_to_files[fn].append(fpath)
    return {fn: files for fn, files in name_to_files.items() if len(files) > 1}


def find_intrafile_duplicates(
    func_map: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Funzioni ripetute DENTRO lo stesso file → {filepath: [func_name, ...]}."""
    result: dict[str, list[str]] = {}
    for fpath, funcs in func_map.items():
        seen: dict[str, int] = defaultdict(int)
        for fn in funcs:
            seen[fn] += 1
        dupes = [fn for fn, count in seen.items() if count > 1]
        if dupes:
            result[fpath] = dupes
    return result


def find_near_duplicates(
    func_map: dict[str, list[str]], min_stem_len: int = 8
) -> list[tuple[str, str, str, str]]:
    """Coppie (file1, fn1, file2, fn2) con stem quasi-identico (differing only in trailing digits/suffix).

    Usa regex: strip trailing _v2, _2, _bis, _copy, _new, _old, _alt.
    """
    _suffix_re = re.compile(r"(_v?\d+|_bis|_copy|_new|_old|_alt|_\d+)$")

    def normalize(name: str) -> str:
        return _suffix_re.sub("", name)

    # Indice {normalized_name: [(file, original_name), ...]}
    norm_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for fpath, funcs in func_map.items():
        for fn in set(funcs):
            norm = normalize(fn)
            if len(norm) >= min_stem_len:
                norm_map[norm].append((fpath, fn))

    results: list[tuple[str, str, str, str]] = []
    for entries in norm_map.values():
        # Almeno due funzioni con stesso stem ma non identiche
        unique_names = {fn for _, fn in entries}
        if len(entries) > 1 and len(unique_names) > 1:
            for i, (f1, n1) in enumerate(entries):
                for f2, n2 in entries[i + 1 :]:
                    if n1 != n2:
                        results.append((f1, n1, f2, n2))
    return results


def find_zero_assertion_tests(tests_dir: Path) -> dict[str, list[str]]:
    """Funzioni test senza assert/raises nel corpo → probabile placeholder."""
    result: dict[str, list[str]] = {}
    for fpath in sorted(tests_dir.glob("test_*.py")):
        try:
            tree = ast.parse(fpath.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        placeholders = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
                continue
            body_src = ast.unparse(node)
            has_assert = "assert" in body_src
            has_raises = "raises" in body_src or "pytest.raises" in body_src
            has_fail = "pytest.fail" in body_src
            if not (has_assert or has_raises or has_fail):
                placeholders.append(node.name)
        if placeholders:
            result[str(fpath)] = placeholders
    return result


def build_report(
    func_map: dict[str, list[str]],
    cross_dupes: dict[str, list[str]],
    intra_dupes: dict[str, list[str]],
    near_dupes: list[tuple[str, str, str, str]],
    zero_assertions: dict[str, list[str]],
    tests_dir: Path,
) -> str:
    total_files = len(func_map)
    total_funcs = sum(len(v) for v in func_map.values())
    lines = [
        "# Tests Deduplication Report",
        "",
        "**Data**: 2026-05-01  **Sprint**: S-61  **Worker**: sonnet2-ts",
        f"**Suite**: `{tests_dir}/` — {total_files} file, {total_funcs} funzioni test",
        "",
        "---",
        "",
    ]

    # 1. Cross-file duplicates
    lines += ["## 1. Duplicati cross-file (stesso nome in file diversi)", ""]
    if cross_dupes:
        lines.append(f"**{len(cross_dupes)} funzioni duplicate** (stesso nome, file diversi):\n")
        lines.append("| Funzione | File |")
        lines.append("|----------|------|")
        for fn, files in sorted(cross_dupes.items()):
            short_files = ", ".join(Path(f).name for f in files)
            lines.append(f"| `{fn}` | {short_files} |")
    else:
        lines.append("Nessun duplicato cross-file trovato.")
    lines.append("")

    # 2. Intra-file duplicates
    lines += ["## 2. Duplicati intra-file (stesso nome nello stesso file)", ""]
    if intra_dupes:
        for fpath, funcs in sorted(intra_dupes.items()):
            lines.append(f"**{Path(fpath).name}**: `{'`, `'.join(funcs)}`")
    else:
        lines.append("Nessun duplicato intra-file trovato.")
    lines.append("")

    # 3. Near-duplicates
    lines += ["## 3. Quasi-duplicati (stem identico, suffisso numerico/variante)", ""]
    if near_dupes:
        lines.append(f"**{len(near_dupes)} coppie** con nome quasi-identico:\n")
        lines.append("| File 1 | Funzione 1 | File 2 | Funzione 2 |")
        lines.append("|--------|-----------|--------|-----------|")
        shown = set()
        for f1, n1, f2, n2 in sorted(near_dupes):
            key = tuple(sorted([(Path(f1).name, n1), (Path(f2).name, n2)]))
            if key in shown:
                continue
            shown.add(key)
            lines.append(f"| {Path(f1).name} | `{n1}` | {Path(f2).name} | `{n2}` |")
    else:
        lines.append("Nessun quasi-duplicato trovato.")
    lines.append("")

    # 4. Zero-assertion tests
    lines += ["## 4. Test senza asserzioni (possibili placeholder)", ""]
    if zero_assertions:
        total_placeholders = sum(len(v) for v in zero_assertions.values())
        lines.append(f"**{total_placeholders} funzioni** senza `assert`/`raises`:\n")
        for fpath, funcs in sorted(zero_assertions.items()):
            lines.append(f"**{Path(fpath).name}**: `{'`, `'.join(funcs)}`")
    else:
        lines.append("Nessun test senza asserzioni trovato.")
    lines.append("")

    # Summary
    lines += [
        "---",
        "",
        "## Riepilogo",
        "",
        "| Categoria | Count |",
        "|-----------|-------|",
        f"| Duplicati cross-file | {len(cross_dupes)} |",
        f"| Duplicati intra-file | {sum(len(v) for v in intra_dupes.values())} |",
        f"| Quasi-duplicati (coppie) | {len(near_dupes)} |",
        f"| Test senza asserzioni | {sum(len(v) for v in zero_assertions.values())} |",
        "",
        "### Raccomandazioni",
        "",
    ]
    if cross_dupes:
        lines.append(
            "- **Duplicati cross-file**: consolidare nel file piu' specifico; "
            "rimuovere la copia generica (es. test_smoke.py vs test_<modulo>.py)."
        )
    if intra_dupes:
        lines.append("- **Duplicati intra-file**: eliminare la seconda occorrenza o rinominare.")
    if near_dupes:
        lines.append(
            "- **Quasi-duplicati**: verificare se testano casi davvero distinti; "
            "se si, aggiungere commento esplicativo; altrimenti unificare con parametrize."
        )
    if zero_assertions:
        lines.append(
            "- **Placeholder**: aggiungere almeno un `assert` o rimuovere se non necessari."
        )
    if not any([cross_dupes, intra_dupes, near_dupes, zero_assertions]):
        lines.append("Suite pulita: nessuna azione richiesta.")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Trova test duplicati/ridondanti nella suite")
    ap.add_argument("--tests-dir", default="tests/")
    ap.add_argument("--report", default="reports/TESTS_DEDUPE.md")
    ap.add_argument("--json-output", default=None, help="Salva JSON raw risultati")
    args = ap.parse_args()

    tests_dir = Path(args.tests_dir)

    func_map = collect_test_functions(tests_dir)
    cross_dupes = find_cross_file_duplicates(func_map)
    intra_dupes = find_intrafile_duplicates(func_map)
    near_dupes = find_near_duplicates(func_map)
    zero_assertions = find_zero_assertion_tests(tests_dir)

    report_md = build_report(
        func_map, cross_dupes, intra_dupes, near_dupes, zero_assertions, tests_dir
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Report → {report_path}")

    raw = {
        "total_files": len(func_map),
        "total_functions": sum(len(v) for v in func_map.values()),
        "cross_file_duplicates": {fn: files for fn, files in cross_dupes.items()},
        "intrafile_duplicates": {f: funcs for f, funcs in intra_dupes.items()},
        "near_duplicates": [
            {"file1": f1, "fn1": n1, "file2": f2, "fn2": n2}
            for f1, n1, f2, n2 in near_dupes
        ],
        "zero_assertion_tests": {f: funcs for f, funcs in zero_assertions.items()},
    }
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(raw, indent=2), encoding="utf-8")
        print(f"JSON → {args.json_output}")

    print(
        f"\nSummary: cross={len(cross_dupes)} intra={sum(len(v) for v in intra_dupes.values())} "
        f"near={len(near_dupes)} placeholders={sum(len(v) for v in zero_assertions.values())}"
    )


if __name__ == "__main__":
    main()
