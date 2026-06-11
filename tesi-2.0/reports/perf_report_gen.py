"""Genera report markdown da risultati benchmark."""

import json
from pathlib import Path


def generate_perf_report(bench_json: str | Path, output_md: str | Path) -> None:
    """Legge benchmark raw JSON e genera MD report con tabella."""
    bench_json = Path(bench_json)
    output_md = Path(output_md)

    with open(bench_json) as f:
        data = json.load(f)

    timings = data["timings"]
    total = sum(timings.values())

    # Genera tabella MD
    lines = [
        "# Performance Benchmark — matchingpennies pipeline",
        "",
        f"**Subject**: {data['subject']}",
        f"**Atlas**: {data['atlas']}",
        f"**Metric**: {data['metric']}",
        f"**Epochs**: {data['n_epochs']}",
        f"**ML Algorithms**: {', '.join(data['ml_algos'])}",
        "",
        "## Step Timings",
        "",
        "| Step | Time (s) | % |",
        "|------|----------|---|",
    ]

    for step, elapsed in timings.items():
        pct = (elapsed / total) * 100
        step_label = step.replace("_", " ").title()
        lines.append(f"| {step_label} | {elapsed:.3f} | {pct:.1f}% |")

    lines.extend([
        "",
        f"| **Total** | **{total:.3f}** | **100%** |",
        "",
        "## Bottleneck Analysis",
        "",
    ])

    # Identifica bottleneck
    bottleneck_step = max(timings, key=timings.get)
    bottleneck_time = timings[bottleneck_step]
    lines.append(
        f"**Bottleneck**: `{bottleneck_step}` ({bottleneck_time:.3f}s, "
        f"{(bottleneck_time/total)*100:.1f}% di tempo totale)"
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report → {output_md}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-json", default="reports/PERF_BENCHMARK_raw.json")
    ap.add_argument("--output-md", default="reports/PERF_BENCHMARK.md")
    args = ap.parse_args()

    generate_perf_report(args.bench_json, args.output_md)
