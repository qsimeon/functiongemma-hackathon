"""
Step 2: Benchmark every modified fork against the local test suite.

Each fork runs in its own temp directory (safe to run with multiple workers).
Results saved to results.json. Top-10 memo printed at the end.

Usage:
    python 02_benchmark_forks.py                    # run all forks (4 parallel workers)
    python 02_benchmark_forks.py --workers 1        # sequential (safer on low RAM)
    python 02_benchmark_forks.py --only ypeng12     # run one specific fork owner
    python 02_benchmark_forks.py --resume           # skip already-scored forks

Requires: GEMINI_API_KEY exported in your shell.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SOLUTIONS_DIR  = Path("fork_solutions")
RESULTS_FILE   = Path("results.json")
BENCHMARK_FILE = Path("benchmark.py")
CACTUS_DIR     = Path("cactus").resolve()   # absolute so symlinks work from tmp dirs
PYTHON         = sys.executable             # whatever python is running this script


# ── Parse benchmark.py output ────────────────────────────────────────────────

def parse_output(stdout):
    score = avg_f1 = avg_time = on_dev = None
    for line in stdout.splitlines():
        if "TOTAL SCORE:" in line:
            m = re.search(r"([\d.]+)%", line)
            if m: score = float(m.group(1))
        if "overall" in line and "avg F1=" in line:
            m = re.search(r"F1=([\d.]+)", line)
            if m: avg_f1 = float(m.group(1))
            m = re.search(r"time=([\d.]+)ms", line)
            if m: avg_time = float(m.group(1))
        if "on-device=" in line and "(" in line:
            m = re.search(r"on-device=\d+/\d+\s+\(([\d.]+)%\)", line)
            if m: on_dev = float(m.group(1))
    return score, avg_f1, avg_time, on_dev


# ── Run one fork ──────────────────────────────────────────────────────────────

def run_fork(fork_path: Path, timeout=600) -> dict:
    owner = fork_path.stem.replace("_main", "")
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix=f"cactus_bench_{owner}_") as tmp:
        # Each fork gets its own directory so parallel runs don't stomp each other
        shutil.copy(BENCHMARK_FILE, Path(tmp) / "benchmark.py")
        shutil.copy(fork_path,      Path(tmp) / "main.py")
        os.symlink(CACTUS_DIR,      Path(tmp) / "cactus")

        env = {**os.environ, "CACTUS_NO_CLOUD_TELE": "1"}
        try:
            result = subprocess.run(
                [PYTHON, "benchmark.py"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {"owner": owner, "error": "timeout", "elapsed_s": timeout}
        except Exception as e:
            return {"owner": owner, "error": str(e), "elapsed_s": time.time() - t0}

    elapsed = time.time() - t0

    if result.returncode != 0:
        # Show last 300 chars of stderr for diagnosis
        return {
            "owner":     owner,
            "error":     result.stderr[-300:].strip(),
            "elapsed_s": round(elapsed, 1),
            "stdout":    result.stdout[-500:],
        }

    score, avg_f1, avg_time, on_dev = parse_output(result.stdout)
    return {
        "owner":        owner,
        "score":        score,
        "avg_f1":       avg_f1,
        "avg_time_ms":  avg_time,
        "on_device_pct": on_dev,
        "elapsed_s":    round(elapsed, 1),
        "stdout_tail":  result.stdout[-2000:],
    }


# ── Print memo ────────────────────────────────────────────────────────────────

def print_memo(results: dict):
    scored  = [(o, r) for o, r in results.items() if r.get("score") is not None]
    failed  = [(o, r) for o, r in results.items() if r.get("error")]
    ranked  = sorted(scored, key=lambda x: x[1]["score"], reverse=True)

    print("\n" + "="*72)
    print("  TOP 10 FORKS — Local Benchmark Results")
    print("="*72)
    print(f"  {'#':>2}  {'Owner':<28}  {'Score':>7}  {'F1':>5}  {'AvgTime':>9}  {'OnDev':>6}")
    print(f"  {'--':>2}  {'-'*28}  {'-'*7}  {'-'*5}  {'-'*9}  {'-'*6}")
    for i, (owner, r) in enumerate(ranked[:10], 1):
        f1      = f"{r['avg_f1']:.3f}"   if r.get('avg_f1')       is not None else "  n/a"
        t       = f"{r['avg_time_ms']:.0f}ms" if r.get('avg_time_ms') is not None else "   n/a"
        od      = f"{r['on_device_pct']:.0f}%" if r.get('on_device_pct') is not None else " n/a"
        print(f"  {i:>2}  {owner:<28}  {r['score']:>6.1f}%  {f1:>5}  {t:>9}  {od:>6}")

    if len(ranked) > 10:
        print(f"\n  ... {len(ranked)-10} more scored results in {RESULTS_FILE}")

    if failed:
        print(f"\n  Failed ({len(failed)}): " + ", ".join(o for o,_ in failed))

    print("="*72)

    # Write MEMO.md
    lines = [
        "# Fork Benchmark Memo",
        f"> Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"> Forks evaluated: {len(scored)} scored, {len(failed)} failed",
        "",
        "## Top 10",
        "",
        "| # | Owner | Score | F1 | Avg Time | On-Device | Notes |",
        "|---|-------|-------|----|----------|-----------|-------|",
    ]
    for i, (owner, r) in enumerate(ranked[:10], 1):
        note = ""
        if r.get("avg_time_ms", 999) < 5:
            note = "pure rule-based (no model)"
        elif r.get("avg_time_ms", 999) < 300:
            note = "fast — likely regex + minimal model call"
        else:
            note = "model-heavy"
        lines.append(
            f"| {i} | `{owner}` | {r['score']:.1f}% | {r.get('avg_f1',0):.3f} | "
            f"{r.get('avg_time_ms',0):.0f}ms | {r.get('on_device_pct',0):.0f}% | {note} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- **<5ms avg time**: Rule-based only — never calls the model. Fast and accurate on",
        "  known patterns, but may not generalize to unseen phrasings on the hidden leaderboard.",
        "- **<300ms avg time**: Regex-guided + minimal model call. Best balance of accuracy,",
        "  speed, and generalization. **These are the ones worth synthesizing from.**",
        "- **>500ms avg time**: Model-heavy. Accurate but slow — loses time bonus.",
        "",
        "## Scoring formula (for reference)",
        "",
        "```",
        "Score = 0.20×easy + 0.30×medium + 0.50×hard",
        "level  = 0.60×F1 + 0.15×time_score + 0.25×on_device_ratio",
        "time_score = max(0, 1 − avg_ms / 500)   # full marks if <500ms",
        "```",
    ]
    Path("MEMO.md").write_text("\n".join(lines))
    print(f"\n  Full memo written to MEMO.md")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default 4)")
    parser.add_argument("--only",    nargs="*",            help="Run only these owner names")
    parser.add_argument("--resume",  action="store_true",  help="Skip already-scored forks")
    parser.add_argument("--timeout", type=int, default=600, help="Per-fork timeout in seconds")
    args = parser.parse_args()

    # Load existing results
    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    # Collect forks to run
    all_forks = sorted(SOLUTIONS_DIR.glob("*_main.py"))
    if args.only:
        all_forks = [p for p in all_forks if any(o in p.stem for o in args.only)]
    if args.resume:
        all_forks = [p for p in all_forks
                     if p.stem.replace("_main","") not in results
                     or results[p.stem.replace("_main","")].get("score") is None]

    if not all_forks:
        print("No forks to benchmark. Run 01_fetch_forks.py first.")
        return

    print(f"Benchmarking {len(all_forks)} forks  (workers={args.workers}, timeout={args.timeout}s)")
    print(f"Python: {PYTHON}")
    print(f"Cactus: {CACTUS_DIR}\n")

    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY not set — cloud fallback calls will fail.\n")

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_fork, p, args.timeout): p for p in all_forks}
        for fut in as_completed(futures):
            r = fut.result()
            owner = r["owner"]
            results[owner] = r
            completed += 1

            if r.get("error"):
                status = f"ERROR: {r['error'][:60]}"
            else:
                status = (f"score={r['score']:.1f}%  F1={r.get('avg_f1',0):.3f}"
                          f"  {r.get('avg_time_ms',0):.0f}ms  "
                          f"{r.get('on_device_pct',0):.0f}% on-dev"
                          f"  [{r['elapsed_s']:.0f}s]")

            print(f"  [{completed:>2}/{len(all_forks)}] {owner:<28} {status}")

            # Save after every result so you can Ctrl-C and resume
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

    print_memo(results)
    print(f"\nAll results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
