"""
Step 1: Fetch main.py from every fork of the hackathon repo.
Saves only forks that have actually modified generate_hybrid to fork_solutions/.

Usage:
    python 01_fetch_forks.py
"""
import subprocess, base64, re, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "cactus-compute/functiongemma-hackathon"
OUT  = Path("fork_solutions")
OUT.mkdir(exist_ok=True)

# The exact baseline generate_hybrid — anything identical to this is skipped
BASELINE_HYBRID = '''\
def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Baseline hybrid inference strategy; fall back to cloud if Cactus Confidence is below threshold."""
    local = generate_cactus(messages, tools)

    if local["confidence"] >= confidence_threshold:
        local["source"] = "on-device"
        return local

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"]
    return cloud'''


def get_all_forks():
    r = subprocess.run(
        ["gh", "api", f"repos/{REPO}/forks", "--paginate", "-q", ".[].full_name"],
        capture_output=True, text=True
    )
    return [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]


def fetch_main_py(repo):
    r = subprocess.run(
        ["gh", "api", f"repos/{repo}/contents/main.py", "-q", ".content"],
        capture_output=True, text=True, timeout=15
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return base64.b64decode(r.stdout.strip()).decode("utf-8")
    except Exception:
        return None


def extract_hybrid_fn(code):
    m = re.search(r"(def generate_hybrid\b.*?)(?=\ndef |\Z)", code, re.DOTALL)
    return m.group(1).strip() if m else None


def is_baseline(code):
    fn = extract_hybrid_fn(code)
    if fn is None:
        return True
    return re.sub(r"\s+", " ", fn) == re.sub(r"\s+", " ", BASELINE_HYBRID)


def process(repo):
    code = fetch_main_py(repo)
    if code is None:
        return repo, None, "fetch_failed"
    if is_baseline(code):
        return repo, None, "baseline"
    return repo, code, "modified"


def main():
    print(f"Listing forks of {REPO} ...")
    forks = get_all_forks()
    print(f"Found {len(forks)} forks.\n")
    print(f"Fetching main.py from each (20 in parallel) ...")

    n_modified = n_baseline = n_failed = 0

    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(process, r): r for r in forks}
        for i, fut in enumerate(as_completed(futures), 1):
            repo, code, status = fut.result()
            owner = repo.split("/")[0]

            if status == "modified":
                out_path = OUT / f"{owner}_main.py"
                out_path.write_text(f"# Source: https://github.com/{repo}\n" + code)
                n_modified += 1
                tag = "✓ MODIFIED"
            elif status == "baseline":
                n_baseline += 1
                tag = "  baseline"
            else:
                n_failed += 1
                tag = "  FAILED  "

            print(f"  [{i:>3}/{len(forks)}] {repo:<50} {tag}")

    print(f"""
{'='*60}
  Total forks : {len(forks)}
  Modified    : {n_modified}  → saved to ./{OUT}/
  Baseline    : {n_baseline}  (skipped)
  Failed      : {n_failed}
{'='*60}
""")


if __name__ == "__main__":
    main()
