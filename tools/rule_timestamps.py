"""GitHub-recorded times of each frozen rule and of its results (round-10 rule, Part F).

Governing rule: docs/frozen-rules/2026-10-07-round10-checks.md, Part F.

Commit dates are set by the committer's machine. GitHub's events API records,
on GitHub's servers, when each push arrived (``PushEvent.created_at``, with
the pushed head and the previous head), and its releases API records when each
release and asset was created. The events API returns a bounded recent window
and can omit pushes, so the responses are saved as files and this script reads
only those files and the local git history:

  gh api repos/yzhao062/QEMScore/events --paginate > EVENTS.json
  gh api repos/yzhao062/QEMScore/releases --paginate > RELEASES.json
  gh api repos/yzhao062/QEMScore/actions/runs --paginate > RUNS.json

The events API omits the push of 24a2cbe, so the continuous-integration runs
that each push triggers are a second record: GitHub stamps a run's created_at
when the push arrives, with the pushed head as its head_sha. A run's time is an
upper bound on when its commit was on GitHub.

For each rule file under docs/frozen-rules/ it finds the commit that added
the file and, from RESULTS below (read from the commit history when this rule
was written), the commit that first recorded its results; for each, the
earliest push whose head contains the commit while the previous head does
not. It reports both push times, the releases whose
RELEASES below assigns to the rule, and whether the rule's push precedes its results' push
and those releases.

Usage:
  python tools/rule_timestamps.py --events EVENTS.json --releases RELEASES.json \\
      --runs RUNS.json --frozen-rule docs/frozen-rules/2026-10-07-round10-checks.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

_REPO = Path(__file__).resolve().parents[1]
RULE_FILE = "docs/frozen-rules/2026-10-07-round10-checks.md"
DEFAULT_OUT = _REPO / "artifacts/rule-timestamps/rule-timestamps.json"
# Rule file stem -> commit that first recorded its results. The results of
# the two 2026-10-01 analyses entered git at 645db53, before their rule files
# entered git at 9b8c727; their ordering before computation rests on file
# modification times.
RESULTS = {
    "2026-10-01-learner-seed-replication": "645db53",
    "2026-10-01-near-clifford-positive-control": "645db53",
    "2026-10-02-descriptor-information": "9b8c727",
    "2026-10-03-strength-indicator": "1bac1e9",
    "2026-10-03-shot-sweep": "6a3dc4e",
    "2026-10-03-strong-learners": "6a3dc4e",
    "2026-10-04-crossed-follow-ups": "0152dd5",
    "2026-10-04-fresh-confirmation": "0152dd5",
    "2026-10-04-mlqem-own-data": "0152dd5",
    "2026-10-06-round9-follow-ups": "87870da",
    "2026-10-06-shift-transfer": "91c39b8",
}
# Rule file stem -> releases that hold the fits or data it governs.
RELEASES = {
    "2026-10-02-descriptor-information": ("descriptor-information-v1",),
    "2026-10-03-strength-indicator": ("descriptor-information-v1",),
    "2026-10-03-shot-sweep": ("descriptor-information-v1",),
    "2026-10-03-strong-learners": ("descriptor-information-v1",),
    "2026-10-04-crossed-follow-ups": ("confirmation-v1",),
    "2026-10-04-fresh-confirmation": ("confirmation-v1",),
    "2026-10-04-mlqem-own-data": ("confirmation-v1",),
    "2026-10-06-round9-follow-ups": ("exact-labels-oracle-v1",),
    "2026-10-06-shift-transfer": ("shift-transfer-v1",),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_REPO), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def is_ancestor(commit: str, head: str | None) -> bool:
    if not head:
        return False
    result = subprocess.run(["git", "-C", str(_REPO), "merge-base", "--is-ancestor", commit, head],
                            capture_output=True)
    if result.returncode not in (0, 1):
        raise SystemExit(f"git merge-base failed for {commit} {head}")
    return result.returncode == 0


def flatten(pages) -> list:
    """``gh api --paginate`` writes one JSON array per page, concatenated."""
    text = pages if isinstance(pages, str) else json.dumps(pages)
    decoder, out, i = json.JSONDecoder(), [], 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        value, i = decoder.raw_decode(text, i)
        out.extend(value if isinstance(value, list) else [value])
    return out


def first_push(commit: str, pushes: list[dict]) -> dict | None:
    for p in sorted(pushes, key=lambda e: e["created_at"]):
        head, before = p["payload"].get("head"), p["payload"].get("before")
        if is_ancestor(commit, head) and not is_ancestor(commit, before):
            return {"created_at": p["created_at"], "head": head, "before": before,
                    "event_id": p["id"]}
    return None


def first_run(commit: str, runs: list[dict]) -> dict | None:
    for r in sorted(runs, key=lambda e: e["created_at"]):
        if r.get("event") == "push" and is_ancestor(commit, r["head_sha"]):
            return {"created_at": r["created_at"], "head_sha": r["head_sha"], "run_id": r["id"],
                    "workflow": r.get("name")}
    return None


def earliest(*records) -> str | None:
    times = [r["created_at"] for r in records if r]
    return min(times) if times else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--releases", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--frozen-rule", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    if args.frozen_rule.resolve() != (_REPO / RULE_FILE).resolve():
        raise SystemExit(f"--frozen-rule must be {RULE_FILE}")
    events = flatten(args.events.read_text(encoding="utf-8"))
    releases = flatten(args.releases.read_text(encoding="utf-8"))
    runs = [r for page in flatten(args.runs.read_text(encoding="utf-8"))
            for r in (page.get("workflow_runs", []) if isinstance(page, dict) else [page])]
    pushes = [e for e in events if e.get("type") == "PushEvent"]
    oldest_event = min(e["created_at"] for e in events) if events else None

    rules = []
    for path in sorted((_REPO / "docs/frozen-rules").glob("*.md")):
        name = path.stem
        rel = str(path.relative_to(_REPO))
        added = git("log", "--diff-filter=A", "--format=%H %cI", "--reverse", "--", rel)
        if not added:
            continue
        add_commit, add_date = added.splitlines()[0].split()
        entry = {"rule": rel, "added_commit": add_commit, "added_commit_date": add_date,
                 "added_push": first_push(add_commit, pushes),
                 "added_ci_run": first_run(add_commit, runs)}
        if name in RESULTS:
            r_commit, r_date, r_subject = git("log", "-1", "--format=%H %cI %s",
                                              RESULTS[name]).split(" ", 2)
            entry.update({"results_commit": r_commit, "results_commit_date": r_date,
                          "results_subject": r_subject[:200],
                          "same_commit_as_rule": r_commit == add_commit,
                          "results_push": first_push(r_commit, pushes),
                          "results_ci_run": first_run(r_commit, runs)})
        named = [rel_ for rel_ in releases if rel_["tag_name"] in RELEASES.get(name, ())]
        # A release's created_at is the date of its tagged commit, which the
        # committer sets; published_at and each asset's created_at (upload
        # time) are recorded by GitHub.
        entry["releases"] = [{
            "tag": r["tag_name"], "tagged_commit_date": r["created_at"],
            "published_at": r.get("published_at"),
            "assets": [{"name": a["name"], "uploaded_at": a["created_at"]}
                       for a in r.get("assets", [])]} for r in named]
        rule_push = earliest(entry["added_push"], entry["added_ci_run"])
        results_push = earliest(entry.get("results_push"), entry.get("results_ci_run"))
        entry["rule_on_github_by"] = rule_push
        entry["results_on_github_by"] = results_push
        checks = {}
        if rule_push and results_push and not entry.get("same_commit_as_rule"):
            checks["rule_push_before_results_push"] = rule_push < results_push
        if rule_push and entry["releases"]:
            checks["rule_push_before_every_named_release_upload"] = all(
                rule_push < min([a["uploaded_at"] for a in r["assets"]]
                                + [r["published_at"] or "9999"])
                for r in entry["releases"])
        entry["checks"] = checks
        rules.append(entry)

    doc = {"schema": "rule-timestamps-v1", "frozen_rule": RULE_FILE,
           "rule_file_sha256": sha256_file(_REPO / RULE_FILE),
           "script_sha256": sha256_file(Path(__file__)),
           "events_file_sha256": sha256_file(args.events),
           "releases_file_sha256": sha256_file(args.releases),
           "runs_file_sha256": sha256_file(args.runs), "n_push_ci_runs": sum(
               r.get("event") == "push" for r in runs),
           "n_push_events": len(pushes), "oldest_event": oldest_event,
           "rules": rules}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(rules)} rules, {len(pushes)} push events since {oldest_event}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
