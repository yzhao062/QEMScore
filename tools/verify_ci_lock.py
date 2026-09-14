"""Verify the exact CI lock before installing it."""

from __future__ import annotations

import argparse
import hashlib
import re
import tomllib
from pathlib import Path

_SHA256 = re.compile(r"[0-9a-f]{64}")


def normalized_lock_bytes(content: bytes) -> bytes:
    """Return lock bytes with checkout line endings canonicalized to LF."""
    if b"\r" in content.replace(b"\r\n", b""):
        raise ValueError("CI lock contains an unsupported bare CR line ending")
    return content.replace(b"\r\n", b"\n")


def lock_sha256(content: bytes) -> str:
    return hashlib.sha256(normalized_lock_bytes(content)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--digest-only",
        action="store_true",
        help="print only the verified canonical-LF digest",
    )
    args = parser.parse_args()

    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    policy = config["tool"]["qemscore"]["reproducibility"]
    recorded_path = Path(policy["ci-lock"])
    recorded_hash = str(policy["ci-lock-sha256"]).lower()
    lock_path = args.lock or recorded_path
    expected_hash = (args.expected_sha256 or recorded_hash).lower()

    if lock_path.as_posix() != recorded_path.as_posix():
        raise SystemExit(
            f"CI lock path mismatch: pyproject.toml records {recorded_path}, "
            f"workflow requested {lock_path}"
        )
    if expected_hash != recorded_hash:
        raise SystemExit(
            "CI lock SHA-256 mismatch between pyproject.toml and the workflow: "
            f"{recorded_hash} != {expected_hash}"
        )
    if _SHA256.fullmatch(expected_hash) is None:
        raise SystemExit(
            "CI lock SHA-256 is still a placeholder; fill the measured lowercase "
            "SHA-256 in pyproject.toml and the workflow"
        )
    if not lock_path.is_file():
        raise SystemExit(f"CI lock is missing: {lock_path}")

    raw_content = lock_path.read_bytes()
    raw_hash = hashlib.sha256(raw_content).hexdigest()
    actual_hash = lock_sha256(raw_content)
    if not args.digest_only:
        print(f"CI lock: {lock_path}")
        print(f"expected SHA-256: {expected_hash}")
        print(f"actual SHA-256:   {actual_hash}")
        if raw_hash != actual_hash:
            print("checkout line endings: CRLF normalized to LF for verification")
    if actual_hash != expected_hash:
        raise SystemExit(
            f"CI lock drift: expected {expected_hash}, got {actual_hash}"
        )
    if args.digest_only:
        print(actual_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
