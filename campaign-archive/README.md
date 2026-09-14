# Campaign archive

The outputs of the frozen controlled campaign reported in the QEM-Bench paper.
This directory carries the manifest and the checksums; the archive itself is a
release asset, because it is 60 MB and does not belong in the git history of a
package people install.

## Get it

```
curl -L -O https://github.com/yzhao062/QEMScore/releases/download/campaign-archive-v1/campaign-archive-v1.tar.gz
```

| | |
|---|---|
| Size | 62,575,902 bytes |
| SHA-256 | `62f1ef1875eaf54100c78fe26164b31d62b5efecfb4740e22a4e2a3c313bf918` |
| Extracts to | `campaign-archive-v1/`, 67 files, 277,466,583 bytes |

## Check it

```
sha256sum campaign-archive-v1.tar.gz
tar -xzf campaign-archive-v1.tar.gz
cd campaign-archive-v1 && sha256sum -c SHA256SUMS.txt
```

All 67 lines should report `OK`. `SHA256SUMS.txt` in this directory is the same
file, so the archive can be checked against a copy that travels in git rather
than one that travels beside the download.

The tree fingerprint over those 67 files, hashing the UTF-8 concatenation of
`<file SHA-256>  <relative POSIX path>\n` in sorted relative-path order, is
`c3b79a0766c1d48117995826a2ee48a1f0b2a1955931f314a9d298dbc9611821`. That is the
SHA-256 of `SHA256SUMS.txt` itself, because the file is that concatenation. It is
one number to carry, not a second independent check.

## What is in it

`MANIFEST.md` in this directory is the archive's own manifest, listing every
component, the campaign's identity and frozen-design hash, what is excluded and
why, and what the archive does not establish. Read it before the data.

In short: the campaign report, tables, audit and frozen manifest; per-setting
rosters carrying every method's test predictions, per-cell metrics and
circuit-evaluation ledger; per-setting records carrying the role assignments,
selected configurations and untouched-test predictions; the fit bindings; and
each setting's dataset manifest.

The generated item streams and sidecars are excluded at 771 MB, because the
deterministic generators in this repository reproduce them from the recorded
seed and dependency lock, and `report.json` carries the twelve dataset hashes a
regeneration must match.

## Versioning

`campaign-archive-v1` is the first version. A later version supersedes it under a
new tag rather than replacing the asset, so a paper citing v1 keeps resolving.
