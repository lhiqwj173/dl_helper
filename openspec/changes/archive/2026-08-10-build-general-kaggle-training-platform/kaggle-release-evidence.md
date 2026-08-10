# Kaggle Release Gate Evidence

## Candidate

- Tested candidate revision: `74d7bbd073ff5d845c20e0c14f10c8a7c8c88885`
- Execution date: `2026-08-10`
- Environment: real Kaggle Notebook sessions
- Authentication: Kaggle Secrets only; no Secret values are recorded in this evidence
- Evidence release: `https://github.com/lhiqwj173/dl_helper/releases/tag/kaggle-evidence-74d7bbd0-20260810`
- Evidence tag: `kaggle-evidence-74d7bbd0-20260810`, resolving to tested candidate revision `74d7bbd073ff5d845c20e0c14f10c8a7c8c88885`
- Candidate boundary: later commits may update only this evidence index; the immutable evidence tag above identifies the clean code revision exercised by every archived manifest and resolved config.

## Torch Multi-GPU Resume

- PREEMPTED archive: `mnist-release-gate-round13-proof2-20260810-preempted-evidence.zip`
- PREEMPTED download: `https://github.com/lhiqwj173/dl_helper/releases/download/kaggle-evidence-74d7bbd0-20260810/mnist-release-gate-round13-proof2-20260810-preempted-evidence.zip`
- PREEMPTED SHA256: `648daade4a95bbef8edf022b0cc7350bb24e24b60d8820edb8e83fa233532571`
- Final archive: `mnist-release-gate-round13-proof2-20260810-evidence.zip`
- Final download: `https://github.com/lhiqwj173/dl_helper/releases/download/kaggle-evidence-74d7bbd0-20260810/mnist-release-gate-round13-proof2-20260810-evidence.zip`
- Final SHA256: `0916fb7c374198efa8321421ad7a454860d49404f985f5be038b3d57c6861e5d`
- Run ID: `mnist-release-gate-round13-proof2-20260810`
- PREEMPTED exit evidence created: `2026-08-10T10:31:40Z`
- Final run manifest created: `2026-08-10T10:35:54Z`
- Final service manifest created: `2026-08-10T10:36:08Z`
- Resume exit evidence created: `2026-08-10T10:36:11Z`
- Doctor exit code: `0`
- First training session exit code: `75` (`PREEMPTED`)
- Resume session exit code: `0`
- Distributed execution: two visible GPUs/ranks
- Final optimizer step: `4380`
- Verified: the PREEMPTED archive has the unique `pause-manifest.json` terminal and `release-gate-exit.json` exit code `75`; the final archive has the unique `run-manifest.json` terminal and exit code `0`; both record the tested candidate revision, two Tesla T4 devices/ranks, service audit, and complete Artifact checksums. The remote checkpoint advances from step `3424` to final step `4380` across sessions.

## Sweep

- Evidence archive: `kaggle-sweep-evidence-20260810-081842.zip`
- Download: `https://github.com/lhiqwj173/dl_helper/releases/download/kaggle-evidence-74d7bbd0-20260810/kaggle-sweep-evidence-20260810-081842.zip`
- Archive SHA256: `e7ced1cd684a8a2b9f26c7a83f3e9507f8fe2ab2b2d65fd52197c9a3706264eb`
- Sweep ID: `kaggle-toy-sweep-20260810-081842`
- Trial run IDs: `kaggle-toy-sweep-20260810-081842--lr-1e-2`, `kaggle-toy-sweep-20260810-081842--lr-1e-3`
- Trial run manifests created: `2026-08-10T08:19:51Z`, `2026-08-10T08:20:15Z`
- Final sweep service manifest created: `2026-08-10T08:20:20Z`
- Doctor exit code: `0`
- Sweep exit code: `0`
- Verified: two successful trial terminal manifests, unique successful sweep terminal manifest, scientific ranking and best trial, service manifests, service audits, offline run reports, offline sweep report, and archive integrity

## Sklearn Incremental CPU

- Evidence archive: `sklearn-incremental-evidence-20260810-081842.zip`
- Download: `https://github.com/lhiqwj173/dl_helper/releases/download/kaggle-evidence-74d7bbd0-20260810/sklearn-incremental-evidence-20260810-081842.zip`
- Archive SHA256: `2ed2bf283a413b17b0c89de9e6c7708e2254d661743501f7d6dbd72529485cef`
- Run ID: `sklearn-incremental-smoke-20260810-081842`
- Final run and service manifests created: `2026-08-10T08:21:03Z`
- Doctor exit code: `0`
- Training exit code: `0`
- Verified: unique successful terminal manifest, incremental model artifacts, evaluation contract, service manifest, service audit, offline HTML report, and archive integrity

## Credential Rotation

On `2026-08-10`, the repository owner confirmed that historical AList, WeCom, and all other identified credentials had been rotated and that the previous credentials were invalidated. No credential values are retained in the repository or evidence archives.
