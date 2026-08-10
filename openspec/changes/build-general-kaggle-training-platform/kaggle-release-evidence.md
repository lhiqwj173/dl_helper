# Kaggle Release Gate Evidence

## Candidate

- Revision: `65f1063ffe3ebef746d080c503bfc27b29999829`
- Execution date: `2026-08-10`
- Environment: real Kaggle Notebook sessions
- Authentication: Kaggle Secrets only; no Secret values are recorded in this evidence
- Evidence release: `https://github.com/lhiqwj173/dl_helper/releases/tag/kaggle-evidence-65f1063f-20260810`
- Evidence tag: `kaggle-evidence-65f1063f-20260810`, resolving to candidate revision `65f1063ffe3ebef746d080c503bfc27b29999829`

## Torch Multi-GPU Resume

- Evidence archive: `mnist-release-gate-parent-publish-20260810-evidence.zip`
- Download: `https://github.com/lhiqwj173/dl_helper/releases/download/kaggle-evidence-65f1063f-20260810/mnist-release-gate-parent-publish-20260810-evidence.zip`
- Archive SHA256: `41101843a28dc2377072e6ef9455921026e6dc1fd525ac59c45c27d7f727005e`
- Run ID: `mnist-release-gate-parent-publish-20260810`
- Final run manifest created: `2026-08-10T04:51:38Z`
- Final service manifest created: `2026-08-10T04:51:58Z`
- Doctor exit code: `0`
- First training session exit code: `75` (`PREEMPTED`)
- Resume session exit code: `0`
- Distributed execution: two visible GPUs/ranks
- Final optimizer step: `4380`
- Verified: remote checkpoint restore, unique successful terminal manifest, service manifest, service audit, offline HTML report, and artifact checksums

## Sweep

- Evidence archive: `kaggle-sweep-evidence-20260810-061545.zip`
- Download: `https://github.com/lhiqwj173/dl_helper/releases/download/kaggle-evidence-65f1063f-20260810/kaggle-sweep-evidence-20260810-061545.zip`
- Archive SHA256: `b97fd4cda7c450a474f3e6718b18ffc152355bfa3ae569ef62a3fdc06b28e0e6`
- Sweep ID: `kaggle-toy-sweep-20260810-061545`
- Trial run IDs: `kaggle-toy-sweep-20260810-061545--lr-1e-2`, `kaggle-toy-sweep-20260810-061545--lr-1e-3`
- Trial run manifests created: `2026-08-10T06:16:41Z`, `2026-08-10T06:17:00Z`
- Final sweep service manifest created: `2026-08-10T06:17:04Z`
- Doctor exit code: `0`
- Sweep exit code: `0`
- Verified: two successful trial terminal manifests, unique successful sweep terminal manifest, scientific ranking and best trial, service manifests, service audits, offline run reports, offline sweep report, and archive integrity

## Sklearn Incremental CPU

- Evidence archive: `sklearn-incremental-evidence-20260810-061545.zip`
- Download: `https://github.com/lhiqwj173/dl_helper/releases/download/kaggle-evidence-65f1063f-20260810/sklearn-incremental-evidence-20260810-061545.zip`
- Archive SHA256: `1fdcda6ed00679295306da1e5fc024f8cc1891fea46eec92215ebe57d1997bf8`
- Run ID: `sklearn-incremental-smoke-20260810-061545`
- Final run and service manifests created: `2026-08-10T06:17:37Z`
- Doctor exit code: `0`
- Training exit code: `0`
- Verified: unique successful terminal manifest, incremental model artifacts, evaluation contract, service manifest, service audit, offline HTML report, and archive integrity

## Credential Rotation

On `2026-08-10`, the repository owner confirmed that historical AList, WeCom, and all other identified credentials had been rotated and that the previous credentials were invalidated. No credential values are retained in the repository or evidence archives.
