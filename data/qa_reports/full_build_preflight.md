# Full-Mode Build Preflight (FB1–FB6)

*Generated: 2026-07-17T12:23:50Z*

- **verdict:** WARN
- **generated_at:** 2026-07-17T12:23:50Z

## Gates

| Gate | Name               | Status | Details                                                                                                                                                                                                                                                                             |
| ---- | ------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FB1  | disk space         | pass   | 304.4 GB free (≥ 150 GB)                                                                                                                                                                                                                                                            |
| FB2  | dvc remote         | pass   | 'localstore' at C:\dvc_remote (304.4 GB free)                                                                                                                                                                                                                                       |
| FB3  | roboflow readiness | warn   | sources.roboflow.datasets is empty — human track H-B (slug selection + per-slug license review BEFORE download) has not landed; public coverage for medicine_bottle, charger, wire, gas_cylinder stays blocked.                                                                     |
| FB4  | gpu                | warn   | torch is installed but reports no CUDA device — the download/merge/split stages run fine, but the auto_annotate stage needs the local NVIDIA GPU (user decision, Phase-5 plan).                                                                                                     |
| FB5  | onedrive hazard    | warn   | Repo (and .dvc/cache) live under OneDrive (C:\Users\haris\OneDrive\Desktop\YOLO_V1) — sync can race large builds (risk R34). Before the full build: pause OneDrive sync or relocate the cache off the synced tree (`dvc cache dir <path>` + `dvc config cache.type hardlink,copy`). |
| FB6  | acquisition mode   | warn   | mode: smoke — this preflight targets the full build; flip configs/dataset_sources.yaml mode to 'full' at M7 (bundled with the WIDER class_caps + Roboflow slug changes).                                                                                                            |
