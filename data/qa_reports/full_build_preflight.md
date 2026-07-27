# Full-Mode Build Preflight (FB1–FB6)

*Generated: 2026-07-27T16:48:14Z*

- **verdict:** WARN
- **generated_at:** 2026-07-27T16:48:14Z

## Gates

| Gate | Name               | Status | Details                                                                                                                                                                                                                                                  |
| ---- | ------------------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FB1  | disk space         | pass   | 216.6 GB free (≥ 150 GB)                                                                                                                                                                                                                                 |
| FB2  | dvc remote         | pass   | 'localstore' at C:\dvc_remote (216.6 GB free)                                                                                                                                                                                                            |
| FB3  | roboflow readiness | warn   | sources.roboflow.datasets is empty — human track H-B (slug selection + per-slug license review BEFORE download) has not landed; public coverage for medicine_bottle, charger, wire, gas_cylinder stays blocked.                                          |
| FB4  | gpu                | pass   | CUDA device: NVIDIA GeForce RTX 3050 6GB Laptop GPU                                                                                                                                                                                                      |
| FB5  | onedrive hazard    | warn   | risk R34: the working tree is under OneDrive (C:\Users\haris\OneDrive\Desktop\YOLO_V1) — data/ grows to tens of GB during the full build; pause OneDrive sync for its duration; the DVC cache is already off OneDrive (C:\dvc_cache) — no action needed. |
| FB6  | acquisition mode   | pass   | mode: full                                                                                                                                                                                                                                               |
