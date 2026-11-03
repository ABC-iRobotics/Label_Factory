| Method | Input | CAD | Training on YCB-V | ADD-S AUC | ADD(-S) AUC | BOP AR | Source |
|---|---|---|---|---|---|---|---|
| PoseCNN | RGB | no | yes | 75.8 | 59.9 |  | FFB6D table, arXiv 2103.02242 (no refinement) |
| DenseFusion | RGB-D | no | yes | 91.2 | 82.9 |  | FFB6D table, arXiv 2103.02242 (no refinement) |
| PVN3D | RGB-D | no | yes | 95.5 | 91.8 |  | FFB6D table, arXiv 2103.02242 |
| FFB6D | RGB-D | no | yes | 96.6 | 92.7 |  | arXiv 2103.02242 (no refinement) |
| GDR-Net | RGB | no | yes | 91.6 | 84.3 |  | CVPR 2021; ADD(-S) 84.3 or 84.4 in two summaries |
| CosyPose (single view) | RGB | yes | yes | 89.8 | 84.5 |  | arXiv 2008.08465 |
| CosyPose (5 views) | RGB multi-view | yes | yes | 93.4 |  |  | arXiv 2008.08465 (ADD-S only) |
| MegaPose | RGB(-D) | yes | no |  |  | 65.3* | *BOP AR; configuration unconfirmed, arXiv 2212.06870 |
| FoundationPose | RGB-D | yes | no | 97.4 | 91.5 |  | arXiv 2312.08344 (model-based setup) |
| SAM-6D | RGB-D | yes | no |  |  | 77.6* | *BOP AR; from one summary, unconfirmed, arXiv 2311.15707 |
