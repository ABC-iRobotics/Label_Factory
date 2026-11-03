| Method | ADD-S AUC | ADD(-S) AUC | AR MSSD | AR MSPD | BOP AR | mIoU | Input | CAD | Training | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| Label Factory (ours) | 97.0 | 92.3 | 93.6 | 88.4 | 88.5 | 84.7 | RGB-D seq. | yes | no | this work |
| Label Factory (ours), low-support excluded | 12.8 | 11.7 | 13.4 | 12.8 |  | 8.6 | RGB-D seq. | yes | no | this work |
| SAM-6D single-frame (our run) | 83.2 | 76.0 | 67.2 | 62.7 |  | 71.7 | RGB-D | yes | no | this work |
| Label Factory, same targets | 96.8 | 91.3 | 92.9 | 88.5 |  | 81.9 | RGB-D | yes | no | this work |
| PoseCNN | 75.8 | 59.9 |  |  |  |  | RGB | no | yes | FFB6D table, arXiv 2103.02242 (no refinement) |
| DenseFusion | 91.2 | 82.9 |  |  |  |  | RGB-D | no | yes | FFB6D table, arXiv 2103.02242 (no refinement) |
| PVN3D | 95.5 | 91.8 |  |  |  |  | RGB-D | no | yes | FFB6D table, arXiv 2103.02242 |
| FFB6D | 96.6 | 92.7 |  |  |  |  | RGB-D | no | yes | arXiv 2103.02242 (no refinement) |
| GDR-Net | 91.6 | 84.3 |  |  |  |  | RGB | no | yes | CVPR 2021; ADD(-S) 84.3 or 84.4 in two summaries |
| CosyPose (single view) | 89.8 | 84.5 |  |  |  |  | RGB | yes | yes | arXiv 2008.08465 |
| CosyPose (5 views) | 93.4 |  |  |  |  |  | RGB multi-view | yes | yes | arXiv 2008.08465 (ADD-S only) |
| MegaPose |  |  |  |  | 65.3* |  | RGB(-D) | yes | no | *BOP AR; configuration unconfirmed, arXiv 2212.06870 |
| FoundationPose | 97.4 | 91.5 |  |  |  |  | RGB-D | yes | no | arXiv 2312.08344 (model-based setup) |
| SAM-6D |  |  |  |  | 77.6* |  | RGB-D | yes | no | *BOP AR; from one summary, unconfirmed, arXiv 2311.15707 |
