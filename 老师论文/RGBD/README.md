# Teacher Paper Reference: SA-Gate RGB-D Segmentation

This folder is for notes about the teacher/reference paper:

`Bi-directional Cross-Modality Feature Propagation with Separation-and-Aggregation Gate for RGB-D Semantic Segmentation`

Official code:

`https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch`

## How It Was Used In This Project

The official SA-Gate code was inspected locally, especially:

- `model/SA-Gate.nyu/net_util.py`
- `FilterLayer`
- `FSP`
- `SAGate`

The attempted v1.1 adaptation replaced c1/c2 `GatedFusion` with a
stage-output version of SA-Gate:

- Feature Separation: bidirectional channel recalibration
- Feature Aggregation: two spatial gates with softmax

Result:

- v1.1 SA-Gate c1/c2 5 runs averaged about `0.4801`
- v1.0 stable baseline 10 runs averaged about `0.4828`

Conclusion:

The SA-Gate idea is meaningful in the original paper because recalibrated
features continue through later backbone stages. In this project, the adaptation
was applied only after pretrained stage outputs, so it did not produce a stable
improvement over v1.0.

The full local download of the official repository and PDF is intentionally not
tracked in Git to avoid pushing third-party generated files, example data, and
compiled artifacts.
