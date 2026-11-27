# Research notes: what the strongest THINGS-EEG2 retrieval papers do (read 27 Nov 2026)

Task: 200-way zero-shot EEG-to-image retrieval, intra-subject, test EEG averaged over 80 repetitions. Numbers are as REPORTED by the papers (preprints unless stated), not verified by us.

| Paper (arXiv) | Plain top-1 / top-5 | Main idea | Caveat noted while reading |
|---|---|---|---|
| RPA, Residual Patch-Token Adapter (2609.31698) | 95.4 +/- 0.1 / 99.8 | Frozen OpenCLIP ViT-H/14 layer-11 PATCH grid (16x16x1280), conv adapter, EEGProjection encoder, 0-700 ms | Layer chosen by highest TEST accuracy (their Appendix D); baselines not re-run |
| AttrEcho, Look Closer Align Better (2611.09123) | 94.8 / 99.7 | CLIP-RN50 regional features + HOG/texture(LBP,GLCM)/contour attributes, 9 regions, gated fusion | Preprocessing not described; CLIP-only 76.0 in their ablation |
| DTHA (2611.12798) | 92.4 / 99.5 | CLIP RN50 multi-blur + DINOv2-L blocks 0-14 (learned layer weights), spatial filter bank encoder, 250 epochs | Under review at a journal; single source |
| HCF / Multi-layer + CSLS (2609.02582) | 86.4 / 98.5 with CSLS (73.6-78.1 plain) | CLIP ViT-H/14 layers 10/12/14 + CSLS at test time | CSLS is transductive |
| Shallow Alignment (2601.21948) | 82.6 / 97.7 | Intermediate layer of frozen backbone + linear projector + MLP encoder | Layer sweep needed |
| MB2L (2605.04680) | 80.5 / 97.6 | Low-level (shallow ResNet) + high-level targets, adaptive blur | Internal inconsistencies between tables |
| LVBP / HVF (2602.07495) | 75.7 / 94.6 | RN50 + B32 + VAE hierarchy, MLP brain projection | |
| UBP | 50.9 / 79.7 | baseline in several papers | |
| Ours E18 (this project) | 80.0 +/- 4.1 / 97.1 +/- 1.7 | CLS tokens of mid DINOv2 layers + CLIP, MLP encoder, 100 Hz released data | one run, validation-gated |

Technique conclusions (from the papers' own ablations):
1. Image side matters most. RPA: CLS token 69.5 -> mean of patch tokens 77.0 -> conv adapter over the patch grid 95.4 (same layer). Permuting patch positions drops 95.4 -> 90.7, so spatial layout carries information.
2. Encoder matters less once targets are good. In RPA, EEG-Conformer 94.9 vs EEGProjection 95.4 (same targets); TSConv 87.9.
3. Intermediate layers beat final layers in every paper (we reproduced this in E15).
4. Low-level descriptors (HOG, LBP/GLCM, contour) added to CLIP: 76.0 -> 94.8 in AttrEcho (single paper).
5. CSLS adds roughly +8 to +12 in papers that report it; it needs the whole unlabelled test set (transductive).
6. Selection hygiene: RPA picked its layer on test accuracy; we select on validation only.
