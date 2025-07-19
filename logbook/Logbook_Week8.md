## Meeting (19 July 2025)

**Present:** Marceline Cheng (MC, student), Christopher Pain (CP, supervisor), Claire E. Heaney (CH, supervisor),  Yueyan Li (YL, Technical Advisor), Aniket C. Joshi (AJ, Technical Advisor)

**Key points discussed:**

- Reviewed progress on multi-stage bird motion generation pipeline using diffusion models and skeleton-based motion synthesis.
- MC presented completed Phase 1 (skeleton detection using HRNet with 17-point COCO structure) and Phase 2 (motion generation using Motion Diffusion Model with mathematical wing simulation).
- Discussed technical challenges including validation failures during training, poor motion quality in generated sequences, and integration between motion generation and video synthesis stages.
- Explained choice of 17-point skeleton structure to maintain compatibility with APT36K mammal motion dataset for cross-species motion transfer.

**Feedback received:**

- Supervisors questioned motion encoding methods and integration strategy between skeleton motion and video generation (will use AnimateDiff with ControlNet).
- YL suggested testing motion-to-video pipeline step-by-step rather than training all stages simultaneously to validate each component works independently.
- CH advised starting report writing in early August, creating outline now, and not leaving all writing until the end.
- Team emphasized documenting all experimental attempts comprehensively for the final report.
- CP mentioned potential collaboration opportunities with AI for Urban Health Consortium website.

**Work plan before next meeting:**

- Complete Stage 3 and Stage 4 to achieve end-to-end pipeline with moving bird videos.
- Test whether existing motion data can successfully drive video generation before focusing on improving motion quality.
- Investigate two bird video datasets with pre-annotated keypoints as alternative to mathematical simulation.
- Begin comprehensive documentation of all experimental approaches and results for report preparation.