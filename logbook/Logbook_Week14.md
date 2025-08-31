**Meeting (Date:27/08/2025)**

**Present:** Marceline Cheng (MC, student), Claire Heaney (CH, supervisor), Christopher Pain (CP, supervisor), Yueyan Li (YL), Aniket

**Key points discussed:**

- **Project extension confirmed:** Report deadline extended to September 3rd (5-day extension approved by James). Viva remains at original scheduled time.
- **Skeleton motion generation progress:** MC has developed a two-stage approach for bird motion generation:
    - Stage 1: Static skeleton image generation from bird images
    - Stage 2: Converting static skeletons to motion sequences using mathematical modeling

- **Control mechanisms implemented:** Two methods for controlling skeleton sequence generation:
        - Input static frame from first stage as starting point
        - Categorized bird motions into 6 types (taking off, colliding, hovering, etc.) with labeled training data

- **Frame control parameters:** Model generates 64 frames with adjustable strictness - can set first 10 frames to strictly follow initial input, then transition to chosen motion category.

- **Human vs. bird skeleton mapping challenge:** OpenPose control net trained on human poses doesn't map well to bird anatomy. MC experimented with:
        - Converting 15-point bird skeleton to 5-point human-like structure
        - 9-point version with inference-added arm points
        - Sacrificing leg detail since bird legs positioned differently than human legs

- **Video generation model testing:** Three approaches tried:
        - Stable Diffusion (image-to-image, single frame generation)
        - Stable Video Diffusion (good video quality but limited control net integration)
        - AnimateDiff (current focus with control net integration)

- **DeepLabCut integration attempt:** Manual keypoint annotation on ~150 bird video frames, but encountered environment setup conflicts across local PC, HPC, and Colab.

**Feedback received:**

- **CH:** Suggested focusing on report writing while GPU access limited. Noted interesting potential for future flocking/swarming applications using physics-based surrogate models.

- **CP:** Highlighted potential for exploring bird flocking behaviors and integrating with surrogate models (UNS diffusion, Rapids model) for physics-based bird flight interactions.

- **CH:** Advised against including detailed loss function notation for all models in report - would be too much notation without substantial analysis benefit.

- **Team:** Discussed evaluation challenges for video generation - human rating system proposed but acknowledged as not entirely reliable metric.

**Current challenges:**

- Limited GPU access due to high demand affecting experimentation pace
- Need for reliable numerical evaluation metrics beyond subjective human rating
- Environment setup issues with DeepLabCut for improved training data
- Control net designed for human poses requires adaptation for bird anatomy

**Work plan before next meeting:**

- Continue report writing while GPU access limited
- Set up alternative computing environment (Colab) for continued experimentation
- Generate comparison results using 5-point vs 9-point skeleton approaches
- Test mirrored/flipped skeleton inputs to address left-right understanding issues
- Develop numerical evaluation metrics for video generation quality
- Send draft report to Yuyan and Aniket for additional feedback

**Technical notes:**

- AnimateDiff with control net successfully working after previous week's technical issues
- Current model generates 16-frame sequences (reduced from 64 for testing)
- Control methods: prompt-based appearance control vs. original image input
- Parameter variations tested: edge, depth, and pose control weights