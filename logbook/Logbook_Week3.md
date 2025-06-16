**Meeting (11 June 2025)**

**Present**: Cheng Marceline (CM, student),Christopher Pain (CP, supervisor) , James Coupe (JC, supervisor)

**Key points discussed**:
- CM reported progress on HPC environment setup and AnimateDiffusion deployment, plus ControlNet and LoRA integration for detailed generation control
- Team approaching research plan deadline with focus on two technical challenges: feature preservation and temporal consistency
- JC explained conceptual framework using Victorian sideshow references to explore "normal vs. abnormal" distinctions rather than typical surrealism approaches
- Discussion of three-LoRA training strategy: (1) bird details using CUB-200 dataset, (2) surreal structures using Pokémon/mythological creatures, (3) Victorian artistic style
- Technical setup using Stable Diffusion XL on HPC with local testing on SD 2.1

**Feedback received**:
  - JC emphasized avoiding making birds "look like art" - goal is photorealistic birds with impossible anatomy that viewers believe are real
  - JC provided prompts based on Jane Eyre characters described as birds, formatted like field guide entries
  - JC clarified Victorian sideshow materials are about understanding societal perceptions of "otherness" to inform bird generation methodology
  - JC approved current plan and suggested sparse zoo/aviary environments for final bird presentations
  - CP suggested using ChatGPT API for dataset generation and exploring better text embedding models than SD 1.5's default

**Work plan before next meeting**:
  - CM to continue training LoRA models and test integration
  - Explore ChatGPT API for custom dataset generation if suitable existing datasets unavailable
  - Implement field guide format prompting structure suggested by JC
  - Research and test improved text embedding models for better prompt understanding
  - Finalize research plan document before deadline