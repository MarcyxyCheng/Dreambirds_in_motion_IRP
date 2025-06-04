**Meeting (04/06/2025)**

**Present**: 
- Marceline Cheng (MC, student), 
- James Coupe (JC, supervisor), 
- Christopher Pain (CP, supervisor), 
- Aniket  Joshi (ACJ, PhD&technical advisor), 
- Yueyan Li (YL, PhD&technical advisor)
  
**Key points discussed**:
  - Reviewed MC's progress on Sora-style video generation research, including analysis of state-of-the-art models with two main architectures: standard diffusion with UNet and transformer-based diffusion.
  - MC successfully implemented Animate Diffusion on Stable Web UI, creating a workflow from text-to-image generation of surreal birds to video generation using generated images as starting frames.
  - Discussed input methodology: decided to start with single image input rather than multiple images, beginning with normal bird images before progressing to surreal variants.
  - JC highlighted the physics challenge of animating surreal birds (e.g., multi-headed or multi-winged) that exceed natural skeletal frameworks and movement patterns.
  - Technical architecture discussion concluded that fine-tuning pre-trained models is preferable to training from scratch due to massive data requirements (300TB+).
  
**Feedback received**:
  - ACJ recommended a progressive experimental approach: normal→normal, surreal→analysis, normal→surreal, surreal→surreal.
  - ACJ advised investigating feature preservation papers, particularly physics-oriented models like flow simulations that deal with boundary layer preservation.
  - YL strongly recommended fine-tuning existing pre-trained video/image generation models rather than training from scratch, suggesting LoRA for architectural modifications.
  - CP and team emphasized prioritizing HPC access due to VRAM requirements and hardware limitations on personal laptop.
  - Team advised focusing on open-source solutions and being mindful of GPU memory constraints when selecting model sizes.
  - ACJ stressed the importance of coordinating with Ziqi's work to ensure model synchronization between image and video generation components.
  
**Work plan before next meeting**:
  - Priority: Set up HPC access and migrate from laptop-based development to high-performance computing environment.
  - Complete research proposal for Friday submission, with team review scheduled for Wednesday.
  - Finalize choice between the two downloaded bird motion datasets and begin preprocessing.
  - Establish weekly coordination meetings with Ziqi to ensure synchronized model development.
  - Investigate publicly available pre-trained diffusion models suitable for fine-tuning on bird video generation task.