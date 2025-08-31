**Meeting (20 August 2025)**

**Present:** Marceline Cheng (MC, student), Christopher Pain (CP), Claire Heaney (CH), Yueyan Li (YL), James Coupe (JC), Aniket

**Key points discussed:**

- **Model switching decision:** MC switched from AnimateDiff to Stable Video Diffusion after fine-tuning challenges with AnimateDiff control nets. SVD provided expected results despite using "terrible skeleton" input initially.

- **Stable Video Diffusion results:** Successfully generated bird gliding videos using skeleton control and prompt combination. Current limitations: 25 frames maximum, 512×412 pixel resolution, 5-8 fps on Google Colab with V100 GPU.

- **Surreal bird video generation:** Tested on Ziki's surreal bird dataset, generating videos of birds with head/neck rotation movements. Results show model captures motion concepts but alignment between skeleton and video not 100% accurate.

- **Image-to-prompt pipeline enhancement:** Added automatic prompt generation stage that creates short descriptions (20-30 words) of both bird appearance and motion to improve generation consistency and reduce conflicts between skeleton and text prompts.

- **Three-model pipeline structure:**
        - Model 1: HR-Net for keypoint detection (trained on CUB-200 with manual skeleton annotations)
        - Model 2: Motion Diffusion Model (MDM) for skeleton sequence generation (trained on custom mathematical modeling dataset of ~4000 movements)
        - Model 3: Video generation (using pre-trained SVD with skeleton control)

- **Training data creation:** Created 35 different bird poses, combined into ~1000 movements, then used data augmentation (rotation, size variations) to reach ~4000 training samples.

**Feedback received:**

- **YL:** Questioned data alignment between skeleton and video for training. Clarified that MC doesn't have bird videos with corresponding skeleton annotations, which limits fine-tuning capabilities for video generation stage.

- **YL:** Suggested exploring MimicMotion model (which uses SVD backbone) and offered to discuss state-of-the-art video generation models separately. Noted MimicMotion requires human pose videos, not suitable for birds.

- **CP:** Suggested potential future applications for bird flocking and swarming behaviors using surrogate models for physics-based interactions.

- **CH:** Advised focusing on best and most consistent results in final report due to 15-figure limit and ~5000-word constraint. Suggested briefly mentioning other approaches tested if space permits.

**Technical challenges identified:**

- **Ground truth absence:** No bird video datasets with corresponding skeleton annotations available, limiting supervised training for video generation stage.

- **Model evaluation:** Using general video generation metrics rather than task-specific ground truth comparisons due to lack of reference data.

- **Skeleton-video alignment:** Acknowledged that skeleton and generated video alignment is not 100% accurate, particularly for certain viewing angles.

- **AnimateDiff integration:** Control net integration with AnimateDiff requires extensive code structure conversion, resulting in inconsistent outputs.

**Current status:**

- Two models (keypoint detection and motion generation) fully trained with recorded loss functions and MSE metrics
- Video generation stage using pre-trained models with skeleton and prompt control
- Testing different approaches: Stable Diffusion, Stable Video Diffusion, and AnimateDiff

**Work plan before next meeting:**

- Focus on final report writing with emphasis on most consistent results (SVD approach)
- Include numerical evaluation metrics from first two trained models
- Consider including brief mention of other approaches tested within figure/word limits
- Follow up with YL regarding state-of-the-art video generation models discussion