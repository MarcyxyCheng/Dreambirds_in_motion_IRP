## Meeting (25 June 2025)

**Present**: Cheng Marceline (CM, student), Pain Christopher (PC, supervisor), James Coupe (JC, supervisor), Li Yueyan (LY, advisor)

**Key points discussed**:
- Reviewed progress on surreal bird video generation project using animated fusion base model with ControlNet for edge detection.
- CM presented objective: generate <22 second surreal videos from single bird image input with bird appearance but non-bird animal motion patterns.
- Demonstrated test results showing parrot input transformed using pose guidance - bird follows generated poses while maintaining bird appearance.
- CM showed pipeline using Caltech bird dataset (170 species) for training custom LoRA model to learn bird appearances from different angles and movements.
  
**Feedback received**:
- PC observed unusual creature appearance in generated video - could identify wings and leg-like features but overall result appeared distorted.
- LY suggested implementing control weight adjustment for motion signals to allow fine-tuning between strict vs. loose motion following.
- LY explained that motion and depth map conditioning uses same methodology as geometry input for video generation models.
- CM acknowledged need to adjust control weights - previously used heavy scaling on depth and edge which limited motion to static poses.
  
**Work plan before next meeting**:
- Continue debugging LoRA training process (currently experiencing 3+ hour training times).
- Implement pose control weight adjustment to better balance motion transfer vs. bird appearance preservation.
- Research and apply LY's suggestions for controlling motion signal weights in video generation pipeline.
- Focus on transferring animal motion patterns to bird structure while maintaining visual bird characteristics.
