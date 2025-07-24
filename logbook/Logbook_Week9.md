**Meeting (Date:23/07/2025)**

**Present:** Cheng Marceline (CM, student), Pain Christopher (PC, supervisor), Joshi Aniket C (AC), Heaney Claire E (HC), James Coupe (JC)

**Key points discussed:** 
- Adjusted dataset from 17‑keypoint format to CUB’s built‑in 15‑keypoint format
- Wrote a script to connect skeleton joints (hard/soft connections)
- Ran a simplified proof‑of‑concept on CPU; preliminary integration with HRNet
- Created mathematical bird‑motion models (take‑off and in‑flight) and trained for a few epochs
  
**Feedback received:** 
- GPU access on the Imperial HPC is unreliable; advised to use the ESC cluster instead
- Suggestion: use an API (e.g. ChatGPT) to generate sequential frames with skeleton annotations
- Compare quality of API‑generated data vs. homemade models; fine‑tune lightweight versions if needed
- Consider integrating audio metadata (e.g. narration or prompts) into the final video

**Technical Issues** 
- HPC GPU Access: Imperial HPC frequently queues jobs, especially on Tuesdays; ESC cluster suggested but environment not yet migrated
- Local Resources: Laptop can only handle lightweight or very small‑scale training, insufficient for full video‑quality models
- Data Pipeline Interruptions: Delays in assessing results yesterday/today due to cluster access problems
- Model Deployment: Difficulty re‑deploying full models locally; maintaining simplified backups for urgent tests
  
**Work plan before next meeting:** 
1. Data & Models
   - Finalize the 15‑keypoint dataset and re‑run on all three model pipelines (including HRNet and mathematical models)
   - Ensure temporal consistency and feature preservation in video outputs
2. API Experimentation
    - Use ChatGPT (or similar) to generate 10–20 consecutive bird‑skeleton frames
    - Measure generation time (<1 week) and evaluate visual quality against current models
3. Audio Integration (Stretch Goal)
    - Explore adding short audio snippets (e.g. bird calls or descriptive text) to the generated videos
4. HPC Strategy
   - Migrate environments and pre‑download all necessary models onto ESC cluster to avoid queuing
   - Maintain lightweight local fall‑backs for urgent testing