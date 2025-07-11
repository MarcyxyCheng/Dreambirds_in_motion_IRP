**Meeting (Date:02/07/2025)**

**Present:** Cheng Marceline (CM, student), Pain Christopher (PC, supervisor), Joshi Aniket C (AC)

**Key points discussed:** 
- CM reported progress on skeleton mapping workflow: successfully found mammal skeleton dataset with 17 keypoints and working on mapping to 8-point bird skeleton system. 
- Technical challenge with dataset access: 50GB dataset only available through OneDrive with download issues, forcing CM to work with extracted JSON files containing joint positions. 
- Video generation problems: Current skeleton-to-video pipeline producing blurry and bizarre results despite successful skeleton pose generation. 
- CM demonstrated understanding of the mapping process and showed that skeleton visualization step is working correctly.
  
**Feedback received:** 
- PC requested demonstration of current results but CM explained only having skeleton pictures and poor-quality video outputs. 
- AC suggested using HPC interface in VS Code for better file management and direct file transfer capabilities. 
- AC confirmed expectation for skeleton showcase in next week's meeting.

**Technical issues identified:** 
- Dataset download limitations: Unable to access full 50GB mammal dataset, working with JSON extracts only. 
- Video generation pipeline failure: Skeleton-to-video conversion producing unsatisfactory blurry results. 
- Workflow inefficiency: Using local terminals for HPC transfers instead of integrated VS Code interface.
  
**Work plan before next meeting:** 
- Prioritize solving video generation issues through step-by-step debugging of image reference + skeleton pipeline. 
- Focus on obtaining visualized results from current skeleton mapping approach. 
- Postpone feature preservation work until skeleton generation is stable and producing acceptable results. 
- Consider switching to alternative datasets (Hugging Face, Kaggle) if current dataset download issues persist. 
- Explore improved HPC workflow using VS Code integrated interface for better file management.