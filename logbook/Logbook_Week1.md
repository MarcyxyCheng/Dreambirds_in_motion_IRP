## Week 1 Meeting (29 May 2025)

**Present**: 
- Marceline Cheng (MC, student)
- Christopher Pain (CP, main supervisor)
- James Coupe (JC, Second supervisor)
- Claire Heaney (CH, supervisor)
- Ziqi Yue (ZY, peer - text-to-image)
- Yueyan Li (YL, PhD & technical advisor)
- Aniket Joshi (AJ, PhD & technical advisor)
  
**Key points discussed**:
  - Project overview: Creating AI-generated videos of surreal birds from static images as part of a text→image→video pipeline
  - My specific role focuses on image-to-video generation, preserving surreal characteristics of AI-generated birds
  - Exhibition context: Birds will be displayed in zoo-like vitrines/cages, requiring natural movement behaviors
  - Technical challenges: temporal consistency, bird-specific movement patterns, computational limitations
  - Coordination with Ziqi's text-to-image work - need to decide on integration level vs. separate development
  
**Feedback received**:
  - JC emphasized the artistic vision: birds should move as if "living their life" in captivity, not just generic animation
  - CH suggested starting with Sora tool exploration, I haven't used it before
  - AJ recommended working collaboratively with Ziqi using similar repositories/architectures rather than completely separate approaches
  - YL offered access to existing architectures but noted they're designed for physics problems, may not directly apply
  - JC raised important questions about data approach: single image input vs. multiple images, need for template bird movement videos
  
**Work plan before next meeting**:
  - Research Sora and other state-of-the-art image-to-video generation tools
  - Investigate existing bird movement datasets and video sources
  - Connect with Ziqi to understand image output format and coordinate technical approaches
  - Get GitHub access from Yueyan Li and Aniket Joshi to explore existing architectures
  - Explore computational requirements and HPC limitations
  - Research bird-specific movement patterns and behaviors for realistic animation
  - Clarify whether to focus on single-image-to-video or multi-image approaches
  
**Open questions for next meeting**:
- How long should video sequences be? (JC mentioned 5+ seconds as a target)
- What specific bird behaviors should be prioritized? (wing movement, head turns, perching, etc.)
- How to handle temporal consistency limitations in current video generation models?
- Should we create our own bird movement templates or rely entirely on existing models?