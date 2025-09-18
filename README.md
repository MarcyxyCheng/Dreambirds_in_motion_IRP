<!-- # IRP Repository

This is your IRP repository. Please follow the guidelines below to ensure all expectations are met. For further information, please refer to the [IRP webpage](https://ese-msc.github.io/irp).

Deleting or modifying the pre-existing GitHub Actions workflows or the directory structure in this repository is strictly prohibited.

## Expectations & Instructions

| Expectation                    | Instructions                                                                                           |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- |
| **Submit all IRP deliverables**   | See [deliverables/README.md](deliverables/README.md)                                                    |
| **Regularly commit**              | Follow the [IRP academic integrity rules and expectations](https://ese-msc.github.io/irp/academic-integrity.html) |
| **Regularly update `logbook.md`** | See [logbook/README.md](logbook/README.md)                                                              |
| **Keep `title.toml` up to date**  | See [title/README.md](title/README.md)                                                                  |

## Inactivity Warnings

Scheduled workflows will periodically check whether `logbook.md` has been updated recently on `main` as well as whether **regular commits were made to the repository (to any branch)**. If an inactivity is detected, a warning will be automatically raised as an issue in this repository.

---

**Tip:** After you have familiarised yourself with this repository, you may delete the content of this file and replace it with your project-specific information. -->



# Dreambirds in Motion: AI-Driven Surreal Bird Video Generation

## 1. Introduction
This project explores whether skeleton-driven generative methods can produce **temporally coherent** and **stylistically consistent** bird motion videos.  
We built a three-stage pipeline:
1. **HRNet** for bird keypoint detection (CUB-15 skeleton).
2. **Motion Diffusion Model (MDM)** for generating pose sequences.
3. **AnimateDiff + ControlNet** for rendering skeleton-guided surreal bird videos.

The aim is to combine scientific modelling of motion with artistic freedom, inspired by the *Birds of the British Empire* project.

---

## 2. Results
We demonstrate controllable and surreal bird motion videos:
- **Motion fidelity**: skeleton-following wing flaps, glides, and landings.
- **Temporal smoothness**: no flicker across frames.
- **Surreal preservation**: artistic prompts (e.g., “purple birds with crystal wings”) remain consistent.  

🖌️ **Total pipeline workflow**

![Pipeline workflow](demo/Workflow.png)

🖼️ <b>Demo frames</b>:

<!-- ![Sample frame 1](demo/frame1.png)
![Sample frame 2](demo/frame2.png) -->
<div align="center">
  <img src="demo/frame1.png" alt="Sample frame 1" width="700"/>
</div>
<div align="center">
  <img src="demo/frame2.png" alt="Sample frame 2" width="700"/>
</div>


<!-- 📹 **Demo Videos**:  
![Demo video](demo/video1.gif)
![Demo video](demo/video2.gif)
![Demo video](demo/video3.gif) -->

📷 <b>Demo Videos</b>:

<p float="left">
  <img src="demo/video1.gif" alt="Demo video 1" width="330"/>
  <img src="demo/video2.gif" alt="Demo video 2" width="330"/>
  <img src="demo/video3.gif" alt="Demo video 3" width="330"/>
</p>

🎨 **Screen shot of the WebUI I built for video quality evaluation**
<div align="center">
  <img src="demo/webUI.png" alt="WebUI" width="600"/>
</div>

---

## 3. Repository Structure
```text
Dreambirds_in_Motion_IRP/
│── Full_Codes_for_IRP/         # Main training and inference scripts
│── MOS_Evaluation_WebUI/       # Web interface for human evaluation
│── Report_LatexFormat/         # LaTeX files for academic report
│── Workflow.png                # Project workflow diagram
│── Project_Report.pdf          # Full project report
│── IRP_vivas.pptx              # Presentation slides
│── README.md                   # This file
```


## 4. References

This project builds upon the following open-source models and libraries:

```bibtex
@article{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  journal={CVPR},
  year={2019}
}

@article{tevet2022motion,
  title={Motion Diffusion Model for Human Motion Generation},
  author={Tevet, Guy and others},
  journal={arXiv preprint arXiv:2209.14916},
  year={2022}
}

@article{guo2023animatediff,
  title={AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning},
  author={Guo, Shuai and others},
  journal={arXiv preprint arXiv:2307.04725},
  year={2023}
}

@article{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Zhang, Lvmin and others},
  journal={arXiv preprint arXiv:2302.05543},
  year={2023}
}
```

We gratefully acknowledge the authors of these works for making their code and models publicly available.



