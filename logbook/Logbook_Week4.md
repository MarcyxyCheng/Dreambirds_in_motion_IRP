# An example of a logbook entry

## Meeting (18 June 2025)

**Present:** Cheng Marceline (CM, student), Pain Christopher (PC, supervisor), james.coupe (JC,supervisor), Li Yueyan (LY, PhD & advisor), Joshi Aniket C (AC, PhD & advisor) 

**Key points discussed:**

- Reviewed current progress on AnimateDiff implementation with ControlNet for bird video generation using motion adapters.

- CM demonstrated generated videos showing bird animation with depth and edge detection controls, though motion is currently limited to head turns and body movements.

- Attempted upgrade to Stable Diffusion XL failed, reverted to working system with standard Stable Diffusion.

- Discussed implementation of pose ControlNet to enable anatomical control of bird structure (wing-body connections, skeletal understanding).

**Feedback received:**

- AC advised switching from Imperial HPC cluster to departmental cluster to reduce queue times and improve development efficiency.

- Supervisor recommended focusing on getting current pose system operational before exploring additional ControlNet features.

- JC suggested using generative AI to create skeletal references for birds with extra appendages (multiple wings/heads) when real reference images are unavailable.

- Team emphasized the importance of solving current dimensional issues in code before proceeding to more complex features.

- AC noted that using ControlNet extensions reduces training requirements compared to custom LoRA approaches.

**Work plan before next meeting:**

- Resolve dimensional issues causing code failures and switch to departmental HPC cluster for faster iteration.

- Implement pose ControlNet system to teach bird anatomical structure and enable wing movement control.

- Test using drawing references instead of requiring real bird skeleton photographs for pose training.

- Explore generating skeletal structures for fantastical bird designs with multiple appendages.

- Send demonstration video via email to show current progress and limitations.