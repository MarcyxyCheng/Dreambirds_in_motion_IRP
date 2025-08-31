## Meeting (13 August 2025)

**Present:** Marceline Cheng (MC, student), Christopher Pain (CP), James Coupe (JC)

**Key points discussed:**

- **Extension application status:** MC applying for medical extension due to health condition. Has contacted James and Marion from department, currently waiting for medical evidence documentation from GP to complete extension form.

- **Extension uncertainty:** Duration of extension and whether viva will also be extended remains unclear - department cannot provide definitive answers until application is processed.

- **Skeleton detection improvements:** Fixed wing detection issues that previously caused confusion between right and left wings. Added computational wing detection to better differentiate wing positions for next processing stage.

- **Motion generation adjustments:** Made minor modifications to second stage:
        - Added additional postures to training data
        - Loosened constraints in skeleton sequence generation
        - Amplified skeleton movements to make motion more obvious and visible

- **Video generation experimentation:** Third phase testing ongoing with addition of:
        - Background removal techniques
        - Latent consistency improvements
        - Results still not optimal

- **Frame consistency challenges:** Generated frames lack sequential relationship and temporal consistency. Current output produces individual frames rather than coherent video sequences.

**Technical challenges identified:**

- **Temporal consistency:** Major issue with frame-to-frame consistency in video generation. Frames appear disconnected rather than forming smooth motion sequences.

- **Sequential relationship:** Generated frames don't maintain proper temporal relationships, making them unsuitable as true video frames.

- **Quality concerns:** Current results described as "not very good" and "below average" compared to direct generation from other models.

**Feedback received:**

- **JC:** Highlighted that current frames lack sequential relationship and don't function as proper video frames. Suggested focusing on filling gaps between frames to improve temporal consistency.

- **JC:** Noted that skeletal model should help fill missing information between frames to enable proper sequence generation from single images.

- **CP:** Encouraged continuing work if health permits, acknowledging medical condition impacts.

**Current status:**

- Report writing has begun despite limited progress due to health issues
- Code adjustments ongoing across all three pipeline stages
- Extension application in progress, awaiting medical documentation
- Frame generation working but lacks temporal consistency

**Immediate priorities:**

- Complete medical extension application once GP documentation received
- Address temporal consistency issues in video generation
- Explore methods to improve frame-to-frame relationships
- Continue report writing within health limitations

**Missing elements identified:**

- Proper temporal consistency mechanism between generated frames
- Sequential relationship preservation in video generation stage
- Improved frame interpolation or gap-filling methods