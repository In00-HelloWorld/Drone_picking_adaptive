# Drone_picking_adaptive
This project leverages images captured by an aerial drone and applies a Visual Foundation Model (VFM) to extract semantically meaningful numerical features from the environme. These features are then incorporated into an adaptive control framework to enhance controller performance and robustness. In short, the perception pipeline (VFM-based feature extraction) informs the control pipeline (adaptive controller), enabling data-driven adaptation to changing operational conditions.

## Acknowledgments & Attributions

This project is based on the **Genesis** physics engine and incorporates modifications to example code provided in the Genesis repository:
- Genesis Physics Engine: <https://github.com/Genesis-Embodied-AI/Genesis>

In addition, portions of the workflow rely on **LLaVA-OneVision 1.5** for multimodal model components and tooling:
- LLaVA-OneVision 1.5: <https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5>

### URDF References
URDF assets and configurations in this repository were prepared with reference to **Gazebo** URDF conventions and examples.
