# BEVFusion Enhancement & Visualization

This repository is built on top of the **BEVFusion baseline** and provides enhanced tools for **visualizing perception and detection results**.

It supports:
- **Single-model visualization** — for either the BEVFusion baseline or an enhanced model
- **Two-model comparison** — side-by-side visualization of outputs from two different models

---

## Input Data

The input to the visualization pipeline is a **recorded ROS bag file**.  
The overall processing and visualization structure is illustrated in the diagram below:

<!-- Insert architecture / pipeline diagram here -->

---

## Code Structure

All visualization-related logic is located in:

```text
tools/my_tools/
---
## Visualization Configuration
Visualization behavior and layout are configurable through:
```infer_cfg.yaml
