---
layout: page
title: NVIDIA Isaac Lab Contributions
description: My open-source contributions to the NVIDIA Isaac Lab robotics simulation framework for massively parallel reinforcement learning and imitation learning. I added distributed hyperparameter tuning, image feature extraction with pretrained frozen models, and standardized environments for vision-based reinforcement learning.
img: assets/img/isaac_lab_thumb.png
importance: 1
category: open-source
--- 

Here are some highlights of my work on Isaac Lab:
- Adding distributed hyperparameter tuning to Isaac Lab that can be seamlessly run on a cluster or on a single machine ([documentation](https://isaac-sim.github.io/IsaacLab/main/source/features/ray.html), [code](https://github.com/isaac-sim/IsaacLab/tree/b1133e0591c2ef3a788c1ca148bb25a3f42562a9/scripts/reinforcement_learning/ray),[YouTube video covering the functionality](https://youtu.be/z7MDgSga2Ho?si=UgH5XngHa2uUu3lN)).

- Adding support for [image feature extraction with pretrained frozen models](https://github.com/isaac-sim/IsaacLab/pull/1191).

- Adding [standardized vision-based reinforcement learning environments](https://github.com/isaac-sim/IsaacLab/pull/995).

- Adding a [tool to check the maximum amount of cameras that can be simulated on a GPU](https://isaac-sim.github.io/IsaacLab/main/source/how-to/estimate_how_many_cameras_can_run.html).

- Supporting the community by helping resolve over 20 issues related to Isaac Lab([comments on issues by either @garylvov or @glvov-bdai](https://github.com/isaac-sim/IsaacLab/issues?q=is%3Aissue%20involves%3Agarylvov%20OR%20involves%3Aglvov-bdai), [comments on discussions by @garylvov](https://github.com/isaac-sim/IsaacLab/discussions?discussions_q=commenter%3Agarylvov), [comments on discussions by @glvov-bdai](https://github.com/isaac-sim/IsaacLab/discussions?discussions_q=commenter%3Aglvov-bdai)).