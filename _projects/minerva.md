---
layout: page
title: My Deep Learning Rig (Minerva)
description: My beloved  deep learning workstation
img: assets/img/minerva/quad_3090_pc.jpg
importance: 4
category: hardware
--- 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/minerva/cased.jpg" title="Results" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/minerva/gpu_out.jpg" title="Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/minerva/screenfetch-v1.png" title="Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

View the parts list [here](https://docs.google.com/spreadsheets/d/1JUgSV6aVaqvW5jET3J5IoqAC02ErcvO4mrwiM4GXQ-0/edit?usp=sharing). Tim Dettmers' [deep learning hardware blog](https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/) and [gpu guide](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) were instrumental in helping me build this rig.

My friends often ask, "Why would you possibly need 4 GPUs?" or "Why would you spend so much on a computer?"

The short answer is that it is an investment to be able to train and run machine learning models as fast as possible locally to encourage me to experiment frequently while owning what I create.

Most modern models support distributed data parallelism, or other forms of parallelism, which allows for training on multiple GPUs while recieving a speedup proportional to the number of GPUs.

My system runs ``2x NVIDIA 3090 TI 24G`` and ``2x NVIDIA 3090 24G`` (96GB VRAM total) at a GPU clock ```~1750Mhz - 1800Mhz``` speed limit and ```300W``` limit per card (limits are for training stability to prevent triggering PSU safety stops). 
Under full load, the system peaks at about ```1550W``` total power draw and ```75C``` internal temperature. Minerva also features a ```16-core 32-thread AMD Threadripper 7955wx``` (```3.5Mhz``` base, ```4.5Mhz``` boost), and ```128G```  DDR5 RAM.


# What sort of workloads can Minvera run?

For all of the following workloads, I set up the GPUs according [to my guide](https://github.com/garylvov/dev_env/blob/main/setup_scripts/nvidia/README.md). Having the NVIDIA Container Toolkit is 
essential for the majority of my desired workloads.

## Case Study I: NVIDIA Isaac Lab Multi-GPU training

I ran multi-gpu training to benchmark against the [Isaac Lab perfomance benchmarks](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html), where
a Single-Node, ```4x NVIDIA L40``` GPU, CPU: Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GHz achieves 390,000 frames per second for environment step (with 8192 environments per GPU), inference and train in the ```Isaac-Repose-Cube-Shadow-Direct-v0``` RL environment. 
In this environment, a robot hand learns to position a cube in the desired orientation.

<details class="highlight">
<summary class="code-dropdown-header" style="font-weight: 900 !important;"><b>Click to see the commands to run multi-GPU training inside of Docker</b></summary>
<div class="language-python highlighter-rouge">
{% highlight bash %}
mkdir -p projects/ && cd projects && git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLabs
echo \
"services:
  isaac-lab-base:
    shm_size: '2gb'" > docker/shm-config.yaml
python3 docker/container.py start --files shm-config.yaml
# [INFO] Using container profile: base
# [INFO] X11 Forwarding is configured as '0' in '.container.cfg'.
# 	To enable X11 forwarding, set 'X11_FORWARDING_ENABLED=1' in '.container.cfg'.
# [INFO] Building the docker image and starting the container 'isaac-lab-base' in the background...
#  ✔ isaac-lab-base            Built                                                                                                     0.0s 
#  ✔ Container isaac-lab-base  Started                                                                                                  11.7s 
python3 docker/container.py enter
# [INFO] Using container profile: base
# [INFO] X11 Forwarding is disabled from the settings in '.container.cfg'
# [INFO] X11 forwarding is disabled. No action taken.
# [INFO] Entering the existing 'isaac-lab-base' container in a bash session...

# Option A: Training (8192 environments per GPU)
OMP_NUM_THREADS=8 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Repose-Cube-Shadow-Direct-v0 --headless --distributed
# Example RL Games output collected
# fps step: 611460 fps step and policy inference: 588767 fps total: 472288 epoch: 174/5000 frames: 90701824
# fps step: 592850 fps step and policy inference: 571132 fps total: 464365 epoch: 175/5000 frames: 91226112
# fps step: 616768 fps step and policy inference: 594105 fps total: 479784 epoch: 176/5000 frames: 91750400
# fps step: 602694 fps step and policy inference: 581137 fps total: 477391 epoch: 177/5000 frames: 92274688

# Option B: Training Benchmark
python scripts/benchmarks/benchmark_rlgames.py --task=Isaac-Repose-Cube-Shadow-Direct-v0 --headless

{% endhighlight %}
</div>
</details>

It would appear, from the RL Games training output, that Minerva runs training and simulation at around ~477,000 FPS, or ~23% faster than the benchmarked ```4x L40``` node, and ~182% faster
than ```1x 4090``` (which achieves 170,000 FPS), while running at a clock limit of ```1750MHz```.  However, a ```1x 4090``` is faster than ```1x 3090TI```.

On Minerva, the hand cube repose task can be solved within 20 minutes with 4 GPUs. It takes longer with one GPU.

<details class="highlight">
<summary class="code-dropdown-header" style="font-weight: 900 !important;"><b>Click to see verbose output of benchmark</b></summary>
<div class="language-python highlighter-rouge">
{% highlight bash %}
root@minerva:/workspace/isaaclab# python scripts/benchmarks/benchmark_rlgames.py --task=Isaac-Repose-Cube-Shadow-Direct-v0 --headless
[INFO][AppLauncher]: Loading experience file: /workspace/isaaclab/apps/isaaclab.python.headless.kit
Loading user config located at: '/isaac-sim/kit/data/Kit/Isaac-Sim/4.5/user.config.json'
[Info] [carb] Logging to file: /isaac-sim/kit/logs/Kit/Isaac-Sim/4.5/kit_20250224_025719.log
2025-02-24 02:57:19 [0ms] [Warning] [omni.kit.app.plugin] No crash reporter present, dumps uploading isn't available.
2025-02-24 02:57:20 [436ms] [Warning] [omni.usd_config.extension] Enable omni.materialx.libs extension to use MaterialX
Authorization required, but no authorization protocol specified
2025-02-24 02:57:20 [508ms] [Warning] [omni.platforminfo.plugin] failed to open the default display.  Can't verify X Server version.
Authorization required, but no authorization protocol specified
2025-02-24 02:57:20 [612ms] [Warning] [omni.datastore] OmniHub is inaccessible
2025-02-24 02:57:20 [760ms] [Warning] [omni.isaac.dynamic_control] omni.isaac.dynamic_control is deprecated as of Isaac Sim 4.5. No action is needed from end-users.
Authorization required, but no authorization protocol specified
Authorization required, but no authorization protocol specified

|---------------------------------------------------------------------------------------------|
| Driver Version: 560.35.03     | Graphics API: Vulkan
|=============================================================================================|
| GPU | Name                             | Active | LDA | GPU Memory | Vendor-ID | LUID       |
|     |                                  |        |     |            | Device-ID | UUID       |
|     |                                  |        |     |            | Bus-ID    |            |
|---------------------------------------------------------------------------------------------|
| 0   | NVIDIA GeForce RTX 3090 Ti       | Yes: 0 |     | 24810   MB | 10de      | 0          |
|     |                                  |        |     |            | 2203      | 2b26c591.. |
|     |                                  |        |     |            | 1         |            |
|---------------------------------------------------------------------------------------------|
| 1   | NVIDIA GeForce RTX 3090          | Yes: 1 |     | 24822   MB | 10de      | 0          |
|     |                                  |        |     |            | 2204      | 49e6f8d4.. |
|     |                                  |        |     |            | 21        |            |
|---------------------------------------------------------------------------------------------|
| 2   | NVIDIA GeForce RTX 3090          | Yes: 2 |     | 24822   MB | 10de      | 0          |
|     |                                  |        |     |            | 2204      | c9400bc1.. |
|     |                                  |        |     |            | c1        |            |
|---------------------------------------------------------------------------------------------|
| 3   | NVIDIA GeForce RTX 3090 Ti       | Yes: 3 |     | 24810   MB | 10de      | 0          |
|     |                                  |        |     |            | 2203      | 080243db.. |
|     |                                  |        |     |            | e1        |            |
|=============================================================================================|
| OS: 22.04.5 LTS (Jammy Jellyfish) ubuntu, Version: 22.04.5, Kernel: 6.8.0-52-generic
| Processor: AMD Ryzen Threadripper PRO 7955WX 16-Cores
| Cores: 16 | Logical Cores: 32
|---------------------------------------------------------------------------------------------|
| Total Memory (MB): 128295 | Free Memory: 101049
| Total Page/Swap (MB): 2047 | Free Page/Swap: 0
|---------------------------------------------------------------------------------------------|
2025-02-24 02:57:24 [4,832ms] [Warning] [gpu.foundation.plugin] IOMMU is enabled.
2025-02-24 02:57:24 [4,832ms] [Warning] [gpu.foundation.plugin] Detected IOMMU is enabled. Running CUDA peer-to-peer bandwidth and latency validation.
Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1      2      3 
     0 890.82  11.31  11.28  11.34 
     1  11.26 862.89  11.31  11.35 
     2  11.31  11.32 832.00  11.30 
     3  11.27  11.37  11.29 831.12 
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1      2      3 
     0   1.68  10.72  11.05  10.55 
     1  12.86   1.66  10.75  14.99 
     2  16.30  16.39   1.66  19.46 
     3  13.61  14.69  17.88   1.67 

   CPU     0      1      2      3 
     0   1.71   5.17   5.04   4.91 
     1   5.06   1.56   4.97   4.42 
     2   4.96   4.61   1.48   4.52 
     3   4.95   4.56   4.56   1.42 
2025-02-24 02:57:25 [5,798ms] [Warning] [gpu.foundation.plugin] CUDA peer-to-peer observed bandwidth: 11.3 GB/s.
2025-02-24 02:57:25 [5,798ms] [Warning] [gpu.foundation.plugin] CUDA peer-to-peer observed latency: 19.5 us.
2025-02-24 02:57:25 [5,798ms] [Warning] [gpu.foundation.plugin] Please verify if observed bandwidth and latency are expected.
2025-02-24 02:57:26 [6,720ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Generating formatted report = True
2025-02-24 02:57:26 [6,720ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Using metrics backend = OmniPerfKPIFile
2025-02-24 02:57:26 [6,720ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Local folder location = /tmp
2025-02-24 02:57:26 [6,720ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Starting
2025-02-24 02:57:26 [6,720ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Test mode = False
[INFO]: Parsing configuration from: isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg:ShadowHandEnvCfg
[INFO]: Parsing configuration from: /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/agents/rl_games_ppo_cfg.yaml
[INFO] Logging experiment in directory: /workspace/isaaclab/logs/rl_games/shadow_hand
2025-02-24 02:57:26 [6,924ms] [Warning] [isaaclab.envs.direct_rl_env] Seed not set for the environment. The environment creation may not be deterministic.
[INFO]: Base environment:
	Environment device    : cuda:0
	Environment seed      : None
	Physics step-size     : 0.008333333333333333
	Rendering step-size   : 0.016666666666666666
	Environment step-size : 0.016666666666666666
[INFO]: Time taken for scene creation : 2.152462 seconds
[INFO]: Scene manager:  <class InteractiveScene>
	Number of environments: 8192
	Environment spacing   : 0.75
	Source prim name      : /World/envs/env_0
	Global prim paths     : []
	Replicate physics     : True
[INFO]: Starting the simulation. This may take a few seconds. Please wait...
2025-02-24 02:57:31 [11,745ms] [Warning] [isaaclab.assets.articulation.articulation] ImplicitActuatorCfg fingers has set both effort_limit_sim and effort_limit.Only effort_limit_sim will be used for ImplicitActuators.
2025-02-24 02:57:31 [11,745ms] [Warning] [isaaclab.assets.articulation.articulation] ImplicitActuatorCfg fingers has set both velocity_limit_sim and velocity_limit.Only velocity_limit_sim will be used for ImplicitActuators.
[INFO]: Time taken for simulation start : 6.141075 seconds
[INFO]: Completed setting up the environment...
self.seed = 42
Setting seed: 42
2025-02-24 02:57:34 [15,364ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Starting phase: sim_runtime
Started to train
Exact experiment name requested from command line: 2025-02-24_02-57-26
seq_length: 4
current training device: cuda:0
/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.10/site-packages/rl_games/common/a2c_common.py:254: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
build mlp: 157
RunningMeanStd:  (1,)
RunningMeanStd:  (157,)
/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.10/site-packages/rl_games/algos_torch/a2c_continuous.py:106: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.mixed_precision):
fps step: 58550 fps step and policy inference: 56747 fps total: 52351 epoch: 1/10 frames: 0
fps step: 184865 fps step and policy inference: 178640 fps total: 152956 epoch: 2/10 frames: 131072
fps step: 187879 fps step and policy inference: 181181 fps total: 154814 epoch: 3/10 frames: 262144
fps step: 190668 fps step and policy inference: 183906 fps total: 156801 epoch: 4/10 frames: 393216
fps step: 194337 fps step and policy inference: 187281 fps total: 159257 epoch: 5/10 frames: 524288
fps step: 197345 fps step and policy inference: 190092 fps total: 161290 epoch: 6/10 frames: 655360
fps step: 199005 fps step and policy inference: 191577 fps total: 162346 epoch: 7/10 frames: 786432
fps step: 197395 fps step and policy inference: 190023 fps total: 161233 epoch: 8/10 frames: 917504
fps step: 188600 fps step and policy inference: 181344 fps total: 154939 epoch: 9/10 frames: 1048576
fps step: 190855 fps step and policy inference: 183552 fps total: 156531 epoch: 10/10 frames: 1179648
=> saving checkpoint '/workspace/isaaclab/logs/rl_games/shadow_hand/2025-02-24_02-57-26/nn/last_shadow_hand_ep_10_rew__-19.991896_.pth'
MAX EPOCHS NUM!
2025-02-24 02:57:48 [29,315ms] [Warning] [isaacsim.benchmark.services.recorders] Detected multiple GPU types: ['NVIDIA GeForce RTX 3090 Ti', 'NVIDIA GeForce RTX 3090 Ti', 'NVIDIA GeForce RTX 3090', 'NVIDIA GeForce RTX 3090'].
2025-02-24 02:57:48 [29,315ms] [Warning] [isaacsim.benchmark.services.recorders] Only recording GPU 0 type: NVIDIA GeForce RTX 3090 Ti
/isaac-sim/exts/isaacsim.benchmark.services/isaacsim/benchmark/services/datarecorders/frametime.py:98: DeprecationWarning: The 'warn' method is deprecated, use 'warning' instead
  logger.warn(f"Unable to calculate frametime stats: {e}")
2025-02-24 02:57:48 [29,375ms] [WARNING] [isaacsim.benchmark.services.datarecorders.frametime] Unable to calculate frametime stats: mean requires at least one data point
2025-02-24 02:57:48 [29,376ms] [WARNING] [isaacsim.benchmark.services.datarecorders.frametime] Unable to calculate frametime stats: mean requires at least one data point
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Created new phase 'startup' and stored SingleMeasurement(name='App Launch Time', value=5973.722584, unit='ms', type='single')
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Python Imports Time', value=179.755971, unit='ms', type='single') for phase 'startup'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Task Creation and Start Time', value=8385.666631, unit='ms', type='single') for phase 'startup'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Scene Creation Time', value=2152.4621120006486, unit='ms', type='single') for phase 'startup'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Simulation Start Time', value=6141.0754440003075, unit='ms', type='single') for phase 'startup'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Total Start Time (Launch to Train)', value=15321.311632, unit='ms', type='single') for phase 'startup'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Created new phase 'runtime' and stored DictMeasurement(name='Step Frametimes', value={'Environment only step time': [2.238626480102539, 0.7090144157409668, 0.6976408958435059, 0.6874349117279053, 0.6744587421417236, 0.6641781330108643, 0.6586358547210693, 0.6640071868896484, 0.694974422454834], 'Environment + Inference step time': [2.3097691535949707, 0.7337219715118408, 0.7234294414520264, 0.7127134799957275, 0.6998662948608398, 0.6895182132720947, 0.6841747760772705, 0.6897702217102051, 0.7227792739868164], 'Environment + Inference + Policy update time': [0.19395899772644043, 0.12320470809936523, 0.12321305274963379, 0.12320137023925781, 0.12315535545349121, 0.12313103675842285, 0.12318849563598633, 0.12316560745239258, 0.12317991256713867], 'Environment only FPS': [58550.1875, 184865.078125, 187878.890625, 190668.234375, 194336.578125, 197344.640625, 199005.265625, 197395.453125, 188599.75], 'Environment + Inference FPS': [56746.796875, 178639.875, 181181.453125, 183905.59375, 187281.484375, 190092.15625, 191576.78125, 190022.703125, 181344.4375], 'Environment + Inference + Policy update FPS': [52350.7265625, 152955.90625, 154813.828125, 156800.65625, 159257.0, 161289.75, 162345.75, 161232.84375, 154938.875]}, type='dict')
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Min Environment only step time', value=0.6586358547210693, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Max Environment only step time', value=2.238626480102539, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Mean Environment only step time', value=0.8543301158481174, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Min Environment + Inference step time', value=0.6841747760772705, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Max Environment + Inference step time', value=2.3097691535949707, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Mean Environment + Inference step time', value=0.8850825362735324, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Min Environment + Inference + Policy update time', value=0.12313103675842285, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Max Environment + Inference + Policy update time', value=0.19395899772644043, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Mean Environment + Inference + Policy update time', value=0.13104428185356987, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Min Environment only FPS', value=58550.1875, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Max Environment only FPS', value=199005.265625, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Mean Environment only FPS', value=177627.11979166666, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Min Environment + Inference FPS', value=56746.796875, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Max Environment + Inference FPS', value=191576.78125, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Mean Environment + Inference FPS', value=171199.03125, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Min Environment + Inference + Policy update FPS', value=52350.7265625, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Max Environment + Inference + Policy update FPS', value=162345.75, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Mean Environment + Inference + Policy update FPS', value=146220.59288194444, unit='ms', type='single') for phase 'runtime'
2025-02-24 02:57:48 [29,382ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Created new phase 'train' and stored ListMeasurement(name='Rewards', length=8)
2025-02-24 02:57:48 [29,383ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Max Rewards', value=-6.726855278015137, unit='float', type='single') for phase 'train'
2025-02-24 02:57:48 [29,383ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored ListMeasurement(name='Episode Lengths', length=8) for phase 'train'
2025-02-24 02:57:48 [29,383ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stored SingleMeasurement(name='Max Episode Lengths', value=104.1601333618164, unit='float', type='single') for phase 'train'
2025-02-24 02:57:48 [29,383ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Stopping
2025-02-24 02:57:48 [29,383ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Writing metrics data.
2025-02-24 02:57:48 [29,383ms] [INFO] [isaacsim.benchmark.services.base_isaac_benchmark] Metrics type = OmniPerfKPIFile
2025-02-24 02:57:48 [29,383ms] [INFO] [isaacsim.benchmark.services.metrics.backend] 
sim_runtime Metrics:
workflow_name: benchmark_rlgames_train
task: Isaac-Repose-Cube-Shadow-Direct-v0
max_iterations: 10
phase: sim_runtime
System Memory RSS: 6.484 GB
System Memory VMS: 96.113 GB
System Memory USS: 6.466 GB
GPU Memory Tracked: 0.0 GB
GPU Memory Dedicated: 0 GB
System CPU iowait: 0.0 %
System CPU system: 2.0 %
System CPU user: 9.0 %
System CPU idle: 89.0 %
num_cpus: 32 
gpu_device_name: NVIDIA GeForce RTX 3090 Ti 
Mean App_Update Frametime: 0 ms
Stdev App_Update Frametime: 0 ms
Min App_Update Frametime: 0 ms
Max App_Update Frametime: 0 ms
Mean Physics Frametime: 18.81 ms
Stdev Physics Frametime: 0.78 ms
Min Physics Frametime: 17.52 ms
Max Physics Frametime: 20.44 ms
Mean GPU Frametime: 0 ms
Stdev GPU Frametime: 0 ms
Min GPU Frametime: 0 ms
Max GPU Frametime: 0 ms
Real Time Factor: 0.0 
Runtime: 10932.26 ms
2025-02-24 02:57:48 [29,383ms] [INFO] [isaacsim.benchmark.services.metrics.backend] 
startup Metrics:
workflow_name: benchmark_rlgames_train
task: Isaac-Repose-Cube-Shadow-Direct-v0
max_iterations: 10
phase: startup
App Launch Time: 5973.722584 ms
Python Imports Time: 179.755971 ms
Task Creation and Start Time: 8385.666631 ms
Scene Creation Time: 2152.4621120006486 ms
Simulation Start Time: 6141.0754440003075 ms
Total Start Time (Launch to Train): 15321.311632 ms
2025-02-24 02:57:48 [29,383ms] [INFO] [isaacsim.benchmark.services.metrics.backend] 
runtime Metrics:
workflow_name: benchmark_rlgames_train
task: Isaac-Repose-Cube-Shadow-Direct-v0
max_iterations: 10
phase: runtime
Min Environment only step time: 0.6586358547210693 ms
Max Environment only step time: 2.238626480102539 ms
Mean Environment only step time: 0.8543301158481174 ms
Min Environment + Inference step time: 0.6841747760772705 ms
Max Environment + Inference step time: 2.3097691535949707 ms
Mean Environment + Inference step time: 0.8850825362735324 ms
Min Environment + Inference + Policy update time: 0.12313103675842285 ms
Max Environment + Inference + Policy update time: 0.19395899772644043 ms
Mean Environment + Inference + Policy update time: 0.13104428185356987 ms
Min Environment only FPS: 58550.1875 ms
Max Environment only FPS: 199005.265625 ms
Mean Environment only FPS: 177627.11979166666 ms
Min Environment + Inference FPS: 56746.796875 ms
Max Environment + Inference FPS: 191576.78125 ms
Mean Environment + Inference FPS: 171199.03125 ms
Min Environment + Inference + Policy update FPS: 52350.7265625 ms
Max Environment + Inference + Policy update FPS: 162345.75 ms
Mean Environment + Inference + Policy update FPS: 146220.59288194444 ms
2025-02-24 02:57:48 [29,383ms] [INFO] [isaacsim.benchmark.services.metrics.backend] 
train Metrics:
workflow_name: benchmark_rlgames_train
task: Isaac-Repose-Cube-Shadow-Direct-v0
max_iterations: 10
phase: train
Max Rewards: -6.726855278015137 float
Max Episode Lengths: 104.1601333618164 float
2025-02-24 02:57:48 [29,383ms] [INFO] [isaacsim.benchmark.services.metrics.backend] Writing metrics to /tmp/kpis_benchmark_rlgames_train.json
|----------------------------------------------------|
|                   Summary Report                   |
|----------------------------------------------------|
| workflow_name: benchmark_rlgames_train             |
| task: Isaac-Repose-Cube-Shadow-Direct-v0           |
| max_iterations: 10                                 |
| num_cpus: 32                                       |
| gpu_device_name: NVIDIA GeForce RTX 3090 Ti        |
|----------------------------------------------------|
| Phase: sim_runtime                                 |
| System Memory RSS: 6.484 GB                        |
| System Memory VMS: 96.113 GB                       |
| System Memory USS: 6.466 GB                        |
| GPU Memory Tracked: 0.0 GB                         |
| Real Time Factor: 0.0                              |
| Runtime: 10932.26 ms                               |
| Frametimes (ms):    mean |  stdev |   min |   max  |
| App_Update          0.00 |   0.00 |  0.00 |  0.00  |
| Physics            18.81 |   0.78 | 17.52 | 20.44  |
| GPU                 0.00 |   0.00 |  0.00 |  0.00  |
|----------------------------------------------------|
| Phase: startup                                     |
| App Launch Time: 5973.722584 ms                    |
| Python Imports Time: 179.755971 ms                 |
| Task Creation and Start Time: 8385.666631 ms       |
| Scene Creation Time: 2152.4621120006486 ms         |
| Simulation Start Time: 6141.0754440003075 ms       |
| Total Start Time (Launch to Train): 15321.311632 ms |
|----------------------------------------------------|
| Phase: runtime                                     |
| Min Environment only step time: 0.6586358547210693 ms |
| Max Environment only step time: 2.238626480102539 ms |
| Mean Environment only step time: 0.8543301158481174 ms |
| Min Environment + Inference step time: 0.6841747760772705 ms |
| Max Environment + Inference step time: 2.3097691535949707 ms |
| Mean Environment + Inference step time: 0.8850825362735324 ms |
| Min Environment + Inference + Policy update time: 0.12313103675842285 ms |
| Max Environment + Inference + Policy update time: 0.19395899772644043 ms |
| Mean Environment + Inference + Policy update time: 0.13104428185356987 ms |
| Min Environment only FPS: 58550.1875 ms            |
| Max Environment only FPS: 199005.265625 ms         |
| Mean Environment only FPS: 177627.11979166666 ms   |
| Min Environment + Inference FPS: 56746.796875 ms   |
| Max Environment + Inference FPS: 191576.78125 ms   |
| Mean Environment + Inference FPS: 171199.03125 ms  |
| Min Environment + Inference + Policy update FPS: 52350.7265625 ms |
| Max Environment + Inference + Policy update FPS: 162345.75 ms |
| Mean Environment + Inference + Policy update FPS: 146220.59288194444 ms |
|----------------------------------------------------|
| Phase: train                                       |
| Max Rewards: -6.726855278015137 float              |
| Max Episode Lengths: 104.1601333618164 float       |
|----------------------------------------------------|
root@minerva:/workspace/isaaclab# 
{% endhighlight %}

<div class="highlight">
</div>
</div>
</details>






## Case Study II: NVIDIA Isaac Lab Hyperparameter Tuning

Minerva can run 4 parallel NVIDIA Isaac Lab training runs at once, one on each GPU. This is enabled by the [Ray functionality](https://isaac-sim.github.io/IsaacLab/main/source/features/ray.html) that I added to Isaac Lab.
The following is an example of tuning quadrupedal gait parameters on flat terrain with 4 parallel similar experiments.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/minerva/isaac-ray-tune.png" title="Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


## Case Study III: YoloV11 Training with Ultralytics

For the Northeastern Mars Rover team, I trained a "mallet" and "bottle" detector. About 27,000 images at ```800p``` resolution were used to train 409 layers, 20,054,550 parameters, 20,054,534 gradients, with 12 epochs taking about 24 minutes.

<details class="highlight">
<summary class="code-dropdown-header" style="font-weight: 900 !important;"><b>Click to see the code snippet of running the model</b></summary>
<div class="language-python highlighter-rouge">
{% highlight python %}
from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11m.pt") 
data_yaml = str(Path(__file__).parent / "dataset/data.yaml")  

model.train(data=data_yaml, 
            epochs=100, 
            imgsz=800, 
            batch=100,
            cache="disk",
            freeze=0, 
            copy_paste=.8,
            hsv_v=.3,
            erasing=.9,
            crop_fraction=.8,
            translate=.9,
            mixup=.4,
            perspective=0.00005,
            patience=20, 
            plots = True, 
            save=True, 
	        workers = 8, 
	        device="0,1,2,3",)

{% endhighlight %}
</div>
</details>

<details class="highlight">
<summary class="code-dropdown-header" style="font-weight: 900 !important;"><b>Click to see the output of running the model</b></summary>
<div class="language-python highlighter-rouge">
{% highlight bash %}
python3 train.py # In conda environment
New https://pypi.org/project/ultralytics/8.3.78 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.75 🚀 Python-3.11.11 torch-2.2.2+cu121 CUDA:0 (NVIDIA GeForce RTX 3090 Ti, 24142MiB)
                                                       CUDA:1 (NVIDIA GeForce RTX 3090 Ti, 24139MiB)
                                                       CUDA:2 (NVIDIA GeForce RTX 3090, 24154MiB)
                                                       CUDA:3 (NVIDIA GeForce RTX 3090, 24154MiB)
engine/trainer: task=detect, mode=train, model=yolo11m.pt, data=/home/garylvov/projects/urc_mallet_model_2025/dataset/data.yaml, epochs=100, time=None, patience=20, batch=100, imgsz=800, save=True, save_period=-1, cache=disk, device=0,1,2,3, workers=8, project=None, name=train37, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=0, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.3, degrees=0.0, translate=0.9, scale=0.5, shear=0.0, perspective=5e-05, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.4, copy_paste=0.8, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.9, crop_fraction=0.8, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train37
Overriding model.yaml nc=80 with nc=2

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1412566  ultralytics.nn.modules.head.Detect           [2, [256, 512, 512]]          
YOLO11m summary: 409 layers, 20,054,550 parameters, 20,054,534 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
DDP: debug command /home/garylvov/.conda/envs/rover/bin/python3 -m torch.distributed.run --nproc_per_node 4 --master_port 52275 /home/garylvov/.config/Ultralytics/DDP/_temp_ugpib82l137958647273360.py
Ultralytics 8.3.75 🚀 Python-3.11.11 torch-2.2.2+cu121 CUDA:0 (NVIDIA GeForce RTX 3090 Ti, 24142MiB)
                                                       CUDA:1 (NVIDIA GeForce RTX 3090 Ti, 24139MiB)
                                                       CUDA:2 (NVIDIA GeForce RTX 3090, 24154MiB)
                                                       CUDA:3 (NVIDIA GeForce RTX 3090, 24154MiB)
Overriding model.yaml nc=80 with nc=2
Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /home/garylvov/projects/urc_mallet_model_2025/dataset/train/labels.cache... 27339 images, 6966 backgrounds, 0 corrupt: 100%|██████████| 27339/27339 [00:
train: WARNING ⚠️ /home/garylvov/projects/urc_mallet_model_2025/dataset/train/images/000000278737.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/garylvov/projects/urc_mallet_model_2025/dataset/train/images/000000301977.jpg: 1 duplicate labels removed
train: Caching images (41.2GB Disk): 100%|██████████| 27339/27339 [00:00<00:00, 112448.68it/s]
val: Scanning /home/garylvov/projects/urc_mallet_model_2025/dataset/valid/labels.cache... 575 images, 6 backgrounds, 0 corrupt: 100%|██████████| 575/575 [00:00<?, ?it/s
val: Caching images (0.7GB Disk): 100%|██████████| 575/575 [00:00<00:00, 86981.09it/s]
Plotting labels to runs/detect/train37/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.00078125), 112 bias(decay=0.0)
Image sizes 800 train, 800 val
Using 32 dataloader workers
Logging results to runs/detect/train37
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      22.3G       1.51      1.979      1.248          5        800: 100%|██████████| 274/274 [01:59<00:00,  2.30it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:02<00:00,  4.08it/s]
                   all        575       1042      0.796      0.735      0.789      0.498

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100      22.4G      1.578      1.431      1.251         13        800: 100%|██████████| 274/274 [01:59<00:00,  2.30it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:02<00:00,  4.31it/s]
                   all        575       1042      0.742      0.669       0.72      0.409

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100      22.4G      1.813        1.8      1.395         15        800: 100%|██████████| 274/274 [01:58<00:00,  2.31it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:02<00:00,  4.30it/s]
                   all        575       1042      0.457      0.406      0.408      0.198

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100      22.4G      1.959      2.035      1.536          9        800: 100%|██████████| 274/274 [01:59<00:00,  2.29it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:02<00:00,  4.30it/s]
                   all        575       1042      0.714      0.517      0.554      0.321

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100      22.5G      1.852      1.842      1.483         11        800: 100%|██████████| 274/274 [02:03<00:00,  2.21it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:02<00:00,  4.28it/s]
                   all        575       1042      0.725      0.584      0.628      0.336

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100      22.4G       1.78      1.717      1.441         21        800: 100%|██████████| 274/274 [02:01<00:00,  2.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:02<00:00,  4.24it/s]
                   all        575       1042      0.737      0.621      0.647      0.386

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100      24.1G      1.732       1.63      1.409         17        800: 100%|██████████| 274/274 [02:04<00:00,  2.20it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:02<00:00,  4.28it/s]
                   all        575       1042      0.814      0.685       0.72      0.431

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100      22.4G      1.693       1.61      1.391         49        800: 100%|██████████| 274/274 [02:03<00:00,  2.22it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:02<00:00,  4.31it/s]
                   all        575       1042      0.835       0.68       0.73      0.457

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100      22.4G      1.661      1.505      1.365         13        800: 100%|██████████| 274/274 [02:03<00:00,  2.22it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:02<00:00,  4.34it/s]
                   all        575       1042      0.808      0.722      0.747      0.459

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100      22.4G      1.636      1.535      1.359         12        800: 100%|██████████| 274/274 [02:01<00:00,  2.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:02<00:00,  4.33it/s]
                   all        575       1042      0.805       0.72      0.743      0.484

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100      23.3G      1.615      1.454      1.329         11        800: 100%|██████████| 274/274 [02:03<00:00,  2.23it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:02<00:00,  4.30it/s]
                   all        575       1042       0.85      0.721      0.764       0.49

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100      22.4G      1.587      1.436      1.334         16        800: 100%|██████████| 274/274 [02:05<00:00,  2.19it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:02<00:00,  4.29it/s]
                   all        575       1042       0.83      0.728      0.781      0.491


{% endhighlight %}
</div>
</details>

## Case Study IV: Running DeepSeek-R1-Distill-Llama-70B

Running Ollama distilled from Deepseek-R1 on NVIDIA GPUs requires the NVIDIA Container Toolkit as shown above. Then it's as easy as increasing the context window
on some 70B parameter distillations of Deepseek's models to push Minerva's limits.

<details class="highlight">
<summary class="code-dropdown-header" style="font-weight: 900 !important;"><b>Click to see the code snippet of running Deepseek-r1:70B with a larger context window</b></summary>
<div class="language-python highlighter-rouge">
{% highlight bash %}
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama bash
echo \
"FROM deepseek-r1:70b
PARAMETER num_ctx 131072
PARAMETER num_predict 31072" > Modelfile
ollama create deepseek-r1-max-context-and-output-size:70b -f Modelfile
# gathering model components 
# writing manifest 
# success 
ollama run deepseek-r1-max-context-and-output-size:70b --verbose
# After sample prompt:
# Spills over into 80gb of RAM, from VRAM so super slow 
# total duration:       16.467195206s
# load duration:        12.409235ms
# prompt eval count:    10 token(s)
# prompt eval duration: 893ms
# prompt eval rate:     11.20 tokens/s
# eval count:           45 token(s)
# eval duration:        15.561s
# eval rate:            2.89 tokens/s

# Reset with ctrl + D pkill ollama and then redo run and exec it (may have to docker rm previous container ID)
# Let's fit this whole model onto VRAM.
echo \
"FROM deepseek-r1:70b
PARAMETER num_ctx 60000
PARAMETER num_predict 30000" > Modelfile
ollama create deepseek-r1-60k-context-and-30k:70b -f Modelfile
ollama run deepseek-r1-60k-context-and-30k:70b --verbose

# After simple sample prompt:
# total duration:       2m5.635874804s
# load duration:        13.383071ms
# prompt eval count:    258 token(s)
# prompt eval duration: 509ms
# prompt eval rate:     506.88 tokens/s
# eval count:           1663 token(s)
# eval duration:        2m4.74s
# eval rate:            13.33 tokens/s

# When prompted with the contents of this post...
# total duration:       2m52.68976111s
# load duration:        13.345338ms
# prompt eval count:    16811 token(s)
# prompt eval duration: 45.742s
# prompt eval rate:     367.52 tokens/s
# eval count:           1118 token(s)
# eval duration:        2m6.854s
# eval rate:            8.81 tokens/s

{% endhighlight %}
</div>
</details>

So, with a 60,000 token length context window, we are able to use ```~92G``` VRAM while running ```deepseek-r1:70B```, while generating ```8.81 tokens/s``` and processing input at ```367.52 tokens/s```.

# Cost Analysis - is it worth it?
An ```1x NVIDIA H100 80G``` node can cost roughly $2-$3/hr to rent. An ```H100``` can't do the ray
tracing required for running [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) with cameras, where an ```2x NVIDIA A6000 48G``` node (rentable on Lamba Labs cloud for ~$1.5/hr, 96GB VRAM total) or ```4x NVIDIA L4 24G``` node (such as ```g6.24xlarge``` on AWS, rentable for ~$6/hr, 96GB VRAM total) is able to do ray tracing and is more comparable. There also is the trouble of provisioning instances online; these nodes are in demand and can't always be created without waiting.

Minerva cost roughly $8.2k due to most parts coming from eBay. Let's assume that whatever comparable cloud solution costs on average, $2/hr.
Minerva uses around 1500W at full load. Minerva's electricity cost, assuming an expensive $0.34/kWh (in Boston), is ~$0.51/hr. So, for Minerva to make sense financially, I need to run training for about 5,503 hours, or about 230 days.
Of course, this doesn't account for the time-value of money (say if I invested what I saved by renting into ``$SPY``), so I whipped up this python script to 
calculate some possible Net Present Value (NPV) outcomes.

<details class="highlight">
<summary class="code-dropdown-header" style="font-weight: 900 !important;"><b>Click to see the NPV estimation</b></summary>
<div class="language-python highlighter-rouge">
<div class="highlight">
{% highlight python %}
import math

def calculate_pure_investment_outcome(initial_minerva_cost, cloud_cost_per_hour, 
                                   electricity_cost_per_hour, hours_per_year, 
                                   annual_return, initial_years, investment_years):
    """
    Calculate outcomes where computer usage stops after initial period and money is purely invested.
    
    Args:
        initial_minerva_cost: Initial cost of Minerva
        cloud_cost_per_hour: Hourly cost of cloud computing
        electricity_cost_per_hour: Hourly cost of electricity
        hours_per_year: Annual usage hours
        annual_return: Expected annual return rate
        initial_years: Years of computer usage
        investment_years: Years of pure investment afterwards
        
    Returns:
        list: List of tuples (year, npv, description) for key points
    """
    if annual_return >= 1:
        raise ValueError("Annual return should be expressed as a decimal")
        
    monthly_return = (1 + annual_return) ** (1/12) - 1
    monthly_hours = hours_per_year / 12
    monthly_savings = (cloud_cost_per_hour - electricity_cost_per_hour) * monthly_hours
    
    def position_at_month(month):
        """Calculate net position after given number of months of computer usage"""
        investment_value = initial_minerva_cost * (1 + monthly_return) ** month
        
        if monthly_return == 0:
            savings_value = monthly_savings * month
        else:
            savings_value = monthly_savings * ((1 + monthly_return) ** month - 1) / monthly_return
            
        return savings_value - investment_value, savings_value
    
    results = []
    
    # Initial position (year 0)
    results.append((0, -initial_minerva_cost, "Initial investment"))
    
    # Position at end of computer usage period
    total_months = initial_years * 12
    final_position, final_savings = position_at_month(total_months)
    npv_at_initial = final_position / (1 + monthly_return) ** total_months if monthly_return > 0 else final_position
    results.append((initial_years, npv_at_initial, "End of computer usage"))
    
    # Pure investment period - just let the final position grow
    for year in range(1, investment_years + 1):
        months = year * 12
        # Growth of final position for additional years
        future_value = final_position * (1 + monthly_return) ** months
        # NPV calculation should only discount back from current point in time
        npv = future_value / (1 + monthly_return) ** months
        
        if year == investment_years:  # Only include final year to keep output clean
            results.append((initial_years + year, npv, "End of investment period"))
    
    return results

def print_pure_investment_analysis(minerva_cost, cloud_cost_per_hour, electricity_cost_per_hour,
                                 usage_scenarios, returns, initial_years, investment_years):
    """Print analysis of computer usage followed by pure investment period."""
    for hours in usage_scenarios:
        print(f"\nAnalysis for {hours} hours per year:")
        print("Return (%) | Year | NPV (Dollars) | Stage")
        print("-" * 60)
        
        for ret in returns:
            try:
                results = calculate_pure_investment_outcome(
                    minerva_cost, cloud_cost_per_hour, electricity_cost_per_hour,
                    hours, ret, initial_years, investment_years
                )
                
                for year, npv, description in results:
                    print(f"{ret*100:9.1f} | {year:4d} | {npv:,.2f} | {description}")
                print()
                
            except ValueError as e:
                print(f"{ret*100:9.1f} | Error: {str(e)}")
        print()

if __name__ == "__main__":
    minerva_cost = 8200  # Minerva initial investment ($)
    cloud_cost_per_hour = 2.00  # Cloud rental cost per hour ($)
    electricity_cost_per_hour = .51  # Minerva electricity cost per hour ($)
    
    usage_scenarios = [1300, 1400, 2800]
    returns = [0.05, 0.10, 0.15, 0.20]
    
    print("\n=== Scenario: 6 years usage + 25 years investment ===")
    print_pure_investment_analysis(minerva_cost, cloud_cost_per_hour, electricity_cost_per_hour,
                                 usage_scenarios, returns, 6, 25)
{% endhighlight %}
</div>
</div>
</details>

<details class="highlight">
<summary class="code-dropdown-header" style="font-weight: 900 !important;"><b>Click to see the NPV results</b></summary>
<div class="language-python highlighter-rouge">
<div class="highlight">
{% highlight bash %}
garylvov@minerva:~$ python3 extended_value.py 

=== Scenario: 6 years usage + 25 years investment ===

Analysis for 1300 hours per year:
Return (%) | Year | NPV (Dollars) | Stage
------------------------------------------------------------
      5.0 |    0 | -8,200.00 | Initial investment
      5.0 |    6 | 1,854.94 | End of computer usage
      5.0 |   31 | 2,485.80 | End of investment period

     10.0 |    0 | -8,200.00 | Initial investment
     10.0 |    6 | 616.14 | End of computer usage
     10.0 |   31 | 1,091.54 | End of investment period

     15.0 |    0 | -8,200.00 | Initial investment
     15.0 |    6 | -378.20 | End of computer usage
     15.0 |   31 | -874.79 | End of investment period

     20.0 |    0 | -8,200.00 | Initial investment
     20.0 |    6 | -1,187.44 | End of computer usage
     20.0 |   31 | -3,545.68 | End of investment period



Analysis for 1400 hours per year:
Return (%) | Year | NPV (Dollars) | Stage
------------------------------------------------------------
      5.0 |    0 | -8,200.00 | Initial investment
      5.0 |    6 | 2,628.40 | End of computer usage
      5.0 |   31 | 3,522.30 | End of investment period

     10.0 |    0 | -8,200.00 | Initial investment
     10.0 |    6 | 1,294.31 | End of computer usage
     10.0 |   31 | 2,292.95 | End of investment period

     15.0 |    0 | -8,200.00 | Initial investment
     15.0 |    6 | 223.48 | End of computer usage
     15.0 |   31 | 516.93 | End of investment period

     20.0 |    0 | -8,200.00 | Initial investment
     20.0 |    6 | -648.01 | End of computer usage
     20.0 |   31 | -1,934.96 | End of investment period



Analysis for 2800 hours per year:
Return (%) | Year | NPV (Dollars) | Stage
------------------------------------------------------------
      5.0 |    0 | -8,200.00 | Initial investment
      5.0 |    6 | 13,456.79 | End of computer usage
      5.0 |   31 | 18,033.39 | End of investment period

     10.0 |    0 | -8,200.00 | Initial investment
     10.0 |    6 | 10,788.62 | End of computer usage
     10.0 |   31 | 19,112.69 | End of investment period

     15.0 |    0 | -8,200.00 | Initial investment
     15.0 |    6 | 8,646.96 | End of computer usage
     15.0 |   31 | 20,000.95 | End of investment period

     20.0 |    0 | -8,200.00 | Initial investment
     20.0 |    6 | 6,903.97 | End of computer usage
     20.0 |   31 | 20,615.15 | End of investment period


{% endhighlight %}
</div>
</div>
</details>

According to my NPV estimation, if I run training for roughly 4-6 days a month at full load on average, I could break even within 6 years. This of course includes assumptions, such as 10% return average market return, and that the cost of compute/electricity will remain similar but in retort I'll reference one of my favorite quotes; "All models are wrong but some are useful." 

An argument that my Dad and some of my coworkers make is that there is no point in purchasing such a machine when I'll likely often have the keys to a much more powerful cluster through an employer. There is some truth to this, as returning to Minerva after running training on many ```8x NVIDIA H100``` nodes can feel like going back to a Mercedes after driving a Koenigsegg. That being said, Minerva has the advantage of always being available for my use, while allowing me to retain complete ownership over what I create.

So to summarize, is Minerva worth it? Financially, maybe. Personally speaking, having such a great resource in my room encourages me to do more training runs more frequently - after all, I have no excuse to not train many of the things I would like to try out. I also really love having complete ownership of my creations, which Minerva allows me to retain. I hope that the work Minerva enables me to complete will have intrinsic value as well ;) 

# Future Upgrades
I designed Minerva to hopefully be my main personal computer for at least the next 10 years - with a motherboard that supports 7 PCIE Gen 5 slots, there is a lot of room to upgrade GPUs. 
Currently the limiting factor in getting more GPUs is my PSU (and my bank account), where all available VGA power connectors are already in use. When I can afford upgrades I plan to get a second PSU (plugged into an outlet on a different breaker to get  around the US 1600W limit), and then add several more GPUs. I've seen some really attractive listings for the ``AMD Radeon Instinct MI60 32GB HBM2 300W``, or I may get more ```NVIDIA 3090 TI 24G``` cards, or maybe even ``NVIDIA 4080 16G`` cards.  If I add more GPUs than my current 4, I'll definitely have to 3D print some more modifications to my case to be able to mount them.

Also, ideally I'd watercool the entire rig, although this would make it more difficult to transport, more complex, more expensive, and harder to maintain, so for now I've done air-cooling only. If I know that I won't be moving for a long time, I'll definitely watercool. This would allow for many more GPUs to fit in the case due to a lower thickness (the current 4x 3-Slot cards take up a lot of room that could be reduced) while maintaining great temperature.

There is also the potential option of bifurcating my PCIE slots to fit more than 7 GPUs, which would be really cool, although past 6 GPUs I'd have to use ``8x`` PCIE lanes instead of the current ``16x`` which I think would totally work well too.
