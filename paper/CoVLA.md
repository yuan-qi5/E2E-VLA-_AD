# CoVLA : Comprehensive Vision-Language-Action Dataset  for Autonomous Driving

> 由于本人较菜(qwq)以及刚接触 vla ,会记录更多的基础知识,以引用方式区分论文部分与基础知识

## Dataset Constrcution 

- [raw data][1]
- [data sampling][2]
- [auto labeling][3]
- [auto captioning][4]

[1]:
[2]:
[3]:
[4]:

### raw data

### data sampling 

sample criteria :

- 汽车处于驱动档时记录

- 最高速度不超过 100 km/h

-  GNSS 数据持续可用

> GNSS (Global Navigation Satellite System) : 全球卫星导航系统，如北斗、GPS等，GNSS 提供典型数据有：经纬度、速度、时间戳、航向等，在自动驾驶中主要用于定位和轨迹对齐。

sample process :

- 选择三个衡量驾驶行为的特征（最大转角、最大加速度、转向灯），前两个特征分箱处理
  
- 统计它们的联合经验分布（即现实中出现频率）

- 先采用加性平滑（平滑参数为 50），再使用分布值的倒数作为采样时的概率权重

> additive smoothing（加性平滑）: 又称 Laplace smoothing，一种用于解决概率估计中 “零概率” 问题的技术，核心思想为在每个概率估计中人为的加上一个常数 $\alpha$，从而避免概率为 0 的情况。

### auto labeling

- trajectory : 使用 GNSS 和 IMU传感器通过卡尔曼滤波器来估计车辆行驶路径

   -  对每一时间步，注释未来三秒的(60 frames)轨迹，轨迹数据使用以车辆为中心的全局坐标系

  -  有时 GNSS 数据的不稳定性会导致错误的轨迹，表现为显著的振动，使用一种[启发式][5]方法去识别并删除不准确的轨迹

[5]:

> Kalman Filter（卡尔曼滤波器）：一种递归算法，用于估计动态系统中的状态。核心思想是融合预测值和实际观测值，在不确定性中找到最优估计，通过 “预测 --> 更新” 来不断迭代。在此场景中，GNSS 更新频率慢但稳定，IMU 高频率但漂移，可通过卡尔曼滤波器有效融合起来。
>
> IMU（惯性测量单元） ：主要用来测量物体加速度和角速度的传感器，并通过这些信息推测物体的速度与位置变化，在短时间内较为精确，但存在漂移问题，即随时间推移，推算出的结果会逐渐偏离实际（由加速度计和陀螺仪测量时误差导致）。

- objects :

  - traffic lights : 采用 OpenLenda-s 模型去注释交通信号灯的颜色和箭头方向

  - leading vehicles : 结合雷达和前向摄像头的数据，记录 leading vehicle 的速度、加速度以及相对于自车的位置

### auto captioning 

- rule-based captioning : 考虑车辆运动和检测到的对象的各个方面，包括速度、加速度、轨迹曲率、前方车辆的存在和红绿灯状态，

- VLM-based captioning : 补充基于规则的字幕忽略的信息（如特定标志或不常见的对象），采用预训练 VideoLLaMA2-7B生成，

  - 在一个 60 帧（即 3 秒）的时间窗口内处理抽取的八张代表帧，包括该窗口的第一帧和最后一帧，每个 30 秒场景被划分为 10 个时间窗口

  - 共生成了 10 万条 captions，结合基于规则的 captions，共得到 600 万条 captions

- hallucination mitigation：用规则生成的 caption 来约束和补充 VLM 的生成输出，以提升准确性，缓解幻觉

  - 将 rule-based captions 作为上下文提供给 VLM，同时再加上一条提示语，要求它补充未涵盖的内容

  - 再查询模型并分析它内部的 token 概率分布选取概率最高词作为补充信息，内容包括：道路类型，天气情况、潜在风险等

  - 最后将 rule-based caption 和生成的补充信息当作事实锚点，引导 VLM 生成自由文本格式的描述

## Data Analysis 

- statistics : 覆盖更多复杂/边缘驾驶场景，不仅是大量无聊的 “正常行驶” 状态

  - 车辆速度和转向角的分布在采样后变得更加均匀，低速数据点明显增加，而转向角为 0 度附近的集中峰值减弱，从而实现在不同区间之间更均衡表示

  - 呈现出更多样化的驾驶动作，其中 16.11% 的帧包含转向灯，22.90% 的帧中出现交通信号灯

- comparision : 与其他数据集对比表现出来的优势

  - 自动化方法，得以构建出一个规模远大于依赖人工标注的数据集
 
  - 包含轨迹标注，即记录车辆运动路径的数据，通常包括每个时刻的位置、速度、加速度等 

## Experiment 


## Supplementary Material

### heuristic trajectory filtering

- significant jumps（显著跳跃）:

  - 通过相邻点间距离过滤轨迹数据，给定 20 HZ 的记录频率和 100 km/h 的最大速度，则两点之间的距离最多应为 1.38 m，采用公差率为 1.15，阈值设为 1.59 m，超过此阈值轨迹将被过略。

- movement in the wrong direction（向错误的方向移动）：

  - 先手动检查了所有场景中的 400 个样本，确定了 43 个无效轨迹(10.75%)，观察显示，这些轨迹中振动频率为 10 HZ。

  - 先使用三点移动平均线对轨迹进行平滑处理，再计算平滑轨迹与原始轨迹线之间的差异，再分析这些差异的方差，如果方差超过某个阈值，则归类为无效。

  - 再测试数据集上验证，产生了 0.64 的精度(precision)和 0.75 的召回率(recall)，可将无效轨迹降至 2.6%，虽然假阳率较高（错误轨迹为阳），但对于数据集规模来讲是可接受的。
 








