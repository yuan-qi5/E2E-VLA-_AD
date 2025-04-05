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

### Auto Labeling

- trajectory : 使用 GNSS 和 IMU传感器通过卡尔曼滤波器来估计车辆行驶路径
  -  对每一时间步，注释未来三秒的(60 frames)轨迹，轨迹数据使用以车辆为中心的全局坐标系
  -  有时 GNSS 数据的不稳定性会导致错误的轨迹，表现为显著的振动，使用一种启发式方法去识别并删除不准确的轨迹

> Kalman Filter（卡尔曼滤波器）：一种递归算法，用于估计动态系统中的状态。核心思想是融合预测值和实际观测值，在不确定性中找到最优估计，通过 “预测 --> 更新” 来不断迭代。在此场景中，GNSS 更新频率慢但稳定，IMU 高频率但漂移，可通过卡尔曼滤波器有效融合起来。
>
> IMU（惯性测量单元） ：主要用来测量物体加速度和角速度的传感器，并通过这些信息推测物体的速度与位置变化，在短时间内较为精确，但存在漂移问题，即随时间推移，推算出的结果会逐渐偏离实际（由加速度计和陀螺仪测量时误差导致）。

- objects :
  -










