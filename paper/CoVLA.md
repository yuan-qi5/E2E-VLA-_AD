# CoVLA : Comprehensive Vision-Language-Action Dataset  for Autonomous Driving

> 由于本人较菜(qwq)以及刚接触 vla ,会记录更多的基础知识,以引用方式区分论文部分与基础知识

## Dataset Constrcution 

### raw data

### data sampling 

sample criteria :

- 汽车处于驱动档时记录

- 最高速度不超过 100 km/h

-  GNSS 数据持续可用

> GNSS (Global Navigation Satellite System) : 全球卫星导航系统，如北斗、GPS等，GNSS 提供典型数据有：经纬度、速度、时间戳、航向等，在自动驾驶中主要用于定位和轨迹对齐。

sample process :

- 选择三个衡量驾驶行为的特征（最大转角、最大加速度、转向灯），前两个特征分箱处理

- 统计它们的联合经验分布（即显示中出现频率）

- 先采用加性平滑（平滑参数为 50），再使用分布值得倒数作为采样时的概率权重

> additive smoothing（加性平滑）: 又称 Laplace smoothing，一种用于解决概率估计中 “零概率” 问题的技术，核心思想为在每个概率估计中人为的加上一个常数 $\alpha$，从而避免概率为 0 的情况。













