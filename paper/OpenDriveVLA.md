# OpenDriveVLA: Towards End-to-end Autonomous Driving with  Large Vision Language Action Model

## Introduction



## Model




## Experiments

### train data 

采用 nuScenes 数据集

![OpenDriveVLA_overview_train_pipeline](./pictures/OpenDriveVLA_overview_train_pipeline.png)

### vision-language alignment

- agent：后处理 instance caption from Tod3cap，提供单个对象的 2D 视觉描述以及目标对应的 BEV 坐标

- scene token : 处理 multi-view scene 描述遵循 Lidar-llm，并合并为统一描述 

- map token : 从 ground-truth 注释中推导结构化语言描述

### driving instruction tuning 

- 采用多个从 nnScenes 派生的面向指令的数据集，统一为基于指令的 QA 格式

- 每个 QA 对都以结构化环境视觉 token 和自我车辆状态为条件，以确保一致性

### motion forecasting and trajectory prediction 

- 规划 ego vehicle 的未来路径和其他 agent 的未来轨迹，都采用 local coordinate

- 预测未来 3s 轨迹，每 0.5s 间隔采样，总共 6 waypoints

> local coordinate（局部坐标系）: 相对于某个特定对象建立的坐标系统
>
> global coordinate（全局坐标）：再整个环境中统一定义的参考坐标系，所有物体的位置都用这个系统来描述

### evaluations

- open-loop planning task on nuScenes benchmark ，评估第 1 秒、第 2 秒和第 3 秒的 L2 位移误差，以及在整个预测时间段内的平均碰撞率

- 在 driving instruction tuning 后直接在 VQA 上评估场景理解能力，采用 BLEU, METEOR, CIDEr, BERT-Score 等

> average collision rate ：用来衡量模型预测轨迹是否会导致车辆与其他目标发生碰撞

### implementation details

- 3D visual perception module 以 ResNet-101 作为骨干进行 2D 特征提取，在 3D 目标检测、目标追踪和地图分割进行多任务学习，生成空间分辨率为 $$200 \times 200$$ 的 BEV 特征图 

- scence token : global SceneSampler 应用二维自适应池化，将各视角池化后的特征拼接起来，形成一个 global scene token

- agent token & map token : 从各自的 Query Transformer 的最后一层中提取出来

- 每种类型的 token 通过一个独立的两层 MLP 进行映射，使用 GeLU 激活函数投影到语言空间

- LLM 采用 Qwen2.5-Instruct，训练时进行全参数微调 

> object tracking ：在连续的图像帧中，对已检测到的物体进行跨时间的身份保持与位置预测
>
> map segmentation ：对道路场景中的不同区域进行像素级别或区域级别的语义分割
>
> adaptive pooling ：不需要提前指定池化窗口大小，会自动调整池化窗口大小和步长，是输入被均匀划分，最终输出固定大小的特征图。

### main results 

![OpenDriveVLA_open-loop_planning_performance_comparison](./pictures/OpenDriveVLA_open-loop_planning_performance_comparison.png)

> ST-P3 (Spatio-Temporal Planning Prediction Protocol)：一个统一的评估协议，用于评估联合的轨迹预测 + 规划输出在自动驾驶场景中表现
>
> UniAD (Unified Autonomous Driving): 既是一个自动驾驶任务的统一框架也是一个大规模评估基础，把感知、预测、规划多个任务统一在一个 pipeline 中进行评估

![OpenDriveVLA_QA_performance](./pictures/OpenDriveVLA_QA_performance.png)


- 自我状态信息在轨迹预测中起着重要作用

- 由于数据存在分布不平衡问题，大量场景都是维持当前状态，模型因此过渡依赖 ego-state history，导致模型在预测时倾向于保守决策

![OpenDriveVLA_ablation](./pictures/%20OpenDriveVLA_ablation.png)





