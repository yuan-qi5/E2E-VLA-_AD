# Asynchronous Large Language Model Enhanced Planner for Autonomous Driving 


## Induction

currrent motion planning framework :

- learning-based real-time motion planning : 利用 vectorized map information 作为 input，employ a decoder 去预测轨迹

   - 易受长尾现象影响，在 rare or unseen scenarios 性能显著下降
 
   - 可控性低
     
- rule-based motion planning : 使用预先制定的规则去预测

  - 不足以捕捉整个潜在的复杂场景，易趋向于极端，要么过度谨慎，要么过度激进
 
  - 可控性低

- LLM-based motion planning : 使用 LLM 去预测，

  - 场景信息通过语言去描述，会受到 input token length 限制，使得捕捉精确且全面场景信息有挑战性
 
  - 语言输出预测轨迹不够精确 ：输出 high-level commands 容易引入误差，控制不够精确；直接用语言生成轨迹坐标，LLM 并不擅长精细的数值预测
 
  - prevalent framework 主要使用 LLMs 作为核心决策者，与 real-time planner 相比推理速度明显降低
    
![different_learning-based_AD_framework](./pictures/different_learning-based_AD_framework.png)

> nuPlan ：由 Motional 发布的大规模自动驾驶规划数据集，主要用于研究路径规划 (planning) 任务，特别强调在闭环训练和评估环境 (closed-loop training / evaluation) 中测试学习驱动的规划模型 (learning-based planner)
>
> learning-based planner : 指用机器学习方法（特别是深度学习）来学习自动驾驶的规划模块
>
> real-time planner : 能够在极短时间内（毫秒级）做出路径或轨迹决策的模块或算法
> 
> vectorized map（向量化地图）: 用几何向量形式（点、线、面）表示地图中的关键元素，而不是像素点的图像（raseter map）来表示地图信息，更像是用 “坐标 + 属性 + 拓扑” 构成的可计算对象
>
> raseter map（栅格地图）: 把地图以像素网格（pixels）形式存储的数据结构，本质是一个二维数据/图像矩阵，每个 pixel 包括 RGB 值（纯图像）和类别标签（语义地图）
>
> asynchronous（异步）：在编程中，异步表示多个任务并发执行而不是一个个排队顺序执行
>
> inference frequency（推理频率）：指 LLM 被调用进行推理的次数或频率
>
> model predictive control（模型预测控制）：一种基于模型的最优控制方法，核心思想：每一个控制时刻，利用系统动力学模型，在**未来一个时间窗口内预测系统行为**，通过**优化控制序列**使未来轨迹尽可能好（如偏离最小、控制能耗最小），只执行第一个控制动作，下一次再重新预测。
>   > intuition : MPC = 预测未来 + 求最优方案 + 实时滚动更新

## Data

nnPlan dataset : 第一个大规模自动驾驶规划基准，由 1200 hours 真实驾驶数据组成。

从 nnPlan Train and Val 中开发 pre-training and fine-tuning dataset ，集中于 14 个官方挑战性场景。

目的是为了增强 LLM 对自动驾驶中指令的理解能力。

### Pre-training Data Generation

- planning-QA
  
   - 由 rule-based 途径生成，为增强 LLM 对 waypoint、high-level instructions、control 间关系的理解。

   - 包含六种问题类型，每一种都侧重于在 waypoint、high-level instructions、control 的转换

- reasoning-1K

   - 由 GPT-4 生成的 1000 条数据，除了包含答案外还包含基于场景的推理与解释

### Fine-tuning Data Generation

- 为了进一步增强多模态理解和对齐，构建了一个基于 10,000 场景的微调数据集，每 8s 捕获一帧，得到了包含 180,000 帧的训练集和 20,000 帧的验证集，每一帧都包含 vectorized map data 和 linguistic prompts

   - 训练集和测试集中的场景类型分布与整个 nuPlan train-val dataset 分布保持一致 

-  对于 vectorized scene information，包含自车本身的信息，在 20 帧历史轨迹中 20 个周围智能体信息和椅子车为中心的全球地图数据

-  对于 LLM prompt ，由 system prompt 和 系列 routing instructions 组成。

   - 关于 routing instructions，使用基于规则的方法将路径点（pathway）转化为一系列带有距离信息的指令。训练集使用自车在未来 8 秒内的真实轨迹作为生成路径指令的基础，仿真时根据当前场景观察，通过人工设定方法，在规定的最大路径长度范围内，找到一条参考路径来生成路径指令。

## Methodology

AsyncDriver (asynchronous LLM-enhanced closed-loop framework): 由两部分组成 ： 

- Scene-Associated Instruction Feature Extraction Module

- Adaptive Injection Block

![asyncdriver_framework](./pictures/asyncdriver_framework.png)

### Scene-Associated Instruction Feature Extraction Module

**Multi-modal Input** : 

- 在每个 planning iteration，vectorized scene information 从模拟环境中获得。其中包括 ego 和 other agents 的历史轨迹和状态信息以及全球地图数据。

- real-time planner 的矢量化场景信息以相同方式提供，所有矢量数据都是相对于 ego 位置的。

- 随后通过 vector map encoder 和 map adapter 处理，map embeddings 和 language embeddings 一起送到 Llama2-13B 以得到 hidden features。

**Alignment Assistance Module**

- 确定自动驾驶过程中多任务预测的五个关键场景，由 5 个独立的 2 层 MLP 预测头预测

- ego vehicle : 使用回归任务估计车辆在 X、Y 方向上的速度和加速度

- map information : 执行分类任务用于识别当前车道左右两侧是否有相邻车道和判断当前车道关联的交通灯状态

- towards future navigation : 进行分类任务，识别是否需要在未来轨迹中进行换道 (lane change)，和未来的速度决策，即加速、减速或保持当前速度三种选项。

- 注意，对齐辅助模块尽在训练阶段用于帮助多模态特征对齐，推理阶段时不使用

### Adaptive Injection Block

- 采用类似 Gameformer 解码器结构作为基础的 

### Asynchronous Inference 

- 利用大语言模型来引导实施规划器，通过一系列灵活组合的语言指令，在不破坏原有结构完整性的前提下，显著提升了系统性能

- 实现可控的异步推理机制，有效地解耦了 LLM 与实施规划器地推理频率，因此 LLM 无需再每一帧都参与推理。

- 在异步时间间隔内，先前由 LLM 得出的高层指令特征仍然会持续引导实时规划器的预测过程，显著提升了整体推理效率并大幅降低了由 LLM 引入的计算成本

- notably，作者的框架支持一系列灵活组合的路径指令，从而能够提供长期的、高层次的路径规划洞见（routing insights），因此即使在异步时间间隔内，先前生成的高层次指令特征仍然能提供有效指导

- 实验结果表明将 LLM 设置为每 3 帧推理一次，能够将 LLM 推理时间减少将近 40%，但准确率仅下降 1%，表明了在准确率与推理速度之间实现了最优平衡。


### Training Details

- pre-training stage :

   - 使用 Reasoning1K 数据集，并结合从 Planning-QA 随机选取的 1500 个样本，采用 LoRA 微调
 
   - 使得 LLM 从一个通用型大语言模型转变为专门针对自动驾驶任务优化的模型

- fine-tuning stage : 

   - VectorMap encoder and decoder 架构保持不变，加载在同一数据集上预训练好的实施规划器的权重，以提升训练稳定性
 
   - 微调时损失由两部分组成：对齐辅助损失（Alignment Assistance Loss）和规划损失（Planning Loss）。对齐辅助损失划分为五个部分，使用 L1 损失预测自车的速度与加速度；使用交叉熵损失预测速度策略； $\tilde{x}_{dec}$
 
![async_ft_loss](async_ft_loss.png) 


## Experiment

$$L_1(\tilde{x}_{val})$$

\( L_{align} = L_1(\tilde{x}_{val}, x_{va} ) \)



