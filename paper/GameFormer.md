# GameFormer: Game-theoretic Modeling and Learning of Transformer-based  Interactive Prediction and Planning for Autonomous Driving

## Induction



## Model

![GameFormer_framework_overview.png](./pictures/GameFormer_framework_overview.png)

**GameFormer** : 交互预测和规划框架，采用 Transformer encoder-decoder architecture  

### Game-theoretic Formulation

定义问题并讨论指导模型设计和学习过程的 level-k game theory 



### Scene Encoding

介绍编码上下文信息的编码器

- **input representation** : 输入数据由 agent 的历史状态信息 $S_{p} \in R^{N \times T_{h} \times d_{s}}$ 和局部 vectorized map polylines $M \in R^{N \times N_{M} \times N_{p} \times d_{p}}$ 组成

  - $d_{s}$ 代表 state 属性的数量；$N_{m}$ 代表临近地图元素（如路线或人行横道）数量，其中每个都包含具有 $d_{p}$ 个属性的 $N_{p}$ 个 waypoints  

  - 所有输入都以当前 ego agent 位置为远点，方向为正方向进行坐标系变换，缺失值用 0 填补

- **agent history encoding** : 用 LSTM 编码每个 agent 的历史状态信息 $S_{p}$，得到 $A_{p} \in R^{N \times D}$，D代表隐藏层维度

- **vectorized map encoding** : 使用 MLP 生成地图特征张量 $M_{p} \in R^{N \times N}$ 以对齐特征维度 D，再将属于同一地图元素的 waypoints 分组，对同一组内的点使用 max-pooling 聚合特征以减少 token 数量，最终得到维度为 $M_{r} \in R^{N \times N_{mr} \times D}$ 的地图特征张量，其中 $N_{mr}$ 代表聚合地图元素数量 

- **relation encoding** : 将每个 agent feature 和对应 local map feature 拼接在一起得到 agent-wise 场景上下文张量 $C^{i} = [A_{p} , M_{p}] \in R^{(N + N_{mr}) \times D}$ ,再用 E 层 Transformer 编码器去建模每个 agent 上下文张量 $C_{i}$ 中所有场景元素间关系，最后输出一个场景上下文编码张量 $C_{s} \in R^{N \times (N + N_{mr}) \times D}$，作为后续解码器网络的通用环境背景输入。

### Feature Decoding with Level-k Reasoning

介绍整合了交互建模的解码器

- **modality embedding**

- **level-0 decoding**

- **interaction decoding**

- 



### Learning Process

提出了一种学习过程，该过程能建模不同推理层次之间的相互影响



## Experiments

