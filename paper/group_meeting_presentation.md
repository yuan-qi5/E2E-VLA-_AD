# 2025/04/10

个人感受 (CoVLA, OpenDriveVLA, AlphaDrive, DriveLMM-o1) ：

自动驾驶核心任务 ：感知、预测、规划

- end-to-end (input --> trajectory)：

  - CoVLA : 提出一个能缓解 “长尾” 问题的数据集 [1] [2] [3] ,而且是全自动
 
  - OpenDriveVLA : 优化模型建模，优化从 2D --> 3D  [4]

- 尝试将推理引入到 VLA 

  - DriveLMM-o1 : 构建逐步推理数据集 [5] 

  - AlphaDrive : 引入强化学习 GRPO [6]

  - OpenDriveVLA : [7]
    
- 数据重要性 

    
[3]:https://github.com/yuan-qi5/VLA/blob/main/paper/CoVLA.md#data-sampling
[2]:https://github.com/yuan-qi5/VLA/blob/main/paper/CoVLA.md#data-analysis
[1]:https://github.com/yuan-qi5/VLA/blob/main/paper/OpenDriveVLA.md#main-results
[4]:https://github.com/yuan-qi5/VLA/blob/main/paper/OpenDriveVLA.md#model
[5]:https://github.com/yuan-qi5/VLA/blob/main/paper/DriveLMM-o1.md#benchmark 
[6]:https://github.com/yuan-qi5/VLA/blob/main/paper/AlphaDrive.md#model
[7]:https://github.com/yuan-qi5/VLA/blob/main/paper/OpenDriveVLA.md#driving-instruction-tuning

