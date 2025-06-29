# nuScenes: A multimodal dataset for autonomous driving

## Induction

### What is the main content ?







## 官网介绍

### 数据整体结构

nuScenes 是包含 1000 个驾驶场景、每个持续 20s 的自动驾驶数据集，在数据集上以 2HZ 的频率标注 23 种物体种类，并使用精确的 3D 边界框进行标注。

nuScenes 是首个提供自动驾驶车辆全部传感器套件数据（6 个摄像头、1 个激光雷达、5 个雷达、GPS、IMU）的大规模数据集，全面考虑所有传感器。

在 nuScenes-lidarseg 中，我们对 nuScenes 中的每个关键帧的激光雷达点标注了 32 种可能的语义标签之一（即激光雷达语义分割）。因此，nuScenes-lidarseg 包含 4 万个点云和 1000 个场景中总共 14 亿个标注点（其中 850 个场景用于训练和验证，150 个场景用于测试）。

### 数据集结构

五个文件夹：`maps`，`samples`，`sweeps`，`v1.0-train`，`v1.0-test` 

`maps` 文件夹：4 张地图照片。

`samples` 文件夹：针对关键帧，传感器（6 个相机、1 个激光雷达、5 个毫米波雷达）所采集到的信息。

`sweeps` 文件夹：结构与 `samples` 文件夹相同，是 intermediate frames（过度帧或中间帧）的传感器数据

`v1.0-train` 文件夹：

`v1.0-test` 文件夹

















