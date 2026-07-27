# 05 IMU 与优化后端

## 5.1 IMU 数据类型

文件：`include/ImuTypes.h`，`src/ImuTypes.cc`  
命名空间：`ORB_SLAM3::IMU`

| 类型 | 含义 |
|------|------|
| `Point` | 单次测量：加速度 `a`、角速度 `w`、时间戳 `t` |
| `Bias` | 加速度计偏置 `(bax,bay,baz)`、陀螺偏置 `(bwx,bwy,bwz)` |
| `Calib` | 外参 `Tbc`/`Tcb`、噪声协方差 `Cov`、随机游走 `CovWalk` |
| `IntegratedRotation` | 单步陀螺积分（ΔR + right Jacobian） |
| `Preintegrated` | 帧间/关键帧间预积分量 |

重力常量：`IMU::GRAVITY_VALUE = 9.81`。

### Preintegrated 关键能力

- 状态增量：ΔR、ΔV、ΔP
- 对偏置的 Jacobian：`JRg, JVg, JVa, JPg, JPa`
- 协方差（通常 15×15）
- API：`IntegrateNewMeasurement`、`Reintegrate`、`MergePrevious`、`SetNewBias`、`GetDelta*`

## 5.2 Tracking 侧惯性处理

1. 外部调用 `Track*` 时传入 `vImuMeas`，或单独 `GrabImuData`。
2. 测量进入 `mlQueueImuData`（互斥保护）。
3. `PreintegrateIMU()`：
   - 取出上一帧到当前帧之间的 IMU
   - 更新 `mpImuPreintegratedFromLastKF` 与 Frame 级预积分
4. `PredictStateIMU()`：用预积分预测当前位姿与速度（丢失短时维持）。
5. 关键帧创建时把预积分挂到 `KeyFrame`，并重置偏置相关量。

位姿优化使用：

- `Optimizer::PoseInertialOptimizationLastFrame`
- `Optimizer::PoseInertialOptimizationLastKeyFrame`

## 5.3 LocalMapping 侧 IMU 初始化

入口：`LocalMapping::InitializeIMU(priorG, priorA, bFirst)`

概念步骤：

```text
收集一段有足够激励的关键帧窗口
  → 估计重力方向 Rwg、尺度 scale、偏置 bg/ba
  → Optimizer::InertialOptimization(...)
  → Map::ApplyScaledRotation（统一尺度与重力对齐）
  → Tracking::UpdateFrameIMU
  → 可选 FullInertialBA（bFIBA）
```

分阶段精炼（与论文/README 一致）：

| 阶段 | 时间尺度（约） | 目标 |
|------|----------------|------|
| 首次 Init | ~2s | 尺度误差通常 <5% |
| VIBA1 | ~5s | 加强惯性 BA |
| VIBA2 | ~15s | 进一步去掉强 prior，尺度约到 ~1% |
| ScaleRefinement | 更晚、周期性 | 单目惯性窗口再精炼 |

单目惯性通常使用更紧的 `priorG/priorA`；双目/RGB-D 惯性尺度已由视觉固定，主要估计重力与偏置。

## 5.4 Optimizer API

文件：`include/Optimizer.h`，`src/Optimizer.cc`（体量最大的模块之一）

### 视觉 BA / 位姿

| 方法 | 用途 |
|------|------|
| `BundleAdjustment` | 给定 KF/MP 集合 BA |
| `GlobalBundleAdjustemnt` | 地图全局视觉 BA |
| `LocalBundleAdjustment` | 局部 BA；另有 welding 重载 |
| `PoseOptimization` | 仅优化当前 Frame 位姿 |

### 视觉惯性

| 方法 | 用途 |
|------|------|
| `FullInertialBA` | 全图 VI-BA |
| `LocalInertialBA` | 局部 VI-BA |
| `PoseInertialOptimizationLastFrame/KeyFrame` | 前端 VI 位姿 |
| `MergeInertialBA` | 地图融合焊接区 |
| `InertialOptimization`（3 个重载） | IMU 初始化 / 尺度 / 偏置 |

### 回环 / 融合图优化

| 方法 | 用途 |
|------|------|
| `OptimizeEssentialGraph` | 7DoF/6DoF 位姿图 |
| `OptimizeEssentialGraph4DoF` | 惯性回环（yaw+位置） |
| `OptimizeSim3` | 两 KF 间 Sim3 |
| `Marginalize` | Schur 补边缘化 |

## 5.5 g2o 自定义类型

### 视觉投影边

文件：`OptimizableTypes.h/.cpp`

- `EdgeSE3ProjectXYZ` / `EdgeSE3ProjectXYZOnlyPose`
- Body/右目变体：`...ToBody`
- Sim3 投影：`EdgeSim3ProjectXYZ` 等

### 惯性相关

文件：`G2oTypes.h/.cc`

**顶点：**

| Vertex | 含义 |
|--------|------|
| `VertexPose` | 6DoF 位姿（含多相机） |
| `VertexPose4DoF` | 惯性位姿图 |
| `VertexVelocity` | 速度 |
| `VertexGyroBias` / `VertexAccBias` | 偏置 |
| `VertexGDir` | 重力方向 |
| `VertexScale` | 尺度 |
| `VertexInvDepth` | 逆深度（部分场景） |

**边：**

| Edge | 含义 |
|------|------|
| `EdgeMono` / `EdgeStereo` | 视觉重投影 |
| `EdgeInertial` | 预积分约束 |
| `EdgeInertialGS` | 含重力方向 + 尺度 |
| `EdgeGyroRW` / `EdgeAccRW` | 偏置随机游走 |
| `EdgePriorPoseImu` / `EdgePriorAcc/Gyro` | 先验 |
| `Edge4DoF` | 4DoF 位姿图边 |

辅助结构：`ImuCamPose`、`ConstraintPoseImu`、`InvDepthPoint`。

## 5.6 因子图直观理解

**前端帧位姿（VI）：**

```text
[MapPoints] --reproj--> [Pose_k]
[Pose_{k-1}, Vel, Bias] --preint--> [Pose_k, Vel_k, Bias]
```

**局部 VI-BA：**

```text
窗口内 KeyFrames 的 Pose / Vel / Bias
  + 观测到的 MapPoints
  + 相邻 KF 预积分边
  → 联合优化
```

**回环（惯性）：**

```text
先用 Sim3/SE3 对齐公共区域
  → 4DoF Essential Graph（位置 + yaw）
  → 可选 FullInertialBA
```

## 5.7 调试与耗时统计

在 `include/Config.h` / 相关宏中启用：

```cpp
#define REGISTER_TIMES
```

可输出各线程耗时统计（终端 + `ExecTimeMean.txt` 等）。详见官方 README 第 7 节。
