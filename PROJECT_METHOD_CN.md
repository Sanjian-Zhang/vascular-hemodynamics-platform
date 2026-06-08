# 血管网络粒子流动近似模型说明

## 1. 项目目标

本项目不是直接做完整 CFD，而是基于 `lc_3.vtk` 中已有的中心线、直径、流量、速度、压强和血细胞压积数据，构建一个可交互的血管网络粒子流动近似模型。

当前版本同时服务于两个目标：

1. 做到分叉口粒子分流在视觉上自然、连续、可解释。
2. 让可视化结果可以反向暴露网络拓扑或后端流体计算中的异常。

## 2. 当前数据基础

`lc_3.vtk` 当前可用字段包括：

- `POINTS`
- `LINES`
- `pressure`
- `diam_um`
- `radius_um`
- `flow_nl_min`
- `velocity_mm_s`
- `hematocrit`
- `shear_stress`

因此当前实现不再只依赖几何近似，而是优先使用原始 `flow_nl_min` 和 `pressure` 参与粒子路由与诊断。

## 3. 当前实现路线

### Step 1. 图结构与分叉识别

血管中心线被转成图结构：

- `Node = POINTS`
- `Edge = LINES` 拆分后的血管段

同时按节点连接度识别：

- `degree = 1` 端点
- `degree = 2` 连续段
- `degree >= 3` 分叉点或汇合点

### Step 2. 基于真实流量的分叉路由

当前版本已经不是单纯的 `diameter^3` 分流，而是采用真实 `flow_nl_min` 主导的路由权重，并叠加几何连续性与直径偏置：

```text
score_i ∝ Q_i^α · d_i^β · continuity_i · occupancy_i
```

当前实现参数取值为：

- `α = 1.0`
- `β = 0.45`

其中：

- `Q_i = |flow_nl_min|`
- `d_i = diam_um`
- `continuity_i` 由分叉角度、管径连续性和速度连续性共同决定
- `occupancy_i` 用于把粒子数稳定拉回目标通量分布

这一步对应的是：

- `flow_nl_min` 主导分流
- `diam_um` 作为直径偏置
- `velocity_mm_s` 调整粒子运动速度

### Step 3. RBC-like 粒子行为

当前红细胞粒子系统包含以下近似行为：

1. 红细胞双凹形实例化渲染，而非简单点粒子。
2. 在分叉口加入短暂停留 `junction lingering`，避免粒子机械瞬时转向。
3. 使用 `hematocrit` 控制粒子大小和局部通量感。
4. 使用目标粒子占有量 `target occupancy` 做粒子生成/消亡调节。

目标占有量使用如下近似：

```text
target_i ∝ |Q_i| · d_i^0.45 · (0.42 + 1.35 · hematocrit_i)
```

它不是细胞级生物物理模拟，但已经能稳定表达：

- 粗分支粒子更多
- 大流量分支粒子更多
- 高血细胞压积分支更“红”更“密”

## 4. 当前诊断能力

当前查看器已经支持以下一致性检查：

### 4.1 质量守恒检查

对每个节点统计：

```text
Σ inflow
Σ outflow
massConservationError = |Σout - Σin| / max(Σin, Σout)
```

用于发现：

- `All inflow / no outflow`
- `All outflow / no inflow`
- `Outflow exceeds inflow`
- `Inflow exceeds outflow`

### 4.2 分支连通度检查

对每个分叉点分析：

- 上游进入方向
- 下游离开方向
- 管径匹配
- 速度连续性

如果分叉几何过于突兀，会被标为：

- `Weak branch continuity`

### 4.3 压力-流量一致性检查

当前版本额外引入了 `pressure`：

- 每条边记录 `pressure_start`
- 每条边记录 `pressure_end`
- 计算 `pressure_drop = pressure_start - pressure_end`

再与 `flow_nl_min` 方向比对：

- 若流向与压降方向明显相反，则记为 `Pressure-flow mismatch`

这可以帮助发现：

- 局部方向编码错误
- 分叉口流向设置错误
- 后端压力/流量场不一致

## 5. 当前版本定位

当前版本最准确的表述是：

```text
Graph-based vascular flow viewer with flow-driven RBC-like particle routing,
junction continuity analysis, mass-conservation diagnostics, and pressure-flow consistency checks.
```

它已经明显超过“纯展示版”，但还没有进入严格 Poiseuille 网络求解或细胞尺度 RBC 直接数值模拟阶段。

## 6. 后续可继续增强的方向

### 6.1 Step 2+：Poiseuille 网络求解

如果后续需要做更强物理解释，可进一步引入：

```text
R = 8 μ L / π r^4
Q = ΔP / R
```

再根据入口/出口边界条件求解节点压力场，得到网络相对流量。

### 6.2 Step 3+：更强 phase separation

如果后续要往科研展示推进，可继续加入：

- 细分支 RBC 进入概率衰减
- 分叉不对称偏置
- 局部滞留时间与剪切应力关联
- 粒子密度与 `hematocrit` 的显式映射

## 7. 建议汇报表述

可以直接使用下面这段中文：

> 我们首先将血管中心线构建为图结构，并利用节点连接度自动识别分叉点与汇合点。在粒子路由阶段，不再仅依赖几何直径估计，而是使用原始 `flow_nl_min` 作为主导分流依据，同时结合分支连续性、血管直径偏置与血细胞压积信息，生成更符合微循环直觉的红细胞样粒子运动效果。进一步地，系统在每个节点同步计算流量守恒误差、分支连通度以及压力-流量一致性，从而使该可视化不仅用于展示，也可作为后端血流模拟结果的结构化校验工具。 

