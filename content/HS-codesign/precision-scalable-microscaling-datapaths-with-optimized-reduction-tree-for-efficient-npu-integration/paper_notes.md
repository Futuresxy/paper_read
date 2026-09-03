# Precision-Scalable Microscaling Datapaths with Optimized Reduction Tree for Efficient NPU Integration 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Stef Cuyckens, Xiaoling Yi, Robin Geens, Joren Dumoulin, Martin Wiesner, Chao Fang, Marian Verhelst

**发表期刊/会议 (Journal/Conference)**: unknown

**发表年份 (Publication Year)**: 2025

**研究机构 (Affiliations)**: ESAT-MICAS, KU Leuven

---

## 1. 摘要

**目的**

- 面向边缘设备**持续学习** 场景（机器人、可穿戴健康监测、自动驾驶），下一代 NPU 需同时支持**训练与推理**，对 MAC 阵列提出**精度可扩展**与**高能效**的双重需求。
- **Microscaling (MX)** 标准通过 32 元素共享 8-bit exponent 的分组机制，以窄位宽元素格式（INT8、FP8、FP6、FP4）兼顾低推理成本与训练所需的动态范围，但现有 MX MAC 设计存在关键权衡：
  - **整数累加**：需将窄浮点乘积进行代价高昂的格式转换；
  - **FP32 累加**：存在量化损失且归一化开销大。
- 核心动机：SotA MX MAC 中**归约树占面积 88.7%、能耗约 85%**（Fig. 1），是优化的首要目标；同时现有 SNAX 数据流器为静态最坏情况带宽配置，在低精度模式下造成内存通道浪费与 bank 争用。

![](images/dc9c4b234b7f43ce68cfd57e2037a6884e07e0650b6b342e533ac30907bf0f8a.jpg) *Fig. 1. Resource breakdown of the state-of-the-art precision-scalable Microscaling (MX) multiply-accumulate (MAC) unit [15], where more than 80% of the resources go to the reduction tree.*

---

**方法**

- **混合精度可扩展归约树**：
  - 输入为四个带 6-bit exponent 的 10-bit significand 及两个 8-bit 共享 exponent；
  - 融合 **early-accumulation** 方案至 [15] 的 FP32 归约树，使 L2 加法器仅需 **28-bit** 信号宽度，避免加法器过配，同时降低 L2 与累加之间的归一化和对齐开销；
  - 通过 **MUX 复用机制**（Fig. 3b）：乘积和的 24-bit 扩展仅在左移或右移其中一种情况需要，二者互斥，将归一化输入宽度从 **77-bit 压缩至 53-bit**。

![](images/a95e4644fdfbd01b687f331a3b348780b10094408b0a7acdc6128538cb512bbb.jpg) *(d)*

- **累加精度优化**：
  - 分析各模块位宽随存储部分和 mantissa 宽度的缩减规律：L2 对齐/加法按比例缩减，累加对齐、加法、归一化以两倍速率缩减；
  - 通过对比**加法误差**与最终 MX 量化误差（归一化至 FP64 基准），在均匀分布与高斯分布、64×64 与 256×256 矩阵规模下寻找误差平衡点，最终选定 **16-bit mantissa** 作为最坏情况下的实现位宽。

![](images/41d28965663a6d74e8dc1ed03a41cf3793380cb88bf4f3b8c542cb4236abaa71.jpg)

- **NPU 系统集成**（SNAX 平台）：
  - 64 个 MX MAC 组成 **8×8 空间阵列**，支持横向与纵向数据复用，按精度模式在 8/2/1 周期内完成 8×8 矩阵 GeMM；
  - 架构包含三个子模块：灵活 **FSM** 控制器、精度可扩展空间阵列、**SIMD 量化单元**（将 64 个浮点输出转换回任意 MX 格式）；
  - 通过 Snitch RISC-V 核心的 **CSR 接口**统一编程：CSR0 选择精度模式，CSR1 定义累加次数，CSR2 定义矩阵 tile 尺寸；
  - 扩展 SNAX 数据流器支持**动态通道门控**：MXINT8、MXFP8、MXFP6、MXFP4 模式分别仅激活 1/4/3/4 个内存通道，减少无效访存能耗与 bank 争用；可编程 **AGU** 支持不同精度模式各自优化的数据布局与访问模式。

![](images/fa523805476e069e2857d5d2aa2e67b7fc2750e6dd41c1e6bd90f07c5dc87e27.jpg) *Fig. 5. System architecture overview with precision-scalable MX tensor core integration.*

---

**结果**

- **MAC 级评估**（GlobalFoundries 22FDX，0.8V，100–1800 MHz 综合）：
  - 全面优于 Long integer addition [24]：面积与能耗在全频段占优，最高频率达 **1800 MHz**（对方仅 1100 MHz）；
  - 对比 FP32 addition [15]：MXFP8 与 MXFP4 模式在 1 GHz 以下能效更优，面积效率在 500–1000 MHz 区间领先；MXINT8 模式下 FP32 方案仍略优。

![](images/07f30bd24fdf22e77a78a904e688697c91698d31d35a3fa5b7b494bfce95bcea.jpg)

- **系统级评估**（500 MHz）：
  - 在 ResNet18 与 Vision Transformer 的推理（INT8）与训练（FP8 E4M3）工作负载上，计算利用率达 **94.41%–99.51%**，证明控制与内存瓶颈被有效抑制。

![](images/1b1201d92346d5ed0b44ae7b6af3960077d9403bbd9e29461eb035b8e2f3f513.jpg)

  - 系统面积 0.60 mm²，MX tensor core 占 29.5%，其余为多 bank SPM 与数据供给；得益于通道门控与时钟门控，能耗由 MX core 主导。

- **SotA 对比**：

| 指标 | [24] | [15] | [25] | **本工作** |
|---|---|---|---|---|
| 频率 | 1000 | 400 | 200 | **500** |
| 工艺 | 12 | 16 | 16 | **22** |
| Area (mm²) | 0.59 | 8.92 | 0.62 | **0.60** |
| Area/MAC (µm²) | 3150 | 2080 | 144 | **2766** |
| 支持精度 | MXFP8 | 全 MX | INT8 | **全 MX (INT8/FP8/FP6/FP4)** |
| 吞吐 | 102 | / | 204 | **MXINT8: 64 / MXFP8/6: 256 / MXFP4: 512** |
| 能效 | 356 | 3597 (MXFP4) | 4680 | **MXINT8: 657 / MXFP8/6: 1438–1675 / MXFP4: 4065** |
| NPU 集成 / 精度可扩展 | ✓ / ✗ | ✗ / ✓ | ✓ / ✗ | **✓ / ✓** |

- 相较前 SotA [15]，能效提升 **1.59×（MXINT8）、3.05×–3.21×（MXFP8/6）、1.13×（MXFP4）**。

---

**结论**

- 提出**混合精度可扩展归约树**，结合整数累加免归一化与浮点累加窄加法器宽度的双重优势，通过 MUX 复用扩展位与 **16-bit mantissa 精度松弛**进一步压缩硬件成本。
- 将 8×8 MX tensor core 集成至 **SNAX** NPU 平台，通过 CSR 统一控制与**动态带宽门控数据流器**，在系统级消除控制与内存瓶颈，实现 94% 以上的计算利用率。
- 最终系统在同时具备**全 MX 精度可扩展性**与**NPU 集成能力**的前提下，于多数精度模式上超越既有 SotA 能效，是面向边缘持续学习训练-推理统一平台的高效解决方案；代码已开源。

---

## 2. 背景知识与核心贡献

**研究背景**

- 边缘端**持续学习**应用（机器人、可穿戴健康监测、自动驾驶）要求设备在保持**能效**与**面积效率**、满足低延迟约束的同时，能适应动态变化的环境。
- 为降低 SoC 面积与成本，**训练**与**推理**负载需运行在同一计算阵列上，构成统一的 **training-inference NPU 平台**。
- 两种负载的精度需求差异巨大：
  - **推理**：依赖 **INT8/INT4** 等紧凑整数格式，最小化硬件成本与能耗。
  - **训练**：需要更大的**动态范围**（传统上由 **FP32** 提供）以保证模型收敛。
- **Microscaling (MX)** 标准通过**共享指数**（8-bit shared exponent）分组（32 元素一组）窄位宽元素（FP8/FP6/FP4/INT8），在保留动态范围的同时大幅降低位宽，成为统一训练-推理的理想数据格式。

![](images/dc9c4b234b7f43ce68cfd57e2037a6884e07e0650b6b342e533ac30907bf0f8a.jpg) *Fig. 1. Resource breakdown of the state-of-the-art precision-scalable Microscaling (MX) multiply-accumulate (MAC) unit [15], where more than 80% of the resources go to the reduction tree.*

**研究动机**

- **归约树是硬件瓶颈**：先前工作表明，累加树占据了 MX MAC 的 **88.7% 面积**与约 **85% 能耗**（Fig. 1）。
- 现有两条 SotA 技术路线各有硬伤：
  - **FP32 addition 方案**（[15]）：需昂贵的**归一化**逻辑，且存在量化损失。
  - **Long integer addition 方案**（[24]）：将窄浮点积转换为宽整数，**格式转换成本高**，累加器与归一化位宽过大（67-bit 整数、77-bit 归一化输入）。
- **系统级瓶颈**：SNAX NPU 集成平台的 data streamer 按静态最坏情况带宽配置，在低精度运算时导致**内存通道部分闲置**，产生动态功耗并加剧 **bank contention**。

![](images/21466dc31e25d1155fd05655f17ff102f7dbed0cef7507c54a62bc30676bbaeb.jpg) *Fig. 2. Overview and issues of the state-of-the-art reduction trees for MX MAC implementations: FP32 addition [15], and Long integer addition [24]. Followed by the solutions proposed in this work.*

---

**核心贡献**

- **算术单元级：提出混合精度可扩展归约树**
  - 融合两种 SotA 方案的优点：利用**整数累加可跳过归一化**的特性 + **浮点累加加法器位宽更窄**的优势。
  - 引入 **early-accumulation** 机制避免 L2 加法器过度位宽配置，同时减少 L2 加法与累加之间的归一化和对齐开销。
  - 通过 **MUX 复用 24-bit 扩展位**（左侧或右侧扩展不可同时使用），将归一化输入位宽从 **77-bit 压缩至 53-bit**。

- **累加精度优化：受控的精度松弛**
  - 通过对比**加法误差**与**量化误差**（以 FP64 为基准），确定临界尾数位宽。
  - 在量化误差占主导的前提下，将存储部分结果的尾数宽度从 **23-bit（FP32）降至 16-bit**，进一步压缩 L2 对齐、加法、累加与归一化模块的位宽。

![](images/41d28965663a6d74e8dc1ed03a41cf3793380cb88bf4f3b8c542cb4236abaa71.jpg)

- **NPU 级：8×8 MX tensor core 集成到 SNAX 平台**
  - 空间阵列支持水平/垂直数据复用，按精度模式在 8/2/1 周期内完成 GeMM（分别对应 INT8、FP8/FP6、FP4）。
  - 通过 **CSR** 提供 32-bit/周期的统一编程接口，运行时动态配置精度模式、累加深度与矩阵分块尺寸。
  - 扩展 SNAX **data streamer** 支持**动态通道门控**：INT8、FP8、FP6、FP4 模式分别仅激活 1、4、3、4 个内存访问通道，减少无效内存流量与能耗。
  - 可编程 **AGU** 支持各精度模式专属的数据布局与访问模式。

**关键性能结果**

| 指标 | MXINT8 | MXFP8/FP6 | MXFP4 |
|---|---|---|---|
| 能效 (GOPS/W) | 657 | 1438–1675 | 4065 |
| 吞吐 (GOPS) | 64 | 256 | 512 |
| 相对 SotA [15] 能效提升 | 1.59× | 3.05×–3.21× | 1.13× |

- 系统在 ResNet18 与 Vision Transformer 的推理/训练负载下实现 **94.41%–99.51%** 的计算利用率，总面积仅 **0.60 mm²**（22 nm 工艺）。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**架构定位**

本文提出一个面向**训练-推理融合边缘 NPU 平台**的完整技术方案，核心是**精度可扩展的 Microscaling (MX) 数据通路**，整体架构自底向上分为三个层次：

- **算术单元层 (MAC Level)**：混合精度可扩展 reduction tree + 累加精度优化
- **张量核层**：8×8 空间阵列 + SIMD 量化单元 + FSM 控制
- **系统集成层**：集成至 SotA NPU 平台 **SNAX**，扩展 CSR 控制接口与动态数据流

---

**一、混合 Reduction Tree (Hybrid Reduction Tree)**

设计动机来自对两种 SotA 方案缺陷的融合改进：

![](images/21466dc31e25d1155fd05655f17ff102f7dbed0cef7507c54a62bc30676bbaeb.jpg) *Fig. 2. Overview and issues of the state-of-the-art reduction trees for MX MAC implementations: FP32 addition [15], and Long integer addition [24]. Followed by the solutions proposed in this work.*

- **FP32 addition 方案 [15]**：完整支持全部六种 MX 格式，但需要昂贵的归一化与对齐逻辑，且累加树占据 MAC 面积的 **88.7%**、能耗的 **约 85%**
- **Long integer addition 方案 [24]**：采用 early-accumulation 免除归一化，但仅支持 MXFP8，且需 67-bit 甚至更宽的整数加法器

本文的混合方案 (如 Fig. 3 所示) 取两者之长：

![](images/a95e4644fdfbd01b687f331a3b348780b10094408b0a7acdc6128538cb512bbb.jpg) *(d)*

- **第一迭代**：将 [24] 的 **early-accumulation** 机制集成进 [15] 的 FP32 reduction tree
  - L2 加法器仅需 **28-bit** 信号宽度，避免过度配置
  - 累加端扩展 24-bit 以支持 FP32 尾数左移对齐，再扩展 24-bit 用于右移重附着，导致归一化需支持 **77-bit** 输入
- **第二迭代 (MUX 优化)**：利用左扩展与右扩展**互斥**的特性，通过 MUX 仅保留一个 24-bit 扩展，将归一化输入宽度缩减至 **53-bit**
- 输入接口：四个 **10-bit significand** (带 6-bit 指数) + 两个 **8-bit shared exponent**

---

**二、累加精度优化**

- **核心思想**：中间结果最终必然被量化回 MX 格式，量化误差不可避免；只要**加法误差 ≤ 量化误差**，缩减尾数位宽即无损
- **误差分析**：以 FP64 为基准，对比 uniform 与 Gaussian 两种分布、64×64 与 256×256 两种矩阵规模下加法误差与量化误差的交点
- **结论**：最终采用 **16-bit mantissa** 作为存储部分和的尾数宽度
- **硬件收益传导**：
  - L2 对齐与加法模块随尾数宽度**成比例缩减**
  - 累加端的对齐、加法、归一化模块**每减少 1-bit 尾数缩减 2-bit** 位宽

![](images/41d28965663a6d74e8dc1ed03a41cf3793380cb88bf4f3b8c542cb4236abaa71.jpg)

---

**三、MX Tensor Core (8×8 空间阵列)**

![](images/fa523805476e069e2857d5d2aa2e67b7fc2750e6dd41c1e6bd90f07c5dc87e27.jpg) *Fig. 5. System architecture overview with precision-scalable MX tensor core integration.*

由三个子模块构成：

- **FSM (有限状态机)**：生成微架构控制与握手信号，根据矩阵尺寸编排整个 GeMM 执行流程
- **Precision-scalable MX 空间阵列**：64 个 MX MAC 组成 **8×8 二维 mesh**，支持水平与垂直数据复用；单次 GeMM 在不同精度下耗时：
  - MXINT8：**8 cycles**
  - MXFP8/FP6：**2 cycles**
  - MXFP4：**1 cycle**
- **SIMD 量化单元**：位于阵列下游，将 64 个浮点输出转换为任意支持的 MX 数值格式 (计算 shared exponent 后再量化)

---

**四、SNAX NPU 系统集成**

**控制通路 —— Unified CSR Control**

- 由 RISC-V **Snitch** 核心 (RV32IMAFD) 通过标准 **CSR 写指令**编程，带宽 **32 bits/cycle**
- 三个 CSR 寄存器构成统一编程接口：

| Register | Function |
|---|---|
| CSR0 | MX 阵列与量化单元的精度模式选择 |
| CSR1 | 单个结果输出的累加次数 |
| CSR2 | 块矩阵尺寸 |

**数据通路 —— Dynamic Streamers**

- 基于 SNAX 数据流引擎，扩展**动态 channel gating**，运行时按精度模式激活所需存储通道子集：

| 精度模式 | 所需通道数 |
|---|---|
| MXINT8 | 1 |
| MXFP8 | 4 |
| MXFP6 | 3 |
| MXFP4 | 4 |

- 每种精度模式采用针对 operand 宽度优化的数据布局；流引擎内**可编程 AGU (地址生成单元)**由 Snitch 核心运行时配置以匹配访问模式
- 基础设施包括：**128 KiB 32-bank 共享 SPM** + 全连接 crossbar、**DMA core** 提供 512-bit 峰值外部带宽

---

**五、整体架构数据流**

- **控制流**：Snitch core → CSR 写入 → FSM 配置精度/累加深度/矩阵尺寸
- **数据流**：外部内存 → DMA → SPM → 动态 streamer (按需门控通道 + AGU 布局适配) → 8×8 MX 空间阵列 → SIMD 量化单元 → 量化后 MX 结果 → 下一层
- **最终实现效果**：GlobalFoundries 22FDX 工艺、500 MHz 下，系统面积 **0.60 mm²**，其中 MX tensor core 占 **29.5%**；吞吐量为 MXINT8 **64 GOPS**、MXFP8/6 **256 GOPS**、MXFP4 **512 GOPS**，能效分别为 **657**、**1438-1675**、**4065 GOPS/W**，并在 ResNet18 与 ViT 上实现 **94.41%-99.51%** 的计算利用率

### 1. 混合精度可扩展归约树

---

**核心定位与设计动机**

- **混合精度可扩展归约树**是本文在算术单元层面的核心贡献，旨在弥合两种 SotA MX MAC 归约树方案（**FP32 addition** [15] 与 **Long integer addition** [24]）之间的关键 trade-off。
- 现有方案的困境：
  - **FP32 addition 方案** [15]：支持全部六种 MX 数据类型，但在 L2 加法器后需要昂贵的 **normalization** 与 **alignment** 逻辑，且 26-bit 显著数扩展导致加法器过配。
  - **Long integer addition 方案** [24]：通过 **early-accumulation** 免去乘积求和后的归一化，但需要 **67-bit 整数累加器** 和 **95-bit 宽归一化**，且不支持 MXFP4。
- 本文的核心洞察：**归约树占据 MX MAC 面积的 88.7%、能耗的约 85%**（Fig. 1），因此优化 L2 adder 与 accumulation 级是收益最大的切入点。

![](images/21466dc31e25d1155fd05655f17ff102f7dbed0cef7507c54a62bc30676bbaeb.jpg) *Fig. 2. Overview and issues of the state-of-the-art reduction trees for MX MAC implementations: FP32 addition [15], and Long integer addition [24]. Followed by the solutions proposed in this work.*

---

**输入输出关系与在整体架构中的位置**

- **输入信号**：
  - 四个 **10-bit significand**（来自 L1 乘法级，携带至多 **6-bit exponent**）。
  - 两个 **8-bit shared exponent**（每个乘法因子各一个，对应 MX 的 2 级指数方案）。
  - 一个存储的 **FP32 partial result**（累加寄存器中的部分和）。
- **输出信号**：
  - 归一化后的浮点累加结果，写入累加寄存器。
  - 在完整 GeMM 累加结束后，64 个未量化的浮点输出送入 **SIMD-based quantization unit**，量化回目标 MX 格式（计算正确的 shared exponent 后进行）。
- **架构作用**：归约树位于精度可扩展乘法阵列（处理 2-bit 显著数乘法单元）与量化单元之间，是 **MXINT8 / MXFP8 / MXFP6 / MXFP4 三种精度模式**（分别累加 1、4、8 个乘积）共用的混合整数-浮点求和通路。

---

**第一迭代：融合 Early-Accumulation 的混合结构**

![](images/a95e4644fdfbd01b687f331a3b348780b10094408b0a7acdc6128538cb512bbb.jpg) *(d)*

- **设计动作**：将 [24] 的 **early-accumulation 方案**（源自 Lutz 等人的 fused FP8 dot product [31]）注入 [15] 的 FP32 归约树骨架中（Fig. 3a）。
- **工作机制**：
  - **28-bit product-sum** 由 L2 加法器产生——该宽度精简、**避免加法器过配**。
  - 该 28-bit 求和结果向**左扩展 24 bits**，使存储的 FP32 部分和的 24-bit significand 可向左移位完成对齐后相加（early-accumulation 将 FP32 部分和对齐到 MXFP8 的 shared exponent 基准，**免除 L2 加法后至累加之间的 normalization 与 alignment 开销**）。
  - 加法输出为 **53-bit**，再向**右扩展 24 bits**——因为 FP32 部分和的整个 24-bit significand 可能被完全移出，需重新接回以正确归一化。
- **遗留问题**：归一化模块需支持 **77-bit 输入宽度**，normalization 开销反而成为新的瓶颈。

---

**第二迭代：基于互斥性的 MUX 优化**

- **关键观察**：左扩展与右扩展的 **24-bit 扩展位在任意时刻至多只有一侧被使用**：
  - 左侧 24-bit 扩展：仅当存储的 FP32 部分和**大于** product-sum、需相对左移时使用。
  - 右侧 24-bit 扩展：仅当 FP32 significand 需要**右移**时使用。
- **实现手段**：插入一个 **multiplexer (MUX)**，根据 FP32 部分和的对齐移位方向，动态地将 28-bit product-sum 向**左或向右**扩展（符号处理电路在图中省略）。
- **效果**：归一化输入宽度从 **77-bit 压缩至 53-bit**，显著削减累加级的归一化硬件成本。Fig. 3c 与 3d 给出了两种累加配置的数值示例以说明 MUX 的选择逻辑。

---

**累加精度优化：16-bit Mantissa 的推导**

- **优化思路**：进一步缩减存储部分和的 **mantissa 位宽**，使计算精度与最终结果的实际精度匹配（最终结果必然要量化回窄位宽 MX 格式）。
- **scaling 规律**（Fig. 3b 中蓝色箭头标注）：
  - **单箭头模块**（L2 alignment、L2 adder）：位宽随 mantissa 宽度**成比例缩减**——L2 加法精度设计上与存储部分和精度一致。
  - **双箭头模块**（累加级的对齐、加法、归一化）：每移除 1 bit mantissa，位宽**减少 2 bits**——因其输入宽度 = L2 输出宽度 + 存储部分和的 mantissa 宽度，二者均随 mantissa 缩放（原为 28-bit 与 24-bit 组合成 52-bit 信号）。
- **误差分析方法**：
  - 假设：只要 **quantization error ≥ addition error**，加法误差即可被忽略。
  - 实验设置：以 **FP64 为基准**，误差按 FP64 结果归一化，避免被个别大数值主导；分别用 **uniform 分布**（shared exponent 限制在 -32 到 32 防溢出）与 **Gaussian 分布**（6σ = 2³²）模拟通用负载。
- **关键数据**（64×64 矩阵、MXFP8 E4M3）：量化误差与加法误差在 **mantissa = 13 bits** 处相当（Fig. 4 left）。
- **参数选择**：跨所有 MX 格式、矩阵尺寸（64×64 与 256×256）及两种分布取最坏情况下的最高临界值，最终硬件采用 **16-bit mantissa**，保证最不利场景下的加法精度。

![](images/88594977bfe3e1d1d4925704bad84a86118ce82593c3aa9897239756f0226c81.jpg)

---

**三种归约树方案的结构性对比**

| 维度 | FP32 addition [15] | Long integer addition [24] | **本工作（Hybrid）** |
|---|---|---|---|
| 支持 MXFP4 | 是 | 否 | 是 |
| Product-sum 宽度 | 26-bit 扩展显著数 | 67-bit 整数 | **28-bit** |
| 累加器/归一化输入 | FP32 归一化 + 每次对齐 | 95-bit 扩展求和 + 宽归一化 | **53-bit（MUX 优化后）** |
| Normalization 时机 | 每周期 L2 输出后 | 仅最终（early-accumulation） | **仅最终（early-accumulation）** |
| 存储 mantissa | 23-bit (FP32) | 23-bit (FP32) | **16-bit** |
| 最高频率 | — | 1100 MHz | **1800 MHz** |

---

**性能收益与量化结果**

- **MAC 级对比**（100–1800 MHz 扫频，Fig. 6）：
  - 相对 Long integer addition [24]：在**全频率范围内**面积与能效均占优，最高频率从 1100 MHz 提升至 **1800 MHz**。
  - 相对 FP32 addition [15]：
    - MXFP8 与 MXFP4 模式：**1 GHz 以下能效更优**，以上相当。
    - MXINT8 模式：FP32 SotA 仍略优（整数通路无需浮点对齐开销）。
    - 面积效率：**500–1000 MHz 区间全面领先**。
- **系统级能效**（集成 8×8 阵列至 SNAX 后）：MXINT8 达 **657 GOPS/W**、MXFP8/6 达 **1438–1675 GOPS/W**、MXFP4 达 **4065 GOPS/W**，较前 SotA [15] 分别提升 **1.59×、3.05×–3.21×、1.13×**。
- **方法论要点**：系统级收益超出 MAC 级对比预期，源于综合流程中采用 **register optimization** 与**更宽松的时序约束**设置。

![](images/bebcdf147ee228f50740636e87c4a8858c1d96d8dfa2f87f468f55014b42ce30.jpg)

---

**技术总结**

- 混合归约树的本质是一次**方案杂交 + 互斥资源复用 + 精度-硬件协同裁剪**的三层优化：
  - 杂交：取 [15] 的全 MX 类型支持与 [24] 的 early-accumulation 免归一化能力。
  - 复用：利用左/右扩展位互斥性，以 MUX 将归一化输入从 77-bit 降至 53-bit。
  - 裁剪：通过量化误差-加法误差对比分析，将存储 mantissa 从 23-bit 削至 **16-bit**，且累加级模块按 2 倍速率获益。
- 该设计是精度可扩展 MX MAC 从孤立算术单元走向高效 NPU 集成（SNAX 平台、动态 channel gating 数据流）的算术基础，支撑了训练（FP8 大动态范围）与推理（INT8 低功耗）在同一计算阵列上的统一执行。

### 2. 基于误差分析的累加精度优化

**核心观点**

该技术的本质是一种**误差预算驱动的硬件精简方法**：在 Microscaling (MX) 混合精度 MAC 的累加路径上，通过定量比较**累加误差（addition error）**与**最终量化误差（quantization error）**的相对大小，找到二者达到平衡的**关键尾数宽度（critical mantissa width）**，并据此将存储部分和的尾数位宽压缩至 16-bit。其核心假设是：**只要量化误差超过累加误差，后者即可被忽略**——因为无论累加精度多高，最终结果都必须被量化回 MX 目标格式（FP8/FP6/FP4/INT8），精度上限天然受限于该量化步骤。

---

**问题背景：为什么要削减累加尾数位宽**

- 传统设计（FP32 addition [15] 与 long integer addition [24]）均将部分和存储为 **FP32（23-bit mantissa）**，导致整个 reduction tree 的位宽被“锚定”在 FP32 精度上。
- 但 MX 架构的**最终输出必然要量化回窄位宽 MX 格式**（例如 MXFP8 E4M3 仅 3-bit mantissa），中间过程保留 23-bit 精度存在**精度冗余（over-provisioning）**。
- 论文指出累加树占 MX MAC 面积的 **88.7%**、能耗的约 **85%**，因此尾数宽度每削减 1 bit，都直接转化为硬件收益。

![](images/dc9c4b234b7f43ce68cfd57e2037a6884e07e0650b6b342e533ac30907bf0f8a.jpg) *Fig. 1. Resource breakdown of the state-of-the-art precision-scalable Microscaling (MX) multiply-accumulate (MAC) unit [15], where more than 80% of the resources go to the reduction tree.*

---

**两种误差源的建模**

- **量化误差**：MX 规范 [20] 规定，累加完成后 64 个未量化的 MAC 输出先计算**共享指数**，再将各元素除以共享尺度并量化为 INT8/FP8/FP6/FP4。该步骤引入的舍入误差是**结构性、不可消除的**——除非下一层计算退回 FP32，代价高昂。
- **累加误差**：由截断的尾数位宽引起，包括 L2 对齐移位时的位丢失、部分和存储的舍入等。
- **判定准则**：当 `quantization error ≥ addition error` 时，累加误差被量化误差“淹没”，对最终精度**无感知影响（negligible）**；尾数宽度可持续削减，直至二者**量级相当**。

---

**算法流程与参数设置**

误差分析实验的完整流程如下：

- **参考基准**：以 **FP64** 计算的结果作为“完美结果”，两种误差均相对 FP64 度量。
- **归一化方式**：误差值除以 FP64 结果本身，得到**相对误差**，避免少数大数值主导统计，使误差度量覆盖所有数据点。
- **实验变量**：
  - 尾数宽度在 **2–23 bit** 之间扫描（部分和存储与运算路径同步采用该位宽）。
  - 矩阵规模：**64×64 与 256×256**。
  - 输入分布：**Uniform（均匀分布）** 与 **Gaussian（高斯分布）** 两种。
- **分布参数设定**：
  - Uniform：共享指数限制在 **−32 到 +32** 的实际范围内，避免溢出。
  - Gaussian：标准差由 $6\sigma = 2^{32}$ 确定，覆盖真实训练数据的动态范围量级。
- **为何采用双分布**：目标应用是** continual learning**，无法针对预定义 workload 调优，因此用两种分布模拟**通用负载**以覆盖最坏情况。

![](images/41d28965663a6d74e8dc1ed03a41cf3793380cb88bf4f3b8c542cb4236abaa71.jpg)

实验结果（Fig. 4）的关键观察：

- 左图（MXFP8 E4M3，64×64，Gaussian）：量化误差曲线基本持平（由目标格式决定，与累加位宽无关），而累加误差随尾数宽度增加而单调下降；两条曲线在 **13-bit** 处相交。
- 右图：汇总各 MX 格式、矩阵规模与分布组合下的**关键尾数宽度**。设计上取所有场景中的**最大值**以保证最坏情况下的累加精度，最终选定 **16-bit mantissa** 作为硬件实现位宽。

---

**输入输出关系与硬件位宽映射**

该分析产生的输入输出关系清晰：

- **输入**：四个 10-bit significand（含 6-bit exponent，来自 L1 乘法输出）+ 存储的 FP32 形式部分和（经缩减后为 16-bit mantissa）。
- **输出**：一条位宽经过系统性缩减的 reduction tree 数据通路。

缩减规则在 Fig. 3b 中以蓝色箭头标注，分两类：

- **单箭头模块（按比例缩减）**：**L2 对齐（alignment）与 L2 加法**的位宽与尾数宽度**成比例缩小**。原因：[15] 中该级加法的精度设计为与存储部分和精度一致。
- **双箭头模块（双倍缩减）**：累加级的**部分和对齐、加法、归一化**模块，尾数每削减 1 bit，其位宽**缩减 2 bit**。原因：这些模块的输入位宽由 **L2 加法输出宽度 + 部分和 mantissa 宽度** 两部分拼接而成（原为 28-bit + 24-bit = 52-bit），二者均随尾数宽度同步缩减。
- 配合早前提出的 **hybrid reduction tree**（MUX 选择左侧或右侧 24-bit 扩展，归一化输入从 77-bit 压缩至 53-bit），尾数缩减进一步将 53-bit 归一化输入压缩。

![](images/a95e4644fdfbd01b687f331a3b348780b10094408b0a7acdc6128538cb512bbb.jpg) *(d)*

---

**在整体架构中的作用与贡献**

- 该优化作用于 MAC 中**能耗与面积占比最高的模块（reduction tree > 85% 能耗）**，是连接“算法可接受精度”与“硬件成本”的桥梁。
- 与 hybrid reduction tree（省去归一化开销）形成**两级协同优化**：
  - 第一级：early-accumulation + MUX 复用扩展位，消除 77-bit 归一化。
  - 第二级：误差分析驱动的尾数缩减，压缩所有累加路径位宽。
- 系统级收益最终体现为：相比前 SotA [15]，能效提升 **1.59×（MXINT8）、3.05×–3.21×（MXFP8/6）、1.13×（MXFP4）**。

---

**方法的审慎性与潜在局限**

- **最坏情况保守设计**：16-bit 取自所有分布/格式/矩阵规模组合的**最高关键尾数**，而非针对单一 workload 调优，牺牲部分压缩空间换取 continual learning 场景的通用性。
- **误差掩盖假设的边界**：该假设在量化误差主导时成立；若某 workload 对中间精度敏感（如极长累加链、大量数值抵消 cancellation），累加误差的累积可能突破单次比较的界限，此时代价需由应用端承担。
- **实验负载的代表性**：Uniform 与 Gaussian 合成数据覆盖了统计特性，但真实 DNN 权重/激活的分布（重尾、稀疏、layer-wise 差异）可能给出不同的关键位宽，这是该类 error-budget 方法普遍的开放性问题。

**总结**：这项技术通过**“量化误差作为精度天花板”**这一洞察，将原本被 FP32 规格绑架的累加数据通路由 23-bit mantissa 压缩至 16-bit，并利用位宽拼接结构使部分模块获得**双倍缩减速率**，是整个 precision-scalable MX MAC 能效突破的关键一环。

### 3. MX张量核心的SNAX集成与CSR统一控制

**核心观点**

该设计的系统级价值在于：将一个 **precision-scalable MX tensor core（8×8 MAC array）** 通过 **hybrid control/data coupling（混合控制/数据耦合）** 策略嵌入 SNAX RISC-V compute cluster。控制路径上，仅用 **3 个 CSR（Configuration and Status Register）** 即可在一个统一编程接口下完成 **precision mode、accumulation depth、matrix tile size** 的运行时动态配置；数据路径上，通过 **dynamic channel gating（动态通道门控）** 与可编程 **AGU（Address Generation Unit）** 使数据流带宽随精度模式弹性伸缩。二者结合使 MX tensor core 在 ResNet18 与 Vision Transformer 的训练/推理混合负载下达到 **94.41%–99.51% 的计算利用率**，逼近理论峰值吞吐。

---

**一、SNAX 集成的整体架构定位**

![](images/fa523805476e069e2857d5d2aa2e67b7fc2750e6dd41c1e6bd90f07c5dc87e27.jpg) *Fig. 5. System architecture overview with precision-scalable MX tensor core integration.*

- **SNAX cluster** 是一个面向快速 NPU integration 的 **RISC-V compute cluster template**，其核心组件包括：
  - 一个 **RV32IMAFD Snitch core**：作为控制主机，通过标准 CSR 指令对 NPU（此处为 MX tensor core）进行配置与 kernel offloading；
  - **128 KiB、32-bank 共享 Scratchpad Memory（SPM）**：经由 fully connected crossbar 提供 operand 存储与高带宽访问；
  - **Dedicated data streamers**：自主、连续地向 NPU 供数，最大化 NPU 利用率；
  - **DMA core**：负责 SPM 与外部 memory 之间的搬移，峰值数据带宽 **512-bit**。
- **hybrid coupling 策略** 的设计意图：
  - **Loosely coupled control**：kernel 卸载不阻塞 Snitch core，控制流与计算流解耦；
  - **Tightly coupled data**：data streamers 直接紧耦合到 shared memory，低延迟、高带宽，且带宽需求可按 NPU workload 在设计期/运行期灵活适配。
- **本工作的针对性扩展**：原始 SNAX streamer 按静态最坏情况带宽配置，在低精度模式下会导致部分 memory channel 空转、消耗动态功耗并加剧 **bank contention**。本设计为此引入 **dynamic channel gating**，这是集成层面的关键增量。

---

**二、MX Tensor Core 的三大子模块**

MX tensor core（Fig. 5 底部）由三个功能解耦的子模块构成，形成“控制—计算—量化”的清晰流水：

- **Flexible FSM（有限状态机）**：
  - 提供微架构级控制与 handshake 信号；
  - 接收 CSR 写入的配置参数，生成基于矩阵尺寸的时序控制信号；
  - 负责对 spatial array 的 precision mode 配置，以及整个 **GeMM（General Matrix Multiplication）** 执行过程的编排。
- **Precision-scalable MX spatial array（空间阵列）**：
  - 由 **64 个 MX MAC unit** 组成的 **8×8 二维 mesh**，同时支持 **horizontal 与 vertical data reuse（横向与纵向数据复用）**；
  - 计算节拍随精度模式变化：对两个 8×8 矩阵执行 GeMM 时，所需周期数为：

| Precision Mode | MAC 模式语义 | 单次 GeMM 周期数 | 系统 Throughput |
|---|---|---|---|
| **MXINT8** | 每周期累加 1 个乘积 | 8 cycles | 64 GOPS |
| **MXFP8 / MXFP6** | 每周期累加 4 个乘积 | 2 cycles | 256 GOPS |
| **MXFP4** | 每周期累加 8 个乘积 | 1 cycle | 512 GOPS |

  - 该节拍特性直接决定了下游数据流的带宽需求（见第四节）。
- **SIMD-based Quantization Unit（量化单元）**：
  - 位于 spatial array 下游，处理 **64 个 FP 浮点输出**；
  - 依据 [20] 规范，先在 64 元素 MX group 中寻找最大元素确定 **shared exponent（共享指数）**，再将各结果除以该 shared scale 后量化为 **INT8 / FP8 / FP6 / FP4**；
  - 量化后的结果作为下一层网络的输入，完成“高精度累加 → 低精度传递”的精度闭环。

**控制、计算、量化的分离式架构**使 tensor core 能在不重构硬件的前提下，动态适配不同精度与矩阵尺寸需求，同时维持峰值吞吐。

---

**三、CSR 统一控制：实现原理与参数语义**

CSR 机制是整个集成方案中“以极低成本换取高灵活性”的核心。其配置带宽为 **32 bits/cycle**，由 Snitch core 通过标准 CSR write 指令驱动。

- **寄存器功能映射**（Table II）：

| Register | 配置对象 | 语义与作用 |
|---|---|---|
| **CSR0** | Precision mode | 同时选择 spatial array 与 quantization unit 的数值精度（INT8/FP8/FP6/FP4 六种 MX 类型之一） |
| **CSR1** | Accumulation times | 单个结果输出所需的累加次数，即 GeMM 中 K 维度（accumulation dimension）的深度 |
| **CSR2** | Matrix tile size | Block matrix 的 row/column 维度，即矩阵 tile 的边长 |

- **配置流程**：
  - Snitch core 发出 CSR write 指令 → **CSR manager**（Fig. 5 左下角的专用模块）锁存参数 → 参数被路由至 **MX tensor core FSM** → FSM 据此重构微架构时序：
    - CSR0 决定阵列内 MAC 的精度模式分支（乘法器拆分方式、reduction tree 配置、streamer 激活通道数）；
    - CSR1 决定累加循环次数，即何时触发 FP32 partial result 到 quantization unit 的输出；
    - CSR2 决定 GeMM tile 的分块尺寸与对应的地址生成模式。
- **输入输出关系**：
  - 输入：Snitch 的 CSR write（3×32-bit 配置字）；
  - 输出：FSM 的全套控制信号（precision 选择信号、循环计数边界、handshake 信号、数据流启停）；
  - 本质上是把“重编程一个异构加速器”压缩为 **3 条指令级别的轻量写操作**，实现 **runtime 动态切换 precision / accumulation depth / matrix size**，无需停机重编译。
- **作用**：
  - 抽象掉硬件细节（MAC 拆分、指数对齐、量化逻辑），向程序员暴露极简接口；
  - 由于 control 是 **loosely coupled** 的，CSR 配置完成后 Snitch 即可继续执行其他任务，kernel 在 tensor core 上自主运行，避免控制流阻塞。

---

**四、Dynamic Streamers：带宽与数据布局的弹性适配**

- **动态通道门控**：
  - 每种精度模式激活的 memory access channel 数量与计算节拍严格匹配：

| Precision Mode | 激活的 Streamer 通道数 | 带宽匹配逻辑 |
|---|---|---|
| **MXINT8** | 1 | 8 cycles 完成一个 tile，带宽需求最低 |
| **MXFP8** | 4 | 2 cycles/tile，需 4 路并发供数 |
| **MXFP6** | 3 | 元素仅 6-bit，有效带宽需求略低于 FP8 |
| **MXFP4** | 4 | 1 cycle/tile，满带宽供数 |

  - 未激活通道被完全关断，消除空转通道的动态功耗，并缓解低精度运行时的 **SPM bank contention**。
- **可编程 AGU（Address Generation Unit）**：
  - 每种精度模式采用针对其 operand width 优化的 **matrix tile data layout** 与差异化的访问模式；
  - AGU 由 Snitch core 在 **run-time** 配置，使同一套 streamer 硬件支持多种 data layout 与 access pattern，无需为每种 MX 类型重复设置存储子系统。
- **作用与收益**：streamer 层的带宽、布局、访问模式三者均随精度模式弹性收缩/扩展，使数据供给子系统与计算阵列的动态精度特性精确同步——这是系统级能效（**MXFP4 高达 4065 GOPS/W**）的关键保障之一。

---

**五、端到端验证：控制与数据协同的量化证据**

- **计算利用率**：在 **ResNet18** 与 **Vision Transformer**、batch size 32 的负载上：
  - 推理采用 **INT8**，训练采用 **FP8 E4M3**（其更大动态范围是收敛所需）；
  - 四种 workload 的利用率达 **94.41%–99.51%**，证明 CSR 控制 + dynamic streaming 有效消除了控制开销与 memory bottleneck。

![](images/1b1201d92346d5ed0b44ae7b6af3960077d9403bbd9e29461eb035b8e2f3f513.jpg)

- **面积与功耗分布**：系统总面积 **0.60 mm²**（500 MHz, GF 22FDX, 0.8V），其中多 bank SPM 与数据供给链路占据主体，MX tensor core 仅占 **29.5%**；
  - 能耗分布上 MX core 反而占主导——得益于 data streamer 的 **dynamic channel gating** 与综合流程中的 **clock gating**，SPM 和数据供给等时序逻辑的能效被显著增强，其相对能耗占比被压低。

![](images/bebcdf147ee228f50740636e87c4a8858c1d96d8dfa2f87f468f55014b42ce30.jpg)

- **与 SotA 对比中的集成价值**：

| 指标 | [24] MXDotP | [15] SotA MX MAC | [25] OpenGeMM | **本工作** |
|---|---|---|---|---|
| Precision-scalable | ✗ | ✓ | ✗ | **✓** |
| NPU integrated | ✓ | ✗ | ✓ | **✓** |
| Energy Efficiency (GOPS/W) | 356 | MXINT8: 412 / MXFP8: 472-521 | 4680 | **MXINT8: 657 / MXFP8/6: 1438-1675 / MXFP4: 4065** |

  - [15] 虽是精度可扩展的 SotA MAC，但 **未做 NPU 集成**，系统级能效无从谈起；本工作在完整集成条件下仍取得相对 [15] **1.59×（MXINT8）、3.05×–3.21×（MXFP8/6）、1.13×（MXFP4）** 的能效提升，说明优化收益同时来自 reduction tree（MAC 级）与 SNAX 集成/streamer gating（系统级）两个层面。

---

**六、总结：控制与数据协同的设计哲学**

- **CSR 统一控制**将异构 precision-scalable 硬件的全部可变性（6 种 MX 数据类型、累加深度、tile 尺寸）压缩为 **3 个 CSR 寄存器**，以 **32 bits/cycle** 的极低配置带宽换取 runtime 全动态适配能力，是"ease of programmability"的直接来源。
- **Dynamic streamers** 则从数据侧镜像了这种可变性：带宽按精度门控、布局由 AGU 运行时重配，使 128 KiB SPM 的带宽资源与 8×8 阵列的计算节拍精确对齐。
- 二者共同构成 **training–inference 统一 NPU 平台**的系统底座：训练时切至 FP8/FP6 大动态范围模式，推理时切至 INT8/FP4 低成本模式，切换成本仅为数次 CSR 写——这正是边缘 continual learning 场景（机器人、可穿戴健康监测、自动驾驶）所要求的“单一 compute fabric 支撑双 workload”能力。

### 4. 动态通道门控数据流

**核心问题：静态带宽供给与动态精度需求之间的矛盾**

- SNAX 平台原始的 **data streamer**（数据流引擎，源自 DataMauler [29]）在**设计时**按照 **worst-case bandwidth**（最坏情况带宽）静态配置 memory access channel，即固定的 **4 条 128-bit 通道**，合计 **512-bit peak data bandwidth**（与 DMA 核心的峰值带宽一致）。
- 然而 **precision-scalable MX MAC** 的带宽需求随精度模式剧烈变化：低精度模式（如 MXFP4）单周期消耗大量窄位宽元素，而 **MXINT8** 模式每周期仅需少量数据。
- 后果是在低精度或低带宽需求的运行阶段，部分 memory channel 处于“被供给但未被有效利用”的状态：
  - 未充分利用的通道仍产生 **dynamic power**（动态功耗）；
  - 多余的访存请求加剧 **SPM bank contention**（存储体竞争），干扰有效数据流的低延迟访问；
  - 该问题在使用精度可扩展 MX MAC 时尤为突出，因为带宽需求在 MXINT8/MXFP8/MXFP6/MXFP4 间动态切换。

---

**实现原理：运行时动态通道门控 (Dynamic Channel Gating)**

- 核心思想：将 streamer 的 memory access channel 从“常开”改为**按需使能**——仅在当前精度模式真正需要的通道子集上发出访存请求，其余通道被 **gating**（门控关闭），不产生切换活动与动态功耗。
- 带宽需求由 **MX tensor core** 的计算特性决定（8×8 spatial array 对两个 8×8 矩阵 tile 执行 GeMM）：
  - **MXINT8** 模式：每 8 个 cycle 完成一个 tile，单周期仅取 16 个 8-bit 元素 ≈ 128-bit → 仅需 **1 条通道**；
  - **MXFP8** 模式：每 2 个 cycle 完成一个 tile，单周期取 64 个 8-bit 元素 = 512-bit → 需 **4 条通道**；
  - **MXFP6** 模式：6-bit 元素位宽下单周期需求 384-bit → 需 **3 条通道**；
  - **MXFP4** 模式：每 1 个 cycle 完成一个 tile，单周期取 128 个 4-bit 元素 = 512-bit → 需 **4 条通道**。

| 精度模式 | 单 tile 计算周期 | 单周期 operand 带宽需求 | 激活通道数 |
|---|---|---|---|
| MXINT8 | 8 cycles | ~128-bit | **1** |
| MXFP8 (E5M2/E4M3) | 2 cycles | 512-bit | **4** |
| MXFP6 (E3M2/E2M3) | 2 cycles | 384-bit | **3** |
| MXFP4 (E2M1) | 1 cycle | 512-bit | **4** |

- 通道使能信息与计算模式绑定：Snitch RISC-V 核心通过 **CSR 写指令**（CSR0 选择精度模式）下发配置，streamer 依据该模式在运行时切换激活的通道子集，无需重新综合或静态重配置。

---

**算法流程**

- 配置阶段：
  - Snitch 核心向 MX tensor core 的 **CSR manager** 写入 **CSR0**（精度模式）、**CSR1**（累加次数）、**CSR2**（matrix tile 维度）；
  - 同一精度配置同步传播至 **data streamer**，触发对应的通道门控策略；
  - streamer 内部的可编程 **AGU (Address Generation Unit)** 被配置为与当前精度模式的 **data layout**（数据排布）和 **access pattern**（访问模式）匹配——因为不同位宽的元素在 128-bit 通道中的打包方式与跨 bank 的分布各不相同。
- 执行阶段：
  - FSM 启动 GeMM，streamer 以自主、连续的方式从 **128 KiB 32-banked SPM**（经 fully connected crossbar）向 tensor core 供给数据；
  - 仅被激活的通道发起 bank 访问，被门控通道保持静默，消除冗余的 **memory traffic**；
  - 计算完成后，64 个 FP 输出由 SIMD quantization unit 量化回目标 MX 格式，进入下一层数据流。
- 模式切换：
  - 训练（FP8 E4M3）与推理（INT8）之间的切换只需重写 CSR，通道门控与 AGU 配置随之即时更新，支撑 **continual learning** 场景下训练/推理任务的动态交替。

![](images/fa523805476e069e2857d5d2aa2e67b7fc2750e6dd41c1e6bd90f07c5dc87e27.jpg) *Fig. 5. System architecture overview with precision-scalable MX tensor core integration.*

---

**输入输出关系**

- 输入侧：
  - 控制输入：来自 Snitch 核心的 CSR 配置流（32-bit/cycle），编码精度模式与矩阵尺寸参数；
  - 数据输入：SPM 中按精度优化排布的 A/B operand tile，经门控后的通道子集读出。
- 输出侧：
  - 数据输出：符合当前精度模式带宽节奏的连续 operand stream，送入 8×8 MX spatial array；
  - 计算输出：64 个未量化 FP 结果 → SIMD 量化单元 → 目标 MX 格式（INT8/FP8/FP6/FP4），供下一层消费。
- 接口契约：streamer 的有效供给带宽与 tensor core 在该精度模式下的**峰值消耗速率精确匹配**，既不欠供（避免 stall）也不过供（避免功耗浪费），这是实现高利用率的根本前提。

---

**在整体系统中的作用与量化效果**

- 打破内存瓶颈：系统级能效的上限由 **memory access bottleneck** 决定，动态通道门控使数据供给侧与计算侧的精度自适应能力对齐，避免“计算可扩展而带宽固定”的失衡。
- 利用率验证：在 **ResNet18** 与 **Vision Transformer**（batch size 32）的推理（INT8）与训练（FP8 E4M3）四类 workload 上，MX tensor core 的 **temporal utilization 达到 94.41%–99.51%**，证明控制与内存瓶颈被有效消除，核心接近理论峰值吞吐运行。

![](images/7eaaaa0662dc7623910fffb2503edd590eb3562cf7ba6fbbe91c4d97037bd6ef.jpg)

- 能耗结构重塑：由于通道门控（配合综合流程中的 **clock gating**）大幅压缩了 SPM 与 data supply 等时序逻辑的动态功耗，**能耗主导权从数据通路转移至 MX tensor core 本身**；而面积上系统共 0.60 mm²（GF 22FDX, 0.8 V），MX tensor core 仅占 29.5%，多数面积由多 bank SPM 与数据供给网络占据。

![](images/1b1201d92346d5ed0b44ae7b6af3960077d9403bbd9e29461eb035b8e2f3f513.jpg)

- 最终系统级指标（500 MHz）：

| 指标 | MXINT8 | MXFP8/6 | MXFP4 |
|---|---|---|---|
| Throughput (GOPS) | 64 | 256 | 512 |
| Energy efficiency (GOPS/W) | **657** | **1438–1675** | **4065** |
| 相对前 SotA [15] 能效提升 | 1.59× | 3.05×–3.21× | 1.13× |

- 关键结论：MXFP8/6 模式获得最大能效增益（约 **3×**），正是因为该模式下原始静态 streamer 的通道闲置与 bank contention 问题最严重，动态通道门控的收益在此充分兑现；配合混合 reduction tree 的 MAC 级优化，共同构成了该精度可扩展 NPU 的系统级效率来源。


---

## 4. 实验方法与实验结果

**一、实验设置总览**

本文的实验评估分为两个层级展开，形成从底层算术单元到完整 NPU 系统的完整验证链条：

- **评估对象**：精度可扩展的 MX MAC 单元（含混合 reduction tree）、MX tensor core（8×8 阵列）、集成到 SNAX 平台的完整 NPU 系统。
- **实现方式**：全部以 SystemVerilog 完成 RTL 实现，代码已开源（GitHub: KULeuven-MICAS/Precision-Scalable MX）。
- **物理实现流程**：
  - 综合工具：**Synopsys Design Compiler®**，工艺节点 **GlobalFoundries 22FDX®**（22nm FDX）。
  - 工作条件：0.8V 标称电压，typical-typical corner。
  - 功耗分析：**Siemens QuestaSim™** 生成 netlist 仿真切换活动（switching activity），**Synopsys PrimeTime PX™** 进行功耗/能耗分析。
  - 综合流程中启用 **clock gating** 以提升能效。
- **频率扫描范围**：MAC 级评估覆盖 **100 MHz 至 1800 MHz**，系统级评估工作点为 **500 MHz**。
- **评估基准（benchmark）**：
  - MAC 级：将两种 SotA adder tree（FP32 addition [15]、Long integer addition [24]）代入相同的 MX MAC 框架中做**控制变量式直接对比**。
  - 系统级：**ResNet18** 与 **Vision Transformer**，覆盖 inference 和 training 两类 workload，batch size 为 32。
  - 精度配置：inference 使用 **INT8**，training 使用 **FP8 E4M3**（因其更大的动态范围）。

---

**二、MAC 级实验结果分析**

实验对比三种 adder tree 方案：本文的 **hybrid reduction tree**、SotA **Long integer addition [24]** 与 SotA **FP32 addition [15]**，结果如 Fig. 6 所示。

![](images/bebcdf147ee228f50740636e87c4a8858c1d96d8dfa2f87f468f55014b42ce30.jpg)

![](images/7eaaaa0662dc7623910fffb2503edd590eb3562cf7ba6fbbe91c4d97037bd6ef.jpg)

![](images/b4f5498bc4b598759cdb4bd044a93a56127c2e1a807bce890685f85aa74a52ae.jpg)

![](images/07f30bd24fdf22e77a78a904e688697c91698d31d35a3fa5b7b494bfce95bcea.jpg)

- **操作计数标准**：每次乘法和加法各计一次 operation，因此各模式吞吐为 **2 ops/cycle（MXINT8）**、**8 ops/cycle（MXFP8/MXFP6）**、**16 ops/cycle（MXFP4）**，保证能效对比的公平性。
- **对比 Long integer addition [24]**：
  - 本文 hybrid 设计在**全频率范围内**的 area 和 energy 均占优。
  - 最高频率：本文达到 **1800 MHz**，而 [24] 仅达 **1100 MHz**（约 **1.64× 频率优势**），说明长整数加法器宽位宽（67-bit/95-bit）严重制约时序收敛。
- **对比 FP32 addition [15]**：
  - **MXFP8 / MXFP4 模式**：低于 1 GHz 时本文设计能效更优；高于 1 GHz 后两者相当。
  - **MXINT8 模式**：FP32 SotA 设计仍保持更高的能效优势（整数模式下归一化开销本身可跳过，hybrid 方案收益减小）。
  - **Area efficiency**：本文设计在 **500–1000 MHz** 区间内最优。
- **结果解读**：hybrid reduction tree 的收益来源于两点——保留整数累加**跳过归一化**的能力（early-accumulation），同时利用浮点累加的**窄加法器位宽**，避免 Long integer 方案的宽归一化与宽加法器开销。

---

**三、累加精度消融实验**

这是本文最核心的一项消融实验，验证**缩减存储 partial result 的 mantissa 位宽**对精度和硬件成本的影响：

![](images/41d28965663a6d74e8dc1ed03a41cf3793380cb88bf4f3b8c542cb4236abaa71.jpg)

- **实验动机**：
  - 存储的 FP32 partial result 含 23-bit mantissa，但最终结果必然会被量化回 MX 格式，产生不可忽略的 **quantization error**。
  - 只要 **addition error < quantization error**，加法误差即被量化误差“淹没”，可以安全缩减 mantissa 位宽以降低硬件成本。
- **实验设计**：
  - 对比两种误差源，均以 **FP64 结果为真值基准**（6σ = 2³² 设定的 Gaussian 分布避免溢出）。
  - 误差归一化到 FP64 结果值，避免少数大值主导误差统计，得到相对误差度量。
  - 为覆盖**通用 workload**（而非针对特定任务过拟合），采用两种分布模拟：**uniform distribution**（shared exponent 限制在 -32 至 32）和 **Gaussian distribution**。
  - 矩阵尺寸测试 **64×64** 和 **256×256**，覆盖不同累加深度。
- **关键实验结论**：
  - Fig. 4（左）以 MXFP8 E4M3、64×64 矩阵为例：mantissa 宽度降到约 **13 bits** 时，addition error 与 quantization error 达到相同量级。
  - Fig. 4（右）汇总各 MX 格式、矩阵尺寸、分布下的临界 mantissa 位宽。
  - 硬件实现取**最保守（最高）的临界值**，最终选定 **16-bit mantissa**——即在保证最坏情况下加法精度仍不低于量化精度的前提下，将累加精度从 23-bit 压缩至 16-bit。
- **硬件成本联动效应**（Fig. 3 中蓝色箭头标注）：
  - L2 alignment 与 L2 addition 模块的位宽随 mantissa 宽度**等比例缩减**（单端箭头）。
  - Accumulation 中的 alignment、addition、normalization 模块位宽缩减幅度为**两倍**（双端箭头），因其输入位宽由 L2 输出宽度与 mantissa 宽度共同决定，两者同时缩小。

![](images/a95e4644fdfbd01b687f331a3b348780b10094408b0a7acdc6128538cb512bbb.jpg)

---

**四、Reduction Tree 架构迭代消融**

本文对 reduction tree 本身进行了两轮迭代设计（Fig. 3a → 3b），可视为架构层面的消融：

![](images/88594977bfe3e1d1d4925704bad84a86118ce82593c3aa9897239756f0226c81.jpg)

- **第一轮（Fig. 3a）**：直接将 [24] 的 **early-accumulation** 方案集成进 [15] 的 FP32 reduction tree：
  - 28-bit product-sum 左侧扩展 24 bits 用于 FP32 significand 的左移对齐。
  - 加法后 53-bit 输出再向右扩展 24 bits，用于被移出位的重新附加（reattachment）。
  - 代价：**归一化逻辑需支持 77-bit 输入**，开销过大。
- **第二轮（Fig. 3b）**：关键观察——左侧 24-bit 扩展仅在 FP32 partial result **大于** product-sum 时需要；右侧扩展仅在需要**右移**时需要。二者**互斥、不可能同时使用**。
  - 解决方案：插入 **MUX**，根据 FP32 significand 的移位方向选择左侧或右侧扩展。
  - 效果：归一化输入宽度从 **77-bit 缩减至 53-bit**，在不损失精度的前提下消除约 24-bit 的归一化硬件开销。
- **示例验证**：Fig. 3c 和 3d 分别给出两种累加配置的数值实例，说明 MUX 双路径的正确性。

---

**五、系统级实验结果分析**

**1. Temporal Utilization 评估**

![](images/6dadf0f8abb6c854867237ae98a8376544cca18eca72023a428f738d00aa7701.jpg)

- 测试 workload：ResNet18 与 ViT 的 inference（INT8）与 training（FP8 E4M3）共四组。
- 结果：计算利用率达 **94.41%–99.51%**，证明 CSR 控制、data streamer 数据供给与 MX tensor core 之间几乎无控制/内存瓶颈，核心可逼近理论峰值吞吐。

**2. Area 与 Power 分解**

![](images/1b1201d92346d5ed0b44ae7b6af3960077d9403bbd9e29461eb035b8e2f3f513.jpg)

- **Area**：系统总占用 **0.60 mm²**，其中多 bank SPM 与数据供给链路占大头，MX tensor core 仅占 **29.5%**。
- **Energy**：MX tensor core 反而主导能耗分布，原因在于：
  - Data streamer 采用 **dynamic channel gating**，按精度模式动态激活存储访问通道（MXINT8/FP8/FP6/FP4 分别需要 **1/4/3/4 通道**），消除了低精度模式下未用通道的动态功耗。
  - 综合流程中的 **clock gating** 进一步压缩了 SPM 与数据供给等时序逻辑的能耗。

**3. SotA 横向对比**

| 指标 | [24] | [15] | [25] | **本文** |
|---|---|---|---|---|
| Frequency (MHz) | 1000 | 400 | 200 | **500** |
| Technology (nm) | 12 | 16 | 16 | **22** |
| Area (mm²) | 0.59 | 8.92 | 0.62 | **0.60** |
| Area/MAC (µm²) | 3150 | 2080 | 144 | **2766** |
| Supported precisions | MXFP8 | MX (INT8/FP8/FP6/FP4) | INT8 | **MX (INT8/FP8/FP6/FP4)** |
| Throughput (GOPS) | 102 | / | 204 | **MXINT8: 64, MXFP8/6: 256, MXFP4: 512** |
| Energy efficiency (GOPS/W) | 356 | MXINT8: 412, MXFP8/6: 472-521, MXFP4: 3597 | 4680 | **MXINT8: 657, MXFP8/6: 1438-1675, MXFP4: 4065** |
| NPU integrated? | ✓ | ✗ | ✓ | **✓** |
| Precision-scalable? | ✗ | ✓ | ✗ | **✓** |

- **vs. [15]**（精度可扩展 SotA）：能效提升 **1.59×（MXINT8）**、**3.05×–3.21×（MXFP8/6）**、**1.13×（MXFP4）**；同时首次实现 NPU 级集成（[15] 仅为 MAC 级设计）。
- **vs. [24]**：支持全部六种 MX 类型（[24] 仅支持 MXFP8），且在更落后两代的 22nm 工艺下仍实现相近面积与更高能效。
- **vs. [25]**（OpenGEMM，非精度可扩展纯 INT8 GeMM core）：本文在能效和面积效率上存在明显差距（[25] 能效 4680 GOPS/W、Area/MAC 仅 144 µm²），作者坦承这是 **precision-scalability 与 MX 全格式支持**的代价。
- **诚实披露的细节**：系统级相对 [15] 的能效提升幅度**大于** MAC 级对比（Fig. 6）预期，作者明确说明原因——本文综合流程采用了**更宽松的约束与寄存器优化设置**，这一归因说明是严谨的。

---

**六、结果可信度与实验方法论评价**

- **优势**：
  - MAC 级对比采用**同一 MAC 框架替换 adder tree** 的控制变量法，隔离了加法树设计本身的贡献。
  - 误差分析以 quantization error 为锚点、FP64 为真值、双分布双矩阵尺寸交叉验证，避免精度消融针对单一 workload 过拟合。
  - 开源 RTL 代码 + 标准工业流程保证可复现性。
- **局限**：
  - 精度消融基于 synthetic 分布，未在真实 continual learning 任务上验证 16-bit mantissa 的端到端模型精度影响。
  - 与 [25] 的对比暴露精度可扩展性的开销成本，但缺少分解该开销的定量分析（多少来自 reduction tree、多少来自量化单元与控制逻辑）。
  - 系统级对比存在工艺节点不一致（12/16/22nm）与综合约束差异，跨平台能效数字需谨慎解读。

---

