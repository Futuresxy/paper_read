# Efficient Precision-Scalable Hardware for Microscaling (MX) Processing in Robotics Learning 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Stef Cuyckens, Xiaoling Yi, Nitish Satya Murthy, Chao Fang, Marian Verhelst

**发表期刊/会议 (Journal/Conference)**: ISLPED

**发表年份 (Publication Year)**: 2025

**研究机构 (Affiliations)**: KU Leuven, Nanjing University

---

## 1. 摘要

**目的**

- 解决自主机器人在边缘端进行**on-device continuous learning**时面临的精度与能效矛盾：训练需要浮点（FP）精度，而边缘设备受限于位宽与能耗预算。
- 针对现有 SotA 连续学习处理器 **Dacapo** 的两大局限提出改进方案：
  - 仅支持 **MXINT**（整数）格式，无法兼容新一代 **MX standard** 中的 MXFP 系列浮点格式；
  - 采用 **vector-based grouping**（32 元素向量分组），导致反向传播（backpropagation）中需要同时存储原始与转置权重的两套量化版本，**内存翻倍**或需反复 requantization。
- 目标是在 **iso-peak-throughput** 条件下，实现支持全部六种 MX 数据类型（MXINT8、MXFP8 E5M2/E4M3、MXFP6 E3M2/E2M3、MXFP4 E2M1）的**精度可扩展（precision-scalable）**硬件，同时降低内存占用、提升有效训练吞吐量。

---

**方法**

*一、算术单元层面：Precision-Scalable MX MAC Unit*

- 以 **16 个 2-bit 乘法器**为基本计算单元，通过灵活互连统一整数与浮点处理路径，相比 1-bit 处理单元（如 Flexibit）降低加法器开销。
- 三种动态重配置工作模式：

| 模式 | 并行度 | 说明 |
|---|---|---|
| **INT8** | 1 路 INT8×INT8 | 全部 16 个 2-bit 乘法器协作产出 16-bit 结果，指数加法器不激活 |
| **FP8/FP6** | 4 路并行 | 每路用 4 个 2-bit 乘法器做 mantissa 乘法 + 1 个 5-bit 指数加法器 |
| **FP4** | 8 路并行 | 每个 2-bit 乘法器配一个 2-bit 指数加法器，带宽与 FP8/6 模式保持一致 |

- **层级式 MX 累加器（Hierarchical Accumulator）** 采用 Sum-Together 方案，复用精度扩展后闲置的乘法器作为部分积加法器：
  - **L1 Adder**：INT8/FP8/FP6 模式下累加 4-bit 整数部分积；FP4 模式下直接按 exponent 移位求和四个 E3M4 结果（利用 E3M4 指数范围仅 0-4 的特性），避免先求最大指数。
  - **L2 Adder**：用 **FP32 加法（26-bit mantissa 加法器）**替代 78-bit 巨型整数加法器；针对窄格式中常见的 **subnormal number**，仅将内部 mantissa 扩展 2-bit 而非引入昂贵的 normalization 电路。
  - **关键路径优化**：INT8 与 FP4 模式可 bypass L2 加法器的对齐逻辑（仅 FP8/6 需要），平衡各模式时延。

*二、架构层面：Square-based MX PE Array 与 GeMM Core*

- 提出 **64 元素方形分组（8×8 square blocks）**取代传统 32 元素向量分组：
  - 方形块在原始矩阵与转置矩阵上量化结果**对称一致**，反向传播无需 requantization，权重只需存储一份；
  - 保持与 MX standard 的兼容性（两个 32 元素块共享同一 exponent）。
- **PE Array** 由 64 个 MAC 单元组成，处理两个 8×8 方块的乘法耗时随精度递减：INT8 需 **8 cycles**，FP8/FP6 需 **2 cycles**，FP4 仅需 **1 cycle**。
- **GeMM Core** 采用 4×16 的 PE Array 网格：
  - 高度 4 对应 batch size 32 ÷ 8（8×8 方块维度），宽度 16 平衡带宽与并行度；
  - 采用 **output-stationary dataflow**，共享指数在 PE 级相加后施加到 L2 输出，最小化 FP32 中间数据搬运；
  - 权重梯度阶段因沿小 batch 维度累加导致 array 利用率下降，通过 stall 周期带宽（最大 5280 bits/cycle ≈ 264 GB/s）回写 FP32 输出至 quantizer；权重更新由独立加速器块完成。

![](images/48b04174a270c3c4ba05580d601e6c6e51a0f88fda297d9589cf5019f8ae052d.jpg) *Fig. 6. The PE Array that handles the multiplication of two 64-element square blocks, for INT8 this is executed in 8 cycles, for FP8/FP6 this takes 2 cycles, and for FP4 in 1 cycle. The PE Array is part of our GeMM core of 4x16 PE Arrays.*

---

**结果**

*算术单元评估（TSMC 16nm FinFET，400 MHz，Table II）*

- **mantissa adder extension + bypass** 为最优实现变体：

| 实现变体 | Area [µm²] | MXFP8 (E5M2) 能耗 [pJ/OP] | MXFP4 能耗 [pJ/OP] |
|---|---|---|---|
| mant. add. ext. | 2341.35 | 1.62 | 0.29 |
| norm. inputs | 3640.78 | 2.46 | 0.60 |
| **ext. + bypass** | **2078.42** | **1.36** | **0.28** |

- PE Array 能耗分解显示 **FP accumulation addition 是最耗能组件**，而共享指数逻辑开销可忽略；面积上 **L1/L2 加法器占比最大**（含模式特定逻辑）。

*内存占用对比（pusher workload，Table III）*

- 方形块方案相对 FP32 基线实现 **3.94× 内存压缩**，相对 Dacapo 的向量块方案减少 **约 51%（2.06×）** 内存footprint，且在 batch size 16/32/64 下压缩比稳定不变。

*与 SotA Dacapo 的综合对比（Table IV，同为 4096 MACs，400 MHz，TSMC 16nm）*

| 指标 | Ours | Dacapo | 对比 |
|---|---|---|---|
| Area [mm²] | 8.92 | 8.52 | 1.05× 略增 |
| Memory [KB] | 179.78 | 370.13 | **2.06× 减少** |
| Max. BW [GB/s] | 264 | 512 | **1.94× 减少** |
| E/op MXINT8 vs MX9 [pJ] | 2.43 | 2.68 | **省 10%** |
| E/op MXFP4 vs MX4 [pJ] | 0.278 | 0.358 | **省 29%** |
| E/op MXFP8/FP6 vs MX6 [pJ] | 1.92–2.12 | 1.46 | 1.32–1.45× 略增 |
| 训练延迟 MXINT8 [µs/batch] | 13.575 | 50.5 | **3.7× 加速** |
| 训练延迟 MXFP4 [µs/batch] | 4.7625 | 25.75 | **5.4× 加速** |

- 整体实现 **4× 有效训练吞吐量提升**，得益于 Dacapo 脉动阵列数据进出移动的开销。
- 训练收敛性验证：**MXFP8 在 pusher 任务上为速度与精度的最优选择**；给定等时间/等能量预算下，MXFP8 的能效与 Dacapo 的 MX9 相当（因需更多 epochs）。
- 机器人学习任务精度实验表明不同任务偏好不同格式：**MXFP8 E4M3** 更适合 pusher/reacher（机器人-物体交互），**MXINT8** 更适合 cartpole/halfcheetah（平衡任务），验证了精度可扩展设计的必要性。

![](images/7a69f9e336d42b97a25f3285749d6233a0968738cb7ba0e87e46da23293d3734.jpg)

---

**结论**

- 提出首个支持**全部六种 MX standard 数据类型**的高效 precision-scalable MAC 单元，通过 **2-bit 乘法器基本单元 + sub-word parallelism + 整数/浮点统一处理**实现硬件高度复用。
- 首创**64 元素方形共享指数分组**的 PE Array 与 learning-enabled GeMM Core，从根本上消除反向传播中的权重双份存储与 requantization 开销。
- 在 TSMC 16nm、400 MHz、iso-peak-throughput 条件下，相对 SotA Dacapo 实现：**51% 内存footprint 降低**、**4× 有效训练吞吐提升**、**相近能效**（MXINT8/MXFP4 更优，MXFP8 略逊），并额外获得对新一代 MX standard 的完整兼容性。
- 该设计为边缘端机器人**持续学习（continual learning）**提供了高效硬件基础，使自主机器人在无云端依赖条件下实时适应新环境成为可能。

---

## 2. 背景知识与核心贡献

**研究背景**

- 自主机器人需要在陌生环境中进行**实时自适应学习**（如导航、灾害响应、人机协作），传统云端学习依赖持续网络连接，且损害**数据隐私**与**实时响应能力**，因此**边缘端训练**成为关键需求。
- 神经网络推理可使用 INT8 等整数格式高效运行，但**训练**需要浮点精度以准确捕捉梯度更新，这形成了机器人边缘计算的根本矛盾：**FP 精度需求 vs. 能耗约束下的最小位宽**。
- **Microscaling (MX) 格式**（Open Compute Project 标准）通过 32 元素分组共享指数（E8M0 格式），结合 INT 与 FP 特性，在训练精度、内存占用与能耗之间提供折中方案。

![](images/8d3a2057e591e4c7ba4b00b8028ad45e75f87df5e841485a5238094ec30f15ab.jpg) *Fig. 1. Overview of goals, challenges, and our contributions: (1) precisionscalable MX MAC unit supporting all six MX formats combining INT and FP operations; (2) 64-element square MX groups for training at the edge and the design of a square-based MX PE Array for GeMM Core.*

MX 标准定义了六种具体格式：

| MX 名称 | MXINT8 | MXFP8 | MXFP8 | MXFP6 | MXFP6 | MXFP4 |
|---|---|---|---|---|---|---|
| 元素格式 | INT8 | E5M2 | E4M3 | E3M2 | E2M3 | E2M1 |
| 位数 | 8 | 8 | 8 | 6 | 6 | 4 |

- 作者在四个机器人学习 workload（pusher、reacher、cartpole、halfcheetah）上验证了 MX 格式的可行性：**MXFP8 E4M3** 在 pusher 和 reacher 任务（机器人-物体交互）中表现最优，**MXINT8** 在 cartpole 和 halfcheetah（平衡任务）中表现最佳，证明**不同任务需要不同精度配置**，即精度可扩展硬件的必要性。

![](images/7f461e7677d1fdf402a14494c4f293a1fae2dc8cb2c67543ec22ce19f519521e.jpg) *Fig. 2. Validation loss curves of the concrete MX data types on 4 robotic training workloads compared with the FP32 baseline, showing that FP32 can be replaced by low-bit MX types towards robotics learning.*

---

**研究动机与现有方案局限**

- **SotA 连续学习处理器 Dacapo** 仅支持 MXINT 类格式（MX9/MX6/MX4，为 MX 标准的前身），**不支持 MXFP**，无法兼容新一代 MX 标准且无法受益于 FP 精度。
- **算术单元层面**的挑战：
  - 现有 MX 硬件要么缺乏精度可扩展能力，要么采用 **1-bit 处理单元**拆分数据通路，导致能耗或面积低效。
  - 六种 MX 格式的 mantissa 长度各异（1、2、3、8 bit），且混合 INT 与 FP 运算，统一支持所有格式的硬件设计复杂度极高。
  - 精确加法对 MXFP8 E5M2 极为低效——已有工作需要 **78-bit 加法器**。
- **架构层面**的挑战：
  - MX 标准的**向量式分组**（32 元素行/列分组）在反向传播中造成严重低效：训练需要权重矩阵的**原始与转置两种形式**，行分组与列分组量化结果不一致。
  - 现有解法要么**存储两份独立量化的权重**（内存翻倍），要么**存储未量化权重并反复量化**（计算与能耗开销），均不利于资源受限的边缘设备。

![](images/e307d006271caa90f4684aec543fc87f4d1ea11ffc04be9d2d05ef04e52ff986.jpg) *Fig. 4. The L1 and L2 adders, the INT8 path is indicated in brown, FP8/FP6 in red and FP4 in blue. (a) Vector Blocks (b) Square Blocks Fig. 5. Training computation graphs with MX vector and square block quantizers. Computational and memory footprint gains can be achieved for square blocks, by enabling storage of block-quantized parameters for backpropagation.*

---

**核心贡献**

- **贡献一：精度可扩展的 MX MAC 算术单元**
  - 首个支持全部六种 MX 数据类型的 MAC 设计。
  - 以 **2-bit 乘法器**为基本计算单元（16 个互联），在单一设计中统一 INT 与 FP 处理，避免 1-bit 方案的能耗/面积低效。
  - 三种工作模式：**INT8**（16 个乘法器协作完成单个 INT8×INT8）、**FP8/FP6**（4 路并行，每路 4 个乘法器 + 5-bit exponent adder）、**FP4**（8 路并行，每路配 2-bit exponent adder）。
  - **Sum-Together 分层累加器**：复用精度扩展中释放的乘法器/加法器作为部分积加法电路，每种精度下每 MAC 仅产生单一输出，简化阵列级数据通路。

![](images/cd2e43c9b1b59adedcbcf2de4d09eaa68569e281b139b5f7e85b2054c88eae4b.jpg)

- **贡献二：方形共享指数块的 PE Array 与学习使能 MX GeMM Core**
  - 用 **64 元素 (8×8) 方形块**取代传统 32 元素向量分组，使同一量化表示可同时服务于前向与反向传播，**消除重复量化**，权重存储减少约 **50%**（两个 32 元素块共享同一 exponent，保持与 MX 标准兼容）。
  - **4×16 PE Array 网格**构成 GeMM Core：高度 4 对应 batch size 32 除以 8×8 方块维度，宽度 16 平衡带宽效率与计算并行度，适配边缘设备小 batch 训练约束。
  - 采用 **output-stationary 数据流**，FP32 中间结果在 PE 内局部累加，最小化数据搬运。

![](images/48b04174a270c3c4ba05580d601e6c6e51a0f88fda297d9589cf5019f8ae052d.jpg) *Fig. 6. The PE Array that handles the multiplication of two 64-element square blocks, for INT8 this is executed in 8 cycles, for FP8/FP6 this takes 2 cycles, and for FP4 in 1 cycle. The PE Array is part of our GeMM core of 4x16 PE Arrays.*

- **贡献三：与 Dacapo 的全面对比评估**（TSMC 16nm FinFET、400MHz、**iso-peak-throughput**、同等 4096 MACs）：

| 指标 | Ours | Dacapo |
|---|---|---|
| 面积 [mm²] | 8.92 | 8.52 |
| 最大带宽 [GB/s] | 264 | 512 |
| 内存 [KB] | **179.78** | 370.13 |
| MXINT8 能耗/OP [pJ] | **2.43** | 2.68 |
| MXFP4 能耗/OP [pJ] | **0.278** | 0.358 |
| MXINT8 训练延迟/Batch [µs] | **13.575** | 50.5 |
| MXFP8/FP6 训练延迟/Batch [µs] | **6.03** | 30.7 |
| MXFP4 训练延迟/Batch [µs] | **4.7625** | 25.75 |

  - 最终实现 **51% 内存占用降低**、**4× 有效训练吞吐提升**、相近能效，并完整支持 MX 标准的全部六种数据格式，为边缘端机器人持续学习提供了高效的硬件基础。

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**整体定位**

本文提出面向机器人边缘端持续学习的 **Microscaling (MX) 训练加速器**，是首个支持 MX 标准全部六种数据类型（MXINT8、MXFP8 E5M2/E4M3、MXFP6 E3M2/E2M3、MXFP4 E2M1）的硬件实现。整体技术架构自底向上分为三个层次：**precision-scalable MX MAC 算术单元** → **64-MAC 方块式 (square-based) PE Array** → **4×16 PE Array 组成的 learning-enabled MX GeMM Core**。

![](images/8d3a2057e591e4c7ba4b00b8028ad45e75f87df5e841485a5238094ec30f15ab.jpg)

---

**层次一：Precision-Scalable MX MAC 算术单元**

- 以 **16 个 2-bit 基础乘法器**为最小计算单元，通过灵活互连统一支持 INT 与 FP 运算，避免 1-bit 处理块设计的面积/能耗低效。
- 支持三种动态重配置工作模式：
  - **INT8 模式**：16 个 2-bit 乘法器全协作，计算单个 INT8×INT8 的 16-bit 乘积，exponent 加法器不激活。
  - **FP8/FP6 模式**：同一硬件并行处理 4 个 FP 乘积，每个乘积占用 4 个 2-bit 乘法器做 mantissa 乘法，外加一个 5-bit exponent 加法器。
  - **FP4 模式**：每个 2-bit 乘法器配对 2-bit exponent 加法器构成完整 FP4 单元，受限于带宽仅并行计算 8 个 FP4×FP4 结果。
- 采用 **Sum-Together 层级化双级累加器**，实现任意精度下每 MAC 单元单输出，简化 array 级数据通路。

![](images/3e1a74d22c4d884b22697dbd805305cef66817745d973ad5a9f568cb2e984034.jpg)

- **L1 Adder**：INT8/FP8/FP6 模式下累加 4-bit 整数部分积形成 8-bit 输出；FP4 模式下直接按 exponent 移位求和四个 E3M4 结果（利用其 exponent 范围仅 0–4 的特性），免除“先求最大 exponent”的对齐步骤。
- **L2 Adder**：对 FP8/FP6 采用 **FP32 加法（26-bit mantissa 加法器）**，替代直接移位所需的 78-bit 整数加法器；以 **mantissa 加法器仅扩展 2 bit** 的方式处理窄格式中的 subnormal 数，保证 FP32 精度且免除昂贵的 normalization 电路。
- **关键路径优化**：INT8（sign-magnitude 转换）与 FP4（可变移位器）模式在 L1 adder 有较长路径，可 **bypass 仅为 FP8/FP6 所需的 L2 对齐逻辑**，平衡各模式时序。

![](images/cd2e43c9b1b59adedcbcf2de4d09eaa68569e281b139b5f7e85b2054c88eae4b.jpg)

---

**层次二：Square-based MX PE Array**

- 核心创新是将传统 MX 的 **32-element 向量分组** 替换为 **64-element (8×8) 方块分组**，一个 8×8 block 内所有元素共享单一 exponent（两个 32-element 组共享同一 exponent，保持与 MX 标准的兼容性）。
- 设计动机：反向传播需同时使用权重矩阵的原始与转置形式做行/列内积，向量分组导致两种形式量化结果不一致，被迫存储双份量化权重或反复 requantization；方块分组具有**对称性**，使同一量化表示可服务所有训练 pass，**消除存储冗余与量化开销**。

![](images/e307d006271caa90f4684aec543fc87f4d1ea11ffc04be9d2d05ef04e52ff986.jpg) *Fig. 4. The L1 and L2 adders, the INT8 path is indicated in brown, FP8/FP6 in red and FP4 in blue. (a) Vector Blocks (b) Square Blocks Fig. 5. Training computation graphs with MX vector and square block quantizers. Computational and memory footprint gains can be achieved for square blocks, by enabling storage of block-quantized parameters for backpropagation.*

- 每个 PE Array 包含 **64 个上述 MAC 单元**，处理两个 8×8 block 的 GeMM（每个输出需累加 8 个乘积），不同精度下吞吐差异化：
  - INT8：**8 个周期**完成一个 block 乘法
  - FP8/FP6：**2 个周期**
  - FP4：**1 个周期**

---

**层次三：Learning-enabled MX GeMM Core**

![](images/48b04174a270c3c4ba05580d601e6c6e51a0f88fda297d9589cf5019f8ae052d.jpg) *Fig. 6. The PE Array that handles the multiplication of two 64-element square blocks, for INT8 this is executed in 8 cycles, for FP8/FP6 this takes 2 cycles, and for FP4 in 1 cycle. The PE Array is part of our GeMM core of 4x16 PE Arrays.*

- 64 个 PE Array 排布为 **4×16 网格**：
  - 高度 4 对应 batch size 32 ÷ 8（square 维度）
  - 宽度 16 在带宽效率与计算并行度间取得平衡
  - 矩形构型适配边缘端小 batch 场景（backward pass 需缓存 activations）
- **Output-stationary 数据流**：partial results 在各 PE 内本地累加，最小化 FP32 中间数据传输；输入 block 的 shared exponents 在 PE 级相加后施加于每个 MAC 的 L2 adder 输出。
- 训练阶段执行特征：
  - **Forward / backward pass**：高利用率
  - **Weight gradient 计算**：沿 batch 维度 32 累加，频繁写回内存，阵列利用率下降
  - 最大带宽 **5280 bits/cycle（约 264 GB/s）**，FP8/FP4 操作时被完全占用；stall 周期用于写回 FP32 输出并送至 quantizer
  - 权重更新（梯度加到权重）在独立 accelerator block 中完成，GeMM Core 专注矩阵乘并维持高效流水

---

**关键性能指标（TSMC 16nm FinFET, 400MHz, iso-peak-throughput）**

| 指标 | Ours | Dacapo [24] |
|---|---|---|
| Area [mm²] | 8.92 | 8.52 |
| Max BW [GB/s] | 264 | 512 |
| Memory [KB] | **179.78** | 370.13 |
| MAC 数量 | 4096 | 4096 |
| E/op MXINT8 (MX9) [pJ] | **2.43** | 2.68 |
| E/op MXFP8/FP6 (MX6) [pJ] | 1.92–2.12 | 1.46 |
| E/op MXFP4 (MX4) [pJ] | **0.278** | 0.358 |
| 训练延迟 (MXINT8) [µs/batch] | **13.575** | 50.5 |
| 训练延迟 (MXFP8/FP6) [µs/batch] | **6.03** | 30.7 |
| 训练延迟 (MXFP4) [µs/batch] | **4.7625** | 25.75 |

- 相较 SotA Dacapo，实现 **51% 内存占用降低**、**4× 有效训练吞吐提升**、**2.06× 内存节省**、**1.94× 带宽需求降低**，能耗效率相当（MXINT8 低 10%、MXFP4 低 29%，MXFP8 略高 1.32–1.45×），同时完整覆盖 MX 标准六种数据格式。

### 1. Precision-Scalable MX MAC Unit (支持全部六种MX格式的统一INT/FP算术单元)

**核心定位与设计动机**

- Precision-Scalable MX MAC Unit 是本文在**算术单元层**的核心创新，目标是单一硬件同时覆盖 MX 标准定义的全部**六种 MX 数据格式**：MXINT8、MXFP8 (E5M2 / E4M3)、MXFP6 (E3M2 / E2M3)、MXFP4 (E2M1)。
- 设计动机源于三个现实约束：
  - Dacapo 等 SotA 方案仅支持 MXINT（且是其前身格式 MX9/MX6/MX4），无法兼容新一代 **OCP MX 标准**，也无法在需要 FP 精度的训练负载上获得精度收益。
  - 现有精度可扩展硬件要么缺乏**precision-scaling 能力**，要么采用 1-bit 处理单元（如 Flexibit）导致**面积/能耗低效**。
  - 精确加法对 MXFP8 E5M2 极不友好——直接对齐指数需要 **78-bit 加法器**（因指数范围达 ±30），硬件代价不可接受。
- 关键洞察：以 **2-bit 乘法器作为基础计算单元**，通过灵活互连在不同精度模式间**复用硬件**，并将精度缩减后“富余”的乘法器**重新用途化为部分积加法器**（Sum-Together 方案），而非让其闲置。

![](images/3e1a74d22c4d884b22697dbd805305cef66817745d973ad5a9f568cb2e984034.jpg)

---

**基础架构与三种工作模式**

- MAC 单元由 **16 个 2-bit 乘法单元**构成，配合层级化加法树（L1 / L2 adder）与一个 FP 累加寄存器，形成三种动态可重构模式：

- **INT8 模式 (Fig. 3a)**：
  - 全部 16 个 2-bit 乘法器协同工作，计算单个 INT8×INT8 乘法，生成 **16-bit 结果**。
  - 指数加法器全部**处于非活跃状态**，此时 L1 加法器负责带移位的 4-bit 部分积累加，构成传统整数部分积树。

- **FP8/FP6 模式 (Fig. 3b)**：
  - 同一硬件**并行处理 4 路** FP8/FP6×FP8/FP6 乘法。
  - 每路乘法占用 **4 个 2-bit 乘法器**做 mantissa 乘法，外加一个专用 **5-bit 指数加法器**。
  - 该模式同时覆盖 E5M2、E4M3、E3M2、E2M3 四种格式——指数/尾数位宽差异由上层量化与对齐逻辑吸收，底层 mantissa 乘法结构统一。

- **FP4 模式 (Fig. 3c)**：
  - 每个活跃 2-bit 乘法器配对一个 **2-bit 指数加法器**，形成完整 FP4 处理单元。
  - 受**带宽限制**，该模式仅输出 **8 路并行 FP4×FP4 结果**，与 FP8/6 模式保持相同数据带宽；这 8 路结果随后在共享指数的 8×8 block 内求和。
  - 设计权衡：不追求满并行度，而是让带宽在所有精度下保持一致，简化 array 层数据通路。

- **吞吐一致性**：INT8 每 MAC 每周期 1 个乘法、FP8/6 为 4 个、FP4 为 8 个（带宽受限），实现**计算密度随精度降低而递增**，三种模式对上层 PE Array 分别对应 8 / 2 / 1 周期完成一个 8×8 block 乘法。

---

**Area-Efficient 层级化 MX 累加器**

![](images/cd2e43c9b1b59adedcbcf2de4d09eaa68569e281b139b5f7e85b2054c88eae4b.jpg)

- **L1 加法器**：
  - INT8/FP8/FP6 模式下，累加带适当移位的 **4-bit 整数部分积**，形成 8-bit 输出。
  - FP4 模式下，直接按各结果的 exponent 对 **E3M4 mantissa 做变移位后求和**，**省去先求最大指数的归一化步骤**——这得益于 E3M4 指数范围极窄（0–4）。
  - 硬件复用：同一个整数加法器服务所有精度模式，FP4 模式仅需额外 **2-bit 扩展**。

- **L2 加法器**：
  - 若对 FP8/FP6 采用直接整数移位累加，将需要不可行的 **78-bit 加法器**。
  - 本文方案：采用 **FP32 加法**，mantissa 加法器仅需 **26-bit**，并针对窄格式中大量 **subnormal number** 的存在，将内部 mantissa 位宽**额外扩展 2-bit**，以**廉价的位宽冗余替代昂贵的归一化电路**，保证即便输入未归一化也能达到 **FP32-accurate addition**。

- **关键路径优化**：
  - INT8 模式因 sign-magnitude 转换、FP4 模式因变移位器，均在 **L1 处形成较长关键路径**。
  - 解决方案：为这两种模式添加**旁路逻辑，绕过仅 FP8/FP6 需要的 L2 对齐逻辑**，实现各模式关键路径的均衡。
  - 该优化由综合工具自动实现，效果直接反映在 Table II 中。

---

**设计变体的量化评估**

- 三种实现变体在 TSMC 16nm FinFET、400 MHz 下综合，能量取 500 周期随机输入的平均值：

| MAC 变体 | 频率 | 面积 [µm²] | E5M2 [pJ/OP] | E4M3 | E3M2 | E2M3 | E2M1 |
|---|---|---|---|---|---|---|---|
| mant. add. ext.（基线） | 400 MHz | 2341.35 | 1.62 | 1.18 | 1.22 | 1.16 | 1.22 | 0.29 |
| norm. inputs（归一化输入） | 400 MHz | 3640.78 | 2.46 | 2.41 | 2.58 | 2.38 | 2.58 | 0.60 |
| **mant. add. ext. + bypass** | 400 MHz | **2078.42** | **1.36** | **1.14** | **1.16** | **1.11** | **1.16** | **0.28** |

- 结论要点：
  - **2-bit mantissa 扩展 + 关键路径旁路**的最终方案相比归一化输入方案，面积降低约 **43%**（3640.78 → 2078.42 µm²），各格式能耗普遍降低 **50% 以上**（如 E5M2: 2.46 → 1.36 pJ/OP）。
  - 旁路逻辑使时序更易收敛，综合工具无需插入大驱动强度单元与 buffer，带来显著面积红利。

![](images/6c8a254d8ec64f81f0ae684673ae53b84d921a04cba8cbc517909e255f44faed.jpg)

- 在 PE Array 层的分解进一步验证：
  - **FP 累加加法是能耗主导项**；L1/L2 加法器因含模式专属组件而占据**最大面积**——面积与能耗呈不对称分布。
  - **Shared exponent 逻辑的能耗开销可忽略不计**，支撑了整个架构将尺度处理融入 MAC 的决策。

---

**输入输出关系与在整体系统中的作用**

- **输入**：来自 8×8 square block 的元素对（INT8 / FP8 / FP6 / FP4）以及 block 级 **E8M0 shared exponent**。
- **输出**：无论何种精度模式，每个 MAC 单元均产生**单一输出**（经 FP32 精度累加），由 Sum-Together 方案保证——这一设计大幅**简化 array 层数据通路**。
- **Shared exponent 的集成方式**：在 PE 级将输入双方的 shared exponent 相加，随后施加于每个 MAC 的 **L2 加法器输出**，完成块级尺度还原，使 output-stationary 数据流下的本地累加无需传输 FP32 中间数据。
- **向上承接**：64 个此类 MAC 组成一个 PE Array（处理 8×8 square block 乘法：INT8 需 8 周期、FP8/6 需 2 周期、FP4 需 1 周期），4×16 个 PE Array 进一步组成 **Learning-enabled MX GeMM Core**。
- **系统级收益**（与 Dacapo iso-peak-throughput 对比，同为 4096 MAC、400 MHz、TSMC 16nm）：
  - MXINT8 模式能耗低 **10%**（2.43 vs 2.68 pJ/OP），MXFP4 低 **29%**（0.278 vs 0.358 pJ/OP）；MXFP8/FP6 为 Dacapo 对应模式的 1.32–1.45 倍，是全格式兼容付出的代价。
  - 结合 square block 架构，整体实现 **4× 有效训练吞吐**、**51% 内存占用削减**，为边缘机器人 **continual learning** 提供从数据格式、算术单元到阵列架构的全栈精度可扩展支撑。

### 2. Square-based MX PE Array and GeMM Core (64元素方形共享指数块架构)

---

**核心定位**

该模块是论文的**架构级核心创新**，由两部分组成：**Square-based MX PE Array（基于方形块的 PE 阵列）**与**Learning-enabled MX GeMM Core（支持训练的 GeMM 核）**。其根本目标是解决传统 MX **32元素向量分组（vector-based grouping）**在反向传播中导致的**权重双重存储或重复量化**问题，同时与算术单元级的精度可扩展 MAC 协同，构成完整的边缘训练加速器。

![](images/e307d006271caa90f4684aec543fc87f4d1ea11ffc04be9d2d05ef04e52ff986.jpg) *Fig. 4. The L1 and L2 adders, the INT8 path is indicated in brown, FP8/FP6 in red and FP4 in blue. (a) Vector Blocks (b) Square Blocks Fig. 5. Training computation graphs with MX vector and square block quantizers. Computational and memory footprint gains can be achieved for square blocks, by enabling storage of block-quantized parameters for backpropagation.*

---

**问题根源：为什么向量分组在训练中失效**

- 神经网络训练的计算图（training computation graph）涉及三类 GeMM 运算：**前向传播**、**误差反向传播**、**权重梯度计算**，后两者要求权重矩阵以**原始形式（W）**和**转置形式（W^T）**分别参与行方向和列方向的点积（dot-product）。
- MX 标准将共享指数（shared exponent）绑定于**32元素向量块**：
  - 对原始矩阵按**行向量**做块量化，得到一组共享指数；
  - 对转置矩阵按**行向量**（即原始矩阵的列）做块量化，得到**另一组完全不同的共享指数**；
  - 两种量化结果**数学上不可互换**。
- 由此产生两种已被采用但均不理想的解法：
  - **双重存储方案**（如 FlexBlock、FAST，即文献 [31][32]）：同时保存两份分别量化的权重版本，**存储需求翻倍**；
  - **重复量化方案**（如文献 [33]）：存储未量化的 FP 权重，每次使用时重新量化，引入**持续的计算开销与能耗**。
- 两种方案对资源受限的**边缘设备（edge device）**均构成关键瓶颈。

---

**方形块设计原理与参数设置**

- **块结构**：以 **64元素（8×8）方形块**取代 32元素向量块，块内所有 64 个元素共享**同一个 E8M0 格式共享指数**（8-bit 指数、无尾数，限定为 2 的幂次）。
- **指数设定规则**：遵循 MX 标准语义——共享指数取块内最大绝对值元素的最大 2 的幂，除以元素格式本身可表示的最大 2 的幂。
- **为什么选择 8×8 而非其他尺寸**：
  - **与 MX 标准兼容**：标准定义分组为 32 的倍数，8×8 = 64 恰好对应**两个 32元素块共享同一指数**，不违反规范语义；
  - **对称性（symmetry）**：方形块的行分组与列分组结果完全一致，这是消除 requantization 的数学基础；
  - **粒度与效率平衡**：块越大，共享指数的量化误差风险越高（大动态范围元素挤占小元素精度）；8×8 在消除转置冗余与保持量化精度之间取得平衡。
- **对称操作的核心收益**：
  - 同一份量化后的权重表示**同时服务前向与反向传播**，转置操作仅是索引重排，**不改变共享指数**；
  - 前向用 W 乘以激活，反向用同一份量化数据以转置布局乘以误差，**无需 requantization**；
  - 权重存储近似**减少 50%**。

![](images/48b04174a270c3c4ba05580d601e6c6e51a0f88fda297d9589cf5019f8ae052d.jpg) *Fig. 6. The PE Array that handles the multiplication of two 64-element square blocks, for INT8 this is executed in 8 cycles, for FP8/FP6 this takes 2 cycles, and for FP4 in 1 cycle. The PE Array is part of our GeMM core of 4x16 PE Arrays.*

---

**PE Array：组织结构与运算流程**

- **基本组成**：单个 PE Array 包含 **64 个精度可扩展 MX MAC 单元**，与 8×8 方形块一一对应；每个 MAC 支持 **INT8 / FP8·FP6 / FP4** 三种精度模式（复用 16 个 2-bit 基础乘法器）。
- **核心运算**：两个 8×8 方形块相乘（block GeMM），每个输出元素需要累加 **8 个乘积**，共 64 个输出，合计 512 次乘法、4096 次乘加操作涉及完整块对。
- **分时复用策略（关键设计）**：通过 MAC 单元的精度可扩展性，不同精度模式下的块乘法周期数不同：

| 精度模式 | 并行度（每 MAC） | 块乘法周期数 | 说明 |
|---|---|---|---|
| **MXINT8** | 1 路 INT8×INT8 | **8 cycles** | 16 个 2-bit 乘法器全数合作用于单个 16-bit 乘积 |
| **MXFP8 / MXFP6** | 4 路并行 FP 乘积 | **2 cycles** | 每路用 4 个 2-bit 乘法器做尾数乘法 + 1 个 5-bit 指数加法器 |
| **MXFP4** | 8 路并行 FP4 乘积 | **1 cycle** | 每 2-bit 乘法器配 2-bit 指数加法器；受**带宽限制**，并行度止步于 8（保持与 FP8/6 模式相同的输入带宽 BW） |

- **FP4 模式的带宽约束权衡**：
  - FP4 理论上可支持更高并行度，但会要求输入端口带宽翻倍；
  - 设计选择**锁定带宽等于 FP8/6 模式**，8 路结果在一个 8×8 共享指数块内求和，保持接口一致性，代价是该模式未用满计算密度。
- **共享指数处理**：输入两个块的共享指数在 **PE 级相加**，随后施加到每个 MAC 的 **L2 加法器输出**上，完成块级缩放，保证累加结果正确的数量级。

---

**GeMM Core：全局架构与数据流**

- **网格配置**：**4×16 = 64 个 PE Array**，即 **4096 个 MAC**，总面积 8.92 mm²（TSMC 16nm FinFET，400 MHz）。
- **维度选择依据**：
  - **高度 4**：对应 **batch size 32 ÷ 8**（方形块维度），使阵列高度天然匹配小 batch 边缘训练场景；
  - **宽度 16**：在**带宽效率**与**计算并行度**之间取最优平衡点。
- **Dataflow：输出驻留（Output-Stationary）**：
  - 部分和（partial sums）在**每个 PE 内本地累加**，最小化 FP32 中间数据的搬运；
  - 这要求共享指数在算术单元的 FP 累加器内完成对齐（见上节 PE 级指数相加机制）。
- **训练各阶段的执行特征**：
  - **前向传播**：高阵列利用率，规则矩阵乘法；
  - **反向传播**：前向的镜像，利用率相当；
  - **权重梯度计算**：累加发生在更小的 **batch 维度（32）** 上，需要**更频繁的内存写回**，阵列利用率显著下降，是整体流水线的瓶颈阶段。
- **内存带宽与写回机制**：
  - 最大带宽 **5280 bits/cycle（约 264 GB/s）**，在 FP8 与 FP4 操作时被完全占用；
  - stall 周期（主要出现在权重梯度阶段）期间，带宽用于**写回 FP32 输出**，随后送至**量化器（quantizer）**重新生成方形块；
  - **权重更新**（梯度加到权重上）由**独立的加速器模块**完成，使 GeMM Core 专注于矩阵乘法，维持流水线效率。

---

**量化性能收益（对比 SotA Dacapo）**

- 评测条件：TSMC 16nm FinFET、400 MHz、**iso-peak-throughput**（双方均为 4096 MAC）。

| 指标 | Ours | Dacapo | 对比 |
|---|---|---|---|
| 面积 | 8.92 mm² | 8.52 mm² | 1.05×（略增） |
| 最大带宽 | 264 GB/s | 512 GB/s | **1.94× 更低** |
| 内存占用 | 179.78 KB | 370.13 KB | **2.06× 更低（-51%）** |
| MXINT8 训练延迟/batch | 13.575 µs | 50.5 µs | **3.72× 更快** |
| MXFP8/FP6 训练延迟/batch | 6.03 µs | 30.7 µs | **5.1× 更快** |
| MXFP4 训练延迟/batch | 4.7625 µs | 25.75 µs | **5.4× 更快** |

- **内存占用细分（pusher workload，4 层全连接网络，batch=32）**：

| 方法 | 权重存储 | 总占用 | 相对 FP32 |
|---|---|---|---|
| FP32 基线 | 576.0 KB（W） | 708.0 KB | 1.00× |
| Dacapo（向量块） | 162.0 + 162.0 KB（W 与 W^T 双份） | 370.1 KB | 1.91× |
| **Ours（方形块）** | **146.3 KB（单份，无 W^T）** | **179.8 KB** | **3.94×** |

- **吞吐优势来源**：Dacapo 采用脉动阵列，数据**脉动移入/移出（systolically shifting）**带来固有开销；本设计的 output-stationary dataflow + 分时精度复用规避了该开销，实现**约 4× 有效训练吞吐提升**。
- **能耗权衡**：MXINT8 与 MXFP4 模式每 OP 能耗低于 Dacapo（2.43 vs 2.68 pJ；0.278 vs 0.358 pJ），但 MXFP8 模式高 1.32–1.45×——这是支持完整浮点尾数处理的代价，最终通过端到端能效（Fig. 8 右）验证 MXFP8 的训练能效与 Dacapo 的 MX9 相当。

---

**在整体系统中的作用与输入输出关系**

- **输入**：
  - 两个待相乘的矩阵分块，每个分块为 8×8 方形块 + 1 个 E8M0 共享指数；
  - 精度模式配置信号（INT8 / FP8·FP6 / FP4，覆盖全部六种 MX 数据类型）。
- **输出**：
  - 本地累加完成的 FP32 部分和/最终和，经共享指数缩放后按 dataflow 写回；
  - 权重梯度阶段的输出经量化器重新块化，供后续迭代使用。
- **系统级角色**：
  - 作为**算术单元层（Sec. III MAC）与系统应用层（机器人持续学习 workload）之间的桥梁**：MAC 提供精度灵活性，PE Array 将其映射到方形块语义，GeMM Core 组织为可训练的完整计算引擎；
  - 消除反向传播的量化冗余 + 提供全六种 MX 格式支持，使边缘机器人能在**不依赖云端**的前提下，按任务特性（如 pusher/reacher 用 MXFP8 E4M3，cartpole/halfcheetah 用 MXINT8）动态选择精度，实现精度-能效-存储的三维优化。


---

## 4. 实验方法与实验结果

**一、实验设置总览**

本文实验围绕三个层级展开：**算术单元级（MX MAC）**、**PE Array 级**、**GeMM Core 级（系统级对比）**，形成由下而上的完整验证链条。

- **工艺与频率**：
  - 全部设计基于 **TSMC 16nm FinFET** 工艺综合实现
  - 时钟频率统一为 **400 MHz**
  - 功耗与能耗分析使用 **Synopsys PrimeTime PX**，基于 switching annotated power 仿真
- **对标基线**：
  - 对比对象为 SotA 连续学习处理器 **Dacapo**（ISCA 2024）
  - 为保证公平性，采用 **iso-peak-throughput** 原则：双方均配备 **4096 个 MAC** 单元
  - Dacapo 采用 8×8 systolic array（共 64 个），数据格式为其自研的 **MX9/MX6/MX4**（MX 标准前身，vector block 为 16 元素 + 8-bit 共享指数 + 2 元素子组的 1-bit 二级指数）
- **测试数据与负载**：
  - MAC 单元能耗测试：**500 cycles 随机输入数据**取平均
  - PE Array 能耗测试：**100 次 block multiplication**，总计 **51,200 次乘法 OP**，随机输入
  - 系统级评估使用 **pusher** 机器人连续控制任务（4 层全连接网络，输入/输出维度 32，隐藏层维度 256）
  - Fig. 2 的精度验证覆盖 **4 个 robotics 训练负载**（pusher、reacher、cartpole、halfcheetah），并与 FP32 baseline 对比 validation loss

![](images/7f461e7677d1fdf402a14494c4f293a1fae2dc8cb2c67543ec22ce19f519521e.jpg) *Fig. 2. Validation loss curves of the concrete MX data types on 4 robotic training workloads compared with the FP32 baseline, showing that FP32 can be replaced by low-bit MX types towards robotics learning.*

- **精度-负载适配性发现**：
  - **MXFP8 E4M3** 在 **pusher / reacher**（机器人-物体交互类任务）中训练效果最优
  - **MXINT8** 在 **cartpole / halfcheetah**（平衡控制类任务）中表现最佳
  - 该发现论证了硬件必须**同时支持全部 6 种 MX 格式**的必要性

---

**二、算术单元级消融实验（Table II）**

Table II 针对精度可扩展 MX MAC 中**非归一化输入处理**与**关键路径平衡**两项设计决策进行消融，共三组配置：

| MX MAC 变体 | 频率 | 面积 [µm²] | E5M2 | E4M3 | E3M2 | E2M3 | E2M1 |
|---|---|---|---|---|---|---|---|
| mant. add. ext. | 400 | 2341.35 | 1.62 | 1.18 | 1.22 | 1.16 | 0.29 |
| norm. inputs | 400 | 3640.78 | 2.46 | 2.41 | 2.58 | 2.38 | 0.60 |
| ext. + bypass | 400 | **2078.42** | **1.36** | **1.14** | **1.16** | **1.11** | **0.28** |

- **消融维度一：子规范数处理策略**
  - 方案 A（**mantissa adder 扩展 2-bit**）：面积 2341.35 µm²
  - 方案 B（**L2 输入归一化**）：面积 3640.78 µm²，比方案 A 增加 **~56%**，能耗增加 **~2 倍**（如 E5M2 从 1.62 升至 2.46 pJ/OP）
  - 结论：扩展 mantissa 加法器位宽显著优于归一化电路，以极低开销保证 **FP32 精度加法**（26-bit mantissa adder 即可，避免了 E5M2 直接整数累加所需的 **78-bit adder**）
- **消融维度二：模式专属 bypass 逻辑**
  - 在方案 A 基础上加入 bypass 后：面积进一步从 2341.35 降至 **2078.42 µm²**（再降 **11%**），能耗全线下降
  - 原因：INT8 与 FP4 模式的关键路径在 L1 adder，可绕过仅 FP8/FP6 需要的 L2 对齐逻辑，使综合工具无需依赖大驱动强度 cell 和 buffer，时序更易收敛
- 注意事项：此消融**尚未包含 shared exponent 逻辑**，为纯粹的算术单元内部对比

![](images/cd2e43c9b1b59adedcbcf2de4d09eaa68569e281b139b5f7e85b2054c88eae4b.jpg)

---

**三、PE Array 面积与能耗分解（Fig. 7）**

![](images/6c8a254d8ec64f81f0ae684673ae53b84d921a04cba8cbc517909e255f44faed.jpg)

- **测试条件**：使用带 bypass 的 MAC 单元，100 次随机 block 乘法，能耗除以 51,200 次乘法 OP
- **能耗侧发现**：
  - **FP accumulation 加法是最耗能组件**（架构级瓶颈）
  - **Shared exponent 处理开销可忽略不计**——这是支撑 square block 方案可行性的关键量化证据
- **面积侧发现**：
  - **L1 与 L2 adder 占据最大面积份额**，因其包含各精度模式专属组件，复用度受限
  - 面积与能耗呈现**不对称性**：加法器面积大但乘法阵列耗能高
- **吞吐特性**（Sec. IV）：单个 8×8 square block 乘法耗时随精度自适应：
  - INT8 模式：**8 cycles**
  - FP8/FP6 模式：**2 cycles**（4 路并行乘积）
  - FP4 模式：**1 cycle**（8 路并行，受带宽限制主动降为 8 并行以保持与 FP8/6 相同的 BW）
- **带宽配置**：GeMM Core 最大 **5280 bits/cycle（约 264 GB/s）**，在 FP8/FP4 操作时被完全占用；stall 周期（典型为 weight gradient 阶段）用于 FP32 结果写回

---

**四、内存占用对比（Table III）**

Table III 在 pusher 负载上对比 **FP32 baseline、Dacapo（vector block）、本文（square block）**三种方案的内存占用：

| Batch Size | 方案 | 总内存 [KB] | 相对压缩比 |
|---|---|---|---|
| 16 | FP32 | 642.0 | 1.00× |
| 16 | Dacapo | 347.1 | 1.85× |
| 16 | **Ours** | **163.1** | **3.94×** |
| 32 | FP32 | 708.0 | 1.00× |
| 32 | Dacapo | 370.1 | 1.91× |
| 32 | **Ours** | **179.8** | **3.94×** |
| 64 | FP32 | 840.0 | 1.00× |
| 64 | Dacapo | 416.3 | 2.02× |
| 64 | **Ours** | **213.4** | **3.94×** |

- **核心机制**：backpropagation 需要 weight 的原始与转置两种形式
  - Dacapo 的 vector block 量化下，行/列方向量化结果不同，必须**存储双份量化权重**（W 与 W^T 各 162.0 KB）
  - 本文的 **64 元素（8×8）square block** 量化具有对称性，转置前后量化结果一致，**W^T 存储归零**
  - 权重存储从 324 KB 降至 146.3 KB，缩减近 **50%**
- **额外收益**：无需存储未量化权重再重复量化，消除了 requantization 的计算与能耗开销
- **压缩比稳定**：无论 batch size 如何变化，本文相对 FP32 始终维持 **3.94×** 压缩，而 Dacapo 在 1.85×–2.02× 之间波动
- 量化说明：本文方案整体比 Dacapo 少 **2.06×** 内存（对应正文所述 **51% 内存占用削减**）；两个 32 元素 block 共享同一 exponent，保持与 MX 标准的兼容性

---

**五、系统级综合对比（Table IV）**

在 iso-peak-throughput（均为 4096 MAC）下，本文 4×16 PE Array GeMM Core 与 Dacapo 的完整规格对比：

| 指标 | Ours | Dacapo [24] | 对比结果 |
|---|---|---|---|
| 频率 [MHz] | 400 | 400 | 持平 |
| 面积 [mm²] | 8.92 | 8.52 | 高 1.05×（可接受代价） |
| Max. BW [GB/s] | 264 | 512 | **降低 1.94×** |
| 内存 [KB] | 179.78 | 370.13 | **降低 2.06×** |
| E/op MXINT8 vs. MX9 [pJ] | 2.43 | 2.68 | **省 10%** |
| E/op MXFP8/FP6 vs. MX6 [pJ] | 1.92–2.12 | 1.46 | 高 1.32×–1.45×（劣势项） |
| E/op MXFP4 vs. MX4 [pJ] | 0.278 | 0.358 | **省 29%** |
| 训练延迟 MXINT8 [µs/batch] | 13.575 | 50.5 | **3.7× 加速** |
| 训练延迟 MXFP8/FP6 [µs/batch] | 6.03 | 30.7 | **5.1× 加速** |
| 训练延迟 MXFP4 [µs/batch] | 4.7625 | 25.75 | **5.4× 加速** |

- **能耗/OP 分析**：
  - MXINT8 与 MXFP4 两种模式能耗优于 Dacapo
  - MXFP8/FP6 模式能耗高于 Dacapo 对应的 MX6（1.32×–1.45×），原因是统一 FP/INT 数据通路带来的开销——这是为**全格式覆盖**付出的可接受代价
- **训练吞吐大幅领先的原因**：
  - Dacapo 采用 systolic array，数据需**脉动式逐级移入/移出**，产生大量通信开销
  - 本文采用 **output-stationary 数据流**，partial result 在 PE 内本地累加，避免 FP32 中间数据频繁搬运
  - 综合三档精度平均约 **4× 有效训练吞吐提升**（论文标题性结论的来源）
- **带宽减半**：512 → 264 GB/s，源于 square block 的双份权重消除与数据复用

---

**六、训练质量与能效对比（Fig. 8）**

![](images/7a69f9e336d42b97a25f3285749d6233a0968738cb7ba0e87e46da23293d3734.jpg)

- **实验设计**：在 **1250 µs 时间预算**与 **200 pJ 能量预算**两个维度下，对比 pusher 任务的 validation loss 收敛曲线
- **时间受限场景（左图）**：
  - **MXFP8 是明显的胜出者**——在限定时间内达到最低 validation loss，兼具快速与精确训练
  - 验证了“不同 robotics 负载适配不同精度”的核心主张：pusher 类交互任务受益于 FP 精度
- **能量受限场景（右图）**：
  - MXFP8 的能效与 Dacapo 的 **MX9 相当**（comparable），尽管 MXFP8 需要更多 epochs
  - 说明本文虽然在 MXFP8/FP6 模式单 OP 能耗偏高，但由于更低的每 batch 延迟与更少的内存访问，**端到端训练能效整体持平**
- **结论支撑**：最终实现 **4× 有效训练吞吐 + 51% 内存削减 + 相当能效** 的三重收益，同时覆盖 MX 标准全部 **6 种数据格式**（MXINT8、E5M2、E4M3、E3M2、E2M3、E2M1）

---

**七、实验体系评述**

- **优势**：
  - 消融实验设计精准，直接量化了 **mantissa 扩展 vs. 归一化**、**bypass 逻辑**两个关键设计点的收益（面积合计优化约 **~43%**：3640.78 → 2078.42 µm²）
  - 对比框架严谨：iso-peak-throughput、同工艺、同频率、同 MAC 数量，变量控制充分
  - 从单 OP 能耗到端到端训练收敛的多层级评估，避免单一指标误导
- **局限与可关注点**：
  - MXFP8/FP6 模式下能耗仍落后 Dacapo MX6 约 **32%–45%**，是统一数据通路的固有代价
  - Table III 仅基于 **pusher 单一负载**（4 层 FC 网络），对 CNN/Transformer 类负载的普适性未展示
  - FP4 模式因带宽限制无法跑满 16 个 2-bit multiplier 的理论并行度，存在设计折中
  - 系统级对比中 Dacapo 的延迟数据引自原文 [24] 计算值而非重新实现，可能存在配置差异

---

