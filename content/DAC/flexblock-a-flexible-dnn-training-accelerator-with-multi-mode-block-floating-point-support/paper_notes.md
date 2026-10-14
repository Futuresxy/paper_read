# FlexBlock: A Flexible DNN Training Accelerator With Multi-Mode Block Floating Point Support 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Seock-Hwan Noh, Jahyun Koo, Seunghyun Lee, et al.

**发表期刊/会议 (Journal/Conference)**: IEEE Transactions on Computers

**发表年份 (Publication Year)**: 2023

**研究机构 (Affiliations)**: Daegu Gyeongbuk Institute of Science and Technology (DGIST), Korea Advanced Institute of Science and Technology (KAIST), Korea University

---

## 1. 摘要

**目的**

- 解决 DNN 训练中 **浮点运算开销大** 的问题：传统 GPU/NPU 使用 FP32 或 bfloat16 进行训练，面对 AmoebaNet-A (469M 参数)、GPT-3 (175B 参数) 等超大模型时，内存占用与训练时间成本过高。
- 突破现有 **BFP (Block Floating Point) 加速器仅支持单一精度** 的局限，实现可在 **activation、weight、gradient 三类张量上独立配置不同 BFP 精度** 的训练加速器。
- 解决现有 precision-scalable MAC array (如 Bit Fusion、BitBlade) 在低精度模式下 **MAC 利用率下降** 以及 **depthwise 操作 (2D 运算) 硬件利用不足** 的问题，逼近理论加速上限 (8-bit 模式 4×，4-bit 模式 16×)。

---

**方法**

- **多模式 BFP 数据格式**
  - 支持 **4-bit / 8-bit / 16-bit ‘sign+mantissa’** 配 **8-bit 共享 exponent**，定义三种基础格式：**FB12** (X4W4)、**FB16** (X8W8)、**FB24** (X16W16)，支持 27 种张量级混合精度组合。
  - 采用细粒度 blocking 策略，按精度与 layer 类型动态调整最小 block size (如 Conv3 层 FB16 为 3×3×12)。

- **层次化核心架构 (Multiplier → PE → PU → Subcore)**
  - 将 Bit Fusion 的 **2D sub-word parallelism 拆分为两个 1D sub-word parallelism**：X 的 sub-word 映射到 PU 内的 4 个 PE，W 的 sub-word 映射到 subcore 内的 4 个 PU，由 **selective adder tree** 按精度模式选择性累加 psums。
  - 8-bit 模式提供 2 条累加路径、4-bit 模式提供 4 条累加路径，并行计算不同 output channel，使累加次数仅随精度降低线性增长 (而非 Bit Fusion 的 4 倍增长)。
  - 9 个 4b×4b multiplier 组成一个 PE 以映射 2D convolution；六个 subcore (54 个 INT16 MAC/core) 可按 Conv3/Conv5/Conv7 灵活聚类，core 利用率达 0.91–0.93。

- **双路径 Reduction Unit**
  - 为 3D 运算 (General Conv、FC) 与 **2D 运算 (DW-Conv、∂L/∂W 计算、Dilated Conv、Up Conv)** 各配置独立的 reduction unit，解决 WU 步骤中 depthwise full convolution 导致的利用率瓶颈；额外 2D reduction unit 的面积/功耗开销仅 **4.16% / 3.89%**。

- **动态精度控制 (Dynamic Precision Control)**
  - 利用 **FP2BFP converter** 统计每个 epoch 各张量的 underflow 比例，通过 **hysteresis controller** (32-bit underflow counter + 两个 32-bit comparator) 与上下阈值 (th_Down, th_Up) 比较，逐层自动调节下一 epoch 的 BFP 精度；控制器开销仅 0.5% 功率 / 0.3% 面积。

![](images/afd66c2569cd3050e03d446f55d6334c6b0c2c7f6b25fcf39de02b77fd9df4d5.jpg)

---

**结果**

- **训练精度 (Table IV 摘要)**
  - FB24/FB16 在 CIFAR-10/100 上精度与 FP32 基线相当；纯 FB12 平均精度下降 6.21% (CIFAR) / 11.46% (ImageNet)。
  - **FB12+BW16** (BW 步骤用 FB16) 在 CIFAR 数据集上恢复至基线水平；ImageNet 需 **FB12+BW24** (ResNet-18 达 68.20% vs FP32 的 69.95%)；Transformer (WMT14) 在 FB12 下 perplexity 仍与高精度格式接近。

- **硬件对比 (65nm, 333MHz, 64 cores)**

| 指标 | TPU-like SA | BitFusion-based (BF) | **FlexBlock (FB)** |
|---|---|---|---|
| 面积 [mm²] | 74.38 | 40.22 | **33.82** |
| 平均功耗 [W] | 9.19 | ~16.4 | **7.69** |
| 吞吐 (FB12) [TFLOPS] | 1.77 | 2.82 | **8.78** |
| 能效 (FB12) [GFLOPS/W] | 192.60 | 164.43 | **1073.30** |

- **性能与能耗**
  - 相比 BF：能耗平均降低 **65.3% (FB24) / 68.2% (FB16) / 79.3% (FB12)**；训练时间平均缩短 **32.0% / 47.5% / 68.4%**。
  - 相比 SA：能耗降低 **44.3%**，训练时间缩短 **52.7%**；总体实现 **1.5–5.3× 训练加速** 与 **2.4–7.0× 能效提升**。
  - 多精度支持的开销：面积 +30.2%、功耗 +41.8%、无延迟开销 (critical path 在 FP32 adder)；扣除开销后 FB16/FB12 有效加速为 **2.8× / 7.3×**。

- **动态精度控制效果**
  - CIFAR 数据集上相比固定 FB16，训练速度平均提升 **22.5%** 且无精度损失；相比静态最优精度选择再加速 **13.1%**。
  - ImageNet 上相比 FB12+BW24，训练时间仅增加 17% 但精度提升 6.7%。
  - 阈值敏感性：weight/activation 张量阈值不敏感 (精度损失 ≤0.21%)；**local gradient 需保持低阈值** (0.55, 0.75 时损失 1.88%)。

![](images/44e9edef3cba2023b49535afd62ba4296dab9efad71d4ea930896d046b1941d4.jpg)

---

**结论**

- FlexBlock 通过 **层次化 sub-word parallelism** 与 **双路径 reduction unit**，在低精度与 depthwise 操作下维持高 MAC 利用率，首次将 precision-scalable 架构有效扩展至 **DNN 训练** 场景。
- 多模式 BFP + 逐张量独立精度配置，使 4-bit 训练 (FB12+BW16/24) 成为可行方案，实现最高 **5.3× 加速** 且精度损失可忽略。
- 基于 underflow 统计的 **动态精度控制** 将精度自动压至最低必需水平，额外带来 **13% 训练加速**，无需人工调参即可适配不同 benchmark。
- 该工作为高灵活性、高能效的 BFP 训练硬件设计提供了方向，有望推动大规模 DNN 训练加速器的精度可扩展架构发展。

---

## 2. 背景知识与核心贡献

**研究背景**

- DNN 训练通常依赖 IEEE **FP32** 浮点运算，成本高昂；模型规模急剧膨胀（如 AmoebaNet-A 的 **469M 参数**、GPT-3 的 **175B 参数**），对内存带宽与算力提出严峻挑战。
- 社区开始探索更高效的数据表示以降低浮点运算负担：
  - **Mixed Precision Training**（FP16 乘法 + FP32 累加），借助 NVIDIA GPU 的 Tensor Cores 实现 2~4 倍加速。
  - **bfloat16**：保留 FP32 的 8-bit 指数位，截断尾数至 16-bit，已被商用训练加速器采用。
  - **Block Floating Point (BFP)**：一组数值共享一个 exponent，仅传输/存储 sign+mantissa，显著压缩内存占用，并允许用更廉价的定点乘累加（MAC）完成内积运算。
- BFP 下内积可分解为：$\langle \vec{w}, \vec{x} \rangle = \langle \vec{\hat{w}}, \vec{\hat{x}} \rangle \cdot 2^{e_w + e_x}$，即尾数部分可用纯定点逻辑计算，共享指数在末端统一处理。

**动机：现有方案的局限**

- 现有 BFP 训练加速器（如 Flexpoint、Hybrid BFP）绑定**特定精度格式**，缺乏灵活性，无法按张量（activation/weight/gradient）独立选择精度。
- 现有 precision-scalable MAC array（如 **Bit Fusion**、**BitBlade**）虽在推理端支持可变精度，但存在两大问题：
  - **2D sub-word parallelism 导致累加数暴增**：精度每降低 2 倍（X、W 双侧），所需的 partial sum 累加数量** quadruples（×4）**，造成 MAC under-utilization，无法兑现理论加速比。实测在 MobileNetV1 训练中，X16W16 利用率为 76.5%，降至 X8W8 再降 13.8%，降至 X4W4 额外损失 22.0%。
  - **不适用于训练的 2D 运算**：weight update (WU) 步骤由大量 depthwise convolution 构成，仅需少量累加，传统架构下 MAC 大量闲置。

![](images/d8aaa8a686a5f159c728d348d42f61343f12ab089e2a2a824ef9ed46598ddf62.jpg)

**FlexBlock 的应对思路**

- 将 Bit Fusion 的 **2D sub-word parallelism 拆分为两层 1D 结构**（PU 内对 X 并行、跨 PU 对 W 并行），使精度降低时累加数仅**翻倍（×2）**而非四倍。
- 为 2D/3D 运算分别配置**双路径 reduction unit**，专为 depthwise 运算优化利用率，额外开销仅 **4.16% 面积 / 3.89% 功耗**。

**核心贡献**

- **Multi-mode BFP 支持**：
  - 支持 **FB12 (4-bit)**、**FB16 (8-bit)**、**FB24 (16-bit)** 三种 sign+mantissa 精度，均搭配 **8-bit 共享指数**（与 FP32/bfloat16 动态范围一致）。
  - Activation、weight、gradient 三类张量可独立配置精度，共 **3³ = 27 种组合**。
  - 降低精度可提升 MAC 并行度：8-bit 模式每 core 的 MAC 数量 **×4**，4-bit 模式 **×16**（相对 16-bit 模式）。
- **高核心利用率**：
  - 通过 **multiplier → PE (9 个乘法器) → PU (4 个 PE) → subcore (4 个 PU)** 的层次化结构映射不同 tensor 维度，覆盖 Conv1/Conv3/Conv5/Conv7、DW-Conv、FC 等层型（Conv5 利用率 0.93，Conv7 利用率 0.91）。
  - 每个 core 含 6 个 subcore，提供 **54 个 INT16 MAC**；按精度与层型自适应调整 BFP block size。
- **低精度训练与动态精度控制**：
  - 实证 **FB12+BW16/BW24** 混合精度方案：FW 与 WU 用 4-bit，仅 backward pass 提升至 8/16-bit，即可在 CIFAR-10/100、ImageNet、WMT14 上保持接近 FP32 的精度。
  - 提出基于 **underflow 计数 + hysteresis 控制器**的动态精度调节机制（硬件开销仅 0.5% 功耗 / 0.3% 面积），自动逐层、逐 epoch 选择最低可用精度，较静态 BFP 选择再提速 **13.1%**，较 FB16 平均提速 **22.5%** 且无精度损失。
- **端到端性能优势**：
  - 65nm 工艺、64 cores、333 MHz 下，与 **TPU-like systolic array (bfloat16)** 和 **BitFusion-based BFP 加速器**对比：

| 指标 | 对比 BF (FB24/FB16/FB12) | 对比 SA (bfloat16) |
|---|---|---|
| 能耗降低 | 65.3% / 68.2% / 79.3% | 44.3% (平均) |
| 训练时间缩短 | 32.0% / 47.5% / 68.4% | 52.7% (平均) |
| 面积 | 1.2× 更小 (33.82 vs 40.22 mm²) | 2.2× 更小 (vs 74.38 mm²) |
| 功耗 | 约 2.1× 更低 (7.69W vs 16.37W 均值) | 约 1.2× 更低 (vs 9.19W) |

  - 整体上实现 **1.5~5.3 倍训练加速**与 **2.4~7.0 倍能效提升**（相对其他训练加速器）。

![](images/b29a0008d2473c3603b2d19a9b79019eeb3d0b18f7a688db48420c4f173f7854.jpg) *SA: Systolic Array (bfloat16) | BF: BitFusion-based | FB: FlexBlock (Proposed) Fig. 15. Comparisons of the performance and energy consumption between the systolic array (SA), the BitFusion-based BFP accelerator (BF), and the proposed FlexBlock (FB in red). For the performance and energy analysis, we evaluated five CNN benchmarks on ImageNet and a Transformer-base model on WMT14 En-De.*

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

**整体技术架构概述**

FlexBlock 是一款支持**多精度 Block Floating Point (BFP)** 的 DNN 训练加速器，核心设计目标是：在 4-bit / 8-bit / 16-bit 三种‘sign+mantissa’精度模式下（均配 **8-bit shared exponent**），最大化 MAC 阵列利用率，并同时高效处理 **3D 操作**（General Conv、Pointwise Conv、FC）与 **2D 操作**（Depthwise Conv、Weight Gradient 计算 ∂L/∂W、Dilated Conv 等）。

![](images/9291d9b82077847ce8f97591fab9f58bb27f00a3373dbeec2d24e53aae8dea7a.jpg)

---

**一、顶层微架构三大模块**

- **Processing Core（处理核心）**：负责卷积 / GEMM 的定点乘加，仅处理 BFP 中的 **‘sign+mantissa’** 部分
- **2D Reduction Unit（2D 归约单元）**：专用路径，服务 Depthwise 等仅需少量累加的 2D 操作，解决 WU 步骤硬件利用率低下的问题
- **3D Reduction Unit（3D 归约单元）**：默认路径，用于常规 Conv / FC 的部分和累加
- 共享指数在归约单元内的 **INT2FP 算术转换器**中处理；计算完成后经由 **FP2BFP Converter** 重新提取子张量 shared exponent，供下一层使用
- **Dual-Path 归约单元**的开销极小：额外面积 **4.16%**、功耗 **3.89%**

---

**二、层次化硬件结构**

![](images/900f5ef17cfec9bbf032a143d07b20e96c0ef62d2473e8b5ee575422f79ae8aa.jpg) *4bx4b Multiplier 9Processing Element (PE) Fig. 6. Hierarchical structure of hardware modules in Flex-Block. Four PUs form a subcore in FlexBlock.*

自底向上构建：

- **Multiplier**：基础 **4b×4b** 乘法器，支持 signed / unsigned 操作数（由全局符号位控制），负责张量的 W/H 维度
- **Processing Element (PE)**：**9 个乘法器**聚类，适配 2D 卷积窗口映射，负责由输入精度决定的 **C_in** 维度
- **Processing Unit (PU)**：**4 个 PE** 聚类，实现 X 张量上的 **1D Sub-Word Parallelism**，负责由权重精度决定的 **C_out** 维度
- **Subcore**：**4 个 PU** 聚类，实现 W 张量上的 **1D Sub-Word Parallelism**
- **Core**：**6 个 Subcore**，共 **54 个 INT16 MAC**；用于支持 Conv5 / Conv7 等大卷积核的 Subcore 聚类，利用率可达 **93%（Conv5）** 与 **91%（Conv7）**

各层模块与张量维度的映射关系：

| FlexBlock 模块 | 映射的张量维度 |
|---|---|
| Subcore | C_in (Conv1/FC, Conv3), C_out (2D Mode), W/H (Conv5, Conv7) |
| Processing Unit | C_out（由权重精度决定） |
| Processing Element | C_in（由输入精度决定） |
| Multiplier | C_in (Conv1/FC), W/H (Conv3/5/7, 2D Mode) |

---

**三、Bit Reconfigurability（精度重构机制）**

与 Bit Fusion 的 **2D Sub-Word Parallelism**（每个 FU 内所有累加必须完成才能传递）不同，FlexBlock 将其拆分为两个 **1D 并行**：

![](images/91964888095c7ac31e9ddfe6ced3ea32d0aa4eef3ce73a66c4bf4532ba25530c.jpg) *(a) Fusion Unit (FU) in [11] Fig. 3. An illustration of performing a 16b 16b multiplication using two precision-scalable multipliers: (a) a fusion unit (FU) in Bit Fusion [11] and (b) a processing unit (PU) in the proposed FlexBlock. Four PUs in FlexBlock split the 2D sub-word parallelism into two 1D sub-word parallelisms.*

- **X 侧并行（PU 内跨 PE）**：
  - **X16**：每个 4-bit sub-word (x3~x0) 映射到对应 PE
  - **X8**：两个 4-bit sub-word，广播 2 个 input channel
  - **X4**：四个 input channel 各映射一个 PE
- **W 侧并行（Subcore 内跨 PU）**：
  - **W16**：四个 sub-word 分布至四个 PU，psum 经 Selective Adder Tree 全量累加
  - **W8**：PU 分为两簇，各自负责不同 C_out，产生两个独立输出
  - **W4**：四个 C_out 分配至四个 PU，psum 直接 Bypass 加法树
- 支持三种基础格式及**混搭配置**（activation / weight / gradient 各自独立设精度），共 **3³ = 27 种组合**：

| 基础格式 | 配置 | 相对 FB24 的 MAC 数增益 |
|---|---|---|
| FB24 | X16W16 | 1× |
| FB16 | X8W8 | 4× |
| FB12 | X4W4 | 16× |

- 关键优势：精度降低 2× 时，累加次数仅**翻倍**（Bit Fusion 为**四倍**），且多条累加路径可分别计算不同 output channel，维持高利用率

---

**四、Layer Reconfigurability（层类型重构机制）**

- **3D 模式映射**：
  - Conv1 / FC：C_in 维度沿 Subcore → PE → Multiplier 展开
  - Conv3：multiplier 映射 fmap 的 W/H 维度
  - Conv5 / Conv7：多个 Subcore 聚类以覆盖整个卷积核窗口
- **2D 模式映射**：
  - 每个 Subcore 输出对应一个 C_out，经 **2D Reduction Unit 的 4-way Adder Tree** 水平累加
  - DW-Conv5 聚类 3 个 Subcore，DW-Conv7 聚类 6 个
- Block Size 随精度与层类型动态调整（如 FB24 + Conv3 为 **3×3×6**，FB12 + Conv1 为 **1×1×216**），保证所有 multiplier 始终处于激活状态

---

**五、存储与系统级配置**

- 片上 Buffer：**512 KB Input Buffer + 512 KB Weight Buffer + 256 KB Output Buffer**，均采用**双缓冲**隐藏 DRAM 访问延迟
- DRAM：双通道 **DDR4-2133**
- 外围单元：BN Unit (FP/BP)、ReLU Unit、Pool Unit、ReLU_Pool Unit、**WU Unit（权重更新）**、FIFOs、Masking Unit、FP2BFP Converter、Quantization Unit
- 执行模型采用 **Weight Stationary Dataflow**，层执行时间由 max(T_DRAM, (T_W_LOAD + T_COMP)·N_TILE) 决定

---

**六、规模扩展与硬件规格**

以 64 核扩展（65 nm CMOS，333 MHz，FB24 模式下 **54×64** 阵列）与基线对比：

| 指标 | TPU-like SA | BitFusion-based (BF) | **FlexBlock (FB)** |
|---|---|---|---|
| 支持精度 | bfloat16 | FB12/16/24 | **FB12/16/24** |
| 面积 [mm²] | 74.38 | 40.22 | **33.82** |
| 功耗 [W] | 9.19 | 15.47~17.15 | **7.25~8.18** |
| 吞吐 [TFLOPS] | 1.77 | 0.66~2.82 | **0.95~8.78** |
| 能效 [GFLOPS/W] | 192.60 | 42.66~164.43 | **131.03~1,073.30** |

- 面积比 SA 小 **2.2×**、比 BF 小 **1.2×**；功耗分别低 **1.2×** 与 **2.1×**
- 多精度支持本身的开销：相比纯 FB24 核心，面积增加 **30.2%**、功耗增加 **41.8%**，但无延迟开销（关键路径在 FP32 累加器）

---

**七、动态精度控制**

- 基于 **FP2BFP Converter** 中 shift 操作产生的 **underflow 计数**，逐 epoch 监控每类张量（weight / activation / gradient）的精度需求
- 采用 **Hysteresis Controller**（32-bit 计数器 + 2 个比较器，阈值 th_Down / th_Up）决定下一 epoch 的 BFP 精度，避免精度抖动
- 控制器开销仅 **0.5% 功耗 / 0.3% 面积**，可在保持精度的前提下将训练时间再缩短 **13.1%**（相比静态精度选择）

### 1. 层级化一维子字并行架构（1D Sub-Word Parallelism Split）

**核心观点**

FlexBlock 将 Bit Fusion 式的 **2D sub-word parallelism**（同时对 X 和 W 两个操作数做子字切分）拆解为**两个正交的 1D sub-word parallelism**：X 方向的切分在 **PU 内部的四个 PE 之间**完成，W 方向的切分在 **subcore 内部的四个 PU 之间**完成。关键差异在于**部分和的累积时机**——Bit Fusion 要求 FU 内所有累积必须在结果传出前完成，导致精度降低时累积数量**平方级增长**；而 FlexBlock 将累积过程**后置到 selective adder tree**，使累积数量仅**线性增长**，从而在低精度模式下释放出独立的计算路径，逼近理论加速比（8-bit 模式 4×，4-bit 模式 16×）。

---

**一、问题背景：Bit Fusion 的固有瓶颈**

- Bit Fusion 的 **Fusion Unit (FU)** 用 16 个 4b×4b multiplier 组成一个 16b×16b 全精度乘法器，每个 partial product 由专属的 4b×4b multiplier 计算。
- 致命限制：**FU 内所有累积必须在结果传递给下一个 FU 之前完成**。当 X 和 W 各降精度 2× 时，需要累积的 partial sum 数量**翻 4 倍（平方增长）**。
- 后果链条：
  - 低精度模式下 MAC array 被迫执行超长向量点积 → 无法并行计算多个输出通道；
  - **MAC 利用率随精度降低而下降**：MobileNetV1 训练中，X16W16 时利用率为 76.5%，X8W8 时平均下降 **13.8%**，X4W4 时再降 **22.0%**；
  - **WU step 加剧恶化**——weight gradient 计算本质是 depthwise full convolution，累积深度小，只能激活 MAC 单元的一个小子集。
- 附带缺陷：Bit Fusion 需要海量 shifter（64×64 阵列约需 **98k 个 4-bit shifter**），即便 BitBlade 通过聚类将 shifter 减少 93.8%，累积数量问题依然存在。

![](images/91964888095c7ac31e9ddfe6ced3ea32d0aa4eef3ce73a66c4bf4532ba25530c.jpg) *(a) Fusion Unit (FU) in [11] Fig. 3. An illustration of performing a 16b 16b multiplication using two precision-scalable multipliers: (a) a fusion unit (FU) in Bit Fusion [11] and (b) a processing unit (PU) in the proposed FlexBlock. Four PUs in FlexBlock split the 2D sub-word parallelism into two 1D sub-word parallelisms.*

---

**二、硬件层级结构：Multiplier → PE → PU → Subcore**

FlexBlock 采用**四级层级化设计**，每级绑定一个 tensor 维度，实现精细化映射控制：

![](images/900f5ef17cfec9bbf032a143d07b20e96c0ef62d2473e8b5ee575422f79ae8aa.jpg) *4bx4b Multiplier 9Processing Element (PE) Fig. 6. Hierarchical structure of hardware modules in Flex-Block. Four PUs form a subcore in FlexBlock.*

| 层级模块 | 构成 | 承载的 Sub-word Parallelism | 映射的 Tensor 维度 |
|---|---|---|---|
| **Multiplier** | 1 个 4b×4b 乘法器（支持 signed/unsigned，全局 sign bit 切换） | 无 | Cin (Conv1/FC)，或 W/H (Conv3/5/7, 2D Mode) |
| **PE (Processing Element)** | 9 个 multiplier 聚类 | — | **Cin**（由输入精度决定），9 对应 3×3 卷积核 |
| **PU (Processing Unit)** | 4 个 PE | **1D SWP on X**（4-bit sub-word 横跨 4 个 PE） | **Cout**（由权重精度决定） |
| **Subcore** | 4 个 PU | **1D SWP on W**（4-bit sub-word 分配到 4 个 PU） | Cin (Conv1/FC, Conv3)、Cout (2D Mode)、W/H (Conv5/7) |
| **Core** | 6 个 subcore + 双 reduction unit (2D/3D) | — | 共 **54 个 INT16 MAC**（9×6） |

- 六个 subcore 的数量经过精心选择，用于**层类型适配**：
  - Conv5：3 个 subcore 聚类处理一个通道，利用率 (5×5)/(3×9) = **93%**；
  - Conv7：6 个 subcore 全聚类，利用率 (7×7)/(6×9) = **91%**。

---

**三、位分配机制：两个正交的 1D 切分**

**(a) X 方向切分（PU 内部跨 PE）**

- **X16 模式**：每个 fmap element 拆为 4 个 4-bit sub-word（x3, x2, x1, x0），分别映射到 PU 内对应 PE，单周期广播；
- **X8 模式**：每个 element 只有 2 个 sub-word，**相同互连带宽下可广播 2 个输入通道**到 PU；
- **X4 模式**：**4 个输入通道同时广播**，每个通道独占一个 PE——通道级并行度随精度降低而提升。

![](images/b93ace5f04a514311ae56ec09f0e5ed52cba05eff440b173be003e9078b21333.jpg)

**(b) W 方向切分（subcore 内部跨 PU）**

- **W16 模式**：16-bit weight 的 4 个 sub-word（w3/w2/w1/w0）分配到 4 个 PU，所有 PU 输出经 selective adder tree **全量累积**，产出单个 partial sum；
- **W8 模式**：PU 分成两个 cluster（PU2-3 与 PU0-1），**Cout 维度跨 cluster 展开**，selective 4-way adder tree 产出**两个独立 partial sum**（对应 Cout=k 与 k+1）；
- **W4 模式**：**4 个输出通道分派到 4 个 PU**，各 PU 直接产出 partial sum 并**旁路 selective adder tree**。

![](images/4dc3bd77fd17dabeadcd7a4a58f8a412fb21e7d7119b8198236e05c3cd9d37e0.jpg) *Fig. 7. Examples on distributing input data to PEs in each PU depending on the input tensor precision (feature maps or local gradients). Inputs are broadcast to PUs in a single cycle. Fig. 8. Examples on distributing weight data to processing units in a subcore depending on the precision of the weight. In this example, 16-bit inputs are assumed (thus, a single input fmap is shared across multiple weight kernels).*

**(c) 累积路径复用——加速比来源**

- **16-bit 模式**：4 条 PU 路径汇成 1 个输出，无额外并行度；
- **8-bit 模式**：提供 **2 条独立累积路径**，理论上 MAC 吞吐 ×4（相对 16-bit）；
- **4-bit 模式**：提供 **4 条独立累积路径**，理论上 MAC 吞吐 ×16。
- 每条路径可被指派计算**不同的输出通道**，而非强迫所有单元做同一超长向量乘法——这是与 Bit Fusion 的本质区别。

![](images/b9b616b5ce3eb5bb34941ed3222b0dbf9fde9a1cf543625b0ab3dca887629cfc.jpg) *Fig. 4. A motivational example explaining the rate of increase in the number of accumulations at a reduced precision: (a-c) Bit Fusion and (d-f) FlexBlock. Al accumulations within an FU in Bit Fusion need to be completed prior to passing the partial sum. However, only a subset of accumulations are completed at each PU in FlexBlock and the remaining accumulations, if required, happen at the end. In addition, FlexBlock provides a better input data reuse requiring less number of new operands.*

---

**四、混合精度分组**

X 与 W 精度可独立配置，通过 PE/PU 灵活分组实现 **27 种组合**（3³，含 gradient tensor）：

![](images/afd66c2569cd3050e03d446f55d6334c6b0c2c7f6b25fcf39de02b77fd9df4d5.jpg)

- **X4W8 示例**：1×2 个 PE 分组，每个 subcore 处理 **72 个 MAC**；
- **X8W16 示例**：2×4 个 PE 分组，每个 subcore 处理 **18 个 MAC**。

![](images/f23d0d72e3e9d7f454021095b63a040eae0c06b9f072850e3f8b4ce658149933.jpg) *(b) X8W16 Fig. 9. Examples of PE grouping when two operands, i.e., X and W, have different mantissa bit-widths.*

---

**五、量化收益：MAC 利用率与系统级指标**

![](images/d8aaa8a686a5f159c728d348d42f61343f12ab089e2a2a824ef9ed46598ddf62.jpg)

- **利用率对比**（MobileNetV1 训练，batch=16）：FlexBlock 在全精度档位下 MAC 利用率显著高于 Bit Fusion 与 BitBlade，且**精度降低时利用率不降反升**，而两个 baseline 呈单调下降趋势；
- **64-core 规模化对比**（65nm，333 MHz）：

| 指标 | TPU-like SA (bfloat16) | BitFusion-based (BF) | **FlexBlock (FB)** |
|---|---|---|---|
| Array Size | 128×128 | 64×64 (FB24) | 54×64 (FB24) |
| Area [mm²] | 74.38 | 40.22 | **33.82** |
| Power [W] | 9.19 | 15.47~17.15 | **7.25~8.18** |
| Throughput (FB12) [TFLOPS] | 1.77 | 2.82 | **8.78** |
| Efficiency (FB12) [GFLOPS/W] | 192.60 | 164.43 | **1,073.30** |
| Efficiency (FB16) [GFLOPS/W] | — | 107.27 | **438.48** |

- **多精度支持的开销**：相比仅支持 FB24 的同规模 core，面积 +30.2%、功耗 +41.8%，**无延迟开销**（关键路径均在 FP32 accumulator）；扣除开销后，FB16/FB12 的有效加速仍达 **2.8×/7.3×**。

---

**六、在整体架构中的作用与输入输出关系**

- **输入**：FP2BFP converter 预先提取 sub-tensor 的 shared exponent，仅将 **‘sign+mantissa’** 送入 processing core，core 内部全部为**定点 MAC 运算**；
- **输出流**：PE 内整数加法累积 → PU 输出 psum → selective adder tree 按精度模式选择性汇流 → **INT2FP converter** 恢复指数（2^(ew+ex)，见 Eq. 3）→ FP32 accumulator → weight update / batch normalization 单元 → FP2BFP converter 重新分块，供下一层使用；
- **2D 模式的协同设计**：层级化切分必须配合**独立的 2D reduction unit** 才能兑现收益——WU step 的 depthwise full convolution 仅需少量累积，专用 4-way adder tree 让每个 subcore 输出对应一个 Cout 通道，避免 3D reduction unit 成为瓶颈（额外 2D reduction unit 的面积/功耗开销仅 **4.16%/3.89%**）；
- **块大小自适应**：最小 BFP block 由“产出一个 partial sum 所需的 MAC 数”决定——精度越低、卷积核越大，参与一个 block 的输入通道数越少（如 Conv7 的 FB24 模式 block size 仅为 7×7×1），保证**所有 multiplier 在各种精度与层类型下均被填满**；
- **下游联动**：层级化架构产出的高利用率是 **dynamic precision control**（基于 FP2BFP converter 的 underflow 计数 + hysteresis controller）得以奏效的前提——只有硬件在 FB12/FB16 下真正跑满吞吐，运行时降精度才能转化为实际训练加速（相比静态 FB12+BW16/BW24 再提速 **13.1%**，相比 FB16 平均提速 **22.5%** 且无精度损失）。

---

**总结**

层级化 1D 子字并行架构的本质，是把 Bit Fusion 中“乘法器级 2D 切分 + 强制同步累积”重构为“**X 在 PU 内切、W 在 PU 间切、累积按需后置**”的松耦合结构。由此将低精度下的累积增长从平方级降为线性级，把省出的算力转化为多输出通道并行，最终在 bfloat16 等效精度下实现 **1.5~5.3× 训练加速**与 **2.4~7.0× 能效提升**，并为逐 tensor、逐 layer 的动态精度调度提供了硬件底座。

### 2. 层类型可重构的张量维度映射

---

**核心观点**

FlexBlock 的**层类型可重构**（Layer Reconfigurability）机制，本质是通过**层次化硬件结构**（Multiplier → PE → PU → Subcore → Core）与**张量维度解耦映射**，使同一块硬件能根据 DNN 层的类型（Conv1/FC、Conv3、Conv5、Conv7、DW-Conv、2D 算子等）动态地将张量的不同维度（$C_{in}$、$C_{out}$、W/H）绑定到不同层级的计算单元上，从而在**多种卷积核尺寸**和**多种精度模式**下均维持 **90% 以上的 core utilization**。其理论依据在于：6 个 subcore 可以灵活**分簇**（clustering），让被卷积核尺寸 W/H 维度占用的乘法器数量恰好逼近硬件总量。

![](images/170428a1503b86703e89d38add2d31affd80f05e935ec1b8803d38963096c2e5.jpg) *Fig. 10. Examples on how FlexBlock groups subcores depending on the layer type and distributes input/weight tensors for (a) Conv3 and (b) Conv5 layers. There are dedicated input (X) and weight (W) buffers for ‘sign+mantissa’ at each subcore. Note that three subcores are grouped together to handle two input channels for the Conv5 layer, to maximize MAC utilization. Since we have six subcores in FlexBlock, we can map two subsequent input channels to subcore 3 5 for the 8-bit BFP mode (FB16).*

---

**一、硬件层次结构与维度绑定的基本原理**

- **层次化组织**：
  - **Multiplier（4b×4b）**：最小计算单元，接受 signed/unsigned 操作数，由一个 global sign bit 控制。
  - **PE（Processing Element）**：**9 个 multiplier** 聚合成一个 PE，恰好映射 **3×3 卷积核的 W/H 空间维度**——这是为 Conv3 “量身定做”的粒度。
  - **PU（Processing Unit）**：**4 个 PE** 组成，在 PU 内部实现 **X（输入/梯度）张量的 1D sub-word parallelism**，即 16-bit 输入的 4 个 4-bit sub-word（x3, x2, x1, x0）分别映射到 4 个 PE。
  - **Subcore**：**4 个 PU** 组成，跨 PU 实现 **W（权重）张量的 1D sub-word parallelism**（w3, w2, w1, w0 分发到 4 个 PU）。
  - **Core**：**6 个 subcore**，提供 $9 \times 6 = 54$ 个 INT16 MAC（FB24 模式），并可根据层类型进行 subcore 级别的**动态分簇**。

![](images/900f5ef17cfec9bbf032a143d07b20e96c0ef62d2473e8b5ee575422f79ae8aa.jpg) *4bx4b Multiplier 9Processing Element (PE) Fig. 6. Hierarchical structure of hardware modules in Flex-Block. Four PUs form a subcore in FlexBlock.*

- **关键设计哲学**：将 Bit Fusion 的 **2D sub-word parallelism 拆分为两个正交的 1D parallelism**（X 在 PE 间、W 在 PU 间），使得不同 tensor 维度可以被独立编译映射到不同硬件层级。

---

**二、Table II：维度到硬件模块的映射规则（3D 与 2D 模式）**

| FlexBlock Module | Mapped Tensor Dimension |
|---|---|
| **Subcore** | $C_{in}$（Conv1/FC、Conv3），$C_{out}$（2D Mode），W/H（Conv5、Conv7） |
| **Processing Unit (PU)** | $C_{out}$（由权重精度决定映射数量） |
| **Processing Element (PE)** | $C_{in}$（由输入精度决定映射数量） |
| **Multiplier** | $C_{in}$（Conv1/FC），W/H（Conv3、Conv5、Conv7、2D Mode） |

- **映射含义解读**：
  - 对于 **Conv1/FC**：累加维度仅为 $C_{in}$，因此 $C_{in}$ 被横向铺开到 subcore、PE 和 multiplier——Subcore0 负责 $C_{in}=0 \sim 17$（18 通道），Subcore1 负责 $C_{in}=18 \sim 35$，以此类推。
  - 对于 **Conv3**：唯一区别是 multiplier 层的映射对象从 $C_{in}$ 切换为 **fmap 的 W/H 空间维度**，每个 PE 的 9 个 multiplier 恰好处理一个 3×3 kernel window。
  - 对于 **Conv5/Conv7**：kernel 空间维度超出单个 subcore 的 3×3 覆盖范围，必须**跨 subcore 分簇**。
  - 对于 **2D Mode**（DW-Conv、∂L/∂W 计算）：由于**不存在跨通道累加**，$C_{out}$ 被映射到 subcore 层级，输出横向流入专用的 **2D reduction unit**。

---

**三、3D 操作的分簇策略与利用率计算（核心算法流程）**

- **Conv3 层（无需分簇）**：
  - 每个 subcore 独立处理一个输入通道的 3×3 卷积，9 个 multiplier 全部激活。
  - 利用率：$(3 \times 3)/(1 \times 9) = 100\%$。

- **Conv5 层（两簇 × 三 subcore）**：
  - **算法流程**：
    - 将 **6 个 subcore 分成 2 个 cluster**，每个 cluster 含 **3 个 subcore**。
    - 3 个 subcore 并行覆盖 **5×5 = 25 个空间位置**，即 $(5 \times 5)/(3 \times 9) = 0.93$ 的利用率（25 个 INT16 MAC 激活，占 3 个 subcore 总计 27 个 MAC 中的 25 个）。
    - 由于 6 个 subcore 可组成 2 个 cluster，可同时处理**两个后续输入通道**（以 FB16/X8W8 为例，输入通道映射数量随精度翻倍）。

- **Conv7 层（单簇 × 六 subcore）**：
  - 将**全部 6 个 subcore 聚合成一个 cluster**，覆盖 **7×7 = 49 个空间位置**。
  - 利用率：$(7 \times 7)/(6 \times 9) = 0.91$（49 MACs 激活 / 54 MACs 总量）。

- **利用率对比汇总**：

| Layer Type | Cluster 配置 | 激活 MACs / 可用 MACs | Core Utilization |
|---|---|---|---|
| Conv1 / FC | 每 subcore 独立 | — | ~100% |
| Conv3 | 每 subcore 独立 | 9/9 | 100% |
| Conv5 | 2 clusters × 3 subcores | 25/27 | **93%** |
| Conv7 | 1 cluster × 6 subcores | 49/54 | **91%** |
| DW-Conv (2D) | 见下文 | — | 显著高于 prior work |

- **为何选择 6 个 subcore**：这是对 Conv5（3 subcore/簇）与 Conv7（6 subcore/簇）的**最小公倍数权衡**——既能让 Conv5 双通道并行，又能让 Conv7 单簇满配，两种 kernel 尺寸的利用率都被推高至 90% 以上。

---

**四、操作数分发机制（输入与权重的差异化通路）**

![](images/b93ace5f04a514311ae56ec09f0e5ed52cba05eff440b173be003e9078b21333.jpg)

- **输入张量 X 的分发**：
  - 输入以**广播** 方式送达 PU，而非将不同输入通道映射到各 PU。
  - **X16 模式**：4 个 4-bit sub-word 分别送入匹配的 PE（颜色对应）。
  - **X8 模式**：同一互连带宽下可同时广播**两个输入通道**。
  - **X4 模式**：广播**四个输入通道**，每个通道占用 PU 内的一个 PE。

![](images/4dc3bd77fd17dabeadcd7a4a58f8a412fb21e7d7119b8198236e05c3cd9d37e0.jpg) *Fig. 7. Examples on distributing input data to PEs in each PU depending on the input tensor precision (feature maps or local gradients). Inputs are broadcast to PUs in a single cycle. Fig. 8. Examples on distributing weight data to processing units in a subcore depending on the precision of the weight. In this example, 16-bit inputs are assumed (thus, a single input fmap is shared across multiple weight kernels).*

- **权重张量 W 的分发**：
  - **W16 模式**：4-bit sub-word（w3）送入 PU3 并转发至 PU 内所有 PE，其余 sub-word（w2/w1/w0）分发到其余 3 个 PU；所有 PU 输出经 **selective adder tree** 累加。
  - **W8 模式**：PU 分为两簇（PU2-3 与 PU0-1），分别计算 $C_{out}=k$ 与 $C_{out}=k+1$，4-way adder tree 输出两个独立 psum。
  - **W4 模式**：4 个输出通道分配到 4 个 PU，各 PU 输出**旁路** adder tree 直接写入 output buffer。

---

**五、2D 操作（DW-Conv 与权重梯度）的专用映射**

![](images/85b81051debc4e94ace46627175556af3943c4fd7801b8162e798ed284a751fd.jpg)

- **问题背景**：WU step 的权重梯度 $\Delta W_{ck}^l$ 计算涉及 $G_{Y_k}^{l+1}$ 与 $X_c^l$ 之间的 **depthwise full convolution**，其累加链极短（channel-wise 计算），导致 prior work（Bit Fusion/BitBlade）的 MAC 阵列严重**欠利用**（Section IV 分析显示 X4W4 下平均再损失 22.0% 利用率）。
- **解决方案——双路径 reduction unit**：
  - **3D reduction unit**：垂直累加 PU 输出，服务 general conv / pointwise conv / FC（FW 与 BW）。
  - **2D reduction unit**：每 subcore 的输出对应一个 $C_{out}$，**横向**流入 2D reduction unit 的 4-way adder tree。
  - DW-Conv5 用 3 个 subcore 分簇、DW-Conv7 用 6 个 subcore 分簇，逻辑与 3D 模式的大 kernel 扩展一致。
  - 额外硬件开销极小：**面积 +4.16%、功耗 +3.89%**。

---

**六、块尺寸联动：Layer 类型 × 精度模式的最小 block size**

- 分块原则：**一个 FlexBlock core 生成一个 partial sum 所需的 MAC 数量**决定共享 exponent 的最小 block size。
- 精度降低时，映射到 core 的输入通道数**翻倍（FB16）或四倍（FB12）**，同时 kernel 尺寸增大时单 block 的通道数减少，以保持给定精度下的总 MAC 数恒定：

| Layer Type | FB12 Block Size | FB16 Block Size | FB24 Block Size |
|---|---|---|---|
| Conv1 / FC | 1×1×216 | 1×1×108 | 1×1×54 |
| Conv3 | 3×3×24 | 3×3×12 | 3×3×6 |
| Conv5 | 5×5×8 | 5×5×4 | 5×5×2 |
| Conv7 | 7×7×4 | 7×7×2 | 7×7×1 |

- **验证示例**：Conv7 + FB24 的 block = $7 \times 7 \times 1 = 49$，与该模式下 49 个激活 MACs 完全吻合——**block size 设计与利用率最大化是同一枚硬币的两面**。

---

**七、输入输出关系与在整体架构中的作用**

![](images/9291d9b82077847ce8f97591fab9f58bb27f00a3373dbeec2d24e53aae8dea7a.jpg) *Fig. 12. The overall microarchitecture of FlexBlock accelerator. There are two reduction units where each unit is dedicated to either 2D or 3D operation mode Our FlexBlock core is capable of processing various block floating point numbers with 8-bit shared exponents (e.g., FB12, FB16 and FB24). The ‘out\_CORE goes to a weight update or a batch normalization unit. Before storing the data to DRAM, we extract shared exponents of sub-tensors at a FP2BFP converter for computing the next layer.*

- **输入**：FP2BFP converter 输出的 **BFP 子张量**（仅 sign+mantissa，附 8-bit 共享 exponent），按 Table II 规则编译映射到 subcore/PU/PE/multiplier。
- **中间处理**：processing core 仅执行定点 MAC（$\langle \vec{\hat{w}}, \vec{\hat{x}} \rangle$），psum 经 2D 或 3D reduction unit 累加。
- **输出**：partial sum 经 **INT2FP 转换器**处理共享 exponent 后进入 FP32 累加链，最终送至 **weight update / batch normalization unit**；写回 DRAM 前由 FP2BFP converter 按**下一层的精度与层类型**重新提取共享 exponent——形成精度与层类型联合自适应的闭环。
- **在整体中的作用**：
  - 该映射机制是 FlexBlock 实现 **1.5–5.3× 训练加速**与 **2.4–7.0× 能效提升**的硬件基石。
  - 在 64-core 规模化对比中（Table V），FlexBlock 相比 BitFusion-based 架构（BF）面积缩小 1.2×、功耗降低 2.1×，FB12 模式下能效达 **1,073.30 GFLOPS/W**（BF 仅 164.43 GFLOPS/W）。
  - 支撑**动态精度控制**：由于每个 core 都能按层类型重构，dynamic precision control 才能在不损失精度的前提下实现额外 **13.1% 平均加速**。

---

**关键结论**

- 层类型可重构映射的本质是**“kernel 空间维度对 subcore 簇的尺寸匹配”**：3×3 → 单 subcore、5×5 → 3-subcore 簇、7×7 → 6-subcore 簇，利用率分别为 **100% / 93% / 91%**。
- 通过 **X 广播 + W 分发** 的非对称数据通路，以及 **2D/3D 双 reduction unit**，FlexBlock 同时解决了**低精度下的 MAC 欠利用**与**depthwise 操作（WU step）的累加链过短**两大痛点。
- Block size（Table III）与维度映射（Table II）的联合设计保证了**任意精度 × 任意层类型**组合下计算资源的满载运行，是多模式 BFP 训练得以在不牺牲精度前提下获得 5.3× 加速的根本保障。

### 3. 面向深度wise操作的专用2D归约单元（双路径归约）

**核心观点**

FlexBlock 在微架构层面引入**双路径归约单元**，即在默认的 3D 归约单元之外，额外配置一条**专用于 2D 操作的归约通路**。该设计的核心动机是：DNN 训练的 **Weight Update (WU)** 步骤完全由 depthwise 卷积构成，而 depthwise 操作的跨通道累加维度极小，若复用面向 3D 操作的垂直累加路径，会导致 MAC 阵列严重空转。通过让 2D 操作走独立、更浅的归约树，FlexBlock 在几乎可忽略的硬件开销下（**面积 +4.16%、功耗 +3.89%**）换回了 WU 步骤接近满配的核心利用率。

![](images/85b81051debc4e94ace46627175556af3943c4fd7801b8162e798ed284a751fd.jpg) *Fig. 11. An example of mapping 2D operations to FlexBlock subcores. Here, we select a widely used depthwise Conv3 as an example. Note that outputs from each subcore move horizontally to the reduction unit for 2D mode.*

---

**问题背景：Depthwise 操作为何天然低利用率**

- **操作分类**（论文 Table I）：FlexBlock 将支持的算子按累加结构划分为两类：

| 模式 | 支持的操作 |
|---|---|
| **2D 操作** | 计算 ∂L/∂W、Depthwise Conv、Dilated Conv、Up Conv |
| **3D 操作** | General Conv、Pointwise Conv、FC（前向与反向） |

- **WU 步骤的计算结构**：权重梯度 ∂L/∂W 通过 **local gradient (G_Y)** 与 **input fmap (X)** 的逐对、**depthwise full convolution** 得到，即每个 的梯度只由 G_Y[k] 与 X[c] 这一对张量决定，**不存在跨 Cin 的输入通道累加**。

- **对硬件的后果**：
  - 3D 操作中，多个 subcore 的输出需要沿垂直方向在归约单元中被深度累加（对应 16-bit 模式下 4 个 PU 的 psum 汇聚、跨 subcore 的 Cin 累加）；
  - depthwise 操作中，单个输出所需的乘积累加数量很小，3D 归约单元的长累加链大部分时间**处于空载**；
  - 论文实测：在 Bit Fusion / BitBlade 架构上训练 MobileNetV1，从 X16W16 降至 X8W8 时利用率平均下降 **13.8%**，降至 X4W4 再降 **22.0%**，且 WU 步骤因全是 depthwise 操作而**恶化最严重**。

![](images/d8aaa8a686a5f159c728d348d42f61343f12ab089e2a2a824ef9ed46598ddf62.jpg)

---

**实现原理：独立的水平归约路径**

- **处理核心与归约单元的解耦**：FlexBlock 核心内部只负责纯定点 ‘sign+mantissa’ 的乘加（即 Eq. (3) 中的 ⟨ŵ, x̂⟩），**所有跨块、跨精度的部分和累加与 BFP 指数处理都后移到归约单元**。这使核心保持计算密集，归约策略可按操作类型灵活切换。

- **双通路结构**：
  - **3D 归约单元**：保持默认的深度累加路径，配套 **selective adder tree**，根据精度模式（FB24/FB16/FB12）选择将 4 个 PU 的 psum 全部累加（16-bit）、两两累加（8-bit，产生两个独立输出）或完全 bypass（4-bit，psum 直接写入输出 buffer）；
  - **2D 归约单元**：**每个 subcore 各自拥有独立的 4-way adder tree**，PU 输出在 subcore 内水平汇入，不再做跨 subcore 的垂直深累加。

- **输出语义的变化**：在 2D 模式下，**每个 subcore 的输出直接对应一个输出通道 C_out 的部分和**（参照论文 Table II，subcore 在 2D 模式下映射的维度为 Cout）。多个 subcore 并行产出多个输出通道，而非为同一输出做冗长累加。

![](images/9291d9b82077847ce8f97591fab9f58bb27f00a3373dbeec2d24e53aae8dea7a.jpg)

---

**算法流程：以 DW-Conv 映射为例**

- **DW-Conv3 层的映射流程**：
  - 每个 subcore 承担一个输出通道 C_out 的计算；
  - 权重核的空间维度（fmap 的 W/H）被映射到 **PE 内的 9 个 4b×4b 乘法器**（3×3 核，每乘法器对应一个空间位置）；
  - 各 PU 的乘积在 subcore 内汇成 psum，再经 **2D 归约单元的 4-way adder tree** 水平累加，输出该通道的最终部分和；
  - 相比 3D 模式需要把 subcore 输出送入深层垂直加法树，2D 路径**所需的累加次数显著减少**，PE 利用率随之提升。

- **大核的 subcore 聚簇**（与 3D 模式对称的缩放策略）：
  - **DW-Conv5**：3 个 subcore 聚成一簇覆盖 5×5 核，六 subcore 核心可同时处理两个后续输入通道（FB16 模式下利用率为 (5×5)/(3×9) ≈ **0.93**）；
  - **DW-Conv7**：全部 6 个 subcore 聚成一簇，利用率 (7×7)/(6×9) ≈ **0.91**。

- **精度模式下的块尺寸适配**（Table III 节选，depthwise/2D 相关）：块尺寸随精度降低而扩大通道覆盖数，保证任意精度下乘法器全部有活干：

| 层类型 | 格式 | 共享指数的最小块尺寸 |
|---|---|---|
| CONV3 | FB12 | 3×3×24 |
| CONV3 | FB16 | 3×3×12 |
| CONV3 | FB24 | 3×3×6 |

---

**输入输出关系与在整体训练流程中的角色**

- **输入侧**：
  - **WU 步骤**：input tensor = 当前层 local gradient G_Y[l]，weight tensor = 前一层 input fmap X[l−1]（论文 Section V-A 明确这一张量角色互换）；
  - **DW-Conv 层（FW/BW）**：输入为 fmap 或 G_Y，权重为 depthwise kernel。
  - 所有输入均以 **FP2BFP 转换器**预先剥离出 8-bit 共享指数，核心只接收对齐后的定点 ‘sign+mantissa’。

- **输出侧**：
  - 2D 归约单元内的 **INT2FP 算术转换器**负责将定点部分和重新附着共享指数（2^{e_w+e_x} 缩放）；
  - WU 路径的输出流向 **Weight Update Unit** 或 Batch Normalization 单元（论文微架构图中 out_CORE 通路），完成权重更新。

- **在整个训练迭代中的地位**：每次训练迭代包含 **FW / BW / WU** 三个步骤，2D 归约单元在 WU 步骤（以及含 DW-Conv 的层如 MobileNetV1）中承担全部主计算。由于 WU 是每个迭代必不可少的一环，其利用率直接决定训练端到端加速比——这正是论文宣称 1.5–5.3× 训练加速、2.4–7.0× 能效提升中不可或缺的一环。

![](images/b9b616b5ce3eb5bb34941ed3222b0dbf9fde9a1cf543625b0ab3dca887629cfc.jpg) *Fig. 4. A motivational example explaining the rate of increase in the number of accumulations at a reduced precision: (a-c) Bit Fusion and (d-f) FlexBlock. Al accumulations within an FU in Bit Fusion need to be completed prior to passing the partial sum. However, only a subset of accumulations are completed at each PU in FlexBlock and the remaining accumulations, if required, happen at the end. In addition, FlexBlock provides a better input data reuse requiring less number of new operands.*

---

**开销量化与设计权衡**

| 指标 | 数值 | 说明 |
|---|---|---|
| 2D 归约单元面积开销 | **+4.16%** | 相对整个 FlexBlock core |
| 2D 归约单元功耗开销 | **+3.89%** | 相对整个 core 功耗 |
| 闲置代价 | 二者归约单元交替闲置 | 计算 2D 或 3D 操作时另一条通路空闲 |

- **为何开销如此之小**：归约单元本质是加法树 + 指数处理逻辑，相比核心内 36% 面积、41% 功耗占比的处理阵列（论文 Fig. 14 面积/功耗分解），加法树是轻量资源；
- **代价接受逻辑**：用 <5% 的静态开销，换回 WU 步骤（占训练约 1/3 计算量）中 depthwise 操作的高利用率，属于典型的**以小博大的非对称收益**；
- **与层级化 sub-word parallelism 的协同**：2D 归约单元与 **PU 分组（X 方向 1D 子字并行）+ 跨 PU 选择性加法树（W 方向 1D 子字并行）**配合，共同保证低精度模式下（FB16 双累加路径、FB12 四累加路径）多输出并行，逼近 4×（8-bit）/ 16×（4-bit）的理论加速上界。

---

**小结**

- **双路径归约**是 FlexBlock 三个核心贡献之一（**High Core Utilization**）的关键硬件载体；
- 它将“按操作类型分流归约路径”这一思想固化为两条物理通路：**3D 深累加路径**服务 Conv/FC 的跨通道规约，**2D 浅累加路径**服务 depthwise/权重梯度计算的通道级独立输出；
- 配合动态精度控制（基于 FP2BFP 下溢计数的 hysteresis 控制器），FlexBlock 得以在 WU 步骤大胆采用 **FB12**（4-bit）精度而不牺牲训练精度，最终实现相对 BitFusion 基线 **65.3%–79.3%** 的能耗削减与最高 **5.3×** 的训练加速。

### 4. 多模式BFP低精度训练方案（按训练步骤/张量差异化精度）

**核心观点**

FlexBlock 的多模式 BFP 低精度训练方案，本质是将 **DNN 训练的三个计算步骤（FW / BW / WU）与三类张量（activation、weight、gradient）解耦**，允许每个维度独立选择 4/8/16-bit 的 'sign+mantissa' 位宽（共享 8-bit exponent），在保证训练精度的前提下将平均精度压至最低，实现 **1.5–5.3 倍训练加速与 2.4–7.0 倍能效提升**。其核心洞察是：**反向传播中 local gradient 的数值分布最宽、对精度最敏感**，而前向特征图计算与权重梯度计算可安全降至 4-bit（FB12）。

---

**一、BFP 数据格式基础与精度模式定义**

- **BFP 表示原理**：一个 block 内的 N 个数值共享同一个 exponent（取自该 block 中绝对值最大元素），块内元素仅需存储 'sign+mantissa'，乘累加（MAC）可完全在**定点逻辑**上完成，无需对中间结果做对齐（alignment），显著降低硬件成本。
  - 数值表达：`x_i = (-1)^s_i · m_i · 2^e_i`
  - 块表达：`x⃗ = x̂⃗ · 2^e_s`，其中 `e_s = ⌊log₂(max(|x₁|,...,|x_N|))⌋`
  - 内积分解：`⟨w⃗, x⃗⟩ = ⟨ŵ⃗, x̂⃗⟩ · 2^(e_w + e_x)`——共享 exponent 在运算前提取，MAC 阵列仅处理定点乘加。

- **FlexBlock 三种基础格式**（所有模式均使用 **8-bit shared exponent**，动态范围与 FP32/bfloat16 一致）：

| 格式 | X 精度 | W 精度 | MAC 吞吐（相对 FB24） | 备注 |
|---|---|---|---|---|
| **FB12** | 4-bit | 4-bit | 16× | 最激进，需精度补偿 |
| **FB16** | 8-bit | 8-bit | 4× | 等效 BFP 版 bfloat16 |
| **FB24** | 16-bit | 16-bit | 1× | 基准精度 |

- **27 种组合空间**：activation、weight、gradient 三类张量各自可独立从 {4, 8, 16}-bit 中选取，共 **3³ = 27 种 mix-and-match 组合**（如 8-bit input + 4-bit weight + 16-bit gradient），这是 FlexBlock 区别于固定精度 FP8/HFP8 加速器（如 NVIDIA HFP8、Intel NNP-T）的**核心灵活性来源**。

---

**二、按训练步骤差异化精度的原理与算法流程**

![](images/3b1395d803605dc2100b2666ccede25c21eaaff4bb23c3cbcbfb317b37d8ec60.jpg) *Fig. 2. Three important computational steps involved during the training of a convolutional layer. Each box represents activation maps $( X / Y ) ,$ weight parameters (W), or gradients $\overset { \cdot } { ( } G _ { X } / G _ { Y } )$ ). The W/H represents the width/height of a feature map, and $C _ { i } / \dot { C } _ { o }$ represents the number of input/output channels.*

- **三个训练步骤的张量参与关系**：
  - **FW（Forward Pass）**：输入 X ⊗ 权重 W → 输出特征图 Y；涉及 activation + weight。
  - **BW（Backward Pass）**：local gradient G_Y ⊗ 转置翻转的权重 W → local gradient G_X；涉及 **local gradient + weight**。
  - **WU（Weight Update）**：G_Y 与 X 做逐对 depthwise full convolution → 权重梯度 ∂L/∂W；涉及 local gradient + activation。

- **精度敏感度不对称的根因**：
  - **BW 步骤最敏感**：local gradient 的数值分布更宽（尤其在 ImageNet 等复杂数据集上），4-bit mantissa 不足以表达其分布，导致明显的精度损失——这与 Micikevicius 混合精度训练、FlexPoint、HFP8 等先前工作对梯度精度重要性的结论一致。
  - **FW 与 WU 步骤容忍度高**：特征图计算和权重梯度计算可在 4-bit 下进行而不破坏收敛性（Transformer 例外情况见下文）。

- **差异化精度配置流程**（以 **FB12+BW16** 为例）：
  - FW 步骤：activation 与 weight 均用 4-bit mantissa（X4W4）。
  - BW 步骤：local gradient 与 weight 提升至 8-bit（X8W8），保证梯度回传的数值保真度。
  - WU 步骤：回落到 4-bit（X4W4），depthwise full convolution 在低精度下执行。
  - 对 ImageNet 规模任务进一步升级为 **FB12+BW24**（BW 步骤用 16-bit），因 ImageNet 上 local gradient 分布比 CIFAR 更宽。

- **格式命名的输入输出语义**：每个训练步骤中，“输入张量”的角色会切换——WU 步骤中，**输入张量变为第 l 层的 local gradient，权重张量变为第 l−1 层的 input fmap**，FlexBlock 的张量映射机制对此透明处理。

---

**三、硬件层面如何支撑多模式 BFP**

![](images/9291d9b82077847ce8f97591fab9f58bb27f00a3373dbeec2d24e53aae8dea7a.jpg) *Fig. 12. The overall microarchitecture of FlexBlock accelerator. There are two reduction units where each unit is dedicated to either 2D or 3D operation mode Our FlexBlock core is capable of processing various block floating point numbers with 8-bit shared exponents (e.g., FB12, FB16 and FB24). The ‘out\_CORE goes to a weight update or a batch normalization unit. Before storing the data to DRAM, we extract shared exponents of sub-tensors at a FP2BFP converter for computing the next layer.*

- **层次化结构（Multiplier → PE → PU → Subcore）**：
  - 9 个 4b×4b multiplier 组成 1 个 **Processing Element (PE)**，用于高效映射 2D convolution。
  - 4 个 PE 组成 1 个 **Processing Unit (PU)**，在 PU 内实现 X 方向的 **1D sub-word parallelism**（16-bit 输入的 4 个 sub-word x3/x2/x1/x0 分别映射到 4 个 PE）。
  - 4 个 PU 组成 1 个 **subcore**，实现 W 方向的 1D sub-word parallelism；6 个 subcore 构成一个 core（54 个 INT16 MAC）。

- **关键架构创新——将 2D sub-word parallelism 拆分为两层 1D**：
  - 对比 Bit Fusion 的 Fusion Unit（FU）：精度降低 2× 时累加的 partial sum 数量**翻 4 倍**（4 倍精度降级时翻 16 倍），导致 MAC 利用率骤降（X8W8 相比 X16W16 平均降 13.8%，X4W4 再降 22.0%）。
  - FlexBlock 的 PU 分组方案：精度降低 2× 时累加数量仅**翻 2 倍**，且各 accumulation path 可分配到**不同的输出通道**，从机制上保住 MAC 利用率。

- **多精度下的块尺寸（block size）自适应**：共享 exponent 的最小 block 尺寸随精度和层类型动态调整，保证任意配置下 multiplier 全部投入计算：

| 层类型 | FB12 | FB16 | FB24 |
|---|---|---|---|
| Conv1/FC | 1×1×216 | 1×1×108 | 1×1×54 |
| Conv3 | 3×3×24 | 3×3×12 | 3×3×6 |
| Conv5 | 5×5×8 | 5×5×4 | 5×5×2 |
| Conv7 | 7×7×4 | 7×7×2 | 7×7×1 |

- **双路径 reduction unit**：3D 操作走默认的 selective adder tree；2D 操作（depthwise conv、WU 步骤的权重梯度计算）走**专用 2D reduction unit**，以更少的累加次数生成 partial sum——面积/功耗开销仅 4.16% / 3.89%。

- **精度切换的数据通路**：FP2BFP converter 在数据写回 DRAM 前按下一层的精度与层类型重新分块并提取 shared exponent；INT2FP 转换器放置于 reduction unit 中处理 exponent 还原；最终累加由 **FP32 adder** 完成，构成“低精度 MAC + FP32 累加”的安全组合。

- **多精度支持的硬件代价**：相比仅支持 FB24 的同规模 core，面积开销 **30.2%**、功耗开销 **41.8%**、**零时延开销**（critical path 均在 FP32 adder）；扣除该开销后，FB16/FB12 的有效加速仍达 **2.8× / 7.3×**。

---

**四、精度配置的实验验证（输入输出精度关系）**

![](images/742c2359706508f1252b3c14bc6a8356b6fb1184749044ee933ca79c0145dad9.jpg)

- **统一低精度的失败模式**：全程 FB12 训练时，CIFAR 数据集平均掉点 **−6.21%**，ImageNet 掉点 **−11.46%**（ResNet-18 仅 58.49% vs FP32 的 69.95%）；而 FB24/FB16 与 FP32 基线基本持平。

- **差异化精度的修复效果**（关键数据，CIFAR-10 Top-1）：

| 模型 | FP32 | FB12（全步骤） | FB12+BW16 |
|---|---|---|---|
| AlexNet | 87.12 | 85.11 | **86.41** |
| VGG16 | 92.83 | 87.43 | **92.71** |
| ResNet-18 | 93.37 | 90.36 | **93.50** |
| MobileNetV1 | 87.07 | 79.80 | **87.22** |
| DenseNet-121 | 93.37 | 89.42 | **93.01** |

- **精度需求随任务难度升级**：
  - ImageNet + ResNet-18：需 **FB12+BW24** 才能恢复至基线附近；FB12+BW16 仍掉 **10.84%**，ImageNet 上 local gradient 分布更宽是主因。
  - WMT14 + Transformer-base：FB12 的 perplexity（4.26）与 FP32（3.87）差距很小，语言模型对该方案天然友好。

- **训练超参数设置**（全部与 FP32 基线**完全一致**，未做任何针对低精度的调参）：
  - CIFAR：200 epochs，batch size 128，SGD（lr=0.1，momentum=0.9，weight decay=5e-4），cosine annealing。
  - ImageNet：90 epochs，lr 每 30 epochs 衰减 0.1。
  - WMT14：batch size 64，10 epochs，Adam（wd=0.01，lr=0.001，每 epoch 衰减 0.95）。
  - 实现方式：自研 PyTorch 框架替换 conv/linear 层为自定义 BFP 模块，用 Numba/CUDA 做 FP32→BFP 的张量转换。

---

**五、动态精度控制算法**

![](images/703d89496bd45c7667b97a1f5cdc50fbd8a047e636dadd5fa7eb26220a959a51.jpg) *Fig. 16. Illustration of the process performed by the FP2BFP converter. The number of underflows are used to determine the precision of each tensor at runtime.*

![](images/360c8ac7472e73c64a96d02ea1482a902d10b0afbee37cff551222434bf942a3.jpg)

- **动机**：静态地为每个 benchmark、每个步骤手工选精度成本高且难以预知，需要 runtime 自动化机制。

- **算法流程（基于 underflow 计数的 epoch 级反馈控制）**：
  - **信号采集**：统计 FP2BFP converter 中 shift 操作引发的 **underflow 次数**——mantissa 位宽不足时，块内小数值会被移位至下溢，underflow 比例直接反映精度不足程度。
  - **逐 epoch 决策**：累计当前 epoch 每个 tensor 的 underflow 比例，与预设阈值比较，决定下一 epoch 的 BFP 精度。
  - **滞回控制器**：使用双阈值 (`th_Down`, `th_Upper`) 的 **hysteresis controller** 缓慢切换精度，避免精度在相邻 epoch 间频繁震荡；硬件实现仅需一个 32-bit underflow counter 和两个 32-bit comparator，**功耗开销 0.5%（0.57 mW）、面积开销 0.3%（1,614.72 μm²）**。

- **默认阈值与敏感性**（ResNet-18 + ImageNet，其余两类张量固定为 (0.15, 0.35)）：

| 阈值 | 默认 (0.15, 0.35) | (0.35, 0.55) | (0.55, 0.75) |
|---|---|---|---|
| Weight | 67.89 | 67.82 (−0.07) | 67.76 (−0.13) |
| Activation | 67.89 | 67.68 (−0.21) | 67.73 (−0.16) |
| **Local Gradient** | 67.89 | 66.24 (**−1.65**) | 66.01 (**−1.88**) |

  - 结论：weight/activation 的阈值不敏感、可放宽；**local gradient 必须保持低阈值**以保证稳定训练——再次印证了梯度是精度瓶颈。

![](images/3a4cc1beae2902a86bed2a78958a19c8fe08587084681f60be8fe6f02a3cec4c.jpg)

![](images/44e9edef3cba2023b49535afd62ba4296dab9efad71d4ea930896d046b1941d4.jpg)

- **控制效果**：
  - 动态控制下大部分计算停留在 **FB12**，仅 BW 步骤大量回落到 FB16/FB24，不敏感层（如 ResNet-18 的 L15）被压至最低精度。
  - CIFAR 上相比全程 FB16 训练提速 **22.5%** 且零精度损失。
  - ImageNet 上相比 FB12+BW24 静态配置，训练时间仅增 17% 但精度提升 6.7%（对比 FB16：时间 +17% vs +51.8%，精度收益几乎持平）。
  - 相比最优静态配置（FB12+BW16 for CIFAR / FB12+BW24 for ImageNet），动态控制额外获得 **13.1% 平均加速，且无精度损失**。

---

**六、在整体系统中的作用与端到端收益**

![](images/b29a0008d2473c3603b2d19a9b79019eeb3d0b18f7a688db48420c4f173f7854.jpg) *SA: Systolic Array (bfloat16) | BF: BitFusion-based | FB: FlexBlock (Proposed) Fig. 15. Comparisons of the performance and energy consumption between the systolic array (SA), the BitFusion-based BFP accelerator (BF), and the proposed FlexBlock (FB in red). For the performance and energy analysis, we evaluated five CNN benchmarks on ImageNet and a Transformer-base model on WMT14 En-De.*

- **规模化的硬件对标**（64 cores，65nm，333 MHz，54×64 全精度 MAC 阵列）：

| 指标 | TPU-like SA (bfloat16) | BitFusion-based BF | **FlexBlock FB** |
|---|---|---|---|
| 面积 | 74.38 mm² | 40.22 mm² | **33.82 mm²** |
| 功耗 | 9.19 W | 15.47–17.15 W | **7.25–8.18 W** |
| 峰值吞吐 | 1.77 TFLOPS | 0.66–2.82 TFLOPS | **0.95–8.78 TFLOPS** |
| 能效 | 192.6 GFLOPS/W | 42.7–164.4 GFLOPS/W | **131.0–1073.3 GFLOPS/W** |

- **端到端训练收益**（ImageNet 五个 CNN + WMT14 Transformer，batch size 128，weight stationary dataflow，DDR4-2133 双通道）：
  - 相比 BF 加速器：FB24/FB16/FB12 档位下平均降低能耗 **65.3% / 68.2% / 79.3%**，缩短训练时间 **32.0% / 47.5% / 68.4%**。
  - 相比 bfloat16 systolic array：平均降低能耗 **44.3%**、训练时间 **52.7%**。
  - 整体表述为 **1.5–5.3 倍训练加速、2.4–7.0 倍能效提升**。

- **方案的系统性定位**：多模式 BFP + 步骤级差异化精度 + 动态控制三者构成闭环——**硬件提供 27 种精度组合的可配置性**（输入），**精度敏感度分析给出按步骤分配的静态策略**（中间态），**underflow 驱动的动态控制将策略选择自动化并进一步压榨性能**（输出为最低可行精度配置），最终在不牺牲训练精度的约束下，把 precision scalability 的理论加速上界（4×/16×）尽可能转化为实际吞吐与能效收益。

### 5. 基于下溢计数的动态精度控制（迟滞控制器）

**核心观点**

FlexBlock 的动态精度控制机制是一个**纯硬件实现、零软件干预的运行时精度自适应方案**。其核心洞察是：**BFP 转换过程中的下溢事件频率是 mantissa 位宽是否充足的可观测信号**。通过统计每个 epoch 中 FP2BFP converter 移位操作引发的下溢次数，并利用一个轻量级**迟滞控制器**在三个 BFP 档位间缓慢切换，系统得以在保持训练精度的前提下，将各 tensor（activation / weight / local gradient）始终维持在**可用的最低精度**，从而最大化吞吐。

---

**一、问题背景：为什么需要动态精度控制**

- 静态精度配置存在两难：FB12 在多数层可用，但 **BW step（反向传播计算 local gradient）对精度极其敏感**——ImageNet 上 FB12+BW16 相比 FP32 baseline 精度下降 10.84%，必须升至 FB12+BW24 才能恢复精度。
- 不同 DNN、不同层、不同 tensor 的精度需求差异巨大，**无法事先预知**每个训练步骤所需的最低精度（论文原话："it is difficult to know in advance which precision to use for each training step for different benchmarks"）。
- 人工为每层调优精度组合的计算开销远超训练本身，因此需要一个**自动化、低开销的运行时信号**来驱动精度决策。

---

**二、监测信号：FP2BFP 转换器中的下溢计数**

![](images/703d89496bd45c7667b97a1f5cdc50fbd8a047e636dadd5fa7eb26220a959a51.jpg) *Fig. 16. Illustration of the process performed by the FP2BFP converter. The number of underflows are used to determine the precision of each tensor at runtime.*

**FP2BFP converter 的工作机制**（上图）：

- 该单元位于 **weight update unit 之后**，负责为下一层的计算准备输入/梯度与权重 tensor，即把 FP32 中间结果转换为 BFP 表示。
- 处理流程：
  - 按**预定义的 block size**（由 Table III 根据 BFP 格式与层类型确定，如 Conv3 层 FB12 为 3×3×24，FB24 为 3×3×6）对子张量分块；
  - 由 **shared exponent extractor** 提取块内最大幅值的指数 $e_s = \lfloor \log_2(\max(|x_1|,\dots,|x_N|)) \rfloor$；
  - 由 **mantissa aligners** 将块内每个数右移对齐至共享指数，得到纯定点 'sign+mantissa'。

**下溢为何是精度的代理指标**：

- BFP 块内所有元素共享最大值的指数，块内**幅值较小的元素必须右移**以对齐；
- 若 mantissa 位宽不足（如 FB12 仅 4-bit 'sign+mantissa'），右移后有效位落入 LSB 之下而被截断，即产生**下溢**；
- 下溢比率越高，说明当前位宽无法表示块内的数值分布离散度，**mantissa 已经“不够用”**，此时应升精度；反之若几乎没有下溢，说明当前精度有余量，可降档。

---

**三、控制算法：迟滞控制器**

![](images/360c8ac7472e73c64a96d02ea1482a902d10b0afbee37cff551222434bf942a3.jpg)

**控制流程**（对应上图 a 部分）：

- **统计粒度**：以 **epoch** 为周期，累计当前 epoch 内每个 tensor 的下溢次数；
- **决策指标**：计算**下溢比率** = 下溢元素数 / 总元素数；
- **双阈值比较**：比率同时与下阈值 $th_{Down}$ 和上阈值 $th_{Up}$ 比较，产生三种动作：
  - 比率 > $th_{Up}$ → mantissa 不足，**上调精度档位**（FB12 → FB16 → FB24）；
  - 比率 < $th_{Down}$ → 精度冗余，**下调精度档位**；
  - 介于两者之间 → **维持当前精度**（迟滞区间）；
- **迟滞设计的目的**：避免在下溢比率在单一阈值附近波动时产生精度档位的**高频振荡**，保证精度切换“缓慢而稳定”；
- **作用对象**：控制可对 **FW、BW、WU 三个训练步骤分别独立施加**，且每个步骤覆盖该步骤涉及的两个 tensor（例如 BW step 的 FB16 表示 local gradient 与 weight 的 'sign+mantissa' 均为 8-bit）；
- **默认参数**：$th_{Down} = 0.15$，$th_{Up} = 0.35$。

**硬件实现与开销**：

- 极简电路：**1 个 32-bit 下溢计数器 + 2 个 32-bit 比较器**；
- 开销相对整个 FlexBlock core：
  - 功率仅 **0.5% (0.57 mW)**；
  - 面积仅 **0.3% (1,614.72 μm²)**。

---

**四、输入输出关系与在整体架构中的位置**

- **输入**：FP2BFP converter 逐块转换时的移位下溢事件流（每个 tensor 一路独立统计）；
- **输出**：下一 epoch 各 tensor / 各训练步骤的 BFP 档位选择信号（FB12 / FB16 / FB24），用于配置 processing core 的 **sub-word parallelism 模式**与 FP2BFP 的 block size；
- **闭环位置**：该控制器与 FP2BFP converter 构成一个**反馈回路**——converter 既是精度选择的“受害者”（按所选档位分块、移位），又是精度需求的“传感器”（产生下溢计数）；处理核心则按新档位重新组织 X/W 的 PE/PU 分组，使 MAC 吞吐按 4-bit ×4 / 8-bit ×2 的比例伸缩；
- **粒度可调**：论文演示中可对单个训练步骤（如仅 BW）单独启用，也可对 FW/BW/WU 全部启用（Fig. 18(a) 显示全部启用时各步骤的档位分布）。

---

**五、实验验证：效果与参数敏感性**

![](images/3a4cc1beae2902a86bed2a78958a19c8fe08587084681f60be8fe6f02a3cec4c.jpg)

**档位分布观察**（上图 a，ResNet-18）：

- **BW step 是最精度敏感的步骤**，其计算大部分落在 FB16/FB24；FW 与 WU 大部分保持在 FB12；
- 控制器能够识别不敏感层（如 CIFAR-10 上 ResNet-18 的 L15）并对其维持最低精度 FB12。

**性能收益**：

- CIFAR 数据集上：相比全程 FB16，动态控制实现 **22.5% 平均训练加速**且**零精度损失**；
- 相比最优静态配置（CIFAR 用 FB12+BW16、ImageNet 用 FB12+BW24）：平均 **13.1% 额外加速**；
- ImageNet（ResNet-18）上：
  - FB12+BW24 相对 FB16：训练时间 +51.8%，精度 +7.0%；
  - 动态控制相对 FB16：训练时间仅 **+17%**，精度 **+6.7%**——以极小的时间代价换取接近高精度配置的模型质量。

**阈值敏感性测试**（Table VI，ResNet-18 on ImageNet，其余两个 tensor 固定为默认阈值 (0.15, 0.35)）：

| Tensor 类型 | (0.15, 0.35) | (0.35, 0.55) | (0.55, 0.75) |
|---|---|---|---|
| Weight | 67.89 | 67.82 (-0.07) | 67.76 (-0.13) |
| Activation | 67.89 | 67.68 (-0.21) | 67.73 (-0.16) |
| **Local Gradient** | 67.89 | 66.24 (**-1.65**) | 66.01 (**-1.88**) |

**结论性规律**：

- **Weight 与 Activation 的阈值高度不敏感**——即使放宽到 (0.55, 0.75)，精度损失仍 ≤ 0.21%，说明这两个 tensor 的阈值“容易设置”；
- **Local Gradient 的阈值必须保守**——放宽阈值直接造成 1.65%~1.88% 的精度退化，印证了梯度计算对数值精度的极端敏感性（与 FB12+BW16 在 ImageNet 上失败的原因一致）；
- 工程意义：实际部署时仅需针对 gradient tensor 精调阈值，其余 tensor 可直接使用默认值，大幅降低了调参成本。

---

**六、机制评述**

- **优势**：
  - 硬件开销近乎为零（<0.5%），无需软件 profiling 或离线精度搜索；
  - 下溢计数是**物理层直接观测到的数值损失信号**，比损失函数监控、梯度范数等软件侧信号更贴近 BFP 表示的真实误差来源；
  - 迟滞设计保证了档位切换的稳定性，且按 epoch 粒度切换避免了周期内精度抖动带来的训练不稳定。
- **局限**：
  - 决策以 epoch 为粒度，对 epoch 内部精度需求变化（如训练早期 vs 末期）响应较粗；
  - 下溢比率只刻画“表示能力不足”这一种失效模式，无法捕捉 mantissa 充足但块划分不合理等其他误差源；
  - 梯度 tensor 的阈值敏感性意味着该机制对最关键的 tensor 仍需人工校准，并非完全免调参。


---

## 4. 实验方法与实验结果

**实验设置分析**

**基准数据集与模型选择**

- **数据集覆盖**：作者选用了四类数据集，覆盖图像分类与机器翻译两大任务
  - **CIFAR-10 / CIFAR-100**：用于训练 AlexNet、VGG16、ResNet-18、MobileNetV1、DenseNet-121 共五个 CNN 模型
  - **ImageNet**：用于 ResNet-18 大规模图像分类训练
  - **WMT14 En-De**：用于 Transformer-base 机器翻译训练
- **超参数设定**：
  - CIFAR 系列：200 epochs、batch size 128、SGD（初始学习率 0.1、momentum 0.9、weight decay 0.0005）、cosine annealing 学习率调度
  - ImageNet：90 epochs、每 30 epochs 学习率衰减 0.1
  - Transformer：10 epochs、batch size 64、Adam（weight decay 0.01、初始学习率 0.001、衰减率 0.95）
- **公平性设计**：BFP 训练与 FP32 基线使用**完全相同的超参数**，验证 BFP 格式无需额外调参即可直接应用

**精度配置体系**

- **三种基础 BFP 格式**：均使用 **8-bit shared exponent**（与 FP32/bfloat16 动态范围一致）
  - **FB24**（X16W16）：16-bit sign+mantissa
  - **FB16**（X8W8）：8-bit sign+mantissa
  - **FB12**（X4W4）：4-bit sign+mantissa
- **混合精度变体**：支持在 **FW（前向）/ BW（反向）/ WU（权重更新）** 三个训练步骤中使用不同精度，如 **FB12+BW16**（BW 用 FB16，其余用 FB12）
- **软件实现**：基于 PyTorch 自定义 BFP 模块替换 Conv/Linear 层，使用 **Numba + CUDA** 加速 FP32→BFP 数据转换

**硬件评估平台**

- **RTL 实现与综合**：65nm CMOS 工艺、Synopsys Design Compiler（333MHz）、IC Compiler 后端 PnR、PrimePower 结合 SAIF 文件进行功耗分析
- **内存配置**：512KB input buffer + 512KB weight buffer + 256KB output buffer（均 double-buffered）、双通道 DDR4-2133 DRAM、SRAM 访问延迟用 CACTI 估算
- **性能仿真**：自研 cycle-approximate 模拟器（类似 SCALE-Sim），执行时间模型为 `max(T_DRAM, (T_W_LOAD + T_COMP)·N_TILE)`，其中 **T_COMP 来自真实 RTL 仿真**而非解析模型，提高了可信度
- **基线设计**：两个对比基线
  - **SA**：TPU-like systolic array，bfloat16，128×128 阵列
  - **BF**：基于 Bit Fusion 架构的 BFP 训练加速器，64×64 阵列（FB24 模式）

![](images/b29a0008d2473c3603b2d19a9b79019eeb3d0b18f7a688db48420c4f173f7854.jpg) *SA: Systolic Array (bfloat16) | BF: BitFusion-based | FB: FlexBlock (Proposed) Fig. 15. Comparisons of the performance and energy consumption between the systolic array (SA), the BitFusion-based BFP accelerator (BF), and the proposed FlexBlock (FB in red). For the performance and energy analysis, we evaluated five CNN benchmarks on ImageNet and a Transformer-base model on WMT14 En-De.*

---

**实验结果数据分析**

**训练精度结果（Table IV）**

| 模型/数据集 | FP32 | FP16+FP32 | FB24 | FB16 | FB12 | FB12+BW16 |
|---|---|---|---|---|---|---|
| AlexNet (C10) | 87.12 | 86.96 | 87.06 | 86.68 | 85.11 | 86.41 |
| VGG16 (C10) | 92.83 | 92.90 | 92.93 | 92.97 | 87.43 | 92.71 |
| ResNet-18 (C10) | 93.37 | 93.68 | 93.79 | 93.75 | 90.36 | 93.50 |
| MobileNetV1 (C10) | 87.07 | 86.85 | 85.98 | 86.67 | 79.80 | 87.22 |
| DenseNet-121 (C10) | 93.37 | 93.61 | 93.06 | 93.32 | 89.42 | 93.01 |
| ResNet-18 (C100) | 77.26 | 76.88 | 77.45 | 77.23 | 66.69 | 76.45 |
| MobileNetV1 (C100) | 67.40 | 66.80 | 66.95 | 67.60 | 57.11 | 67.06 |
| ResNet-18 (ImageNet) | 69.95 | 69.23 | 69.92 | 61.20 | 58.49 | 68.20 |

- **关键结论一：FB24/FB16 精度基本无损**
  - CIFAR 数据集上，FB24 与 FB16 的 Top-1 accuracy 与 FP32 基线差距均在 **±1%** 以内，部分甚至超过基线（如 VGG16 FB16 为 92.97 vs FP32 的 92.83）
  - Transformer 在 WMT14 上，FB24（4.33）与 FB12（4.26）的 perplexity 均与 FP32（3.87）接近，说明 Transformer 对低精度不敏感
- **关键结论二：FB12 全局低精度会造成精度退化**
  - CIFAR 数据集平均退化 **-6.21%**，ImageNet 退化 **-11.46%**，其中 MobileNetV1 退化最严重（C10 上 -7.27%，因其 depthwise 结构对精度更敏感）
- **关键结论三：BW 步骤是精度瓶颈，混合精度可补救**
  - **FB12+BW16** 在 CIFAR 上几乎完全恢复精度（ResNet-18: 93.50 vs 93.37；MobileNetV1: 87.22 甚至超过 FP32 的 87.07）
  - ImageNet 需要更保守的 **FB12+BW24**（68.20）才能接近基线；FB12+BW16 仅达 59.11%（退化 10.84%），原因是 **local gradient 在 ImageNet 训练中分布更宽**，需要更高 mantissa 位宽

![](images/742c2359706508f1252b3c14bc6a8356b6fb1184749044ee933ca79c0145dad9.jpg)

**硬件架构对比（Table V）**

| 指标 | SA (Systolic) | BF (BitFusion) | FlexBlock |
|---|---|---|---|
| 工艺 | 65nm | 65nm | 65nm |
| 面积 | 74.38 mm² | 40.22 mm² | **33.82 mm²** |
| 功耗 | 9.19 W | 15.47–17.15 W | **7.25–8.18 W** |
| 吞吐 (FB12) | 1.77 TFLOPS | 2.82 TFLOPS | **8.78 TFLOPS** |
| 吞吐 (FB16) | 1.77 TFLOPS | 1.77 TFLOPS | **3.35 TFLOPS** |
| 能效 (FB12) | 192.60 GFLOPS/W | 164.43 GFLOPS/W | **1,073.30 GFLOPS/W** |

- FlexBlock 面积比 SA 小 **2.2×**、比 BF 小 **1.2×**；功耗比 SA 低 **1.2×**、比 BF 低 **2.1×**
- **FB12 模式下吞吐达 8.78 TFLOPS**，是 SA 的 4.96 倍、BF 的 3.11 倍；能效高达 **1,073 GFLOPS/W**，是 SA 的 5.6 倍
- FlexBlock 优势的核心来源：低精度下**多条独立累加路径**避免了 Bit Fusion 式 2D sub-word parallelism 导致的累加链膨胀问题（精度每降 2×，累加数仅翻倍而非四倍）

**端到端训练性能与能耗（Fig. 15）**

- 相比 BF（同等精度对比）：FlexBlock 在 FB24/FB16/FB12 下分别降低能耗 **65.3% / 68.2% / 79.3%**，缩短训练时间 **32.0% / 47.5% / 68.4%**
- 相比 SA（bfloat16 vs FB16）：平均降低能耗 **44.3%**、缩短训练时间 **52.7%**
- 论文宣称的总体加速范围为 **1.5–5.3×**，能效提升 **2.4–7.0×**

---

**消融实验分析**

**消融一：MAC 利用率对比（Fig. 5，动机验证）**

![](images/d8aaa8a686a5f159c728d348d42f61343f12ab089e2a2a824ef9ed46598ddf62.jpg)

- 实验设置：MobileNetV1 训练（batch size 16），对比 Bit Fusion、BitBlade 与 FlexBlock 在 X16W16/X8W8/X4W4 下的 MAC 利用率
- 结果：
  - X16W16 下三者利用率均约 **76.5%**（说明 16-bit 时架构差异不大，主要受层形状限制）
  - 精度降至 X8W8：Bit Fusion/BitBlade 利用率平均再降 **13.8%**
  - 降至 X4W4：再降 **22.0%**
- **结论**：FlexBlock 的分层 sub-word parallelism 设计在高精度（FB24/FB16）下基本消除了利用率损失，验证了 PU 分组 + 多累加路径的必要性

**消融二：多精度支持的硬件开销**

- 对比对象：仅支持 FB24（X16W16）的 FlexBlock 精简版 vs 完整多精度版（均 54 MACs/core）
- 结果：

| 项目 | FB24-only 版 | 多精度版 | 开销 |
|---|---|---|---|
| 面积 | 406,002 μm² | 528,426 μm² | **+30.2%** |
| 功耗 | 85.1 mW | 120.7 mW | **+41.8%** |
| 延迟 | — | — | **无开销**（关键路径均在 FP32 adder） |

- 由于无多精度时可在同面积放置 **1.3× 更多 MAC**，FB16/FB12 的有效加速比从理论 4×/16× 修正为实际 **2.8× / 7.3×**（平均）
- **结论**：多精度收益（2.8–7.3× 加速）远大于 30–42% 的面积/功耗开销，设计权衡合理

**消融三：2D Reduction Unit 的必要性**

- 针对 depthwise conv（WU 步骤的核心操作，需少量累加）设置独立 reduction unit
- 开销：面积 **+4.16%**、功耗 **+3.89%**（相对整个 core，代价极小）
- 收益：解决 WU 步骤中 2D 操作核心利用率低的问题——这正是 BF/Bit Fusion 架构在训练场景下的固有缺陷

**消融四：动态精度控制的阈值敏感性（Table VI）**

| 张量类型 | (0.15, 0.35) | (0.35, 0.55) | (0.55, 0.75) |
|---|---|---|---|
| Weight | 67.89 | 67.82 (-0.07) | 67.76 (-0.13) |
| Activation | 67.89 | 67.68 (-0.21) | 67.73 (-0.16) |
| Local Gradient | 67.89 | **66.24 (-1.65)** | **66.01 (-1.88)** |

- **Weight 与 Activation 的阈值不敏感**：即使放宽至 (0.55, 0.75)，精度损失仍 <0.25%，说明这两个张量可激进地用低阈值（FB12）
- **Local Gradient 高度敏感**：阈值放宽即造成 1.65–1.88% 的精度退化，证实 **BW 步骤是精度关键路径**，必须保守设置（与 Table IV 的 FB12+BW24 结论相互印证）
- 控制器本身开销极小：仅 **0.5% 功耗（0.57mW）/ 0.3% 面积（1,614.72 μm²）**，由 32-bit underflow counter + 两个 32-bit comparator 构成

![](images/3a4cc1beae2902a86bed2a78958a19c8fe08587084681f60be8fe6f02a3cec4c.jpg)

**动态精度控制的收益量化（Fig. 18）**

- CIFAR 上相比固定 FB16：训练提速 **22.5%**（平均），无精度损失
- ImageNet 上相比 FB12+BW24：训练时间仅增加 17%（vs 静态方案的 51.8%），同时精度反超 FB16 方案 **6.7%**
- 相比最优静态精度选择（FB12+BW16/FB12+BW24）：再提速 **13.1%**（平均）
- **层间异构性**得到利用：不敏感层（如 ResNet-18 的 L15）保持 FB12，敏感层自动升至 FB16/FB24

---

**综合评价**

**优势**

- **实验链路完整**：从算法精度（PyTorch 仿真）→ RTL 综合 → PnR → SAIF 功耗分析 → cycle-approximate 性能仿真，且 T_COMP 基于真实 RTL 仿真，数据可信度高
- **消融覆盖全面**：多精度开销、2D reduction unit、阈值敏感性、MAC 利用率均有独立量化分析，每个设计决策都有数据支撑
- **核心洞察明确**：识别出 BW 步骤（local gradient）是精度瓶颈、WU 步骤（depthwise conv）是利用率瓶颈，硬件设计精准对应这两个痛点

**局限**

- ImageNet 上仅评估 ResNet-18，MobileNetV1/DenseNet 等复杂模型未在大规模数据集验证
- 4-bit 训练（FB12）依赖大 block 内共享 exponent，对梯度分布异常尖锐的模型（如 GAN、大语言模型）泛化性未验证
- 与 FP8/HFP8 类商用方案（如 NVIDIA H100、IBM HFP8）缺乏直接对比，仅对比了 SA 与 BF 两个学术基线

---

