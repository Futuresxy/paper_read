# FlexBlock: A Flexible DNN Training Accelerator With Multi-Mode Block Floating Point Support 图表详解

### 73545598204f95242ad691050060c9bfb00084cf9d61939c1adf36e7eddf7107.jpg

![73545598204f95242ad691050060c9bfb00084cf9d61939c1adf36e7eddf7107.jpg](images/73545598204f95242ad691050060c9bfb00084cf9d61939c1adf36e7eddf7107.jpg)

- **图像定位与概述**
    - 该图为 **Fig. 1(a)**，展示了 **FlexBlock** 加速器的**核心架构概览**（Proposed Training HW）
    - 图像位于论文引言部分，用于直观呈现 FlexBlock 的**三大设计特征**

- **架构组成解析**
    
    | 模块/组件 | 功能描述 | 技术细节 |
    |-----------|----------|----------|
    | **输入接口** | 接收块浮点格式的数据 | 分为 **Sign & Mantissa**（符号+尾数）和 **Shared Exp**（共享指数）两路 |
    | **Precision-Scalable MAC Array** | 核心计算单元 | 支持 **INT4, INT8, INT16** 三种精度模式的可重构乘加阵列 |
    | **INT2FP Unit** | 输出转换模块 | 将定点运算结果转换回浮点格式 |

- **关键设计理念解读**
    
    - **Block Floating Point（块浮点）机制**
        - 采用**共享指数**策略：一组数值共享同一个指数位，仅保留各自的符号和尾数
        - **优势**：将昂贵的浮点运算转化为廉价的**定点乘累加（MAC）操作**，显著降低硬件复杂度和功耗
    
    - **Multiple BFP Support（多模式BFP支持）**
        - **精度灵活性**：同一硬件可配置为 **FB12（4-bit）、FB16（8-bit）、FB24（16-bit）** 三种基本格式
        - **张量级异构性**：Activation、Weight、Gradient 三类张量可**独立设置不同精度**，实现 $3^3 = 27$ 种组合可能
        - **理论性能增益**：相比16-bit模式，8-bit模式下MAC处理能力提升**最高4倍**，4-bit模式下提升**最高16倍**
    
    - **High Energy-efficiency（高能效特性）**
        - 通过**精度缩放**减少数据移动量和计算复杂度
        - 实验表明：相比其他训练加速器，能效提升 **2.4× – 7.0×**

- **在论文论证中的作用**
    - 作为**概念性架构图**，为后续详细微架构设计（Fig. 12）提供高层抽象
    - 与 Fig. 1(b)（DNN训练三步骤）和 Fig. 1(c)（性能提升对比）共同构成 FlexBlock 的**完整价值主张**
    - 强调 FlexBlock 不仅支持 BFP，更关键的是支持**多精度动态切换**，这是区别于 prior work（如 FlexPoint、Hybrid BFP）的核心差异

- **技术实现关联**
    - 图中的 **Precision-Scalable MAC Array** 对应论文第 V 节详细阐述的**层次化设计**（PE → PU → Subcore）
    - **Shared Exp** 的处理对应第 VI 节的 **Reduction Units** 和 **FP2BFP Converter** 模块
    - **INT2FP Unit** 对应 arithmetic converters，负责在定点累加后恢复浮点表示

### e852c3dfbaac1389375152cc6f8725a4fc1730490abd28403baac33f509a82c0.jpg

![e852c3dfbaac1389375152cc6f8725a4fc1730490abd28403baac33f509a82c0.jpg](images/e852c3dfbaac1389375152cc6f8725a4fc1730490abd28403baac33f509a82c0.jpg)

- **图像核心内容解析**
  - 该图为 **DNN Training Steps**（DNN训练步骤）示意图，展示了深度神经网络单次训练迭代中的**三个核心计算阶段**及其数据流关系。
  - 图中采用**颜色编码箭头**区分不同阶段的计算流向：
    - **黑色箭头 (FW)**: 代表 **Forward Pass**（前向传播），数据从Input输入，经Layer 1至Layer k逐层前向计算，最终产生Output和Loss。
    - **红色箭头 (BW)**: 代表 **Backward Pass**（反向传播），梯度从Loss端反向回传至各层，用于计算局部梯度（Local Gradients）。
    - **蓝色箭头 (WU)**: 代表 **Weight Update**（权重更新），基于计算的梯度对各层权重参数进行更新（ΔW_in, ΔW₁, ..., ΔWₖ）。

- **架构细节与数据流**
  - **网络层级结构**: 输入层(Input) → 隐藏层(Layer 1, ..., Layer k) → 输出层(Output) → 损失函数(Loss)
  - **权重更新位置**: 每层（包括输入层连接权重）均有对应的权重更新量（ΔW标注），体现**逐层参数优化**机制。

- **与FlexBlock设计的关联性**
  - 此图是FlexBlock加速器设计动机的关键依据：**不同训练步骤对数值精度的敏感度不同**。
  - FlexBlock的核心创新在于允许对**激活(Activation)、权重(Weight)、梯度(Gradient)** 张量分别配置不同的BFP（Block Floating Point）精度模式：
    - **FW步骤**: 可使用较低精度（如FB12/FB16）加速特征图计算
    - **BW步骤**: 通常需要较高精度（如FB24）以保证梯度计算准确性
    - **WU步骤**: 可恢复低精度进行权重梯度计算
  - 这种**异构精度配置**策略是FlexBlock实现**1.5×-5.3×性能提升**且不牺牲训练精度的理论基础。

- **技术意义总结**
  - 该图直观阐释了DNN训练的**计算范式**：前向计算→反向传播→参数更新的三阶段循环。
  - 为FlexBlock的**动态精度控制（Dynamic Precision Control）**机制提供了算法层面的支撑——通过监测各步骤的下溢（underflow）情况，自动调整每层的BFP精度模式。

### 17db113c7b262594691692a8af8ce3cdd2acc3eef68709841d61ab95ad4fc751.jpg

![17db113c7b262594691692a8af8ce3cdd2acc3eef68709841d61ab95ad4fc751.jpg](images/17db113c7b262594691692a8af8ce3cdd2acc3eef68709841d61ab95ad4fc751.jpg)

- **图表类型与主题**
  - 该图为**分组柱状图**（Grouped Bar Chart），标题为 **"Relative Runtime vs. 16-bit Mode"**（相对于16位模式的运行时间）
  - 核心目的：展示 **FlexBlock** 加速器在不同 DNN 训练步骤中，采用不同 BFP 精度模式时的**性能提升效果**

- **坐标轴定义**
  - **X 轴**：DNN 训练的三个计算步骤及其对应的精度模式
    - **FW (Forward Pass, 前向传播)**：采用 **4-bit Mode** (FB12)
    - **BW (Backward Pass, 反向传播)**：采用 **8-bit Mode** (FB16)
    - **WU (Weight Update, 权重更新)**：采用 **4-bit Mode** (FB12)
  - **Y 轴**：**Relative Runtime**（相对运行时间），以 16-bit 模式 (FB24) 为基准（归一化为 1.0），数值越低表示加速比越高

- **数据系列说明**
  | 柱子颜色/样式 | 含义 | 代表内容 |
  |---|---|---|
  | 白色/空心柱 | **Theoretical Limit** | 理论极限（理想情况下完全利用硬件资源时的最小运行时间） |
  | 深灰色实心柱 | **Prior Work** | 先前工作（如 Bit Fusion、BitBlade 等现有精度可扩展架构）的实际运行时间 |
  | 黑色实心柱 | **Proposed** | 提出的 **FlexBlock** 架构的实际运行时间 |

- **关键数据解读**
  
  **各训练步骤的相对运行时间对比：**
  
  | 训练步骤 | 精度模式 | 理论极限 | Prior Work | FlexBlock (Proposed) | FlexBlock vs Prior Work 提升 |
  |---|---|---|---|---|---|
  | **FW** | 4-bit | ~0.06 | ~0.15 | ~**0.10** | **~33% 更快** |
  | **BW** | 8-bit | ~0.24 | ~0.35 | ~**0.27** | **~23% 更快** |
  | **WU** | 4-bit | ~0.06 | ~0.15 | ~**0.10** | **~33% 更快** |

- **核心洞察与分析**
  
  **① 性能差距的根源——MAC 利用率问题**
  - **理论极限 vs 实际表现**：在 4-bit 模式下，理论运行时间应约为 16-bit 的 **1/4**（即 0.25），但图中显示理论极限仅约 **0.06**，这表明该比较基准可能基于**绝对时钟周期数**或**归一化吞吐量**而非简单的时间比例
  - **Prior Work 的瓶颈**：先前的精度可扩展 MAC 阵列（如 Bit Fusion）在降低精度时，由于**累加路径数量激增**（如 16-bit→4-bit 时累加量增加 16 倍），导致 **MAC 单元严重欠利用**（under-utilization），实际加速远低于理论值
  - **FlexBlock 的突破**：通过**层次化子字并行设计**（Hierarchical Sub-word Parallelism），将 2D 并行拆分为两个 1D 并行（Processing Unit 级别处理输入子字并行，Subcore 级别处理权重子字并行），有效减少了低精度模式下的累加瓶颈
  
  **② 多模式 BFP 支持的价值**
  - 图表明确展示了 FlexBlock 可针对不同训练步骤**独立配置最优精度**：
    - **FW 和 WU 步骤**：可采用激进的 **4-bit (FB12)** 模式，因为这些步骤对数值精度的敏感度较低
    - **BW 步骤**：需保持较高的 **8-bit (FB16)** 甚至 16-bit 模式，因为**局部梯度（Local Gradient）计算对精度极为敏感**（如表 IV 所示，FB12 在 BW 步骤会导致显著精度下降）
  - 这种**异构精度配置**（Mixed-Precision Training）是 FlexBlock 实现 **1.5×–5.3× 整体训练加速**的关键策略之一

  **③ 与论文其他部分的呼应**
  - 该图对应论文 **Figure 1(c)**，直观验证了 Abstract 中提到的核心贡献："**maximizes the core utilization at various precision levels**"
  - 图中标注的 **①** 和 **②** 分别对应论文 Section V 提出的两大技术手段：
    - ① **高 MAC 利用率设计**：多路径累加（Multiple Accumulation Paths）+ 专用 Depthwise Reduction Unit
    - ② **多模式 BFP 支持**：允许 Activation、Weight、Gradient 张量采用不同精度（共 3³=27 种组合可能）

- **工程意义总结**
  - **量化收益**：相比 Prior Work，FlexBlock 在 4-bit 模式下将运行时间从 **0.15 降低至 0.10**（额外获得 **50% 加速**），在 8-bit 模式下从 **0.35 降低至 0.27**（额外获得 **30% 加速**）
  - **能效影响**：结合 Table V 数据，这种运行时间的减少直接转化为**能量效率的提升**（Energy Efficiency 达到 **1073.30 GFLOPS/W** in FB12 模式，远超 Bit Fusion 的 164.43 GFLOPS/W）
  - **设计权衡**：虽然 FlexBlock 仍未达到理论极限（存在约 **40-67% 的 gap**），但这主要受限于 DNN 层类型的多样性（如 Depthwise Convolution 的计算密度低）以及内存带宽瓶颈，而非架构本身的缺陷

### Fig. 2. Three important computational steps involved during the training of a convolutional layer. Each box represents activation maps $( X / Y ) ,$ weight parameters (W), or gradients $\overset { \cdot } { ( } G _ { X } / G _ { Y } )$ ). The W/H represents the width/height of a feature map, and $C _ { i } / \dot { C } _ { o }$ represents the number of input/output channels.

![3b1395d803605dc2100b2666ccede25c21eaaff4bb23c3cbcbfb317b37d8ec60.jpg](images/3b1395d803605dc2100b2666ccede25c21eaaff4bb23c3cbcbfb317b37d8ec60.jpg)

- **图像基本信息**
    - 图像编号：Fig. 2
    - 标题：Three important computational steps involved during the training of a convolutional layer（卷积层训练中的三个关键计算步骤）
    - 核心内容：展示 DNN 训练过程中单个卷积层涉及的**前向传播(FW)、反向传播(BW)、权重更新(WU)**三个阶段的张量流动与计算模式

- **步骤①：前向传播 (Forward Pass, FW)**
    - 目标：计算损失 (compute the loss)
    - 输入张量：
      | 张量符号 | 含义 | 维度 |
      |---------|------|------|
      | X | 输入激活图 | $C_i \times H \times W$ ($C_i$ 为输入通道数) |
      | $W_0 \sim W_{C_o-1}$ | 卷积核权重 | 共 $C_o$ 个滤波器 |
    - 输出张量：Y（输出激活图，维度 $C_o \times H \times W$）
    - 计算本质：**标准 3D 卷积运算**，输入特征图与多组权重核进行跨通道的滑动窗口乘累加

- **步骤②：反向传播 (Backward Pass, BW)**
    - 目标：获取局部梯度 (get local gradients)
    - 输入张量：
      | 张量符号 | 含义 | 维度 |
      |---------|------|------|
      | $G_Y$ | 输出端局部梯度 $\partial\mathcal{L}/\partial Y$ | $C_o \times H \times W$ |
      | $W^{flip}$ | 翻转/转置后的权重 | 对应 FW 步骤权重的空间反转 |
    - 输出张量：$G_X$（输入端局部梯度 $\partial\mathcal{L}/\partial X$，维度 $C_i \times H \times W$）
    - 计算本质：**转置卷积 (Transposed Convolution)**，将梯度通过权重矩阵的转置反向传播至输入侧

- **步骤③：权重更新 (Weight Update, WU)**
    - 目标：计算权重梯度 (get weight gradients)
    - 计算模式：**逐深度全卷积 (Depthwise Full Convolution)**
    - 核心操作：对于每一对输入通道 $c$ 和输出通道 $k$：
      - 输入：$G_Y[k]$（第 $k$ 个输出通道的梯度）与 $X[c]$（第 $c$ 个输入通道的激活）
      - 输出：$\Delta W_{ck}$（对应权重位置的梯度）
    - 关键特性：该步骤属于 **2D 运算**（通道间独立计算），与 FW/BW 的 3D 运算形成对比

- **架构设计启示（结合 FlexBlock 论文背景）**
    - **运算类型分化**：FW 和 BW 属于 **3D 运算**（涉及跨通道累加），而 WU 本质上是 **2D 运算**（Depthwise 特性），这直接催生了 FlexBlock 中 **双路归约单元 (Dual Reduction Units)** 的设计需求
    - **精度敏感度差异**：图中显示 BW 步骤处理的是梯度张量 $G_Y$，其对数值精度的要求通常高于 FW 和 WU 步骤，这支持了 FlexBlock **混合精度训练策略**（如 FB12+BW16 模式）的合理性
    - **数据复用模式**：三个步骤共享相同的权重张量 W（或其变体 $W^{flip}$），但数据流方向和累加维度不同，要求加速器具备灵活的数据通路重构能力

### (a) Fusion Unit (FU) in [11] Fig. 3. An illustration of performing a 16b 16b multiplication using two precision-scalable multipliers: (a) a fusion unit (FU) in Bit Fusion [11] and (b) a processing unit (PU) in the proposed FlexBlock. Four PUs in FlexBlock split the 2D sub-word parallelism into two 1D sub-word parallelisms.

![91964888095c7ac31e9ddfe6ced3ea32d0aa4eef3ce73a66c4bf4532ba25530c.jpg](images/91964888095c7ac31e9ddfe6ced3ea32d0aa4eef3ce73a66c4bf4532ba25530c.jpg)

- **图像主题**: 图3展示了执行16位乘法的两种精度可扩展乘法器架构对比：(a) **Bit Fusion [11]** 的 **Fusion Unit (FU)** 采用**2D子字并行**；(b) **FlexBlock** 的 **Processing Unit (PU)** 采用**1D子字并行**（将2D并行拆分为两个1D并行）。

- **左上角：16b×16b乘法原理**
    - 将16位操作数 **W** 和 **X** 各自分解为4个4位子字：`W = {w3, w2, w1, w0}`，`X = {x3, x2, x1, x0}`
    - 产生 **16个部分积**（partial products），用不同颜色编码标识：
        - 黄色：`w0×x0`
        - 绿色：`w1×x0`, `w0×x1`
        - 蓝色：`w2×x0`, `w1×x1`, `w0×x2`
        - 粉色/红色：更高位的组合
    - 所有部分积经移位后累加，最终生成 **32位结果 (W·X)**

- **(a) Fusion Unit (FU) in Bit Fusion [11] — 2D并行架构**
    - **并行维度**: 同时在 **W** 和 **X** 两个维度上实现子字并行（**2D Parallelism on W & X**）
    - **结构组织**: 采用 **2×2 阵列** 形式
        - 上半部分处理 `{w3:w2}` 与 `{x1:x0}` 的组合
        - 下半部分处理 `{w1:w0}` 与 `{x3:x2}` 的组合
    - **关键组件**:
        - **4b×4b Mult**（白色方框内带×）：基础乘法单元
        - **4b Shifter**（双线框）：用于部分积的对齐移位
        - **(Shared) 4b Shifter**（灰色）：共享移位器
    - **累加特点**: **所有部分积必须在一个FU内部完全累加**后才能传递到下一级，导致低精度模式下**累加次数激增**（精度每降低2倍，累加次数增加4倍）

- **(b) Processing Unit (PU) in FlexBlock — 1D并行架构**
    - **并行维度**: 仅在 **W** 维度上实现子字并行（**1D Parallelism on W**），X维度的并行通过PE层级实现
    - **结构组织**: 采用 **1×4 线性阵列** 形式，包含4个并列处理通路，分别对应 `w3`, `w2`, `w1`, `w0`
    - **关键组件**:
        - 每个 PU 包含 **4b×4b 乘法器** 和 **4b 移位器**（部分移位器可共享）
        - 底部配备 **Selective Adder w.r.t. Precision**（精度相关的选择性加法器）
    - **核心创新 — 分层累加机制**:
    
    | 精度模式 | 累加策略 | 输出数量 | 硬件利用 |
    |:---:|:---:|:---:|:---:|
    | **16-bit (FB24)** | 4个PU的部分积**全部累加** | 1个输出 | 完全利用 |
    | **8-bit (FB16)** | 每**2个PU**的部分积分别累加 | **2个输出** | 利用率保持 |
    | **4-bit (FB12)** | 各PU**直接旁路**加法树，不累加 | **4个输出** | 利用率最大化 |

- **架构对比与优势分析**
    - **Bit Fusion (2D) 的局限**: 当精度从16-bit降至8-bit或4-bit时，由于需要在单个FU内完成更多部分积的累加，导致 **MAC单元利用率显著下降**（论文数据显示：MobileNetV1训练时，X8W8模式下利用率下降13.8%，X4W4模式下再降22.0%）
    - **FlexBlock (1D) 的优势**:
        - 将2D并行拆分为**两个独立的1D并行**（PU内部实现X维度的1D并行，多个PU之间实现W维度的1D并行）
        - 低精度模式下，通过**选择性加法器**将不同PU的计算结果导向不同输出通道，而非强制累加
        - **精度减半时，有效输出数量翻倍**，理论吞吐量可提升**4倍（8-bit）或16倍（4-bit）**
        - 特别适合 **DNN训练中的权重更新(WU)步骤**，该步骤涉及大量depthwise操作（2D操作），传统架构难以充分利用MAC资源

- **设计意义**: 这种**层次化子字并行**设计解决了精度可扩展加速器在低精度模式下的**硬件欠利用问题**，使得FlexBlock能够在不同精度级别下均保持高核心利用率，从而在实际DNN训练中实现接近理论值的性能加速（**1.5×–5.3×**）。

### Fig. 4. A motivational example explaining the rate of increase in the number of accumulations at a reduced precision: (a-c) Bit Fusion and (d-f) FlexBlock. Al accumulations within an FU in Bit Fusion need to be completed prior to passing the partial sum. However, only a subset of accumulations are completed at each PU in FlexBlock and the remaining accumulations, if required, happen at the end. In addition, FlexBlock provides a better input data reuse requiring less number of new operands.

![b9b616b5ce3eb5bb34941ed3222b0dbf9fde9a1cf543625b0ab3dca887629cfc.jpg](images/b9b616b5ce3eb5bb34941ed3222b0dbf9fde9a1cf543625b0ab3dca887629cfc.jpg)

- **图表核心主题**: 对比 **Bit Fusion** 与 **FlexBlock** 架构在降低数值精度时，**部分和（psum）累加速度**及 **MAC 单元利用率**的差异。
- **对比维度**: 分为三组精度模式进行横向与纵向对比：
    - **(a) vs (d)**: 16-bit 输入/权重 (X16W16)
    - **(b) vs (e)**: 8-bit 输入/权重 (X8W8)
    - **(c) vs (f)**: 4-bit 输入/权重 (X4W4)

- **架构特征**: 采用 **2D Sub-word Parallelism**（二维子字并行）。一个 **Fusion Unit (FU)** 内部的所有乘法器必须协同工作以生成一个输出。
- **累加负担呈指数级增长**:
    | 精度模式 | 需累加的 Psum 数量 | 增长倍数 |
    | :--- | :--- | :--- |
    | **X16W16** | 2 | 基准 (1x) |
    | **X8W8** | 8 | **4倍** |
    | **X4W4** | 32 | **16倍** |
- **关键瓶颈 (Under-utilization)**:
    - 在 **(c) X4W4 模式**下，由于一个 FU 必须内部完成所有 32 个 psum 的累加才能输出，若输入通道数 ($C_{in}$) 不足（例如只有 16 而非 32），将导致 **50% 的 MAC 硬件空闲**（图中红色标注 "Becomes idle"）。
    - 这种强制性的全累加要求严重限制了低精度模式下的并行输出能力。

- **架构特征**: 采用 **Hierarchical 1D Sub-word Parallelism**（分层一维子字并行）。引入 **Processing Unit (PU)** 概念，将 2D 并行拆分为两个 1D 并行维度。
- **累加负担线性/温和增长**:
    | 精度模式 | 需累加的 Psum 数量 | 增长倍数 | 输出通道数 |
    | :--- | :--- | :--- | :--- |
    | **X16W16** | 2 | 基准 (1x) | 1 个输出 |
    | **X8W8** | 4 | **2倍** | 2 个独立输出 |
    | **X4W4** | 8 | **4倍** | 4 个独立输出 |
- **核心优势**:
    - **选择性累加 (Selective Accumulation)**: 如图 (e) 和 (f) 所示，在低精度模式下，每个 **PU** 可以独立产生一个部分输出（$Out_0, Out_1...$），无需等待所有 psum 累加完毕。
    - **最大化硬件利用率**: 通过将累加路径分散，FlexBlock 在精度降低时，不是增加单个输出的累加深度，而是**增加了并行的输出通道数量**。这避免了 Bit Fusion 中因数据不足导致的硬件空闲问题。
    - **数据复用效率提升**: FlexBlock 的数据流设计使得输入数据 $X$ 可以被更好地复用，减少了对新操作数的获取需求（"better input data reuse requiring less number of new operands"）。

- **设计哲学差异**: 
    - **Bit Fusion** 是“深度优先”的累加，低精度导致计算图变深，延迟增加且易造成流水线气泡。
    - **FlexBlock** 是“宽度优先”的分配，低精度转化为更多的并行输出，完美契合 DNN 训练中高吞吐量的需求。
- **对训练加速的意义**: 在 DNN 训练的 **Weight Update (WU)** 步骤中涉及大量 **Depthwise (2D) 操作**，这些操作通常累加需求较少。FlexBlock 的架构能确保在这些场景下 MAC 阵列依然保持高利用率，从而实现论文中提到的 **1.5x ~ 5.3x 的训练速度提升**。

### d8aaa8a686a5f159c728d348d42f61343f12ab089e2a2a824ef9ed46598ddf62.jpg

![d8aaa8a686a5f159c728d348d42f61343f12ab089e2a2a824ef9ed46598ddf62.jpg](images/d8aaa8a686a5f159c728d348d42f61343f12ab089e2a2a824ef9ed46598ddf62.jpg)

- **图表类型与核心主题**
  - 该图为**柱状图 (Bar Chart)**，展示了在DNN训练过程中，不同精度配置下三种架构的**MAC利用率 (MAC Utilization)** 对比。
  - 核心论点：**验证了现有精度可扩展架构（Bit Fusion, BitBlade）在低精度模式下存在严重的硬件利用率不足问题，而提出的FlexBlock架构通过层次化设计有效解决了这一瓶颈。**

- **坐标轴定义**
  - **X轴（横轴）**：按训练步骤和精度模式分组
    - 训练步骤：**FW** (Forward Pass, 前向传播)、**BW** (Backward Pass, 反向传播)、**WU** (Weight Update, 权重更新)、**Avg** (平均利用率)
    - 精度配置（三组）：
      - **X16W16**：16位 'sign+mantissa' 输入 + 16位权重
      - **X8W8**：8位输入 + 8位权重  
      - **X4W4**：4位输入 + 4位权重
  - **Y轴（纵轴）**：**MAC Utilization (%)**，范围0%-100%

- **数据系列（图例）**
  | 颜色 | 架构名称 | 设计特点 |
  |------|----------|----------|
  | 浅灰色 | **Bit Fusion** | 基于2D子字并行的精度可扩展MAC阵列 |
  | 深灰色 | **BitBlade** | Bit Fusion的改进版，优化移位器数量 |
  | 黑色 | **FlexBlock (Proposed)** | 本文提出的层次化1D子字并行架构 |

- **关键数据分析**

  **1. 高精度模式 (X16W16) - 基准线**
  - 三种架构表现接近，利用率均在**85%-95%**区间
  - FlexBlock在FW/BW/WU各步骤均达到或超过90%
  - 结论：在高精度下，传统架构与FlexBlock效率相当

  **2. 中等精度模式 (X8W8) - 分化开始显现**
  - **Bit Fusion & BitBlade**：利用率出现明显下降
    - FW步骤：降至约**70%-75%**
    - BW步骤：维持在**70%**左右
    - **WU步骤（关键瓶颈）**：急剧下降至**45%-55%**
  - **FlexBlock**：保持高利用率
    - 各步骤均维持在**85%-90%**
    - 特别是WU步骤仍保持**~85%**的高效利用

  **3. 低精度模式 (X4W4) - 差异最大化**
  - **Bit Fusion & BitBlade**：严重欠利用
    - FW步骤：降至**45%-50%**
    - BW步骤：**40%-45%**
    - **WU步骤（最严重）**：暴跌至**25%-30%**（Bit Fusion仅约27%）
    - 平均利用率：仅**38%-42%**
  - **FlexBlock**：显著优势
    - FW步骤：**~72%**
    - BW步骤：**~73%**
    - WU步骤：**~58%**（虽有所下降但仍远超竞品）
    - 平均利用率：**~68%**，约为竞品的**1.6-1.8倍**

- **深度技术解读**

  **根本原因分析**（结合论文Section IV）：
  - **Bit Fusion/BitBlade的局限性**：采用**2D子字并行**（同时拆分输入和权重的位宽），当精度降低时，单个Fusion Unit (FU)内部需要累积的**部分和 (partial sums) 数量呈平方级增长**（精度减半→累积数翻4倍）。这导致大量MAC单元处于等待状态，造成硬件空闲。
  - **FlexBlock的创新**：将2D并行分解为**两个层次的1D子字并行**：
    - 第一层（PU内部）：实现输入张量X的子字并行
    - 第二层（Subcore内部）：实现权重张量W的子字并行
    - 关键优势：**选择性加法树 (Selective Adder Tree)** 可根据精度模式动态配置输出路径，避免不必要的内部累积等待。

  **WU步骤的特殊性**：
  - 权重更新主要涉及**Depthwise Convolution（深度卷积）**，属于**2D操作**（见表格I），其计算特点是需要累积的部分和数量本来就少。
  - 在低精度下，传统架构的MAC阵列规模相对过大，而计算需求小，导致**极端的利用率崩溃**（X4W4时仅25-30%）。
  - FlexBlock通过**专用的2D Reduction Unit**（见图11和Section V-C）专门优化此类操作，有效缓解了这一问题。

- **量化性能提升总结**

  | 精度模式 | Bit Fusion/BitBlade 平均利用率 | FlexBlock 平均利用率 | **FlexBlock 相对提升** |
  |----------|-------------------------------|---------------------|------------------------|
  | X16W16 | ~90% | ~93% | **+3%** (基准相当) |
  | X8W8 | ~65% | ~86% | **+32%** (显著优势) |
  | X4W4 | ~40% | ~68% | **+70%** (决定性优势) |

- **对论文贡献的支撑作用**
  - 该图直接验证了**Contribution 2 (High Core Utilization)** 的有效性。
  - 为**Section VII-B** 中声称的"训练速度提升1.5-5.3倍"提供了**硬件层面的微观解释**——并非仅仅依靠降低精度减少计算量，更重要的是**通过架构创新避免了低精度下的硬件空转**。
  - 解释了为何FlexBlock能在**4-bit算术 (FB12)** 下进行DNN训练（见Section VII-A），而传统架构在如此低的精度下会因效率过低而失去实用价值。

### 4bx4b Multiplier 9Processing Element (PE) Fig. 6. Hierarchical structure of hardware modules in Flex-Block. Four PUs form a subcore in FlexBlock.

![900f5ef17cfec9bbf032a143d07b20e96c0ef62d2473e8b5ee575422f79ae8aa.jpg](images/900f5ef17cfec9bbf032a143d07b20e96c0ef62d2473e8b5ee575422f79ae8aa.jpg)

- **图像核心内容解析**
    - 该图为 **FlexBlock 加速器的 Subcore（子核）内部层次化架构示意图**，展示了从底层乘法器到顶层处理单元的组织方式。
    - 整体遵循 **Multiplier (乘法器) → PE (Processing Element, 处理单元) → PU (Processing Unit, 处理单元组) → Subcore (子核)** 的四级层级结构。
    - 图中明确标注了 **X16W16（16位输入×16位权重）** 模式下的数据流与计算路径。

- **详细模块分解**

| 层级 | 组件名称 | 数量/配置 | 功能描述 |
| :--- | :--- | :--- | :--- |
| **Subcore** | 顶层计算单元 | 包含 4 个 PU (PU0-PU3) | 负责整合所有 PU 的输出，通过底部的 **Selective Adder Tree（选择加法树）** 完成最终累加。 |
| **PU (Processing Unit)** | 处理单元组 | 每个 Subcore 含 4 个 PU | 实现**权重维度（W）的子字并行**。在 16 位模式下，每个 PU 负责处理权重的一个 4 位切片（如 PU3 处理 w3）。 |
| **PE (Processing Element)** | 处理单元 | 每个 PU 含 9 个 PE | 实现**输入维度（X）的子字并行**。每个 PE 内部集成一个 **4b×4b 乘法器**、加法器和移位器。 |
| **Multiplier** | 基础乘法器 | 4b×4b 定点乘法器 | 执行最底层的定点乘法运算，是整个架构的计算原子。 |

- **数据流动与并行机制分析**
    - **输入数据 X 的广播（Broadcast）**：16 位输入 $X_{0:8}$ 被拆分为四个 4 位子字 $\{x3, x2, x1, x0\}$，并**同时广播**到所有 PU 中的对应 PE。这实现了**空间上的输入复用**。
    - **权重数据 W 的分配（Distribute）**：16 位权重的四个 4 位子字 $\{w3, w2, w1, w0\}$ 被**分别分配**给 PU3, PU2, PU1, PU0。这种分配策略使得不同的 PU 可以并行计算权重不同部分的贡献。
    - **累加路径（Accumulation Path）**：
        - 在 **X16W16 模式**下，由于需要完整的 16 位精度，四个 PU 计算出的部分和（Partial Sums）必须全部送入 **Selective Adder Tree** 进行求和，最终输出 $\sum_{i=0}^{3} W_i \cdot X_i$。
        - 在**低精度模式（如 X8W8 或 X4W4）**下，该加法树可以**选择性旁路或拆分**，允许 PU 直接输出多个独立结果，从而成倍提升吞吐量（这是 FlexBlock 相比 Bit Fusion 等前作的核心优势）。

- **架构设计亮点总结**
    - **2D 并行解耦**：将传统的 2D 子字并行（如 Bit Fusion）**解耦为两个独立的 1D 并行**（X 维度在 PU 内部，W 维度跨 PU）。这种设计使得在降低精度时，不需要强制所有计算单元等待长向量累加完成，而是可以让不同的 PU 产生独立的输出通道，极大提升了**硬件利用率（MAC Utilization）**。
    - **精度可伸缩性（Precision Scalability）**：通过配置 Selective Adder Tree 的行为以及 PE/PU 的分组方式，同一套硬件可以在 **FB12 (4-bit), FB16 (8-bit), FB24 (16-bit)** 三种 BFP 模式间无缝切换。
    - **深度可分离卷积优化**：该层次化结构特别针对 DNN 训练中的 **Weight Update (WU) 步骤**进行了优化。WU 步骤涉及大量的 Depthwise（深度可分离）操作，计算密度低。FlexBlock 通过这种灵活的 PU 分组，能够更高效地映射这类 2D 操作，避免了传统脉动阵列在此类操作上的资源浪费。

### b93ace5f04a514311ae56ec09f0e5ed52cba05eff440b173be003e9078b21333.jpg

![b93ace5f04a514311ae56ec09f0e5ed52cba05eff440b173be003e9078b21333.jpg](images/b93ace5f04a514311ae56ec09f0e5ed52cba05eff440b173be003e9078b21333.jpg)

这张图（Figure 7）详细展示了 **FlexBlock 加速器中输入张量（特征图或局部梯度）如何根据不同精度模式分发到处理单元（PEs）和处理单元组（PUs）**，体现了 FlexBlock 的核心设计理念——**输入维度的子字并行机制（Sub-Word Parallelism on X）**。

以下是针对该图的深度分析：

*   **图表整体结构**
    *   该图采用**上下对比**的方式，展示了三种不同的输入精度模式：**X16（16位）、X8（8位）和X4（4位）**。
    *   每种模式均包含左右两部分：
        *   **左侧（Input fmap）**：展示输入特征图的数据布局及**子字（sub-word）分割**方式。
        *   **右侧（PU0~3）**：展示硬件架构中 **4个PU（Processing Unit）** 及其内部 **PE（Processing Element）** 的数据映射关系。

*   **模式一：X16 模式（高精度/基准模式）**
    *   **数据表示**：每个输入元素为 **16位**，被拆分为 **4个4位的子字**（x[15:12], x[11:8], x[7:4], x[3:0]）。
    *   **映射策略**：
        *   图中显示了一个包含 X0-X7 的 2x4 数据块。
        *   这 **4个子字** 被分别映射到 **同一个PU内部的不同PE**（如 PU0 内的 PE3, PE2, PE1, PE0）。
        *   **关键特性**：**单个输入通道（$C_{in}$）被广播到所有的 PUs**。这意味着所有 PU 都在处理同一组输入数据的不同部分或进行冗余计算准备。
    *   **硬件利用率**：在此模式下，FlexBlock 的行为类似于传统的定点阵列，主要用于处理对数值范围要求较高的层。

*   **模式二：X8 模式（中等精度/性能加速模式）**
    *   **数据表示**：每个输入元素为 **8位**，仅包含 **2个4位的子字**（x[7:4], x[3:0]）。
    *   **映射策略**：
        *   由于每个元素占用的硬件资源（PE数量）减半，FlexBlock 利用空闲的 PE 资源来处理**更多的输入通道**。
        *   **关键特性**：**两个输入通道（$C_{in}$）被同时广播到 PUs**。如图所示，X0-X4 属于一个通道，X5-X8 可能属于另一个通道（或同一通道的不同空间位置），它们被并行送入 PU 阵列。
    *   **性能增益**：相比 X16 模式，**理论吞吐量翻倍**（2x），因为在一个时钟周期内处理了双倍的输入通道数据。

*   **模式三：X4 模式（低精度/极限加速模式）**
    *   **数据表示**：每个输入元素仅为 **4位**（即单个子字 x[3:0]）。
    *   **映射策略**：
        *   硬件资源进一步释放，每个 PE 可以独立处理一个完整的 4位操作数。
        *   **关键特性**：**四个输入通道（$C_{in}$）被同时广播到 PUs**。图中左侧显示了多层堆叠的特征图切片，代表四个独立的通道数据流。
    *   **性能增益**：相比 X16 模式，**理论吞吐量提升至 4 倍（4x）**。这是 FlexBlock 实现“灵活块浮点”加速的核心物理基础。

*   **设计洞察与总结**
    *   **空间可扩展性（Spatial Scalability）**：该图完美诠释了 FlexBlock 如何通过**静态的硬件分组**实现**动态的精度切换**。不需要改变时钟频率或复杂的时序控制，仅需改变数据分发逻辑（Dataflow）即可。
    *   **利用率最大化**：这种设计的精髓在于，当操作数位宽降低时，不会让乘法器闲置，而是将其重新分配给其他维度（此处为输入通道维度 $C_{in}$）的计算任务，从而解决了传统可变精度加速器在低精度下**硬件利用率低下**的痛点（如文中提到的 Bit Fusion 的局限性）。
    *   **与权重并行（W维度）的协同**：结合论文 Fig. 8 的描述，X 维度的这种并行机制与 W 维度（权重）的子字并行相结合，构成了 FlexBlock 的 **2D 子字并行架构**，使其能够在 FB12 (4-bit), FB16 (8-bit), FB24 (16-bit) 等多种 BFP 格式下均保持极高的 MAC 阵列利用率。

| 精度模式 | 输入位宽 | 子字数量 | 广播通道数 ($C_{in}$) | 相对 X16 吞吐量倍数 | 硬件映射特点 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **X16** | 16-bit | 4 个 (4-bit each) | 1 个 | 1x (基准) | 4个子字填满1个PU内的4个PE |
| **X8** | 8-bit | 2 个 (4-bit each) | 2 个 | **2x** | 2个通道复用1个PU内的4个PE |
| **X4** | 4-bit | 1 个 (4-bit) | 4 个 | **4x** | 4个通道分别占用1个PE |

综上所述，这张图是理解 FlexBlock **如何通过数据重配置（Data Reconfiguration）在不增加硬件面积的前提下换取性能提升**的关键示意图，它直观地证明了**低精度训练在硬件层面的效率优势来源**。

### Fig. 7. Examples on distributing input data to PEs in each PU depending on the input tensor precision (feature maps or local gradients). Inputs are broadcast to PUs in a single cycle. Fig. 8. Examples on distributing weight data to processing units in a subcore depending on the precision of the weight. In this example, 16-bit inputs are assumed (thus, a single input fmap is shared across multiple weight kernels).

![4dc3bd77fd17dabeadcd7a4a58f8a412fb21e7d7119b8198236e05c3cd9d37e0.jpg](images/4dc3bd77fd17dabeadcd7a4a58f8a412fb21e7d7119b8198236e05c3cd9d37e0.jpg)

- **图像核心内容解析**
    - 该图为 **Fig. 8**，展示 FlexBlock 架构中**权重张量（Weight Tensor）** 根据不同精度模式（W16/W8/W4）分配至**处理单元（Processing Unit, PU）** 的具体机制。
    - 图示基于**16位输入（X16）** 的假设前提，即单个输入特征图（fmap）在多个权重内核间共享广播。

- **三种精度模式的详细数据流分析**

| 精度模式 | 输出通道分配 | 数据位宽分布 | PU分组策略 | 累加路径 |
|---------|------------|------------|-----------|---------|
| **W16 mode** | 单通道 (C<sub>out</sub>) | w[15:12], w[11:8], w[7:4], w[3:0] 四个子字 | 4个PU协同处理**同一个**输出通道 | 全部4路PU输出经**选择加法树**（Selective Adder Tree）累加后输出 |
| **W8 mode** | 双通道 (C<sub>out</sub>) | 每8位分为高低两部分 | PU3+PU2 处理第1通道；PU1+PU0 处理第2通道 | 分组内2路累加，产生**两个独立部分和** |
| **W4 mode** | 四通道 (C<sub>out</sub>) | 直接使用4位表示 | 每个PU独立负责**一个**完整输出通道 | **无跨PU累加**，各PU直接旁路（Bypass）输出至Subcore |

- **关键设计机制与技术要点**

    - **子字并行ism（Sub-word Parallelism on W）**：
        - 在 **W16** 模式下，16位权重被拆解为4个4位片段（w[15:12]至w[3:0]），分别映射至PU3至PU0，实现**空间上的子字级并行计算**。
        - 每个 PU 接收 **36b (9×4b)** 数据，对应其内部9个乘法器的位宽配置。

    - **精度-吞吐量线性关系**：
        - **精度减半，吞吐量翻倍**的理论基础在此具象化：W16→W8 时，可同时计算的输出通道数从1增至2；W8→W4 时，进一步增至4。
        - 这种**层次化子字并行**（Hierarchical Sub-word Parallelism）避免了传统 Bit Fusion 架构中低精度下**累加路径拥塞**导致的硬件利用率下降问题（如论文Fig. 5所述）。

    - **选择性累加网络（Selective Addition Network）**：
        - 图中 ⊕ 符号代表可配置的加法树结构。根据当前精度模式动态调整：
            - **高精度（W16）**：激活全加法树深度，完成多PU部分和归约；
            - **中/低精度（W8/W4）**：截断或旁路加法树，减少关键路径延迟，直接输出多通道结果。

    - **与输入数据流的协同**：
        - 结合 Fig. 7（输入分配），当输入为X16时，单通道输入广播至所有PU；而此处权重的**通道维度（C<sub>out</sub>）** 被切片分配至不同PU。
        - 这种**输入广播+权重分发**的数据流实现了**权重静止（Weight Stationary）** 或类似的数据复用策略，最大化片上缓存命中率。

- **对训练加速的物理意义**
    - **前向传播（FW）/反向传播（BW）**：对于标准卷积层（3D操作），此机制允许在保持相同硬件资源（54个INT16 MAC/Core）的前提下，通过降低权重精度**成倍提升有效算力**（W8模式下等效108个MAC，W4模式下等效216个MAC）。
    - **权重更新（WU）**：虽然WU主要涉及2D操作（Depthwise Conv），但该分配逻辑同样适用，确保在梯度计算时能高效利用PU阵列处理多通道梯度。

### afd66c2569cd3050e03d446f55d6334c6b0c2c7f6b25fcf39de02b77fd9df4d5.jpg

![afd66c2569cd3050e03d446f55d6334c6b0c2c7f6b25fcf39de02b77fd9df4d5.jpg](images/afd66c2569cd3050e03d446f55d6334c6b0c2c7f6b25fcf39de02b77fd9df4d5.jpg)

- **图像基本信息**
  - **图号与标题**：Fig. 9 (a) X4W8
  - **所属章节**：V. DESIGN OF A FLEXBLOCK CORE → B. Bit Reconfigurability
  - **核心主题**：FlexBlock 加速器在**混合精度模式（X4W8）**下的 PE 分组策略示意

- **架构层次结构解析**

  | 层级 | 组件名称 | 数量/配置 | 功能说明 |
  |------|----------|-----------|----------|
  | 最外层 | **Subcore 'k'** | 1个 | 基本计算单元，包含多个PU |
  | 中间层 | **Processing Unit (PU)** | 4个 (PU0-PU3) | 实现权重维度的子字并行 |
  | 内层 | **Processing Element (PE)** | 每个PU含4个 (PE0-PE3) | 实现输入维度的子字并行 |
  | 最内层 | **4b×4b Multiplier** | 每个PE含9个 | 基础计算单元 |

- **X4W8 混合精度模式的硬件映射机制**
  - **精度定义**：**X4** = 输入张量（激活值/梯度）使用 **4-bit** 符号+尾数；**W8** = 权重张量使用 **8-bit** 符号+尾数
  - **PE 分组策略**：采用 **1×2 PE 分组** 方式（即横向每2个PE为一组）
    - 由于权重为8位（需拆分为2个4位子字），而输入为4位（仅需1个子字）
    - 每个 PU 内部将 **PE0+PE1**、**PE2+PE3** 分别配对，共同处理一个8位权重的不同子字部分
  - **计算能力**：该配置下每个 Subcore 可处理 **72次 MAC 运算**（9 multipliers × 4 PEs × 2 组有效输出路径）

- **数据并行性与利用率优化**
  - **输入维度（X方向）子字并行**：4位输入直接映射到单个PE的4b×4b乘法器，**无需跨PE拆分**
  - **权重维度（W方向）子字并行**：8位权重需拆分为高4位和低4位，分别由**配对的两个PE**处理
  - **关键设计优势**：通过这种**非对称分组**（1×2而非2×2或4×4），避免了低精度模式下MAC单元的空闲，确保在混合精度场景下仍能维持**高硬件利用率**

- **与其它精度模式的对比定位**
  - 此图与 Fig. 7（纯输入精度缩放）、Fig. 8（纯权重精度缩放）、Fig. 9(b)（X8W16模式）形成完整系列
  - **独特性**：展示了 FlexBlock 支持 **27种可能的精度组合**（3³）中的一种典型非对称配置，体现了架构的**灵活可重构性**

- **工程实现意义**
  - **面积效率**：相比固定精度设计，这种动态分组允许同一硬件阵列适配多种精度需求，避免为每种精度单独设计专用电路
  - **能效提升**：在训练过程中，对精度不敏感的层（如某些卷积层的前向传播）可采用X4W8等低精度模式，**显著降低功耗**（论文报告最高可达**7倍能效提升**）
  - **训练精度保障**：虽然前向传播（FW）和权重更新（WU）可使用低精度，但反向传播（BW）的梯度计算仍可保持较高精度（如FB16/FB24），**平衡性能与收敛性**

### (b) X8W16 Fig. 9. Examples of PE grouping when two operands, i.e., X and W, have different mantissa bit-widths.

![f23d0d72e3e9d7f454021095b63a040eae0c06b9f072850e3f8b4ce658149933.jpg](images/f23d0d72e3e9d7f454021095b63a040eae0c06b9f072850e3f8b4ce658149933.jpg)

这张图展示了 **FlexBlock 加速器在 X8W16 混合精度模式下的 PE 分组策略**，具体分析如下：


| 层级 | 组件 | 数量 (本图) | 功能 |
|------|------|-------------|------|
| **Subcore** | 子核心 | 1 个 (标记为 'k') | 基本计算单元 |
| **PU** | Processing Unit | 4 个 (PU0-PU3) | 处理单元 |
| **PE** | Processing Element | 每个 PU 含 4 个 (PE0-PE3) | 处理元素 |
| **Multiplier** | 4b×4b 乘法器 | 每个 PE 含 1 个 | 基础计算单元 |


- **操作数位宽配置**：
  - **X (Input/Activation)**: **8-bit** mantissa (由 2 个 4-bit 子字组成)
  - **W (Weight)**: **16-bit** mantissa (由 4 个 4-bit 子字组成)

- **关键设计目标**: 当两个操作数位宽不匹配时，如何**重组 PE 资源以最大化利用率**


图中**虚线框**揭示了具体的硬件映射方式：

- **水平分组 (跨 PU 合并)**:
  - **左侧组**: **PU3 + PU2** (共 8 个 PE)
  - **右侧组**: **PU1 + PU0** (共 8 个 PE)

- **物理含义**:
  - 由于 X 仅需 **8-bit** (2 个子字)，而 W 需要 **16-bit** (4 个子字)
  - 原 16-bit 模式下需 4 个 PU 处理 W 的 4 个子字
  - 现 X 减半至 8-bit，**每个 PU 内只需 2 个 PE** 即可处理完整的 X
  - 因此可将 **2 个 PU 配对**，每组独立处理**不同的输出通道**

- **计算密度**:
  - 该配置下每个 Subcore 可处理 **18 个 MAC 操作**
  - 相比纯 FB24 (X16W16) 模式的 54 个 MAC，此处为**混合精度优化**


| 模式 | X 位宽 | W 位宽 | PU 分组方式 | MACs/Subcore |
|------|--------|--------|-------------|---------------|
| **FB24** | 16-bit | 16-bit | 4 PU 合并为 1 组 | 54 |
| **FB16** | 8-bit | 8-bit | 2 PU 合并为 1 组 (共 2 组) | 36 |
| **X8W16 (本图)** | 8-bit | 16-bit | 2 PU 合并为 1 组 (共 2 组) | 18 |
| **FB12** | 4-bit | 4-bit | 每个 PU 独立工作 (共 4 组) | 54 |


- **灵活性**: 支持非对称位宽组合 (**3³ = 27 种可能组合**)，允许为 Activation、Weight、Gradient 分别设置不同精度
- **利用率优化**: 避免"因 X 位宽较窄而导致 PE 空闲"的问题，通过**动态重组 PU** 保持高吞吐量
- **能效提升**: 在保持训练精度的前提下，对不敏感的张量使用更低精度，**减少内存访问和计算开销**

此图核心展示了 FlexBlock 的 **Spatial Scalability (空间可扩展性)** 特性——通过**静态重组**硬件计算单元而非简单的时钟门控或时序复用，实现了真正的**精度自适应加速**。

### Fig. 10. Examples on how FlexBlock groups subcores depending on the layer type and distributes input/weight tensors for (a) Conv3 and (b) Conv5 layers. There are dedicated input (X) and weight (W) buffers for ‘sign+mantissa’ at each subcore. Note that three subcores are grouped together to handle two input channels for the Conv5 layer, to maximize MAC utilization. Since we have six subcores in FlexBlock, we can map two subsequent input channels to subcore 3 5 for the 8-bit BFP mode (FB16).

![170428a1503b86703e89d38add2d31affd80f05e935ec1b8803d38963096c2e5.jpg](images/170428a1503b86703e89d38add2d31affd80f05e935ec1b8803d38963096c2e5.jpg)

- **图像核心内容解析**：该图为 **Fig. 10**，详细展示了 **FlexBlock** 加速器针对不同卷积层类型（**Conv3** 与 **Conv5**）的 **子核（Subcore）分组策略** 及 **张量数据映射方式**。演示基于 **FB16 (X8W8)** 精度模式（即 8位符号+尾数）。

- **架构层级映射关系**：
    - **处理核心 (Processing Core)**：由 **6个 Subcore** 组成。
    - **处理单元 (Processing Unit, PU)**：每个 Subcore 内部包含 **4个 PU** (PU0-PU3)，负责权重的子字并行。
    - **缓冲区配置**：每个 Subcore 配备独立的 **Sign+Mantissa Buffer** 分别用于存储输入 **(X)** 和权重 **(W)** 的定点部分。
    - **归约单元 (Reduction Unit)**：根据操作维度使用 **Adder Tree**（如 6-way 或 3-way）累加部分和。

- **场景 (a)：Conv3x3 层映射 (X8W8 模式)**：
    - **输入分布**：输入特征图 **Input X (8b)** 的通道被切分。如图所示，**Subcore 0** 负责处理 **Cin[0~1]**（2个输入通道），**Subcore 5** 负责处理 **Cin[10~11]**。
    - **权重分布**：**3x3 Weights (8b)** 被分发至各 PU。由于是 3x3 卷积，计算密度适中。
    - **并行度与利用率**：在 FB16 模式下，**1个 Subcore 即可高效处理 2个输入通道**。所有 6 个 Subcore 可同时工作，覆盖 12 个输入通道。
    - **数据流**：数据从 Buffer 流入 PU 阵列进行乘累加（MAC）运算，最终通过 **6-way Adder Tree** 归约产生输出 **Y[k]** 和 **Y[k+1]**。

- **场景 (b)：Conv5x5 层映射 (X8W8 模式)**：
    - **资源聚合需求**：由于 **5x5 卷积核** 增大了空间维度的计算量（需 25 次 MAC 运算），单 Subcore 的资源不足以在保持高利用率的情况下处理 2 个通道。
    - **跨 Subcore 分组**：FlexBlock 采用 **“3个 Subcore 聚合”** 策略。例如，**Subcore 0, 1, 2** 联合处理 **Cin[0~3]**（2个输入通道）；**Subcore 3, 4, 5** 联合处理后续通道。
    - **利用率优化**：这种分组确保了即使在大核卷积下，MAC 单元也能被填满。理论利用率为 **(5×5) / (3×9) ≈ 93%**（即 25 个活跃 MAC / 27 个可用 MAC）。
    - **归约路径变化**：由于涉及多个 Subcore 协作，归约结构调整为 **3-way Adder Tree** 先进行局部归约，再进行全局归约。

- **关键设计洞察总结**：

| 特性 | Conv3x3 映射 | Conv5x5 映射 |
| :--- | :--- | :--- |
| **Subcore 分组粒度** | 1 个 Subcore / 2 通道 | 3 个 Subcore / 2 通道 |
| **处理的输入通道数** | 每组 2 通道 (如 Cin[0-1]) | 每组 2 通道 (如 Cin[0-3]) |
| **Adder Tree 类型** | 6-way | 3-way (两级级联) |
| **核心设计目标** | **最大化硬件利用率**，避免因卷积核大小不同导致的 PE 空闲 |

- **结论**：Fig. 10 核心论证了 FlexBlock 通过 **动态重构 Subcore 之间的拓扑连接** 和 **调整归约树深度**，成功解决了传统加速器在处理不同尺寸卷积层时面临的 **硬件利用率下降** 问题，实现了在多种层类型下的 **高效灵活计算**。

### Fig. 11. An example of mapping 2D operations to FlexBlock subcores. Here, we select a widely used depthwise Conv3 as an example. Note that outputs from each subcore move horizontally to the reduction unit for 2D mode.

![85b81051debc4e94ace46627175556af3943c4fd7801b8162e798ed284a751fd.jpg](images/85b81051debc4e94ace46627175556af3943c4fd7801b8162e798ed284a751fd.jpg)

- **图像基本信息**
  - 图像编号：**Fig. 11**
  - 图像标题：*An example of mapping 2D operations to FlexBlock subcores*
  - 示例场景：**DW Conv3**（3×3 Depthwise Convolution，深度可分离卷积）
  - 核心特点：展示 **2D 操作**在 FlexBlock 架构中的数据映射与并行处理方式

- **架构组成解析**

  | 组件层级 | 具体内容 | 数据位宽/规格 |
  |---------|---------|--------------|
  | **Input** | 输入特征图（Input fmap） | **16b**（FB24 模式） |
  | **输入通道数** | 6 input channels | 对应 Subcore0~Subcore5 |
  | **子核阵列** | **Subcore0 至 Subcore5** | 共 **6 个子核**并行工作 |
  | **处理单元** | 每个 Subcore 包含 **PU3, PU2, PU1, PU0** | 4 个 Processing Unit |
  | **Weight** | 权重参数 | **16b** |
  | **Output** | 输出特征图（Ofmap） | 对应 **C_out = k** 至 **C_out = k+5** |

- **数据流向与映射机制**
  - **输入分发**：6 个输入通道分别映射到 **6 个独立的 Subcore**（Subcore0 处理第 k 通道，Subcore5 处理第 k+5 通道）
  - **数据位宽**：
    - 输入到每个 Subcore 的数据总线宽度为 **144b**
    - 每个 PU 输出的部分和（partial sum）为 **36b**
  - **2D 操作特性**：
    - **水平数据流**：各 Subcore 的输出**水平移动**至专用的 **2D Reduction Unit**（2D 归约单元）
    - 与 3D 操作（垂直累加）不同，2D 操作中每个 Subcore **独立计算一个输出通道**，无需跨通道累加
    - 这对应 Depthwise Convolution 的本质：**逐通道**进行空间卷积，无跨通道聚合

- **设计意图与技术意义**
  - **解决核心利用率问题**：传统精度可扩展架构（如 Bit Fusion）在处理 2D 操作（如 DW-Conv、权重梯度计算 WU 步骤）时，因**累加需求少**导致 MAC 单元严重闲置
  - **专用归约路径**：FlexBlock 为 2D 操作设计了**独立的 Reduction Unit**，减少不必要的累加级数，提升 PE 利用率
  - **训练加速关键**：DNN 训练的 **WU（Weight Update）步骤**主要由 Depthwise Convolutions 组成，优化 2D 操作对整体训练性能至关重要（如 MobileNetV1 等网络包含大量 DW-Conv 层）
  - **硬件开销权衡**：增加 2D Reduction Unit 仅带来 **4.16% 面积开销**和 **3.89% 功耗开销**，但显著提升了训练吞吐量

- **精度模式关联**
  - 图示采用 **16b 输入/权重**（FB24 模式），即 **X16W16** 配置
  - 在更低精度模式（FB16/X8W8 或 FB12/X4W4）下，同一硬件可通过 **Sub-word Parallelism**（子字并行）同时处理更多通道：
    - FB16 模式：每个 Subcore 可处理 **2 个输入通道**
    - FB12 模式：每个 Subcore 可处理 **4 个输入通道**
  - 这种**精度-吞吐量线性缩放**是 FlexBlock 相比固定精度加速器的核心优势

### Fig. 12. The overall microarchitecture of FlexBlock accelerator. There are two reduction units where each unit is dedicated to either 2D or 3D operation mode Our FlexBlock core is capable of processing various block floating point numbers with 8-bit shared exponents (e.g., FB12, FB16 and FB24). The ‘out\_CORE goes to a weight update or a batch normalization unit. Before storing the data to DRAM, we extract shared exponents of sub-tensors at a FP2BFP converter for computing the next layer.

![9291d9b82077847ce8f97591fab9f58bb27f00a3373dbeec2d24e53aae8dea7a.jpg](images/9291d9b82077847ce8f97591fab9f58bb27f00a3373dbeec2d24e53aae8dea7a.jpg)

- **整体架构概览**
    - 图中展示了 **FlexBlock Core** 的完整微架构，这是支持多模式 **Block Floating Point (BFP)** 的 DNN 训练加速器
    - 架构采用**双路径归约设计**，分别针对 **2D 操作**（如 Depthwise Convolution）和 **3D 操作**（如标准 Convolution、FC 层）优化
    - 支持 **FB12**（4-bit 尾数）、**FB16**（8-bit 尾数）、**FB24**（16-bit 尾数）三种 BFP 格式，均使用 **8-bit 共享指数**

- **左侧控制与存储子系统**
    - **Control Signal Distributor**：从主机 CPU 接收编程信号，分发控制指令
    - **FSM Block**：有限状态机，控制训练流程的状态转换
    - **Buffer 系统**：
      | 缓冲区类型 | 功能 |
      |------------|------|
      | Input Buffer | 存储激活值/梯度（'sign+mantissa' 部分） |
      | Weight Buffer | 存储权重参数 |
      | Output Buffer | 存储计算结果 |
    - **Shared Exponent Handler**：管理张量的 8-bit 共享指数，实现 BFP 格式的指数对齐

- **中央处理核心（Processing Core）**
    - **层次化结构**：采用 **Multiplier → PE → PU → Subcore** 四级层次
    - **Subcore 配置**：包含 **Subcore0 至 Subcore5** 共 6 个子核心
    - **PU（Processing Unit）组织**：
      - 每个 Subcore 包含 **PU0-PU3** 四个处理单元
      - 每个 PU 前端配有 **2:1 多路选择器**（标记为 0/1），用于根据精度模式选择数据通路
      - **子字并行机制**：
        - **16-bit 模式**：跨 4 个 PE 实现输入张量的子字并行
        - **8-bit 模式**：跨 2 个 PE 分组
        - **4-bit 模式**：每个 PE 独立处理一个通道

- **3D 模式归约路径（上方绿色区域）**
    - 适用场景：**General Convolution**、**Pointwise Convolution**、**Fully Connected** 层的前向/反向传播
    - **数据流**：
      1. 各 Subcore 输出 → **4-way Adder Tree (Int)**：整数域内初步归约
      2. **Selective Bit-Truncation Unit**：根据当前精度模式选择性截断位数
      3. **Selective 6-way Adder Tree (Int)**：跨 Subcore 的二次归约（6 个 Subcore 对应 6 路）
      4. 最终进入 **Arithmetic Converter** 转换为浮点表示

- **2D 模式归约路径（下方橙色区域）**
    - 适用场景：**Depthwise Convolution**、**Weight Gradient 计算**（WU 步骤）、**Dilated Convolution**
    - **关键设计动机**：2D 操作（如深度可分离卷积）的累积维度较短，传统 3D 路径会导致 MAC 利用率低
    - **数据流**：
      1. 各 Subcore 输出 → **6-way Adder Tree (Int)**：每个 Subcore 对应一路 6 输入加法树
      2. **Shared Shifter**：共享移位器，处理 BFP 的指数对齐移位操作
      3. **Arithmetic Converter**：整数转浮点转换
      4. **FP32 Adder**：高精度累积，防止训练过程中的数值溢出/下溢
      5. **Selective 4-way Adder Tree (FP32)**：最终归约为单个部分和（psum）

- **右侧后处理与输出流水线**
    - **Arithmetic Converter + FP32 Adder**：将 2D 路径的多个 psum（图中标注 **psum x6**）合并
    - **Core Output Buffer**：缓存最终计算结果
    - **后续处理单元**（按数据流顺序）：
      | 单元名称 | 功能 |
      |----------|------|
      | Batch Norm Unit | 批量归一化 |
      | ReLU-Pool Unit | 激活函数与池化 |
      | Masking Unit | 掩码操作（用于稀疏训练等） |
      | FIFO | 先入先出队列，平滑数据流 |
      | Weight Update Unit | 权重更新（SGD/Adam 等） |
      | **FP2BFP Converter** | **关键单元**：将 FP32 结果转换回 BFP 格式，提取下一层所需的共享指数 |
      | Quantization Unit | 量化处理 |
    - **To DRAM**：最终结果写回主存

- **架构创新点总结**
    - **双归约单元设计**：通过独立的 2D/3D 路径解决深度可分离卷积等操作的硬件利用率问题（相比 Bit Fusion 等前序工作提升显著）
    - **选择性加法树**：根据运行时精度模式（FB12/FB16/FB24）动态重构加法树拓扑，在低精度模式下产生多个独立输出以提升吞吐
    - **混合精度算术路径**：内部使用**定点运算**（Integer Adder Tree）进行高密度 MAC，仅在最终归约阶段转为 **FP32**，兼顾效率与数值稳定性
    - **细粒度块大小控制**：通过 Shared Exponent Handler 和 FP2BFP Converter 实现**子张量级别**的指数共享（如表 III 所示，不同层类型的块大小从 1×1×2 到 7×7×4 不等）

- **性能影响量化**
    - 该架构使 FlexBlock 在 **FB16 模式**下相比 bfloat16 Systolic Array 实现 **平均 1.9x 能效提升**
    - 在 **FB12 模式**下可实现理论 **16x 吞吐密度**（实际因利用率因素约为 **5.3x** 训练加速）
    - 2D 归约单元的面积开销仅占核心总面积的 **4.16%**，功耗开销 **3.89%**，性价比极高

### 742c2359706508f1252b3c14bc6a8356b6fb1184749044ee933ca79c0145dad9.jpg

![742c2359706508f1252b3c14bc6a8356b6fb1184749044ee933ca79c0145dad9.jpg](images/742c2359706508f1252b3c14bc6a8356b6fb1184749044ee933ca79c0145dad9.jpg)

- **图表类型与内容**
  - 该图为 **Fig. 13**，展示不同 BFP 精度格式下的 DNN 训练曲线对比
  - 包含三个子图，分别对应不同模型和数据集组合
  - 横轴为 Training Epoch（训练轮次），纵轴为 Accuracy（准确率）或 Perplexity（困惑度）

- **图例说明（六种精度配置）**

| 线条样式 | 颜色 | 精度格式 | 说明 |
|---------|------|---------|------|
| 实线 | 黑色 | FP32 (Baseline) | 标准单精度浮点基线 |
| 实线 | 绿色 | FB24 | 16位尾数+8位共享指数 |
| 实线 | 橙红 | FB16 | 8位尾数+8位共享指数 |
| 实线 | 蓝色 | FB12 | 4位尾数+8位共享指数 |
| 虚线 | 黄绿 | FB12+BW24 | FW/WU用FB12，BW用FB24 |
| 点划线 | 黄色 | FB12+BW16 | FW/WU用FB12，BW用FB16 |

- **子图一：MobileNetV1 on CIFAR 数据集**
  - 同时展示 **CIFAR-10**（上曲线组）和 **CIFAR-100**（下曲线组）
  - **关键观察**：
    - **FB24（绿色）** 和 **FB16（橙红）** 与 FP32 基线几乎重合，最终准确率无显著损失
    - **FB12（蓝色）** 出现明显准确率下降，尤其在 CIFAR-100 上更为严重（圈出区域显示明显分离）
    - **FB12+BW16（黄色）** 相比纯 FB12 有显著提升，曲线接近 FP32 水平
    - 训练后期（100-200 epoch），混合精度策略的优势更加明显

- **子图二：ResNet-18 on ImageNet**
  - ImageNet 为大规模复杂数据集，训练 90 个 epoch
  - **关键观察**：
    - **FB24（绿色）** 表现最佳，最终 Top-1 准确率约 **69.9%**，甚至略超 FP32 的 69.95%
    - **FB16（橙红）** 表现稳定，准确率约 **61.2%**
    - **FB12（蓝色）** 训练过程出现**剧烈震荡**（epoch 20-40 期间尤为明显），最终准确率仅约 **58.5%**，较基线下降约 **11.5%**
    - **FB12+BW24（黄绿虚线）** 显著改善训练稳定性，最终准确率回升至约 **68.2%**
    - **FB12+BW16（黄色）** 在 ImageNet 上改善有限，准确率仍较低（约 59%）

- **子图三：Transformer-base on WMT14 (En-De)**
  - 使用 **Perplexity（困惑度）** 作为评价指标，**值越低越好**
  - **关键观察**：
    - 所有 BFP 格式（包括 FB12）均能收敛到与 FP32 接近的困惑度水平（约 3.9-4.3）
    - **FB12（蓝色）** 在此任务上未表现出显著的性能退化
    - Transformer 架构对低精度训练的容忍度似乎高于 CNN
    - 插图放大了后期训练阶段（epoch 100-200），显示各曲线趋于一致

- **核心结论与设计启示**
  - **精度敏感性分层**：不同训练步骤对精度要求不同——**反向传播（BW）步骤**对精度最敏感，前向传播（FW）和权重更新（WU）可使用更低精度
  - **混合精度策略有效性**：**FB12+BW16/FB24** 通过仅在 BW 步骤提升精度，在保持高准确率的同时大幅降低计算开销
  - **任务依赖性**：简单任务（CIFAR）可接受更低精度；复杂任务（ImageNet）需要更高精度或混合策略；Transformer 任务对精度相对不敏感
  - **FlexBlock 设计验证**：实验结果支持 FlexBlock 支持**多模式 BFP** 的硬件设计理念，允许针对不同张量（activation, weight, gradient）独立配置精度

### SA: Systolic Array (bfloat16) | BF: BitFusion-based | FB: FlexBlock (Proposed) Fig. 15. Comparisons of the performance and energy consumption between the systolic array (SA), the BitFusion-based BFP accelerator (BF), and the proposed FlexBlock (FB in red). For the performance and energy analysis, we evaluated five CNN benchmarks on ImageNet and a Transformer-base model on WMT14 En-De.

![b29a0008d2473c3603b2d19a9b79019eeb3d0b18f7a688db48420c4f173f7854.jpg](images/b29a0008d2473c3603b2d19a9b79019eeb3d0b18f7a688db48420c4f173f7854.jpg)

- **图表总体概述**
  - 图15展示了三种DNN训练加速器在**六个基准模型**上的**性能加速比**（上半部分）与**能耗对比**（下半部分）。
  - 横坐标代表不同的硬件配置与精度模式，包括：BF(24)、FB(24)、SA、BF(16)、FB(16)、BF(12)、FB(12)。
  - 纵坐标分别对应**加速倍数**（Speed Up）和**绝对能耗值**（Energy Consumption, 单位焦耳J）。

- **对比对象定义**
  - **SA (Systolic Array)**：基于bfloat16的脉动阵列加速器（作为FP16/BF16精度的参考基线）。
  - **BF (BitFusion-based)**：基于BitFusion架构的BFP训练加速器（支持多精度但存在硬件利用率问题）。
  - **FB (FlexBlock)**：本文提出的**FlexBlock加速器**（红色标注），支持多模式Block Floating Point (BFP)。
  - **精度模式标识**：(24)代表FB24（16-bit尾数）、(16)代表FB16（8-bit尾数）、(12)代表FB12（4-bit尾数），均搭配8-bit共享指数。

- **性能加速比（Speed Up）详细数据分析**

| 模型名称 | FB(24) 加速比 | FB(16) 加速比 | FB(12) 加速比 | 关键观察 |
| :--- | :--- | :--- | :--- | :--- |
| **AlexNet** | 1.0x (基线) | **2.7x** | **16.5x** | 低精度下加速极其显著 |
| **VGG16** | 1.0x (基线) | **3.2x** | **14.0x** | 16位模式下已有3倍以上提升 |
| **ResNet-18** | 1.0x (基线) | **2.3x** | **9.3x** | 残差网络在极低精度下增益略低 |
| **MobileNetV1** | 1.0x (基线) | **3.1x** | **14.8x** | 轻量级网络受益明显 |
| **DenseNet-121** | 1.0x (基线) | **2.1x** | **14.8x** | 密集连接网络在4-bit下爆发式提速 |
| **Transformer** | 1.0x (基线) | **4.0x** | **16.5x** | 注意力机制模型对精度缩放响应极佳 |

  - **核心规律**：随着BFP精度从16-bit降至8-bit再降至4-bit，FlexBlock的理论吞吐量呈**平方级增长**（因MAC阵列可处理更多并行操作）。实际加速比在FB12模式下普遍达到**9.3x ~ 16.5x**，远超传统固定精度架构。
  - **对比优势**：在相同精度等级下（如FB16 vs BF16或SA），FlexBlock凭借**更高的硬件利用率**（解决前述Bit Fusion的累加路径瓶颈）仍能获得额外加速。

- **能耗消耗（Energy Consumption）深度解析**
  - **堆叠柱状图构成**：每个柱子由六部分组成，分别代表前向传播(FW)、反向传播(BW)、权重更新(WU)三个阶段的**计算核心(Core)**与**数据缓冲区(Buffer)**功耗。
  - **相对节能效率**（以BF(24)为基准的百分比降幅）：

| 模型名称 | FB(24) 节能率 | FB(16) 节能率 | **FB(12) 节能率** | 能耗特征 |
| :--- | :--- | :--- | :--- | :--- |
| **AlexNet** | -69.9% | -73.8% | **-86.4%** | Buffer占比显著，低精度大幅削减访存能耗 |
| **VGG16** | -59.3% | -63.5% | **-79.8%** | 计算密集型，Core节能贡献大 |
| **ResNet-18** | -69.0% | -71.6% | **-84.1%** | 均衡型网络，各组件同步优化 |
| **MobileNetV1** | -77.8% | -55.6% | **-71.7%** | Depthwise操作较多，受限于2D归约单元开销 |
| **DenseNet-121** | -72.1% | -65.6% | **-91.2%** | **全模型最低能耗**，得益于高并行度利用 |
| **Transformer** | -43.8% | -46.7% | **-62.3%** | 序列模型内存带宽压力大，相对节能空间较小 |

  - **关键技术洞察**：
    - **Buffer能耗占比**：在低精度模式（特别是FB12）下，由于数据位宽减半或四分之**三**，**存储访问能耗急剧下降**（绿色/黄色/浅蓝色部分明显缩小）。
    - **Core能耗优化**：FlexBlock通过**层次化子字并行**（Hierarchical Sub-word Parallelism）和**双路归约单元**（Dual-path Reduction Units），确保在低精度下MAC阵列不空闲，从而提升每瓦特性能（GFLOPS/W）。
    - **极端案例**：DenseNet-121在FB12模式下实现了**91.2%的能耗降低**，几乎仅需原BF(24)架构十分之一的能量即可完成训练。

- **综合评估与设计意义**
  - **灵活性与效率的统一**：FlexBlock证明了**动态精度切换**不仅是理论可能，更能在真实硬件上转化为**数量级的性能收益**。用户可根据不同层（Layer-wise）或不同张量（Activation/Weight/Gradient）的数值敏感性，选择最优BFP格式。
  - **训练步骤差异化价值**：图中隐含的信息是，**权重更新(WU)步骤**（通常包含Depthwise Convolution）在FlexBlock中通过专用2D归约单元得到了特别优化，避免了传统可变精度加速器在此类操作上的严重欠利用率。
  - **实际部署启示**：对于边缘设备或数据中心，使用FlexBlock进行**混合精度训练**（如FW/WU用FB12，BW用FB16/FB24）可在**不损失模型精度**的前提下（参见论文Table IV），将训练时间缩短**13%~50%**，同时大幅降低运营成本（OPEX）。

### Fig. 16. Illustration of the process performed by the FP2BFP converter. The number of underflows are used to determine the precision of each tensor at runtime.

![703d89496bd45c7667b97a1f5cdc50fbd8a047e636dadd5fa7eb26220a959a51.jpg](images/703d89496bd45c7667b97a1f5cdc50fbd8a047e636dadd5fa7eb26220a959a51.jpg)

- **图像核心功能解析**：该图详细展示了 **FP2BFP Converter**（浮点转块浮点转换器）的内部工作流程，这是 FlexBlock 架构中实现动态精度控制的关键硬件模块。

- **输入数据结构**：
  - 输入为来自 **FlexBlock Core** 的四个输出值（X₀ 至 X₃）
  - 每个数值采用分离式存储：**Exponent**（指数部分，蓝色标识）+ **Mantissa**（尾数部分，黄色标识）
  
  | 输入变量 | Exponent (4-bit) | Mantissa (3-bit) |
  |---------|------------------|------------------|
  | X₀ | 1010 | 101 |
  | X₁ | 1000 | 110 |
  | X₂ | 1010 | 001 |
  | X₃ | 0011 | 111 |

- **处理流水线阶段**：

  - **Stage 1: Shared Exponent Extractor（共享指数提取器）**
    - 功能：扫描当前数据块内所有数值的指数域
    - 策略：选取**最大指数值**作为该块的 **Shared Exponent**（本例中为 **1010**，即十进制 10）
    - 目的：确保块内所有数值能统一表示而不发生上溢

  - **Stage 2: Mantissa Aligner（尾数对齐器）**
    - 功能：根据提取的共享指数，对各数值的尾数进行右移对齐
    - 对齐规则：`shift_amount = shared_exponent - original_exponent`
    - 对齐后生成新的尾数表示（X̂₀ 至 X̂₃）

- **输出与异常检测**：

  | 输出变量 | 对齐后 Mantissa | 状态 |
  |---------|----------------|------|
  | X̂₀ | 101 | 正常 |
  | X̂₁ | 001 | 正常 |
  | X̂₂ | 001 | 正常 |
  | X̂₃ | **000** | **Underflow!**（下溢）**

  - **关键现象——Underflow（下溢）**：
    - X₃ 的原始指数（0011，即 3）与共享指数（1010，即 10）差距过大（差值为 7）
    - 尾数右移 7 位后，有效信息完全丢失，变为全零
    - 图中以**红色高亮**标注此异常状态

- **与动态精度控制的关联机制**：
  - **Underflow Counter（下溢计数器）**：统计每个张量在训练过程中的下溢发生频率
  - **精度决策依据**：当下溢比例超过预设阈值（th_Up），系统自动将对应张量的 BFP 精度提升（如从 FB12 升至 FB16 或 FB24）
  - **Hysteresis Control（滞回控制）**：采用双阈值（th_Down 和 th_Up）避免精度频繁抖动，确保训练稳定性

- **设计意义**：
  - 该模块实现了**运行时精度自适应**：无需人工干预即可根据数值分布特征动态调整精度
  - 通过监控下溢事件，在**训练速度**与**模型精度**之间取得最优平衡点
  - 硬件开销极低：仅占 FlexBlock Core 面积的 **0.3%** 和功耗的 **0.5%**

### 360c8ac7472e73c64a96d02ea1482a902d10b0afbee37cff551222434bf942a3.jpg

![360c8ac7472e73c64a96d02ea1482a902d10b0afbee37cff551222434bf942a3.jpg](images/360c8ac7472e73c64a96d02ea1482a902d10b0afbee37cff551222434bf942a3.jpg)

- **图像基本信息与定位**
    - 该图为论文 **Fig. 17(a)**，标题为 **Dynamic Precision Control**（动态精度控制机制）
    - 位于论文 **Section VII-C** 章节，是 FlexBlock 加速器实现**运行时自动精度调节**的核心控制逻辑流图
    - 图示状态：当前精度基准为 **FB16**（8-bit 尾数 + 8-bit 共享指数）

- **流程图结构与决策逻辑解析**

| 组件/节点 | 功能描述 | 判定条件与动作 |
|-----------|---------|---------------|
| **第一判定节点 (th_Up)** | 上阈值比较器 | 若 **N > th_Up**（下溢过多）：**Set to higher precision (FB24)**，提升至16位尾数以保证数值稳定性 |
| **第二判定节点 (th_Down)** | 下阈值比较器（仅当第一判定为 no 时进入） | 若 **N < th_Down**（下溢很少）：**Set to lower precision (FB12)**，降至4位尾数以提升吞吐量 |
| **保持状态** | 滞回区间（Dead Zone） | 若 **th_Down ≤ N ≤ th_Up**：**Keep precision (FB16)**，维持当前精度不变 |

- **核心设计机制：滞回控制（Hysteresis Control）**
    - 采用**双阈值（th_Up, th_Down）** 设计而非单阈值，形成**滞回环（Hysteresis Loop）**
    - **目的**：防止精度在相邻模式间**频繁振荡（Chattering）**，确保系统稳定性
    - **硬件开销极低**：仅需 **32位下溢计数器 + 两个32位比较器**，面积开销仅占 Core 的 **0.3%**，功耗开销 **0.5%**

- **精度模式映射关系**

| 精度标识 | 尾数位数 (Sign+Mantissa) | 共享指数位数 | 适用场景 |
|---------|------------------------|-------------|---------|
| **FB12** | 4-bit | 8-bit | 对数值误差不敏感层（如浅层特征提取），**最高吞吐量模式** |
| **FB16** | 8-bit | 8-bit | 默认/中间态，平衡精度与性能 |
| **FB24** | 16-bit | 8-bit | 梯度计算（BW步骤）等对精度敏感的操作，**保精度模式** |

- **算法语义与物理意义**
    - **Underflow（下溢）** 的物理含义：当使用较低精度的 BFP 格式表示张量时，部分数值因**尾数位宽不足**而被截断为0，导致信息丢失
    - **控制策略本质**：基于**在线监测的数值误差信号**（下溢率），动态调整**量化粒度**，实现**精度-性能帕累托最优**
    - **与训练步骤的关联**：该控制器可针对 **FW（前向）、BW（反向）、WU（权重更新）** 三个训练步骤**独立设置阈值**，论文实验表明 **BW 步骤对梯度精度最敏感**（需设置更保守的阈值）

- **实验验证效果（结合论文数据）**
    - 在 ResNet-18/CIFAR 数据集上，动态控制相比静态 FB16 模式实现 **22.5% 训练加速**，且**无精度损失**
    - 在 ImageNet 上，相比静态 FB12+BW24 策略，动态控制进一步带来 **13.1% 加速**，同时 accuracy 仅降低 0.3%（相比 FB16 基线降低 6.7% vs 静态策略的 7.0%）
    - **阈值敏感性分析**（Table VI）：局部梯度（Local Gradient）的阈值设置最敏感（阈值过宽导致 >1.5% 精度下降），而权重（Weight）和激活值（Activation）的阈值容忍度较高

### 244041ccfeec87d73d2b966a59d7040e411b895ed8b7ea7e5bd0d71c0c0ec5f4.jpg

![244041ccfeec87d73d2b966a59d7040e411b895ed8b7ea7e5bd0d71c0c0ec5f4.jpg](images/244041ccfeec87d73d2b966a59d7040e411b895ed8b7ea7e5bd0d71c0c0ec5f4.jpg)

- **图表核心主题**：展示 **ResNet-18** 在 **CIFAR-10** 数据集训练过程中，**Dynamic Precision Control** 机制的层级精度自适应行为

- **上层精度配置条形图**：
  
  | 层级 (Layer) | BFP 配置模式 | 含义解读 |
  |-------------|-------------|---------|
  | **L1** | **FB12+BW16** | 前向/权重更新用4位尾数，反向传播用8位尾数 |
  | **L9** | **FB12+BW16** | 同L1，保持较高梯度精度 |
  | **L15** | **FB12(+BW12)** | 全流程统一使用4位尾数（含反向传播） |

- **下层 Underflow Ratio 时序曲线**：
  - **横轴**：Training Epoch（1-200 epochs）
  - **纵轴**：Underflow Ratio（0.0-0.8 范围）
  - **三条特征曲线**：
    - **L1（橙色）**：呈现**高波动性**，多峰结构，峰值多次触及 **th_Up (~0.6)** 阈值线，表明该层数值分布动态范围大，频繁触发精度提升需求
    - **L9（绿色）**：中等波动幅度，整体位于 **0.4-0.7** 区间震荡，介于两阈值之间，处于**临界稳定状态**
    - **L15（蓝色）**：**低平稳曲线**，全程维持在 **th_Down (~0.35)** 附近或以下，证明深层网络可安全降级至最低精度

- **双阈值滞回控制机制**：
  - **th_Up (上阈值 ~0.6)**：红色虚线标记，当 **Underflow Ratio > th_Up** 时，系统判定当前 mantissa 位宽不足，**自动升级精度**（如从 FB12 升至 FB16）
  - **th_Down (下阈值 ~0.35)**：红色虚线标记，当 **Underflow Ratio < th_Down** 且持续稳定时，**允许降级精度**以提升吞吐量
  - **Hysteresis 特性**：两阈值间存在 **~0.25 的死区**，防止训练过程中精度模式**频繁抖动切换**

- **关键工程洞察**：
  - **层级敏感性差异**：浅层卷积层（L1/L9）的 local gradient 分布更广泛，需保留 **8-bit mantissa (BW16)**；深层特征已高度抽象，**4-bit (BW12)** 即可满足收敛需求
  - **资源优化潜力**：L15 层成功维持 **FB12 全流程** 模式，相比统一 FB16 配置可节省约 **50% 的 MAC 操作位宽**，对应理论 **2x 吞吐量提升**
  - **监控实现位置**：Underflow 统计源自 **FP2BFP Converter** 模块（见图16），该单元在每 epoch 结束后统计移位操作中的下溢事件数，作为精度决策依据

- **与论文主旨关联**：此图实证了 FlexBlock 架构的 **per-layer precision scalability** 能力，通过硬件级的 underflow 计数器 + 软件滞回控制器，实现了 **"精度按需分配"** 的自适应训练策略，最终达成 **13.1% 训练加速**（相对于静态 FB12+BW16 配置）且**零精度损失**的目标

### 76ed6c5dda4d2e9fd621cc249adc56e1fa84d400f5b2a6116faf115ae4336f7c.jpg

![76ed6c5dda4d2e9fd621cc249adc56e1fa84d400f5b2a6116faf115ae4336f7c.jpg](images/76ed6c5dda4d2e9fd621cc249adc56e1fa84d400f5b2a6116faf115ae4336f7c.jpg)

- **图表核心主题**：该图展示了 **ResNet-18** 在 **CIFAR-100** 数据集训练过程中，**动态精度控制机制**（Dynamic Precision Control）对特定层（L1, L9, L15）的**下溢比率**（Underflow Ratio）监控及**精度模式自适应调整**过程。

- **图表结构解析**：
    - **上部条形图**：直观呈现三个代表性层在训练过程中实际采用的 **BFP 精度配置**：
        | 层 (Layer) | 精度配置模式 | 含义解释 |
        | :--- | :--- | :--- |
        | **L1** | **FB12(+BW12)** | 前向传播(FW)与权重更新(WU)使用 **4-bit 尾数** (FB12)，反向传播(BW)也使用 **4-bit 尾数** |
        | **L9** | **FB12+BW16** | FW/WU 使用 **4-bit 尾数**，但 BW 步骤提升至 **8-bit 尾数** 以保证梯度计算稳定性 |
        | **L15** | **FB12+BW16** | 同 L9，对 BW 步骤进行了精度提升 |

    - **下部折线图**：量化展示各层随 **Training Epoch**（1-200轮）变化的 **Underflow Ratio** 动态轨迹：
        - **纵坐标 (Y-axis)**：**Underflow Ratio**（下溢比率），范围 0.0 - 0.8。
        - **横坐标 (X-axis)**：**Training Epoch**（训练轮次），范围 1 - 200。
        - **监控曲线**：
            - **橙色线 (L1)**：第一层的下溢比率波动特征。
            - **绿色线 (L9)**：第九层的下溢比率波动特征。
            - **蓝色线 (L15)**：第十五层的下溢比率波动特征。
        - **控制阈值线 (Hysteresis Thresholds)**：
            - **th_Up (上阈值)**：约 **0.65**（红色虚线）。当某层下溢比率**持续高于此值**时，控制器判定当前尾数位宽不足，将在下一 epoch **提升精度**（如从 FB12 升至 FB16）。
            - **th_Down (下阈值)**：约 **0.35**（红色虚线）。当比率**低于此值**时，表明当前精度有冗余，可尝试**降低精度**以换取更高吞吐量。

- **关键现象与深度分析**：
    - **分层敏感性差异**：图中清晰揭示了 DNN 不同深度层对数值精度的敏感度异质性。**L1（浅层）** 能够全程维持在最低精度的 **FB12(+BW12)** 模式，其下溢比率虽在训练初期（Epoch 0-50）有剧烈波动并触及上阈值，但整体可控；相比之下，**L9 与 L15（深层）** 在训练中后期（Epoch 50之后）的下溢比率更容易突破 **th_Up**，迫使系统将 BW 步骤动态切换至更高精度的 **FB16** 模式（对应上部条形图中的灰色区域扩展）。
    - **滞后控制策略 (Hysteresis Control)**：采用双阈值（th_Up / th_Down）而非单阈值设计，有效避免了精度模式在临界点附近的**频繁抖动（Thrashing）**。只有当下溢比率显著恶化或显著改善时才触发切换，保证了训练过程的**数值稳定性**与**硬件效率**的平衡。
    - **训练阶段特性**：在训练早期（Epoch 1-50），所有层的下溢比率均经历快速上升期，这与模型参数随机初始化、梯度方差较大有关。随着训练进入收敛阶段（Epoch 100+），各层比率趋于平稳，但深层网络（L9, L15）仍需保留较高的 BW 精度以维持 **Top-1 准确率** 不退化。

- **工程意义总结**：该图验证了 FlexBlock 提出的**硬件级动态精度控制**的可行性。通过实时监测 **FP2BFP 转换器**产生的下溢统计量，系统能够以极低的硬件开销（仅 **0.5% 功耗** 和 **0.3% 面积** 增加，如论文所述）实现**逐层、逐步骤（FW/BW/WU）** 的精度自适应，最终在不损失模型准确率的前提下，相比静态 FB16 配置实现了平均 **13.1% 的训练加速**（参考论文 Fig. 18(b) 数据）。

### 83ec7eabd8814e9d749708fd157252afef486371326a5adb902fc158739764b4.jpg

![83ec7eabd8814e9d749708fd157252afef486371326a5adb902fc158739764b4.jpg](images/83ec7eabd8814e9d749708fd157252afef486371326a5adb902fc158739764b4.jpg)

- **图像主题与定位**
    - 该图为 **ResNet-18 在 ImageNet 数据集上的动态精度控制 (Dynamic Precision Control) 实验结果**。
    - 对应论文第 VII-C 节，旨在展示 FlexBlock 加速器如何通过运行时监控 **Underflow Ratio (下溢比率)** 来自动逐层调整 BFP (Block Floating Point) 精度。

- **图表结构解析**
    - **上层 (Precision Selection)**: 以色带形式展示了 **Layer 1 (L1)**、**Layer 9 (L9)** 和 **Layer 15 (L15)** 在不同训练阶段所选用的 BFP 精度模式。
        - **FB12+BW24**: 前向传播(FW)和权重更新(WU)使用 4-bit 尾数，反向传播(BW)使用 16-bit 尾数。
        - **FB12+BW16**: FW/WU 使用 4-bit，BW 使用 8-bit。
        - **FB12 (+BW12)**: 全流程使用 4-bit 尾数（最低精度）。
        - **白色/灰色区域**: 可能表示其他精度状态或未明确标注的模式。
    - **下层 (Underflow Ratio Curve)**: 折线图量化了精度调整的依据。
        - **X 轴**: Training Epoch (训练轮次, 范围 1-60)。
        - **Y 轴**: Underflow Ratio (下溢比率, 范围 0-8)。
        - **曲线含义**:
            - **橙色线 (L1)**: 第 1 层的下溢比率变化，早期波动较大，后期稳定在较低水平。
            - **绿色线 (L9)**: 第 9 层的下溢比率，呈现周期性波动，多次触及阈值线。
            - **蓝色线 (L15)**: 第 15 层（较深层）的下溢比率，在第 10-20 epoch 间出现显著峰值（接近 0.6），表明该层对低精度更敏感。

- **关键机制与阈值逻辑**
    - **控制原理**: 系统通过 **FP2BFP Converter** 统计每个 tensor 的下溢次数。当下溢比率超过预设阈值 (**th_Up**, 上红色虚线) 时，控制器会在下一 epoch **提升精度**；当比率低于下阈值 (**th_Down**, 下红色虚线) 时，则尝试 **降低精度** 以提升速度。
    - **层敏感性差异**:
        - **L1 (浅层)**: 对精度相对不敏感，可较早切换至低功耗模式 (如 FB12+BW16 或 FB12)。
        - **L9 (中层)**: 表现出明显的震荡特性，需要频繁在 FB12+BW16 和更高精度间切换以维持数值稳定性。
        - **L15 (深层)**: 出现高下溢峰值，说明梯度信息在深层网络中分布更广或对截断更敏感，因此需要保持较高的 BW (Backward) 精度（如 BW16 或 BW24）。

- **实验结论与设计意义**
    - **自适应优化**: 该图验证了 **Hysteresis Controller (滞回控制器)** 的有效性。它避免了精度在阈值附近的频繁抖动，实现了平滑的精度迁移。
    - **性能与精度的平衡**: 通过这种动态机制，FlexBlock 能够在保证训练收敛（Accuracy 不损失）的前提下，将大部分计算量维持在 **FB12** (4-bit) 模式，仅在关键层（如 L15）或关键步骤（BW）动态回退到 **FB16/FB24**，从而最大化硬件利用率和能效。
    - **实际加速效果**: 根据论文数据，相比静态设置 (Static BFP Mode)，这种动态控制能带来平均 **13.1% 的额外训练速度提升**。

### 3a4cc1beae2902a86bed2a78958a19c8fe08587084681f60be8fe6f02a3cec4c.jpg

![3a4cc1beae2902a86bed2a78958a19c8fe08587084681f60be8fe6f02a3cec4c.jpg](images/3a4cc1beae2902a86bed2a78958a19c8fe08587084681f60be8fe6f02a3cec4c.jpg)

- **图表类型与主题**
    - 该图为**堆叠水平条形图 (Stacked Horizontal Bar Chart)**，对应论文中 **Fig. 18(a)**。
    - 展示了在启用 **Dynamic Precision Control（动态精度控制）** 机制后，不同 DNN 训练步骤（FW/BW/WU）在各数据集上所选择的 **BFP 精度分布比例**。

- **坐标轴定义**
    - **Y 轴（纵向）**：按数据集分组，包含 **CIFAR-10**、**CIFAR-100** 和 **ImageNet** 三类基准；每组内细分为三个训练阶段：
        - **FW (Forward Pass)**：前向传播，计算特征图。
        - **BW (Backward Pass)**：反向传播，计算局部梯度 (Local Gradients)。
        - **WU (Weight Update)**：权重更新，计算权重梯度 (Weight Gradients)。
    - **X 轴（横向）**：**Ratio of Each BFP Precision**，表示各精度模式占总计算量的百分比（0%-100%）。
    - **图例 (Legend)**：三种 BFP 精度模式以灰度区分：
        - ■ **FB12**：4-bit 尾数 + 8-bit 共享指数（最低精度）
        - ■ **FB16**：8-bit 尾数 + 8-bit 共享指数（中等精度）
        - ■ **FB24**：16-bit 尾数 + 8-bit 共享指数（最高精度）

- **详细数据分析**

| 数据集 | 训练步骤 | FB12 占比 | FB16 占比 | FB24 占比 |
| :--- | :--- | :--- | :--- | :--- |
| **CIFAR-10** | FW | **~99%** | ~1% | ~0% |
| | BW | **57%** | **38%** | ~5% |
| | WU | **~99%** | ~1% | ~0% |
| **CIFAR-100** | FW | ~5% | **~95%** | ~0% |
| | BW | **70%** | ~17% | ~13% |
| | WU | **88%** | ~12% | ~0% |
| **ImageNet** | FW | **72%** | **26%** | ~2% |
| | BW | 20% | **27%** | **53%** |
| | WU | **57%** | **25%** | **18%** |

- **核心洞察与工程意义**
    - **前向/更新步骤的激进低精度化**：在 **FW** 和 **WU** 阶段，动态控制器倾向于将精度压至最低的 **FB12**（尤其在 CIFAR-10 上接近 99%）。这验证了特征图计算和权重梯度计算对数值精度相对不敏感，允许硬件以 **4-bit 定点运算** 主导，从而获得理论上的 **16倍吞吐量提升**（相对于 16-bit 模式）。
    - **反向传播的精度敏感性**：**BW** 阶段（局部梯度计算）显著依赖更高精度。在简单的 CIFAR 数据集上，FB12 仍占主导（57%-70%）；但在复杂的 **ImageNet** 上，**FB24 占比飙升至 53%**，表明大尺度数据集的梯度分布更广，需要更多尾数位来保持训练稳定性（与 Table IV 中 FB12 在 ImageNet 上准确率大幅下降的结论一致）。
    - **任务复杂度的自适应缩放**：从 CIFAR-10 → CIFAR-100 → ImageNet，整体精度需求呈上升趋势。动态控制机制成功实现了 **"能低则低，该高则高"** 的精细化调度，避免了静态配置下要么损失精度（全用 FB12）要么浪费算力（全用 FB24）的两难困境。
    - **性能收益量化**：结合论文 Fig. 18(b)，这种动态策略在 CIFAR 数据集上相比静态 **FB16** 模式实现了平均 **22.5% 的训练加速**，且**无准确率损失**；在 ImageNet 上相比静态 **FB12+BW24** 策略仍有 **13.1% 的加速**，同时将准确率差距从 7.0% 缩小至 6.7%（相比 FB16 基线）。

- **方法论关联**
    - 该图是论文 **Section VII-C** 提出的 **Hysteresis Controller（滞回控制器）** 的直接实验证据。控制器通过监测 **FP2BFP 转换器中的下溢 (Underflow) 比例**（Fig. 16），在预设阈值 ($th_{Down}$, $th_{Up}$) 之间滞回切换精度，实现了**层级粒度 (Layer-wise)** 的自适应精度调整（如 Fig. 17(b) 所示的 ResNet-18 各层精度变化轨迹）。

### 44e9edef3cba2023b49535afd62ba4296dab9efad71d4ea930896d046b1941d4.jpg

![44e9edef3cba2023b49535afd62ba4296dab9efad71d4ea930896d046b1941d4.jpg](images/44e9edef3cba2023b49535afd62ba4296dab9efad71d4ea930896d046b1941d4.jpg)

- **图表类型与主题**
  - 该图为**分组柱状图（Grouped Bar Chart）**，展示了不同 **Block Floating Point (BFP)** 精度配置下，DNN 训练的**每批训练时间（Training Time per Batch）**与**测试准确率（Acc.）**的对比关系。
  - 核心研究问题：验证 **Dynamic Precision Control（动态精度控制）** 策略能否在保持模型精度的前提下，最小化训练耗时。

- **坐标轴定义**
  - **X 轴**：Training Time per Batch [ms]，范围 0–450 ms，衡量训练吞吐量的倒数（值越小越快）。
  - **Y 轴**：三个评估基准数据集，分别为 **CIFAR-10**、**CIFAR-100** 和 **ImageNet**。

- **图例与实验变量**
  - **FB12+BW16**（深蓝）：前向传播（FW）和权重更新（WU）使用 4-bit 尾数（FB12），反向传播（BW）使用 8-bit 尾数（FB16）。
  - **FB12+BW24**（浅蓝）：FW/WU 使用 FB12，BW 使用 16-bit 尾数（FB24）。
  - **FB16**（绿色）：所有训练步骤统一使用 8-bit 尾数（FB16）。
  - **Dynamic**（黄色/米色）：采用论文提出的**动态精度控制机制**，运行时自适应调整各层精度。
  - **FB12**（红/橙色）：所有步骤激进地使用 4-bit 尾数（FB12），作为低精度下限参考。
  - **右侧数值标签**：对应各配置的最终 **Top-1 测试准确率（%）**。

- **定量数据分析**

| 数据集 | 配置 | 训练时间 (ms) | 准确率 (%) | 关键观察 |
| :--- | :--- | :--- | :--- | :--- |
| **CIFAR-10** | FB12+BW16 | ~120 | **93.5** | 高精度基准 |
| | FB12+BW24 | ~125 | **93.7** | 最高准确率 |
| | FB16 | ~100 | 93.3 | 速度较快 |
| | **Dynamic** | **~95** | 90.4* | **最快（-24%）** |
| | FB12 | ~110 | 90.4 | 低精度极限 |
| **CIFAR-100** | FB12+BW16 | ~120 | **77.2** | 高精度基准 |
| | FB12+BW24 | ~125 | 76.5 | |
| | FB16 | ~105 | **76.9** | 均衡选择 |
| | **Dynamic** | **~98** | 66.7* | **最快（-21%）** |
| | FB12 | ~115 | 66.7 | 低精度极限 |
| **ImageNet** | FB12+BW16 | ~380 | 61.2 | BW16不足 |
| | FB12+BW24 | ~400 | **68.2** | 最高准确率 |
| | FB16 | ~280 | **67.9** | 速度优势明显 |
| | **Dynamic** | **~340** | 58.5* | 折中方案（-24%） |
| | FB12 | ~220 | 58.5 | 极速但损精度 |

  *注：Dynamic 在 CIFAR 数据集上的准确率数值（90.4%/66.7%）与纯 FB12 相同，暗示动态控制器在该任务上为大部分层选择了 FB12；而在 ImageNet 上则更为保守。*

- **核心趋势与洞察**
  - **精度-速度权衡（Precision-Speed Tradeoff）**：降低尾数位宽（如从 FB16 降至 FB12）理论上可提升吞吐量（处理更多 MAC 操作），但**反向传播（BW）步骤对精度极度敏感**。如图所示，ImageNet 上使用 FB12+BW16 导致准确率暴跌至 61.2%（相比 FB16 的 67.9%），证明梯度计算需要更高精度。
  - **动态控制的效能**：图中**黄色箭头（-24%, -21%, -24%）** 明确标示出 **Dynamic** 模式相比静态 **FB16** 基线所节省的训练时间。该策略通过**滞后控制器（Hysteresis Controller）** 监测 FP2BFP 转换器的**下溢（Underflow）** 比例，自动为不敏感层（如浅层或特定卷积层）分配更低精度（FB12），而为敏感层保留高精度（FB16/FB24）。
  - **数据集复杂度影响**：在简单的 **CIFAR** 数据集上，Dynamic 甚至可以逼近纯 FB12 的速度；但在复杂的 **ImageNet** 上，Dynamic 必须频繁回退到更高精度以维持收敛性，因此其训练时间（~340ms）反而高于静态 FB16（~280ms），但显著优于保守的 FB12+BW24（~400ms），且准确率损失可控。

- **架构设计意义**
  - 该图直接验证了 **FlexBlock** 加速器支持**混合精度训练（Mixed-Precision Training）** 的硬件价值：通过为激活（Activation）、权重（Weight）和梯度（Gradient）张量独立配置 BFP 模式（共 3³=27 种组合），配合软件层面的动态调度，实现了**比固定精度配置更优的帕累托前沿（Pareto Frontier）**。
  - 图中 **FB12** 的存在证明了 FlexBlock 硬件具备执行 **4-bit 算术** 的能力（理论峰值吞吐量可达 FB16 的 4 倍），而 **Dynamic** 模式则是解锁这一潜力的关键软件协同技术。

### d521931e1a62d7ce1b941796c9cd8c7aa0cc155c9471335bbba47abe4e3eca0d.jpg

![d521931e1a62d7ce1b941796c9cd8c7aa0cc155c9471335bbba47abe4e3eca0d.jpg](images/d521931e1a62d7ce1b941796c9cd8c7aa0cc155c9471335bbba47abe4e3eca0d.jpg)

*   **图像类型**: 人物肖像照 / 学术证件照
*   **图像主体**: 论文第一作者 **Seock-Hwan Noh** (Student Member, IEEE)
*   **外貌特征**:
    *   **性别**: 男性
    *   **年龄**: 年轻（根据论文信息，2019年获得学士学位，推测拍摄时为研究生阶段）
    *   **发型**: 黑色短发，整齐的刘海
    *   **面部表情**: 面带微笑，表情自然平和，直视镜头
*   **着装细节**:
    *   上身穿着**深色格纹衬衫**（深绿/黑色调的格子图案）
    *   衬衫领口为标准的翻领设计
    *   整体着装风格偏向**商务休闲** (Business Casual)，符合学术场合的正式但不过分严肃的要求
*   **摄影参数与构图**:
    *   **背景**: 纯白色或极浅灰色背景，典型的证件照/学术简介照背景处理方式，突出主体
    *   **构图**: 标准的**胸上至头部** (Bust Shot) 构图，人物居中对称
    *   **光线**: 正面柔光照明，面部光线均匀，无明显阴影，属于专业的**影棚人像布光**
    *   **画质**: 清晰度高，色彩还原自然，符合学术出版物对作者照片的质量要求
*   **学术身份关联**:
    *   根据论文信息，该照片对应作者简介中的 **Seock-Hwan Noh**
    *   **教育背景**: 2019年于 **Ulsan National Institute of Science and Technology (UNIST)** 获得电气工程学士学位
    *   **当时就读院校**: **Daegu Gyeongbuk Institute of Science and Technology (DGIST)** 电气工程与计算机科学系，攻读综合硕士及博士学位
    *   **研究方向**: 深度学习训练加速器 (DNN training accelerators) 与安全相关的电路设计 (circuit design for security)
*   **图像用途推断**:
    *   用于 IEEE 等学术出版物的**作者简介** (Author Biography) 配图
    *   可能用于实验室官网、学术会议演讲稿或个人学术主页
    *   符合 IEEE 出版物对作者照片的标准化要求（白底、正装、清晰正面照）

### aa33be7ca4efb99b7cc2743eeeca737b45e306c656a81e592e39e82eb49b05db.jpg

![aa33be7ca4efb99b7cc2743eeeca737b45e306c656a81e592e39e82eb49b05db.jpg](images/aa33be7ca4efb99b7cc2743eeeca737b45e306c656a81e592e39e82eb49b05db.jpg)

- **图像类型**: 专业学术肖像照 / 作者简介照片
- **拍摄对象**: **Jahyun Koo** (本文第二作者, Student Member, IEEE)

- **外貌特征**: 
  - 亚洲男性，年龄约20-25岁
  - 佩戴**圆形金属细框眼镜**
  - 黑色短发，发型整齐利落，侧分造型
  - 面部轮廓清晰，表情**沉稳专业**

- **着装规范**:
  - 上装：**深灰色/炭灰色西装外套** (质地看似为羊毛混纺)
  - 内搭：**浅蓝色正装衬衫** (领口整洁，未系领带)
  - 整体风格：**商务休闲 (Business Casual)**，符合学术界年轻研究者形象

- **拍摄参数**:
  - 背景：**纯白色/浅灰白色** 无缝背景 (Studio Backdrop)
  - 光线：**正面柔光照明**，面部光线均匀，无强烈阴影
  - 构图：**胸部以上半身像 (Bust Shot)**，居中构图
  - 视角：平视角度 (Eye-level Shot)

根据论文作者简介部分，该照片对应以下学术身份：

| 属性 | 详细信息 |
|------|----------|
| **学历** | 2020年获庆北国立大学 (Kyungpook National University) 学士学位 |
| **当前状态** | 大邱庆北科学技术院 (**DGIST**) 电气工程与计算机科学系<br>集成硕士+博士项目在读 (Integrated Master's & PhD) |
| **研究方向** | **Energy-efficient hardware accelerators for deep learning**<br>(面向深度学习的节能硬件加速器) |
| **学术身份** | IEEE 学生会员 |

- **技术规格**: 分辨率适中 (约 200×250 像素)，符合数字出版物标准
- **色彩表现**: 色彩还原准确，肤色自然，服装颜色真实
- **适用场景**: 学术论文作者栏、会议演讲者介绍、研究团队主页
- **专业程度**: ★★★★☆ (标准的学术机构官方肖像照水准)

该图像位于论文末尾 **Authors' Biography** 部分，用于：
- 建立作者学术身份的可信度
- 方便读者/审稿人识别研究者
- 体现研究团队的正规性与专业性
- 符合 IEEE 出版物的作者展示规范

### 0001b06bcf615da6163dd6a540a6abcf595d3acc3c1669195a595af540306871.jpg

![0001b06bcf615da6163dd6a540a6abcf595d3acc3c1669195a595af540306871.jpg](images/0001b06bcf615da6163dd6a540a6abcf595d3acc3c1669195a595af540306871.jpg)

- **图像基本信息**
  - **类型**: 人物肖像照片 (Portrait Photograph)
  - **主体**: 论文第三作者 **Seunghyun Lee** (Student Member, IEEE)
  - **拍摄风格**: 正式学术头像照，背景为纯白色/浅灰色
  - **着装**: 浅蓝色衬衫 (Light Blue Shirt)，衣领整洁
  - **面部特征**: 东亚面孔，黑色短发，面带微笑

- **作者身份背景**
  - **当前学历**: 正在攻读 **Integrated Master's & PhD Degrees** (硕博连读)
  - **就读院校**: **Daegu Gyeongbuk Institute of Science and Technology (DGIST)**, 大邱, 韩国
  - **专业领域**: Electrical Engineering and Computer Science (电子工程与计算机科学)
  - **学士学位**: 2020年获得于 **DGIST**

- **研究兴趣方向**
  - **核心领域1**: **Block Floating-Point Optimization on Deep Learning** (深度学习中的块浮点优化)
    - 这与论文主题 **FlexBlock** 直接相关，该论文正是关于支持多模式块浮点的DNN训练加速器
  - **核心领域2**: **Neural Architecture Search** (神经架构搜索)
    - 涉及自动化设计神经网络结构的算法研究

- **在论文中的贡献角色**
  - 作为 **第三作者** 参与了 FlexBlock 加速器的研发工作
  - 研究重点聚焦于 **BFP (Block Floating Point)** 数据表示法在深度学习训练中的硬件优化
  - 该工作涉及 **低精度训练 (Low-Precision Training)** 和 **精度可扩展性 (Precision Scalability)** 等前沿技术

- **图像质量特征**
  - **分辨率**: 标准学术期刊要求的头像尺寸
  - **光线**: 均匀正面照明，无明显阴影
  - **构图**: 胸以上半身像，居中构图，符合 IEEE 学术出版物标准格式

### 30070ac1fa597f3d9430bc7601715a1a80b45a01e3d97b01984c742e4c9b7ec4.jpg

![30070ac1fa597f3d9430bc7601715a1a80b45a01e3d97b01984c742e4c9b7ec4.jpg](images/30070ac1fa597f3d9430bc7601715a1a80b45a01e3d97b01984c742e4c9b7ec4.jpg)

- **图像基本信息**
  - **类型**：人物肖像照（证件照风格）
  - **主体**：男性，亚洲面孔
  - **背景**：浅色（白色或浅灰色）纯色背景
  - **着装**：蓝色圆领上衣（T恤或Polo衫）

- **人物特征描述**
  - **发型**：黑色短发，整齐的刘海，发型较为蓬松自然
  - **面部特征**：
    - 脸型偏圆/椭圆
    - 眉毛浓密且平直
    - 眼睛为深棕色，双眼皮
    - 鼻梁挺直
    - 嘴唇厚度适中
    - 下颌线条清晰
  - **表情**：中性表情，面部放松，直视镜头

- **图像技术参数**
  - **拍摄角度**：正面平视拍摄
  - **光线**：正面均匀照明，无明显阴影
  - **构图**：头部和肩部特写（Headshot），居中构图
  - **图像质量**：分辨率适中，色彩还原正常

- **身份关联信息**
  - 根据论文文档，此照片对应作者 **Seunghyun Lee**
  - **学术身份**：IEEE 学生会员 (Student Member, IEEE)
  - **教育背景**：
    
    | 学位 | 学校 | 时间 |
    |------|------|------|
    | BS | DGIST (大邱庆北科学技术院) | 2020年 |
    | 在读硕博连读 | DGIST | 进行中 |
  
  - **研究方向**：
    - **Block floating-point optimization on deep learning**（深度学习中的块浮点优化）
    - **Neural architecture search**（神经架构搜索）

- **在论文中的角色**
  - 作为 **FlexBlock** 论文的第三作者
  - 主要贡献领域涉及 **BFP (Block Floating Point)** 格式优化与 DNN 训练加速器设计
  - 该论文提出了支持多模式 BFP 的灵活 DNN 训练加速器架构

### e82042ea5558a2b57f2ee6ef3a555e1cc5d1f14e1544f0f911eba5a76a46255c.jpg

![e82042ea5558a2b57f2ee6ef3a555e1cc5d1f14e1544f0f911eba5a76a46255c.jpg](images/e82042ea5558a2b57f2ee6ef3a555e1cc5d1f14e1544f0f911eba5a76a46255c.jpg)

- **图像基本信息**
    - 图像类型：**人物肖像照（Author Photo）**
    - 图像主体：**Jaeha Kung**（本文通讯作者之一）
    - 视觉特征：男性，亚洲面孔，穿着**蓝色针织毛衣**搭配浅色衬衫，背景为简洁的浅灰色/白色，呈现专业的学术形象

- **作者学术背景**（基于论文元数据）
    
    | 属性 | 详细信息 |
    |------|----------|
    | **当前职位** | Associate Professor, School of Electrical Engineering, **Korea University** |
    | **IEEE身份** | Member, IEEE |
    | **教育经历** | <ul><li>**BS**: Electrical Engineering, Korea University (2010)</li><li>**MS**: Electrical Engineering, KAIST (2012)</li><li>**PhD**: ECE, Georgia Institute of Technology (2017)</li></ul> |
    | **研究领域** | Digital accelerators for emerging applications, **efficient AI**, system design for artificial intelligence, **approximate computing**, low-power VLSI design |

- **在 FlexBlock 研究中的角色**
    - 作为**资深作者（Senior Author）**指导研究方向
    - 研究聚焦于**高效AI加速器设计**，与 FlexBlock 的核心目标（灵活DNN训练加速器）高度契合
    - 在**近似计算（Approximate Computing）**和**低功耗VLSI设计**方面的专业知识为 FlexBlock 的**多精度BFP支持**和**能效优化**提供了理论基础

- **学术服务与影响力**
    - 曾担任多个顶级会议的技术程序委员会成员：
        - **IEEE DAC** (Design Automation Conference)
        - **IEEE/ACM ISLPED** (International Symposium on Low Power Electronics and Design)
        - **ISCAS** (International Symposium on Circuits and Systems)
        - **AICAS** (IEEE International Conference on Artificial Intelligence Circuits and Systems)

- **图像功能定位**
    - 该图像位于论文**作者简介（Author Biography）**部分
    - 用于建立作者学术身份的可信度
    - 符合 IEEE 学术出版物对作者肖像的标准规范（正式着装、中性背景、高清晰度）

