# Efficient Precision-Scalable Hardware for Microscaling (MX) Processing in Robotics Learning 图表详解

### Fig. 1. Overview of goals, challenges, and our contributions: (1) precisionscalable MX MAC unit supporting all six MX formats combining INT and FP operations; (2) 64-element square MX groups for training at the edge and the design of a square-based MX PE Array for GeMM Core.

![8d3a2057e591e4c7ba4b00b8028ad45e75f87df5e841485a5238094ec30f15ab.jpg](images/8d3a2057e591e4c7ba4b00b8028ad45e75f87df5e841485a5238094ec30f15ab.jpg)

- **图像整体结构分析**
    - 该图为论文核心思想的**三维概览图**，采用三栏式布局，从左至右依次呈现研究目标、面临的技术挑战以及提出的创新解决方案。
    - 视觉编码采用**色彩分区法**：左侧 Goals 使用蓝色系代表理想状态；中间 Challenges 使用红/粉色调突出瓶颈；右侧 Solutions 采用黄色系展示技术实现路径。

- **Goals 栏（研究动机与目标）**
    - **应用场景定位**：明确指向 **On-device Learning**（边缘端学习）与 **Environment Adaptation**（环境自适应），配以机器人导航图标，强调自主机器人无需云依赖的实时学习能力。
    - **核心设计目标**：在 **accuracy & efficiency**（精度与效率）之间取得平衡。
    - **技术基础**：基于 **OCP MX Format**（开放计算项目微缩放格式），具体涵盖 **MXINT**（整数型）和 **MXFP**（浮点型）两大类数据表示。

- **Challenges 栏（现有技术瓶颈）**
    - **数据格式多样性难题**：列出 **MXINT8、MXFP6、MXFP4** 等多种格式，指出存在 **Diverse mantissa lengths (1-8 bits)**（尾数长度从1位到8位不等）的问题。
    - **混合精度计算复杂性**：明确标注 **INT/FP mixed operations**（整数与浮点混合运算）带来的硬件设计挑战。
    - **传统向量分组（Vector Block）的架构缺陷**：
        - 图示展示了32元素向量块（A₁...A₃₂ × W₁...W₃₂）的传统组织方式。
        - **存储冗余问题**：在反向传播（Backpropagation）阶段，由于矩阵转置需求，传统方法需要 **Training with Vector block: 2×W**（存储两份权重），导致 **Memory** 开销翻倍。
        - **资源约束矛盾**：在 **Limited resources at edge**（边缘资源受限）条件下，这种存储浪费成为关键瓶颈。

- **Our MX Solutions 栏（双层创新方案）**

| 创新层级 | 技术名称 | 核心机制 | 图示细节 |
|---------|---------|---------|---------|
| **Arithmetic Unit Level**<br>（算术单元层） | **Precision-Scalable MX MAC Unit**<br>（精度可扩展MX乘加单元） | 通过子字并行（Sub-word Parallelism）统一整数与浮点处理 | - 展示完整的MAC数据通路<br>- 包含 **L1 Adder**（一级加法器）、**L2 Adder**（二级加法器）、**FP Accumulation Addition**（浮点累加）<br>- 支持动态重构以适应不同位宽（8/6/4-bit） |
| **Architecture Level**<br>（架构层） | **Square-based MX PEs & GeMM Core**<br>（基于方块的MX处理阵列与通用矩阵乘法核心） | 用64元素（8×8）方块替代32元素向量，消除转置存储冗余 | - **Square Blocks** 结构：A 和 W 均按8×8方块组织，共享指数（Shared Exponent）<br>- **PE Array**：8×8 的处理单元阵列，支持对称计算<br>- **GeMM Core**：4×16 的PE阵列网格配置<br>- **On-chip memory**：集成片上存储层次 |

- **关键技术突破点的视觉对比**
    - **从线性到二维的范式转变**：Challenges 栏中的 **Vector Block**（一维32元素向量）与 Solutions 栏中的 **Square Block**（二维8×8矩阵）形成鲜明对比，直观解释了如何通过空间重组解决 **2×W** 的存储问题。
    - **精度与吞吐量的协同优化**：Solutions 栏右侧的 GeMM Core 示意图暗示了不同精度模式下的 **throughput scaling**（吞吐量缩放）：FP4 模式下单周期完成8×8方块计算，而 INT8 模式需8个周期，体现了 **precision-throughput tradeoff**（精度-吞吐量权衡）的硬件实现。

- **与论文正文的技术映射关系**
    - 该图是全文的 **visual abstract**（可视化摘要），直接对应第三章（Arithmetic Unit Innovations）和第四章（Architectural Innovations）的核心内容。
    - 图中 **Precision-Scalable MAC** 的内部结构（L1/L2加法器、累加器）与论文 Fig. 3、Fig. 4 的详细电路图形成宏观-微观的对应关系。
    - **Square-based** 设计理念贯穿于 Fig. 5（计算图对比）和 Fig. 6（PE Array 详细架构），构成了相对于 SotA（Dacapo 处理器）的 **key differentiator**（关键差异化优势）。

### Fig. 2. Validation loss curves of the concrete MX data types on 4 robotic training workloads compared with the FP32 baseline, showing that FP32 can be replaced by low-bit MX types towards robotics learning.

![7f461e7677d1fdf402a14494c4f293a1fae2dc8cb2c67543ec22ce19f519521e.jpg](images/7f461e7677d1fdf402a14494c4f293a1fae2dc8cb2c67543ec22ce19f519521e.jpg)

- **图表概览**
  - 图表类型：多子图折线图（Line Plot with Subplots），共4个子图
  - 坐标轴定义：X轴为Training Epochs（训练轮次），Y轴为Validation Loss（验证损失，对数尺度）
  - 对比维度：7种数据格式的训练收敛曲线对比

- **实验工作负载（Workloads）**
  | 子图位置 | 任务名称 | 任务类型 | 训练轮次范围 |
  |---------|---------|---------|------------|
  | 左上 | Halfcheetah | 运动控制/平衡 | 0-40 |
  | 右上 | Reacher | 机械臂到达任务 | 0-160 |
  | 左下 | Cartpole | 倒立摆平衡 | 0-200 |
  | 右下 | Pusher | 物体推挤交互 | 0-200 |

- **数据格式对比（Data Formats）**
  | 格式名称 | 位宽 | 特征 | 线型/颜色 |
  |---------|------|------|----------|
  | **FP32** | 32-bit | 基线浮点 | 蓝色实线 |
  | **MXINT8** | 8-bit | 纯整数 | 橙色虚线 |
  | **MXFP8(E5M2)** | 8-bit | 宽动态范围(5exp+2mant) | 红色虚线 |
  | **MXFP8(E4M3)** | 8-bit | 高精度尾数(4exp+3mant) | 绿色虚线 |
  | **MXFP6(E3M2)** | 6-bit | 中等配置 | 棕色虚线 |
  | **MXFP6(E2M3)** | 6-bit | 宽尾数 | 紫色虚线 |
  | **MXFP4(E2M1)** | 4-bit | 极低比特 | 粉色虚线 |

- **关键发现与性能分析**

  **1. 任务依赖的精度偏好（Task-Specific Precision Preference）**
  - **平衡类任务（Halfcheetah, Cartpole）**：**MXINT8** 表现优异，验证损失接近甚至匹配FP32基线
    - Halfcheetah: MXINT8最终损失约-3.5 vs FP32的-5
    - Cartpole: MXINT8最终损失约-7.5 vs FP32的-9.5
  - **交互类任务（Reacher, Pusher）**：**MXFP8(E4M3)** 占据优势地位
    - Reacher: MXFP8(E4M3)达到约-10，显著优于其他低比特格式
    - Pusher: MXFP8(E4M3)与FP32差距最小（约-11 vs -11.5）

  **2. 精度-性能权衡层次（Precision-Accuracy Hierarchy）**
  - **第一梯队（接近FP32）**：MXFP8(E4M3)、MXFP8(E5M2)、MXINT8
  - **第二梯队（可接受退化）**：MXFP6系列（E2M3/E3M2），损失比FP32高1-3个单位
  - **第三梯队（显著退化）**：**MXFP4(E2M1)** 在所有任务中均表现最差，损失值比FP32高4-8个单位

  **3. 收敛特性分析（Convergence Behavior）**
  - **早期收敛（0-50 epochs）**：所有格式均呈现快速下降趋势，MXFP4初始下降较慢
  - **平台期表现**：
    - FP32持续缓慢优化至最低损失
    - MXFP8(E4M3)在Reacher/Pusher任务中展现稳定收敛
    - MXFP4(E2M1)在所有任务中过早进入高损失平台期（early plateau）
  - **稳定性**：MXINT8在Cartpole任务中曲线最平滑，无明显振荡

- **工程意义与设计启示**

  **内存-精度最优配置策略**
  - 对于**运动控制/平衡类**机器人应用 → 优先选择 **MXINT8**（零浮点开销，整数运算高效）
  - 对于**精细操作/交互类**机器人应用 → 推荐 **MXFP8(E4M3)**（兼顾动态范围与尾数精度）
  - **避免使用MXFP4**进行端到端训练，除非配合特殊训练技术（如论文后续提到的oscillation-reduced training）

  **硬件设计指导价值**
  - 该图验证了**precision-scalable（精度可扩展）**硬件设计的必要性：单一固定精度无法适配所有机器人学习场景
  - 支持**动态切换**INT8/FP8/FP6的MAC单元（如论文Fig. 3所示）可根据任务特性实时优化能效比
  - MXFP4虽训练效果差，但在**推理阶段**可能仍有价值（图中未展示 inference 性能）

- **数据完整性备注**
  - Y轴采用**对数尺度或非线性映射**（数值范围为-12至+2），便于观察不同量级损失的差异
  - 所有曲线基于相同网络架构和超参数，仅改变数据表示格式
  - 训练轮次因任务复杂度不同而异（Halfcheetah仅需40 epochs，其他任务需160-200 epochs）

### Fig. 3. The precision-scalable MAC unit in the 3 scaling modes: (a) INT8, (b) FP8/FP6, and (c) FP4, respectively. The multiplication is indicated in yellow, the L1 adder in purple, the L2 adder in red, the FP accumulation addition in orange, and the accumulation register in green.

![3e1a74d22c4d884b22697dbd805305cef66817745d973ad5a9f568cb2e984034.jpg](images/3e1a74d22c4d884b22697dbd805305cef66817745d973ad5a9f568cb2e984034.jpg)

- **图像基本信息**
    - 标识符：Fig. 3，展示 **precision-scalable MAC unit**（精度可扩展乘累加单元）的三种工作模式架构
    - 来源：论文《Efficient Precision-Scalable Hardware for Microscaling (MX) Processing in Robotics Learning》
    - 核心功能：通过硬件复用支持 MX 格式的多种数据类型（INT8、FP8/FP6、FP4）

- **颜色编码与组件映射**

| 颜色 | 组件符号 | 功能描述 |
|------|----------|----------|
| **黄色** | X (2-bit) | 基础 **2-bit 乘法器**，共 16 个，作为计算基本单元 |
| **紫色** | + (L1) | **Level-1 加法器**，负责部分积的初步累加 |
| **红色** | + (L2) | **Level-2 加法器**，执行高精度或跨模式累加 |
| **橙色** | + (FP Acc) | **FP 累加加法器**，处理浮点格式的最终累加 |
| **绿色** | 寄存器 | **累加寄存器**，存储中间和最终结果 |

- **模式 (a)：INT8 模式详细分析**
    - **计算资源分配**：全部 **16 个 2-bit 乘法器** 联合工作，通过级联方式计算单个 **INT8 × INT8 = 16-bit** 乘积
    - **数据流路径**：
        - 输入 INT8 操作数被分解为多个 2-bit 片段，分别送入 16 个乘法器
        - **L1 加法器**（紫色）对 4-bit 整数部分积进行带移位累加（<<2, <<4 移位操作可见）
        - **L2 加法器**（红色）进一步合并 L1 输出，形成完整 16 位结果
        - 最终经 **FP 累加加法器**（橙色）写入 **累加寄存器**（绿色）
    - **指数处理**：此模式下 **指数加法器保持非激活状态**（纯整数运算）

- **模式 (b)：FP8/FP6 模式详细分析**
    - **并行度提升**：同一硬件被动态重配置为处理 **4 个并行的 FP8/6 × FP8/6 乘积**
    - **资源分组策略**：
        - 每 **4 个 2-bit 乘法器** 组合处理一个浮点数的尾数乘法
        - 配套 **5-bit 专用指数加法器**（图中隐含于 FP8/FP6 标签区域）
    - **层次化累加机制**：
        - **L1 加法器**：累加 4-bit 浮点部分积，执行适当移位对齐
        - **L2 加法器**：采用 **FP32 加法**（26-bit 尾数加法器）处理大范围指数（±30 for E5M2），避免 78-bit 整数加法的面积开销
        - **非规格化数处理**：通过扩展内部尾数位宽 2 bit 替代昂贵的归一化电路

- **模式 (c)：FP4 模式详细分析**
    - **配置变更**：每个激活的 **2-bit 乘法器** 与 **2-bit 指数加法器** 配对，构成独立 **FP4 处理单元**
    - **带宽约束下的并行度**：
        - 理论最大并行度为 16，但受限于 **带宽一致性**（与 FP8/6 模式保持相同带宽）
        - 实际激活 **8 个并行 FP4 × FP4 通路**，对应 8×8 共享指数块的累加需求
        - 图中 **灰色淡化部分** 表示非激活的乘法器和加法器（资源闲置以匹配 I/O 带宽）
    - **L1 加法器特殊优化**：直接对 4 个完整的 E3M4 乘法结果进行 **指数差值移位相加**，利用 E3M4 指数范围有限（0-4）的特性，**无需预先查找最大指数**

- **关键路径与旁路优化**
    - **模式特定延迟平衡**：
        - **INT8 模式**：关键路径在 **L1 加法器**（符号-幅值转换开销），但可 **旁路 L2 的对齐逻辑**
        - **FP4 模式**：关键路径在 **L1 加法器**（变量移位器延迟），同样 **旁路 L2 对齐逻辑**
        - **FP8/FP6 模式**：必须经过 **L2 对齐逻辑**，但 L1/L2 路径较短
    - **综合工具实现**：通过模式特定的 **bypass 多路选择器**，使三种模式的 **关键路径长度趋于均衡**，在 400MHz 时钟频率下稳定工作

- **架构创新价值总结**
    - **硬件复用率**：单一物理阵列通过 **子字并行性**（sub-word parallelism）支持 6 种 MX 数据类型（MXINT8, MXFP8-E5M2, MXFP8-E4M3, MXFP6-E3M2, MXFP6-E2M3, MXFP4-E2M1）
    - **能效优势**：相比基于 1-bit 处理单元的设计（如 Flexibit），**2-bit 基础粒度** 减少了加法器级联层级，降低动态功耗
    - **统一数据处理**：**消除整数/浮点运算的硬件隔离**，通过 L1/L2 分层结构和模式旁路实现 INT/FP 统一 datapath

### cd2e43c9b1b59adedcbcf2de4d09eaa68569e281b139b5f7e85b2054c88eae4b.jpg

![cd2e43c9b1b59adedcbcf2de4d09eaa68569e281b139b5f7e85b2054c88eae4b.jpg](images/cd2e43c9b1b59adedcbcf2de4d09eaa68569e281b139b5f7e85b2054c88eae4b.jpg)

- **图像整体概述**
    - 该图为论文中 **Fig. 4**，展示了**层次化 MX 累加器 (Hierarchical MX Accumulator)** 的核心架构，分为 **(a) L1 Adder** 和 **(b) L2 Adder** 两部分。
    - 该设计是 **Precision-Scalable MX MAC Unit** 的关键组件，旨在**统一处理所有六种 MX 数据格式**（MXINT8, MXFP8 E5M2/E4M3, MXFP6 E3M2/E2M3, MXFP4 E2M1）的累加操作。

- **L1 Adder 详细分析（左半部分）**
    - **功能定位**：执行**第一级累加**，负责将多个并行乘法的部分积进行初步合并。
    - **多精度输入支持**：
        - 接收来自不同精度模式的乘法结果：**FP8、INT8、FP4、FP6** 等格式。
        - 图中标注了不同的元素格式标识：**M4（4位尾数）、E3（3位指数）、M3、E2** 等。
    - **核心计算路径**：
        - **Exp Shifter（指数移位器）**：根据共享指数对尾数进行移位对齐。图中显示多个并行的移位单元，分别处理不同格式的数据。
        - **位宽变化**：输入数据位宽动态变化，如 **2b:8b**（2位扩展至8位）、**8b**、**10b** 等，体现了**子字并行 (Sub-word Parallelism)** 特性。
    - **针对 FP4 的特殊优化**：
        - 对于 **FP4 (E3M4)** 模式，L1 加法器直接对**四个完整的乘法结果**求和。
        - 利用 **E3M4 指数范围有限（0-4）** 的特点，通过**直接移位尾数**的方式对齐，**无需先查找最大指数**，显著简化了硬件逻辑。

- **L2 Adder 详细分析（右半部分）**
    - **功能定位**：执行**第二级累加**，将 L1 的输出进行最终汇总，生成符合精度要求的累加结果。
    - **核心模块组成**：
        - **Alignment（对齐单元）**：
            - 处理 **FP8/FP6** 模式下的大范围指数对齐。
            - 输入位宽包括 **10b**（来自 INT8/FP4 路径）和 **26b**（来自 FP8/FP6 路径）。
            - 关键创新：对于 FP8/FP6 模式，**避免了使用 78 位整数加法器**（这是传统精确加法方式的瓶颈），转而采用 **FP32 加法** 方案。
        - **26b:2b 加法器**：
            - 执行 **26 位尾数加法**（另有 2 位扩展位），相比 78 位整数加法大幅节省面积。
        - **Normalize（归一化单元）**：
            - 对结果进行归一化处理，输出 **32 位** 数据（兼容 FP32 精度存储）。
            - 输入包含最大指数信息 **(E largest) 6b**。
    - **非规格化数处理策略**：
        - 针对窄位浮点格式中常见的**非规格化数 (Subnormal Numbers)** 问题，采用**仅扩展内部尾数位宽 2 位**的策略，而非实现复杂的归一化电路。
        - 该方案在**保证 FP32 级精度**的同时，最小化了硬件开销。

- **跨级协同与关键路径优化**
    - **模式感知旁路 (Mode-Specific Bypassing)**：
        - **INT8 模式**和 **FP4 模式**在 L1 加法器中具有较长的关键路径（分别由于符号-幅度转换和可变移位器）。
        - 这两种模式可以**旁路 (Bypass) L2 加法器中的对齐逻辑**（该逻辑仅为 FP8/FP6 所需）。
        - 这种**条件旁路策略**平衡了不同精度模式下的关键路径长度，使整体设计能够在 **400MHz** 频率下稳定工作（TSMC 16nm 工艺）。
    - **硬件复用**：
        - 同一套整数加法器硬件通过配置服务于所有精度模式，仅在 FP4 模式下需要 **2 位扩展**，实现了极高的**面积效率 (Area Efficiency)**。

- **设计意义总结**
    | 设计特性 | 技术方案 | 带来的优势 |
    | :--- | :--- | :--- |
    | **统一架构** | L1/L2 分级加法器 | 支持 INT/FP 混合运算，覆盖全部 6 种 MX 格式 |
    | **FP8/FP6 优化** | FP32 尾数加法替代 78-bit 整数加法 | 大幅减少面积和功耗 |
    | **非规格化数处理** | 尾数扩展 2-bit | 避免复杂归一化电路，保证计算精度 |
    | **关键路径平衡** | 模式相关旁路逻辑 | 在多精度模式下均能维持高工作频率 |
    | **FP4 高效支持** | L1 直接移位求和 | 利用指数范围小的特点简化对齐逻辑 |

### Fig. 4. The L1 and L2 adders, the INT8 path is indicated in brown, FP8/FP6 in red and FP4 in blue. (a) Vector Blocks (b) Square Blocks Fig. 5. Training computation graphs with MX vector and square block quantizers. Computational and memory footprint gains can be achieved for square blocks, by enabling storage of block-quantized parameters for backpropagation.

![e307d006271caa90f4684aec543fc87f4d1ea11ffc04be9d2d05ef04e52ff986.jpg](images/e307d006271caa90f4684aec543fc87f4d1ea11ffc04be9d2d05ef04e52ff986.jpg)

- **图像主题与定位**：该图为 **Fig. 5**，标题为 *"Training computation graphs with MX vector and square block quantizers"*，核心对比 **Vector Blocks（向量块）** 与 **Square Blocks（方形块）** 两种 MX 量化策略在神经网络训练（尤其是反向传播阶段）的计算图结构与内存效率差异。

- **子图 (a) Vector Blocks 分析**：
    - **计算图结构**：展示标准训练流程，包含前向传播 ($A_{L-1} \times W_L \rightarrow A_L$) 与反向传播（梯度计算 $A_{L-1}^T \times E_L$ 及权重梯度 $E_L \times W_L^T$）。
    - **关键瓶颈**：
        - 需要**两套独立的量化器**：$Q_{row}(L)$ 用于行方向（处理 $W_L$），$Q_{col}(R)$ 用于列方向（处理 $W_L^T$）。
        - **存储冗余**：由于向量块（32 元素一维结构）在转置后共享指数分布改变，必须同时存储**原始量化权重**与**转置后重新量化的权重**（图中 "Backprop memory" 星标处暗示额外内存需求）。
        - **重量化开销**：反向传播时需对转置矩阵执行实时 requantization 或维护双份存储，增加计算与访存成本。
    - **底部示意**：可视化 32 元素的一维向量块结构，指数 (exp) 沿单一维度共享。

- **子图 (b) Square Blocks 分析**：
    - **计算图结构**：相同的训练算子，但统一使用单一量化器 $Q_s(L/R)$。
    - **核心创新与优势**：
        - **消除存储冗余**：采用 **8×8 的二维方形块**（64 元素），其结构在行列转置后保持对称性，**无需存储两份权重**。
        - **消除重量化开销**：通过硬件转置操作 (**T**) 直接复用同一组量化参数，避免反向传播阶段的 requantization 计算。
        - **右侧明确标注**："**Remove storage redundancy & requantization overhead**"（移除存储冗余与重量化开销）。
    - **底部示意**：展示 8×8 方形块的二维网格结构，FP/INT 数据在方块内共享指数，天然支持行列双向访问。

- **技术意义总结**：

| 维度 | Vector Blocks (a) | Square Blocks (b) | 改进幅度 |
| :--- | :--- | :--- | :--- |
| **块结构** | 32 元素 (1×32 向量) | 64 元素 (8×8 矩阵) | 二维化 |
| **量化器数量** | 2 套 ($Q_{row}$ + $Q_{col}$) | 1 套 ($Q_s$) | **减少 50%** |
| **权重存储** | 需存 $W$ 和 $W^T$ 两版 | 仅需存一版（转置复用） | **节省 ~51%** (见表 III) |
| **反向传播开销** | 需要 Requantization 或双倍内存 |仅需硬件 Transpose | **显著降低延迟与能耗** |

- **与论文贡献的关联**：此图直观论证了第 IV 节 **Architectural Innovations** 中提出的 **Square-based MX PE Array** 的必要性。通过将 MX 标准的 32 元素向量块扩展为兼容标准的 64 元素 (8×8) 方形块，解决了现有 SotA 处理器（如 Dacapo）在持续学习场景下**内存占用过高**与**反向传播效率低下**的关键痛点，是实现边缘机器人高效 on-device training 的架构级基石。

### Fig. 6. The PE Array that handles the multiplication of two 64-element square blocks, for INT8 this is executed in 8 cycles, for FP8/FP6 this takes 2 cycles, and for FP4 in 1 cycle. The PE Array is part of our GeMM core of 4x16 PE Arrays.

![48b04174a270c3c4ba05580d601e6c6e51a0f88fda297d9589cf5019f8ae052d.jpg](images/48b04174a270c3c4ba05580d601e6c6e51a0f88fda297d9589cf5019f8ae052d.jpg)

基于提供的论文内容和图表（Fig. 6），以下是对该硬件架构设计的深入分析：

该图展示了论文提出的 **Microscaling (MX) GeMM Core** 的核心架构，分为三个主要部分：
- **顶部**：展示了不同精度模式下（INT8, FP8/FP6, FP4）的计算调度与数据流。
- **左下**：展示了基本的 **PE Array (Processing Element Array)** 微架构，即一个 8×8 的乘累加（MAC）单元阵列。
- **右下**：展示了完整的 **GeMM Core** 宏观布局，包含多个 PE Array、片上存储器（On-chip memory）以及用于训练的辅助单元（Quantizer, Weight Update）。

这部分直观地展示了硬件如何通过 **Precision-Scalable（精度可扩展）** 设计来适配不同的 MX 数据格式，从而在低精度下获得更高的吞吐量。

*   **计算目标**：两个 8×8 的矩阵块（Block A 和 Block B）相乘得到结果 O。这对应于论文中提出的 **64-element square blocks（64元素方块组）** 概念。
*   **模式对比**：

| 数据模式 | 执行周期 | 并行度特征 | 视觉解读 |
| :--- | :--- | :--- | :--- |
| **INT8** | **8 cycles** | 低并行度 | 计算被分散在多个周期内完成（图中显示为分步处理），适用于需要较高精度的整数运算。 |
| **FP8/FP6** | **2 cycles** | 中等并行度 | 利用浮点格式的特性增加了并行处理能力（图中红色区域增大），显著缩短了延迟。 |
| **FP4** | **1 cycle** | 高并行度 | 充分利用位宽极窄时的 **sub-word parallelism（子字并行）**，在单个时钟周期内完成整个 8×8 块的计算（图中几乎全红）。 |

*   **关键洞察**：这种设计使得硬件在处理低精度数据（如 MXFP4）时，有效训练吞吐量呈指数级提升，解决了边缘设备算力不足的问题。

这是构成整个加速器的基本计算单元，专门针对 **Square-based MX（基于方块的MX）** 格式进行了优化。

*   **拓扑结构**：一个 **8×8 的 MAC 阵列**，共包含 64 个 MAC 单元。
*   **数据流设计**：
    - **输入**：采用 **Row Vector(s) of B**（B的行向量）和 **Column Vector(s) of A**（A的列向量）作为输入。这种设计直接支持矩阵乘法的点积运算。
    - **计算逻辑**：这 64 个 MAC 单元协同工作，处理一个 64 元素的方块组。正如论文所述，这种方形分组（8×8）取代了传统的 32 元素向量分组。
*   **优势**：方形结构使得在前向传播和反向传播（需要转置权重）时，无需重新量化或存储两份权重数据，从而大幅节省了 SRAM 存储开销。

该部分展示了如何将上述 PE Array 组合成一个完整的、支持端侧训练的通用矩阵乘法（GeMM）核心。

*   **阵列规模**：由 **4 × 16** 个 PE Array 组成（注：图中的每个 "PE" 方块代表左下角那个 8×8 的阵列）。
    - **维度选择依据**：
        - **高度 (×4)**：对应于 Batch Size 32 除以方块维度 8，适配边缘设备常用的小批量训练。
        - **宽度 (×16)**：在内存带宽和计算并行度之间取得的平衡。
*   **存储层次 (On-chip memory)**：
    - **Weights/Error buffer**：存储权重和误差梯度。
    - **Activation/Error buffer**：存储激活值和反向传播的误差。
    - 这种分离的缓冲区设计支持 **Output-Stationary（输出驻留）** 数据流，减少中间数据的频繁搬运，降低能耗。
*   **训练专用模块**：
    - **Quantizer（量化器）**：负责将 FP32 的梯度或权重量化回 MX 格式，这是端侧训练流程中的关键步骤。
    - **Weight Update（权重更新）**：执行 SGD 等优化算法，更新模型参数。

结合论文描述，Fig. 6 所示的设计带来了两方面的核心提升：
1.  **存储效率提升 (51% reduction)**：得益于 **8×8 Square Blocks** 设计，消除了传统向量分组在反向传播时的存储冗余。
2.  **吞吐量提升 (4× higher throughput)**：得益于 **Precision-Scalable PE Array**，在 FP4 等低精度模式下，通过单周期爆发式计算，极大缩短了训练时间。

该架构成功地在受限的边缘设备资源下，实现了支持全六种 MX 数据类型的、高效的机器人持续学习（Continual Learning）硬件加速。

### 6c8a254d8ec64f81f0ae684673ae53b84d921a04cba8cbc517909e255f44faed.jpg

![6c8a254d8ec64f81f0ae684673ae53b84d921a04cba8cbc517909e255f44faed.jpg](images/6c8a254d8ec64f81f0ae684673ae53b84d921a04cba8cbc517909e255f44faed.jpg)

根据提供的论文内容和图表（图7），以下是对该精度可扩展 MX MAC 单元及 PE Array 的详细分析：

*   **整体图表构成**
    该图由三个核心部分组成，旨在量化评估所提出的 Precision-Scalable MX MAC 单元及其组成的 PE Array 的硬件开销：
    *   **顶部表格**：对比了三种不同设计变体（Design Variants）在面积和能效上的差异。
    *   **左下图 (a) Area Breakdown**：展示了 PE Array 中各硬件模块的面积占用百分比。
    *   **右下图 (b) Energy Breakdown**：展示了在不同 MX 数据格式下，各模块的能量消耗分布。

*   **算术单元设计变体对比分析 (顶部表格)**
    表格对比了处理非归一化输入的三种策略，所有设计均在 **400 MHz** 频率下综合：
    
    | 设计变体 | 面积 [µm²] | 关键特点与能效表现 |
    | :--- | :--- | :--- |
    | **mant. add. ext.** (尾数加法器扩展) | 2341.35 | 基准方案，通过扩展尾数加法器位宽来处理非归一化数。 |
    | **norm. inputs** (输入归一化) | **3640.78** | **面积最大**（比基准大55%）。需要对每个输入进行昂贵的归一化电路处理，导致能效最差（如 E5M2 能耗达 2.46 pJ/OP）。 |
    | **mant. add. ext. + bypass** (扩展+旁路) | **2078.42** | **最优方案**。通过引入模式特定的旁路逻辑平衡关键路径，不仅**面积最小**（比基准减小11%），且**能效最高**（如 E5M2 降至 1.14 pJ/OP）。 |

*   **PE Array 面积分解分析 (图 a)**
    饼图揭示了硬件资源的瓶颈所在，基于采用 "bypass" 优化的 MAC 单元：
    *   **L2 Adder (二级加法器)** 占据最大面积比重，高达 **35.90%**。这是因为 L2 加法器需要支持不同精度模式下的复杂对齐逻辑（特别是 FP8/FP6 模式下的浮点累加）。
    *   **FP Accum Add (浮点累加加法器)** 次之，占 **27.10%**，是处理浮点精度的核心开销。
    *   **L1 Adders (一级加法器)** 占 **25.70%**，负责初步的部分积求和。
    *   **Mul Units (乘法单元)** 仅占 **3.62%**，说明在该设计中，**加法/累加逻辑的面积开销远大于乘法逻辑**。
    *   **Shared Exp Overhead (共享指数开销)** 极低，仅为 **2.97%**，证明了引入共享指数机制带来的额外硬件成本非常有限。

*   **PE Array 能量分解分析 (图 b)**
    柱状图反映了在不同 MX 格式运行时的动态功耗特征：
    *   **FP Accum Add (橙色)** 是绝对的**能耗热点**。在 INT8 和 FP8 等高精度模式下，它占据了总能量的绝大部分。
    *   **精度与能耗的关系**：随着位宽降低，总能耗显著下降。
        *   **MXINT8 / MXFP8**：能耗较高，约在 **1.8 - 2.5 pJ/OP** 区间。
        *   **MXFP6**：能耗适中，约 **1.9 - 2.1 pJ/OP**。
        *   **MXFP4 (E2M1)**：**能效极高**，总能耗骤降至约 **0.3 pJ/OP** 左右，非常适合对能效极度敏感的场景。
    *   **L2 Adder (红色)** 在高精度模式下也是主要的能耗贡献者，但在低精度（如 FP4）模式下其占比相对缩小。

*   **核心结论与工程意义**
    *   **设计优化验证**：数据有力地证明了 **"mantissa extension + bypass"** 策略的有效性。它避免了高昂的归一化电路（节省约 1500 µm² 面积），并通过时序平衡进一步压缩了面积并降低了能耗。
    *   **架构瓶颈识别**：**加法器树（尤其是 L2 浮点累加器）** 是该精度可扩展设计的面积和功耗瓶颈，而非乘法阵列。未来若需进一步优化，应聚焦于简化高精度模式下的浮点累加逻辑。
    *   **MX 格式的硬件收益**：硬件实现完美体现了 MX 格式的优势——**切换到低精度格式（如 MXFP4）可带来数量级的能效提升**，这为边缘设备根据任务需求（如图2所示的不同机器人任务偏好不同精度）动态调整能效提供了坚实的硬件基础。

### 7a69f9e336d42b97a25f3285749d6233a0968738cb7ba0e87e46da23293d3734.jpg

![7a69f9e336d42b97a25f3285749d6233a0968738cb7ba0e87e46da23293d3734.jpg](images/7a69f9e336d42b97a25f3285749d6233a0968738cb7ba0e87e46da23293d3734.jpg)

- **图表类型与目的**
  - 图8为双子图对比折线图，用于评估 **pusher workload**（推手任务）上的训练性能
  - 左图展示**时间约束**下的收敛特性（给定 **1250 µs 时间预算**）
  - 右图展示**能量约束**下的收敛特性（给定 **200 pJ 能量预算**）
  - 核心对比对象：**Dacapo** 处理器（SotA MX连续学习处理器）与 **本文提出的 MX GeMM Core**

- **对比配置说明**
  
  | 配置标识 | 所属系统 | 数据格式 | 精度特点 |
  |---------|---------|---------|---------|
  | Dacapo (MX9) | Dacapo | 类MXINT8 | 9位等效整数 |
  | Dacapo (MX6) | Dacapo | 类MXFP6 | 6位浮点 |
  | Dacapo (MX4) | Dacapo | 类MXFP4 | 4位浮点 |
  | Ours (MXINT8) | 本文设计 | MXINT8 | 8位纯整数 |
  | Ours (MXFP8) | 本文设计 | MXFP8 (E4M3) | 8位浮点(4位尾数) |
  | Ours (MXFP4) | 本文设计 | MXFP4 (E2M1) | 4位浮点(1位尾数) |

- **左图分析：时间效率对比（Validation Loss vs Time）**
  - **横轴**：训练时间 [µs]，范围 0–1250
  - **纵轴**：验证损失（对数尺度），范围约 –12 至 –2（越低越好）
  - **关键发现**：
    - **Ours (MXFP8)** 呈现**绝对优势**：在相同时间内达到最低验证损失（约 –10 以下），收敛速度最快且最终精度最高
    - **Ours (MXINT8)** 表现次优，明显超越 Dacapo (MX9)，证明**方形分组架构**的时间效率优势
    - **Dacapo 系列普遍滞后**：由于 systolic array 的数据移位开销，在 iso-peak-throughput 条件下实际训练延迟更高（见表IV：Dacapo 50.5µs vs Ours 13.575µs 每批次）
    - **低精度格式行为**：MXFP4/MX4 虽然单操作能耗最低，但收敛曲线较平缓，最终损失高于 FP8/INT8 配置

- **右图分析：能量效率对比（Validation Loss vs Energy）**
  - **横轴**：累积能量消耗 [pJ]，范围 0–200
  - **关键发现**：
    - **Ours (MXFP8) 与 Dacapo (MX9) 能量效率相当**：两者曲线几乎重叠，表明在 pusher 任务上，MXFP8 格式虽需更多 epoch 训练（如文中所述），但每 epoch 能耗更低，总体能量效率与整数格式持平
    - **Ours (MXFP4) 能量效率突出**：在相同能量预算下可完成更多迭代，但受限于格式精度，最终损失略高
    - **Dacapo (MX6/MX4) 能量效率偏低**：在相同能量下验证损失高于本文对应配置

- **核心结论与技术意义**
  - **MXFP8 是机器人交互类任务的最优选择**：对于 pusher/reacher 等需要精细梯度表示的任务，MXFP8 (E4M3) 在**时间效率**和**精度**之间取得最佳平衡
  - **4× 有效训练吞吐量提升**：结合表IV数据，本文设计在 pusher 任务上的训练延迟仅为 Dacapo 的 **1/4–1/5**（MXFP8: 6.03µs vs Dacapo MX6: 30.7µs），直接体现为左图中更陡峭的收敛曲线
  - **架构创新的价值**：**方形共享指数分组（Square Blocks）** 消除了反向传播时的重量化开销和存储冗余（内存减少51%），使得在相同硬件资源下能执行更多有效计算
  - **精度可扩展性的实用价值**：不同机器人任务适配不同精度——平衡任务（cartpole）适合 MXINT8，交互任务（pusher）适合 MXFP8，本文硬件统一支持全部6种MX格式，实现**单一硬件平台的多任务最优适配**

