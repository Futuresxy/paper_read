# High-Level Design of Precision-Scalable DNN Accelerators Based on Sum-Together Multipliers 图表详解

### FIGURE 1. Reference ST multiplier, modified from [7]. The 16-bit inputs (A and B) are partitioned in 4-bit chunks to enable multiple operations, as defined by the 3-bit configuration input (CONFIG) and shown in Table 1. The X/· symbol indicates that the multiplier is capable of multiplication and dot-product operations, depending on the configuration.

![c942a09f3a724c9aeb36856d897aa4083a20db99b12d6246ee9aa920be6802bc.jpg](images/c942a09f3a724c9aeb36856d897aa4083a20db99b12d6246ee9aa920be6802bc.jpg)

- **图像基本信息**
  - 图像标识：FIGURE 1，Reference ST multiplier（参考求和乘法器）
  - 来源：修改自文献 [7]
  - 核心功能：展示 **Sum-Together (ST)** 类精度可扩展乘法器的通用架构

- **端口与信号定义**
  
  | 信号名称 | 位宽 | 方向 | 功能描述 |
  |---------|------|------|----------|
  | **CONFIG** | 3-bit | 输入 | 配置信号，控制乘法器的工作模式与精度 |
  | **A** | 16-bit | 输入 | 第一个操作数输入 |
  | **B** | 16-bit | 输入 | 第二个操作数输入 |
  | **P** | 32-bit | 输出 | 运算结果输出 |

- **输入数据分区机制**
  - **16位输入拆分策略**：两个16位输入（A 和 B）均被划分为 **四个4位块（4-bit chunks）**
  - **输入 A 的分区结构**：
    - `[15:12]`：最高有效4位（第1块）
    - `[11:8]`：次高4位（第2块）
    - `[7:4]`：次低4位（第3块）
    - `[3:0]`：最低有效4位（第4块）
  - **输入 B 的分区结构**：采用完全相同的4位分区方式

- **核心运算单元**
  - **运算符号 `X/·`**：表示该单元具备双重运算能力
    - **`X` 模式**：执行标准乘法运算（Multiplication）
    - **`·` 模式**：执行点积运算（Dot-product）
  - **ST 特性体现**：在低精度配置下，该单元不仅能并行执行多个乘法，还能**在乘法器内部将结果自动求和**（Sum-Together），无需外部加法器

- **支持的精度配置模式**（对应 Table 1）

  | CONFIG 值 | 模式名称 | 运算类型 | 操作说明 |
  |-----------|----------|----------|----------|
  | **000b** | 16×16 | 单次乘法 | $P_{[31:0]} = A_{[15:0]} \times B_{[15:0]}$ |
  | **100b** | 16×8 | 非对称乘法 | $P_{[31:0]} = A_{[15:0]} \times B_{[7:0]}$ |
  | **010b** | 8×8 | 双路点积 | $P_{[16:0]} = A_{[15:8]} \times B_{[7:0]} + A_{[7:0]} \times B_{[15:8]}$ |
  | **011b** | 8×4 | 双路点积 | $P_{[16:0]} = A_{[15:8]} \times B_{[3:0]} + A_{[7:0]} \times B_{[11:8]}$ |
  | **001b** | 4×4 | 四路点积 | $P_{[9:0]} = \sum_{i=0}^{3} A_{[4i+3:4i]} \times B_{[\text{交叉映射}]}$ |

- **架构设计特点**
  - **子字并行（Subword-parallel）**：通过将16位输入动态分割为多个低精度子字（subwords），实现数据级并行
  - **精度-并行度反比关系**：操作数位宽越低（如4bit），并行度 N 越高（N=4）；全精度时 N=1
  - **内存带宽恒定**：无论何种精度模式，输入/输出总线宽度保持不变（16位输入，32位输出），避免了低精度下的**内存带宽爆炸问题**
  - **SIMD 友好**：天然支持单指令多数据流（Single Instruction Multiple Data）计算范式

- **在 DNN 加速器中的应用价值**
  - **混合精度量化（Mixed-Precision Quantization, MPQ）支持**：可根据 DNN 不同层对量化误差的敏感度，动态调整每层的计算精度（16/8/4 bit）
  - **MAC 操作加速**：对于 2D-Convolution 和 Fully-Connected 层，ST 特性可节省 **N-1 次 MAC 加法运算**，显著降低延迟
  - **边缘设备优化**：特别适合资源受限的边缘推理场景（Tiny ML），在保持模型精度的同时减少内存占用和能耗

### (e) FIGURE 2. (a) Our version of BW ST [6]: architecture overview (left), PPM reconfiguration (right). (b) Our version of ST multiplier [17]: 16-bit high-precision multiplier (left), 8-bit and 4-bit low-precision dot-product units (middle, right). (c) Our version of D&C ST [21]: four FUs with four 4-bit BitBricks each, interconnected by shift-and-add logics. (d) Our version of ST multiplier [22]: four 8-bit Booth multipliers interconnected by muxes ending with an adder tree. (e) Radix-4 Booth ST (as in [7]): reconfiguration logic (blue), 16-bit Booth multiplier (white and gray). (f) HLS ST derived from HLS (proposed in this paper): four multipliers and three adders interconnected by a network of muxes and concatenations.

![79a7894766db49c76f253071d0795a318779f16724a3c06338f2865bfd25ef8e.jpg](images/79a7894766db49c76f253071d0795a318779f16724a3c06338f2865bfd25ef8e.jpg)

- **整体概述**：该图展示了 **六种不同架构的 Sum-Together (ST) 乘法器**的详细电路结构对比，涵盖了 **State-of-the-Art (SoA)** 设计以及本文提出的两种新型结构。所有设计均支持 **16×16, 16×8, 8×8, 8×4, 4×4** 等多种精度配置模式。

- **(a) BW ST [6] - Baugh-Wooley 子字并行架构**
    - **核心组成**：左侧为整体架构，包含 **BW PPM (Baugh-Wooley Partial Product Matrix)**、最终加法器（圆圈中的 ⊕）、输出拼接逻辑（&）以及 **right shift & ext（右移与符号扩展）** 模块。
    - **PPM 重配置机制**（右侧三列）：通过门控（红色 × 符号表示被禁用的部分积）实现精度切换：
        - **16×16 / 16×8 模式**：整个 PPM 激活，产生 32 位有效输出；
        - **8×8 / 8×4 模式**：仅对角线上的两个 8×8 子矩阵激活（黄色高亮），其余灰色部分被门控关闭，输出需右移对齐；
        - **4×4 模式**：仅四个 4×4 子乘法器激活，输出位数进一步减少。
    - **基本构建块**：采用 **4×4 BW 乘法器** 作为最小计算单元。

- **(b) 基于 [17] 的 ST 乘法器（RISC-V SIMD 扩展风格）**
    - **架构特征**：采用 ** mutually exclusive datapaths（互斥数据通路）** 设计，分为三个独立部分：
        - **左侧**：标准 **16-bit 高精度定点乘法器**，处理 16×16 和 16×8 配置；
        - **中间**：**8-bit 低精度点积单元**，包含两个 9-bit 乘法器后接压缩树，处理 8×8 点积；
        - **右侧**：**4-bit 低精度点积单元**，包含四个 5-bit 乘法器后接压缩树，处理 4×4 点积。
    - **实现方式**：使用 **behavioral RTL description（行为级 RTL 描述）**，给予综合工具较大的优化自由度。

- **(c) D&C ST [21] - 分治法 (Divide and Conquer) 架构**
    - **核心组成**：包含 **四个 Fusion Units (FUs)**，每个 FU 内部由 **四个 4-bit BitBricks（位砖）** 组成（如虚线框所示）。
    - **互连逻辑**：FUs 之间通过复杂的 **shift-and-add logics（移位加法逻辑）** 连接，支持动态组合与分解。
    - **数据流**：输入操作数被分割后送入不同的 BitBricks，中间结果通过移位和加法逐步合并，最终通过右移扩展模块对齐输出。
    - **特点**：属于 **bit-level reconfigurable（比特级可重构）** 架构，灵活性高但互连逻辑较复杂。

- **(d) 基于 [22] 的 ST 乘法器（可配置定点乘法器）**
    - **核心组成**：**四个 8-bit Booth 乘法器**（注意：图中标注 "without final adder"，即不含最终加法器，仅生成部分积）。
    - **路由网络**：通过密集的 **multiplexer (mux) 网络** 将输入操作数路由到不同的 8-bit 乘法器，并根据配置选择有效的部分积。
    - **归约阶段**：末端接一个 **Adder Tree（加法树）**，将选中的部分积求和得到最终结果。
    - **配置支持**：通过对称/非对称组合支持 16×16, 8×8 点积等模式（本文修改了原始设计的点积计算顺序以匹配参考 ST 行为）。

- **(e) Radix-4 Booth ST [7] - 本文先前提出的基-4布斯架构**
    - **创新亮点**：采用 **lightweight reconfiguration logic（轻量级重配置逻辑）**，以极小的面积开销实现精度缩放。
    - **蓝色区域（重配置逻辑）**：
        - **RADIX-4 BOOTH ENCODER（基-4布斯编码器）**：根据 CONFIG 信号（CONFIG[2:0]）对输入操作数进行编码，生成 SINGLE/DOUBLE/NEQ 控制信号；
        - **BOOTH SELECTOR（布斯选择器）**：根据编码结果选择部分积（PP0-PP7），在低精度模式下复用高精度硬件。
    - **白色/灰色区域（标准 Booth 乘法器）**：包含 **[4:2] Compressor Tree（4:2 压缩树）** 和带 **Prefix Network（前缀网络）** 的 **Final Adder（最终加法器）**，这是标准的 16-bit Booth 乘法器核心。
    - **优势**：重配置逻辑仅占很小比例，主要路径延迟接近标准 Booth 乘法器。

- **(f) HLS ST - 本文提出的新型 HLS 导出架构**
    - **生成方式**：完全由 **High-Level Synthesis (HLS) 工具（Siemens Catapult）** 从 C/C++ 高层描述（Listing 1）自动推断生成，无需手动 RTL 设计。
    - **硬件组成**（从生成的 RTL 反推）：
        - **乘法器阵列**：包含 **一个 16-bit、两个 8-bit 和两个 4-bit 乘法器**（红色 × 符号）；
        - **加法器链**：**三个加法器**（蓝色 + 符号），位宽分别为 8/12/16-bit，用于合并低精度结果；
        - **路由网络**：**多路选择器 (mux) 网络和 concatenation（拼接）块**（& 符号），负责将 16-bit 输入操作数拆分并分发到对应精度的乘法器，并将结果合并为 32-bit 输出。
    - **特点**：**结果已自然右对齐**（无需额外的 right shift & ext 模块）；架构呈现明显的 **数据通路复制** 特征，不同精度模式使用独立的硬件资源。

- **关键架构对比总结**

| 架构类型 | 重配置策略 | 基本计算单元 | 面积效率特征 | 适用频率范围 |
|---------|-----------|-------------|-------------|-------------|
| **BW ST [6]** | PPM 门控 | 4×4 BW 阵列 | **低频下最优**，高频因长关键路径退化 | ≤ 600 MHz |
| **BW-ADD** (改进版) | PPM 门控 + 优化加法器 | 4×4 BW 阵列 | 中频范围改善 | 700-800 MHz |
| **ST [17]** | 独立数据通路 | 多个独立乘法器 | 面积较大（冗余逻辑），但高频下综合工具优化空间大 | ≥ 1.5 GHz |
| **D&C ST [21]** | 移位加法组合 | 4-bit BitBricks | 面积效率较低（复杂互连） | 全范围 |
| **ST [22]** | Mux 路由 + 加法树 | 8-bit Booth 阵列 | 中等，依赖 mux 开销 | 中高频 |
| **Booth ST [7]** | 轻量级编码选择 | 16-bit Booth 核 | **高频下 PPA 均衡最佳** | 600-1400 MHz |
| **HLS ST** (新) | HLS 自动推断 | 混合精度乘法器组 | 接近 Pareto 前沿，设计周期短 | 取决于 HLS 约束 |

- **设计权衡洞察**：
    - **单一可重构乘法器 vs. 独立多精度通路**：如 [6], [7], BW-ADD 等采用单一硬件资源时分复用，**逻辑门利用率更高**，面积更紧凑；而 [17], [22], HLS ST 等为每种精度配备独立硬件，虽面积稍大但在特定频率下可能获得更好的时序收敛性。
    - **关键路径来源**：BW 架构受限于 **PPM 对角线链路 + RCA 进位链**；Booth 架构受益于 **Wallace 树压缩**，关键路径更短；D&C 架构受限于 **多级移位加法级联**。
    - **HLS 设计的启示**：HLS ST 能自动生成具有竞争力的架构（出现在 DSE 的 Pareto 前沿上），证明了 **高层综合在 PS 加速器设计中的可行性**，显著缩短了设计迭代周期。

### FIGURE 3. Working principle of our ST-based DNN accelerators: 2D-Convolution (2D-Conv), Depth-wise Convolution (DW-Conv) and Fully-Connected (FC).

![a53e42701033823b111d26c21760c7e8f9132cc71596c64983afca0e3f5625d7.jpg](images/a53e42701033823b111d26c21760c7e8f9132cc71596c64983afca0e3f5625d7.jpg)

- **图表核心主题**：该图（FIGURE 3）系统性地展示了**Sum-Together (ST) 乘法器**在三种核心 DNN 层——**2D-Convolution (2D-Conv)**、**Depth-wise Convolution (DW-Conv)** 和 **Fully-Connected (FC)**——中的工作原理与数据访问模式。

- **结构化布局分析**：
    - **行维度（Precision Scalability）**：按 ST 配置的并行度 $N$ 分为三档，体现了**精度可扩展性 (Precision-Scalable)** 的核心优势：
        - **$N = 1$ (高精度模式)**：CONFIG 为 $16 \times 16$ 或 $16 \times 8$。此时 ST 乘法器执行标准的高精度乘法或非对称精度乘法。
        - **$N = 2$ (中精度模式)**：CONFIG 为 $8 \times 8$ 或 $8 \times 4$。ST 乘法器同时处理两对低精度操作数（如两个 8-bit 或 4-bit 数据），并在内部执行**点积 (Dot-Product)** 操作。
        - **$N = 4$ (低精度模式)**：CONFIG 为 $4 \times 4$。ST 乘法器并行处理四对 4-bit 操作数，实现最大程度的并行计算加速。
    - **列维度（Layer Specificity）**：针对三种不同的神经网络层展示了差异化的数据流（Dataflow）：
        - **2D-Conv**：涉及空间维度（Height, Width）和通道维度（Channel）的滑动窗口操作。
        - **DW-Conv**：仅在空间维度进行卷积，通道间独立，无跨通道累加。
        - **FC**：纯粹的矩阵-向量乘法，无空间维度概念。

- **视觉符号与数据流解析**：
    - **颜色编码**：
        - **蓝色块**：代表**输入激活张量 (Input Activation Tensor)**。
        - **橙色块**：代表**权重张量 (Weight Tensor)**。
        - **绿色块**：代表**输出激活/结果 (Output Activation)**。
    - **红色边框**：高亮显示在当前时钟周期内被**打包 (Packed)** 进 ST 乘法器 16-bit 输入端口（A 和 B）的具体数据元素。
    - **箭头指向**：明确了数据从内存/缓冲区读取、进入 ST 乘法器运算、再写入输出缓冲区的完整路径。

- **各层工作原理详述**：
    - **2D-Conv 加速器原理**：
        - 在 $N=1$ 时，MAC 单元每次从输入特征图的一个通道和对应的权重核中各取一个 16-bit 元素进行相乘。
        - 在 $N>1$ 时（如 $N=2$ 或 $N=4$），加速器利用 ST 特性，**同时从相邻的多个输入通道**读取低精度数据。例如在 $N=4$ ($4 \times 4$) 模式下，一次读取 4 个通道的 4-bit 数据打包进一个 16-bit 操作数。这使得 MAC 循环的迭代次数理论上减少为 $C/N$（$C$ 为总通道数），从而显著降低延迟。
    - **DW-Conv 加速器原理**：
        - 由于 DW-Conv 不进行跨通道累加，其利用 ST 的方式与 2D-Conv 不同。它将 ST 用于**感受野 (Receptive Field)** 内的空间维度累加。
        - 在 $N=2$ 或 $N=4$ 时，它尝试将卷积核（Kernel）内的多个空间位置（如 $3 \times 3$ 核中的多个点）的低精度数据并行处理。
        - **效率瓶颈**：如图所示，当 Kernel 大小（如 $3 \times 3=9$）不能被 $N$ 整除时（如 $N=4$ 时需 3 次迭代，最后一次仅处理 1 个有效数据），会导致 ST 乘法器的部分硬件空闲（需填充零），产生**利用率不足 (Under-utilization)** 的开销。
    - **FC 加速器原理**：
        - 工作逻辑与 2D-Conv 类似，但作用于向量和矩阵之间。
        - 在 $N>1$ 时，ST 乘法器同时从输入向量中取 $N$ 个低精度激活值，以及从权重矩阵的同一行中取 $N$ 个对应的低精度权重值进行并行点积运算。这同样将循环次数缩减至 $IA/N$（$IA$ 为输入激活数量）。

- **设计启示与性能影响**：
    - **延迟缩放**：对于 2D-Conv 和 FC 层，理论延迟随 $N$ 的增加近似呈 **$1/N$** 缩放。
    - **DW-Conv 的特殊性**：其延迟缩放因子为 $\lceil K^2 / N \rceil / K^2$（$K$ 为核大小），由于取整操作和硬件空闲问题，其加速比通常劣于理想的 $1/N$。
    - **内存带宽优化**：通过将多个低精度数据打包进一个 16-bit 字宽的操作数，ST 架构有效**降低了内存访问次数**，缓解了边缘设备的带宽瓶颈。

- **总结**：该图直观地论证了 ST 乘法器作为一种**可重构算子 (Reconfigurable Operator)**，能够根据 DNN 层的混合量化需求（Mixed-Precision），动态调整数据并行度，在保持硬件复用的同时，为不同类型的神经网络层提供差异化的加速策略。

### FIGURE 4. General architecture of the ST-based accelerators (bottom right), HLS flow (bottom left), and pseudo-code of the high-level C/C++ description (top) that produces the general architecture.

![b3bb3b133c0b738d5575c2d9feb117febfeb9119a65b4488da83116f36200545.jpg](images/b3bb3b133c0b738d5575c2d9feb117febfeb9119a65b4488da83116f36200545.jpg)

对 **FIGURE 4** 的详细分析如下：

**一、整体结构概览**

该图展示了基于 Sum-Together (ST) 乘法器的 DNN 加速器的完整设计框架，由三个核心组成部分构成：

*   **顶部区域**：加速器的高层次 C/C++ 伪代码描述（Pseudo-code）
*   **左下区域**：从高层次描述到硬件实现的 HLS（High-Level Synthesis）设计流程
*   **右下区域**：生成的 ST-based 加速器的通用硬件架构

---

**二、伪代码逻辑分析（顶部）**

该伪代码定义了 `st_based_accelerator` 函数的核心计算逻辑，体现了不同层类型（2D-Conv, DW-Conv, FC）的统一控制流：

*   **L1 循环（最外层，流水线化）**：对应于输出特征图的空间维度（如高度 oh、宽度 ow），适用于 2D-Conv 和 DW-Conv。
*   **L2 循环（OBUF 重置）**：在处理每层的**第一个 tile** 时执行，将输出缓冲区（OBUF）清零。
*   **L3 循环（外层循环，流水线化）**：这是精度可扩展性发挥作用的**关键循环**。
    *   对于 **2D-Conv**：遍历输入通道（IC）；
    *   对于 **DW-Conv**：遍历卷积核元素（Kernel size K²）；
    *   对于 **FC**：遍历输入激活值（IA）。
    *   *关键优化*：在低精度配置下（N=2 或 N=4），该循环的最大迭代次数会除以 N，从而减少总周期数。
*   **L4 循环（最内层，完全展开 Unrolled）**：
    *   从内部缓冲区读取操作数 A（对于 DW-Conv 在此循环内读取，对于其他在 L3 内读取）和权重 B；
    *   调用 **`st_multiplier_function(CONFIG, A, B)`** 执行 ST 乘法或点积运算；
    *   将结果 P 累加到 **OBUF[output_index]** 中。
    *   *HLS 指令映射*：此处的 `unroll` 指令导致硬件生成 M 个并行的 PSMAC 单元。
*   **提前终止机制**：当 L3 的索引达到 `(L3 max iterations / N)` 时触发 `break`，这是实现低精度加速（1/N 延迟降低）的软件控制逻辑。
*   **L5 循环（量化与 ReLU，流水线化）**：仅在处理**最后一个 tile** 时执行，调用相应的 UIQ 公式（Eq. 14/16/6）进行量化，并可选地融合 ReLU 激活函数。

---

**三、HLS 设计流程分析（左下）**

该流程图阐述了从算法描述到物理实现的自动化路径：

| 输入组件 | 描述 | 作用 |
| :--- | :--- | :--- |
| **C/C++ (top)** | 加速器顶层功能描述 | 定义数据流、循环结构和内存访问模式 |
| **RTL/ C/C++ (ST)** | ST 乘法器的具体实现 | 可作为 IP 核（IP Mode）或内联代码（Inline Mode）被调用 |
| **Hardware configuration knobs** | 硬件配置旋钮 | 包括并行度（M）、时钟频率、缓冲区大小等（见表 3） |
| **Constraints** | 实现约束 | 时序约束、面积/功耗目标等 |

*   **处理阶段**：
    1.  **HLS 阶段**：工具（如 Siemens Catapult）综合 C/C++ 代码，根据指令（如 `pipeline`, `unroll`, `partition`）调度操作并生成 RTL。
    2.  **Implem.（实现）阶段**：使用逻辑综合工具（如 DC）将 RTL 转换为门级网表，进行时序分析和面积/功耗优化。

---

**四、硬件架构详细解析（右下）**

生成的 ST-based Accelerator 采用模块化设计，各模块协同工作：

*   **Input/Weight Double Buffers（输入/权重双缓冲）**：
    *   采用**双缓冲（Double Buffering）**技术，允许 DMA 在加速器计算当前 tile 数据的同时预取下一个 tile 的数据，隐藏内存访问延迟。
    *   通过 **`#partitioning`**（分区）指令，将大缓冲区拆分为多个小的存储体（Bank），以支持 PSMAC 阵列的并行数据访问。

*   **Memory Addressing & Concatenating Logic（寻址与拼接逻辑）**：
    *   负责根据当前循环索引（oh, ow, oc, ic 等）计算正确的内存地址。
    *   将从多个 4-bit 存储体中读取出的数据**拼接（Concatenate）**成 ST 乘法器所需的 16-bit 操作数 A 和 B（如图 3 所示的数据打包方式）。

*   **Reconfigurable ST-based PSMAC Array（可重构 ST-PSMAC 阵列）**：
    *   这是加速器的**计算核心**，包含 **M 个并行**的 MAC 单元（由 L4 循环的 `unroll` 生成）。
    *   每个 MAC 单元包含：
        *   一个 **ST 乘法器**（支持 16x16, 8x8 dot-product, 4x4 dot-product 等模式）；
        *   一个 **加法器**（用于累加）；
        *   一个 **累加寄存器（Acc）**。
    *   通过 **CONFIG** 信号动态切换精度模式，实现 SIMD 式的并行计算。

*   **Quantization Logic and ReLU（量化与激活逻辑）**：
    *   实现 UIQ 公式的硬件电路，包括缩放（乘法）、偏移（加法）、舍入（Round）和裁剪（Clip）。
    *   支持 **Layer Fusion（层融合）**，将量化与 ReLU 合并在同一硬件单元中执行，减少中间数据的存储与读取开销。

*   **Output Double Buffers（输出双缓冲）**：
    *   存储 PSMAC 阵列计算的最终结果（或部分累积结果）。
    *   同样采用 **`#partitioning`** 以匹配输出通道的并行写入需求。

---

**五、核心设计洞察总结**

*   **软硬件协同设计**：伪代码中的循环结构直接决定了硬件的并行度（Unrolling 决定阵列大小 M）和流水线深度（Pipelining 决定吞吐量）。
*   **精度可扩展性的硬件体现**：ST 乘法器通过单一硬件资源在不同 bit-width 下复用，配合 L3 循环的动态终止，实现了计算效率随精度降低而提升的特性。
*   **数据流优化**：双缓冲和内存分区技术有效解决了边缘设备常见的“内存墙”问题，确保计算单元不会因等待数据而停顿。
*   **灵活性**：通过改变 Hardware Configuration Knobs（如 M 值、时钟频率、ST 乘法器类型），可以从同一个高层次描述中衍生出面向**低面积（Low-Area）**、**低功耗（Low-Power）**或**低延迟（Low-Latency）**的不同 Pareto 最优硬件实现。

### FIGURE 5. Accelerators design flow.

![d03a34d99ccce295470e0208279338af846080e06f062b97ae212a85c91601b1.jpg](images/d03a34d99ccce295470e0208279338af846080e06f062b97ae212a85c91601b1.jpg)

对 **FIGURE 5. Accelerators design flow** 的详细分析如下：


该图展示了一个**三阶段的硬件-软件协同设计流程**，用于生成基于 Sum-Together (ST) 乘法器的精度可扩展 DNN 加速器。整个流程从浮点模型出发，经过混合精度量化、算子位宽优化，最终通过高层次综合 (HLS) 生成硬件 RTL 实现。

---


**核心目标**：将浮点 DNN 模型转换为混合精度 (MP) 整数量化模型，同时保持模型精度并最小化总比特数。

- **输入**：FP models（原始浮点模型）
- **关键步骤**：
    - **插入 QActivation 层**：实现均匀整数量化公式 (Eqns.1-4)，并进行 Batch Normalization (BN) 折叠
    - **AutoQKeras 自动搜索**：使用贝叶斯优化在 **INT16/8/4** 位宽空间中搜索每层的最优激活值和权重位宽（偏置约束为 **INT31/16**）
    - **收敛判断**：基于验证指标是否稳定决定是否终止搜索
    - **最终微调**：使用 QKeras 对搜索到的最优 MP 结构进行量化感知训练微调
- **输出**：Final MP models（详见 Appendix B 和 Table IV）
- **设计约束**：
    
    | 参数类别 | 约束条件 |
    | :--- | :--- |
    | 权重/激活值 | INT16, INT8, 或 INT4 |
    | 偏置 | INT31 或 INT16 |
    | 参考模型 | 16-bit flat quantized model |

---


**核心目标**：通过消融实验 (Ablation Study) 最小化硬件实现中 UIQ 公式所需的定点变量位宽，以降低加速器面积和功耗。

- **输入**：阶段一输出的 Final MP models
- **关键步骤**：
    - **代码替换**：将 QKeras 量化层替换为对应加速器的 **C/C++ 高层描述**（即 Lsts. 2-4 中的伪代码）
    - **统计分析与位宽削减**：
        - 在校准子集上运行推理，分析变量动态范围
        - 迭代减少 UIQ 公式中 C/C++ 变量的整数和小数部分位宽
    - **退化检查**：在全测试集上评估，若精度损失超过阈值（如 **-0.05%**）则回退
- **输出**：
    - 实现了最优位宽的 **HLS C/C++ 代码**（Lsts.2-4）
    - 硬件配置参数集（Table III，如 IC, OC, 并行度 M 等）
    - 最优变量位宽表（Table V）

---


**核心目标**：利用 HLS 工具和逻辑综合工具，将优化后的 C/C++ 描述转换为物理可实现的硬件电路，并进行广泛的设计空间探索 (DSE)。

- **输入**：阶段二优化后的 HLS 代码及硬件配置参数
- **关键步骤**：
    - **High-Level Synthesis (HLS)**：使用 Siemens Catapult 工具，根据目标时钟频率、ST 乘法器类型（IP模式或内联模式）及各种编译指令（如 `unroll`, `pipeline`, `interleave`）生成 RTL
    - **Logic Synthesis**：使用 Synopsys Design Compiler 在 **28-nm CMOS** 工艺下进行综合，得到具体的面积、功耗和时序数据
- **输出与成果**：
    - **Pareto-optimal 加速器集合**：在 **Latency vs Area** 和 **Latency vs Power** 空间中的最优解（如 Figs. 7-8 所示的散点图和 Pareto 前沿）
    - **完整的 PPA 对比数据**：用于指导 SoC 设计师在不同场景（低面积、低功耗、低延迟）下选择最佳加速器配置

---


- **闭环迭代**：阶段一和阶段二都包含“评估-判断-调整”的反馈循环，确保在硬件效率与模型精度之间取得最佳平衡。
- **自动化程度高**：借助 AutoQKeras 和 HLS 工具，大幅减少了手动调优的工作量。
- **端到端优化**：从算法层面的量化策略到底层硬件的门级实现，实现了跨层次的联合优化。
- **场景适应性**：最终的 DSE 结果提供了多样化的 Pareto 最优解，能够满足边缘设备对不同 PPA 指标的差异化需求。

### 87988e48911353c1b0641f2293f725ce58c182af6546dd451c845cef824b4533.jpg

![87988e48911353c1b0641f2293f725ce58c182af6546dd451c845cef824b4533.jpg](images/87988e48911353c1b0641f2293f725ce58c182af6546dd451c845cef824b4533.jpg)

针对该论文中 **Figure 6: DSE of the SoA and newly proposed ST multipliers** 的深入分析如下：

*   **图表整体概述**
    *   该图展示了**七种不同Sum-Together (ST) 乘法器架构**在28nm CMOS工艺下的性能、功耗和面积(PPA)对比。
    *   采用**设计空间探索(DSE)**方法，通过改变目标时钟频率（500MHz至1500MHz）来评估各架构的优劣。
    *   图表包含两个子图：
        *   **上图：面积(Area) vs. 时钟周期(Clock Period)** —— 衡量面积效率与速度的权衡。
        *   **下图：功耗(Power) vs. 时钟周期(Clock Period)** —— 衡量能效与速度的权衡。
    *   图中的**颜色渐变**代表目标时钟频率（蓝色为低频~500MHz，深红色为高频~1500MHz）。
    *   **黑色实线(Pareto front)** 连接了在给定约束下的帕累托最优点（即无法在不恶化另一指标的情况下改善某一指标的解集）。

*   **参与对比的ST乘法器架构识别**
    下表总结了图中各符号所代表的架构及其来源：

    | 符号 | 架构名称 | 来源/性质 | 核心特点 |
    | :--- | :--- | :--- | :--- |
    | □ (方形) | **BW [6]** | 现有技术(SoA) | Baugh-Wooley子字并行阵列，使用行波进位加法器(RCA)作为最终加法器。 |
    | ○ (圆形) | **D&C [21]** | 现有技术(SoA) | 分治法(Divide-and-Conquer)架构，基于BitBricks的移位相加逻辑。 |
    | ✕ (叉号) | **Booth [7]** | 作者先前工作 | Radix-4 Booth编码架构，具有轻量级重配置逻辑。 |
    | ✳ (星号) | **[17]** | 现有技术(SoA) | 源自RISC-V处理器(RI5CY)的SIMD乘法单元，采用行为级描述。 |
    | ▽ (倒三角) | **[22]** | 现有技术(SoA) | 面向DSP的可配置定点乘法器，包含多个Booth乘法器和加法器树。 |
    | ◇ (菱形) | **BW-ADD (ours)** | **本文新提出** | 改进的BW乘法器，**优化了最终加法器**（允许综合工具自动选择最优加法器而非固定RCA）。 |
    | + (加号) | **HLS ST (ours)** | **本文新提出** | 完全由**高层次综合(HLS)工具**从C/C++描述自动生成的ST乘法器。 |

*   **上图深度解析：面积 vs. 时钟周期 (Area vs. Clock Period)**
    *   **X轴**：面积（单位 μm²），范围约 1000 – 2000。
    *   **Y轴**：时钟周期（单位 ns），范围约 0.67 (≈1.5 GHz) – 2.0 (≈500 MHz)。
    *   **关键趋势与结论**：
        *   **Booth [7] (✕) 占据主导地位**：在**中高频率范围（约600 MHz至1400 MHz，即时钟周期0.71ns-1.67ns）**，Booth架构是绝对的面积效率冠军，构成了帕累托前沿的主体。这归功于其较低的重配置开销和高效的Wallace树压缩结构。
        *   **低频领域的竞争者**：在**低频区域（≤ 800 MHz，即≥1.25ns）**，原始的 **BW [6] (□)** 和新提出的 **BW-ADD (◇)** 表现优异，非常接近或位于帕累托前沿上。这证实了BW阵列结构在时序宽松时具有极高的面积利用率。
        *   **BW-ADD (◇) 的改进效果**：相比原始BW [6]，**BW-ADD在高频端（如700-800 MHz）的表现更接近帕累托前沿**，证明了用优化加法器替换固定RCA能有效缓解BW阵列长关键路径导致的面积膨胀问题。
        *   **专用乘法器的面积劣势**：采用**独立专用乘法器路径**的架构（如 **[17], [22], HLS ST (+)**）普遍偏离帕累托前沿较远，面积较大。这是因为它们在不同精度模式下存在大量**互斥的冗余逻辑门**，硬件复用率低于子字并行式架构（如BW和Booth）。
        *   **D&C [21] (○) 的表现**：面积效率排名靠后，主要受限于其复杂的移位相加互联逻辑带来的额外开销。

*   **下图深度解析：功耗 vs. 时钟周期 (Power vs. Clock Period)**
    *   **X轴**：功耗（单位 mW），范围约 0.2 – 2.0。
    *   **Y轴**：时钟周期（单位 ns），同上。
    *   **关键趋势与结论**：
        *   **整体差异较小**：与面积图相比，各架构在**功耗上的差异相对不那么悬殊**，数据点分布更为密集。
        *   **低频能效之王**：在**400-800 MHz** 频率区间，基于 **BW架构的设计（[6] 和 BW-ADD）** 在功耗上表现出色，通常位于或接近帕累托前沿。
        *   **中高频能效表现**：从 **1000 MHz 至 1300 MHz**，**Booth [7] (✕)** 逐渐展现出功耗优势，并开始主导帕累托前沿。**D&C [21] (○)** 和 **[22] (▽)** 在此频段也具有一定的竞争力。
        *   **极高频下的选择**：在**最高频率（约1500 MHz，0.67ns）** 附近，**[17] (✳)** 表现出最佳的功耗特性。这得益于其行为级描述给予逻辑综合工具更大的优化自由度，使其能推断出满足苛刻时序且功耗较低的实现。
        *   **HLS生成设计的竞争力**：值得注意的是，**HLS ST (+)** 在某些设计点（特别是中低频、中等功耗区域）展现出了不错的能效，甚至接近帕累托前沿，证明了HLS工具在生成能效合理的算术电路方面的潜力。

*   **综合结论与设计启示**
    *   **不存在“万能”的最优ST乘法器**：选择何种架构**严格取决于具体的PPA约束目标**（是追求极致面积、最低功耗，还是最高性能）。
    *   **按需选型指南**：
        *   若目标是**中高性能与良好面积效率**（如大多数边缘AI加速器场景）：**首选 Booth [7]**。
        *   若目标是**超低频、极小面积或低功耗**（如微控制器内部加速）：**BW [6] 或 BW-ADD** 是更佳选择。
        *   若需要**极限工作频率**（>1.4 GHz）：可考虑 **[17]** 或 **Booth [7]**。
    *   **本文新贡献的价值**：
        *   **BW-ADD** 成功改善了经典BW乘法器在高频下的面积退化问题，扩展了其适用范围。
        *   **HLS ST** 证明了通过HLS快速探索和生成可行ST乘法器的可能性，虽然其绝对PPA可能未达顶尖，但大幅降低了设计门槛，且在某些点展现了意外竞争力，为自动化硬件设计流程提供了有力支持。

### 5aaf3d8fd1342b677197119ca4bc34b23233ae0f3b09dd25fc9b3380e5d9eb40.jpg

![5aaf3d8fd1342b677197119ca4bc34b23233ae0f3b09dd25fc9b3380e5d9eb40.jpg](images/5aaf3d8fd1342b677197119ca4bc34b23233ae0f3b09dd25fc9b3380e5d9eb40.jpg)

针对该 **Latency vs Area 设计空间探索（DSE）结果图** 的详细专业分析如下：


*   **图表性质**：该图为 **三维 PPA（Performance-Power-Area）空间向二维 Latency-Area（LA）空间的投影**，展示了三种 ST-based DNN 加速器（2D-Conv, DW-Conv, FC）的综合优化结果。
*   **坐标轴定义**：
    *   **X 轴 (Area)**：硬件面积，单位 mm²，反映硅片成本。
    *   **Y 轴 (Normalized Latency)**：归一化延迟（对数刻度），值越小代表推理速度越快。
    *   **颜色映射 (Clock freq.)**：数据点颜色对应 **目标时钟频率 (MHz)**，从蓝色 (100 MHz) 过渡到深红/黑色 (1000 MHz)。高频通常伴随更高的面积开销以满足时序约束。
*   **符号与图例体系**：
    *   不同几何符号代表 **PSMAC 阵列中所集成的 ST 乘法器类型**（如 BW [6], Booth [7], D&C [21], HLS ST 等）。
    *   **实线 (Pareto front)**：连接 **LA 空间下的帕累托最优解**。这些点代表了在给定面积下无法再进一步降低延迟的非支配解集合。
    *   **数据标签**：格式为 `(m, n)`，分别对应各加速器的并行度参数（如 IC/OC 或 IA/OA）。


*   **设计空间特征**：呈现出典型的 **“面积-延迟”权衡曲线**。左下角为极小面积但高延迟的设计（如点 `32,4`），向右上方移动时，通过增加输出通道并行度 (OC) 和时钟频率，延迟呈指数级下降。
*   **帕累托前沿分布**：前沿线覆盖了从 **超低面积 (~0.02 mm²)** 到 **相对高面积 (~0.2 mm²)** 的广阔区间。
*   **关键标记点解读**：
    *   **点 A / B / C**：位于不同面积预算约束下的候选最优解。例如，**点 B (32,8)** 在中等面积下实现了较低的延迟。
    *   **点 L**：代表 **低延迟 (Low-Latency) 场景** 的优选方案，具有较大的面积预算（>0.12 mm²）和极高的并行度 (`32,32`)。

*   **算法特性影响**：由于 DW-Conv 缺乏输入通道维度的累加，其 ST 乘法器的利用率受限于卷积核大小 ($K^2$)。这导致其 **延迟降低的斜率相比 2D-Conv 更为平缓**，即在增加相同面积的情况下，性能提升幅度较小。
*   **设计点聚类**：数据点在低频区（绿色/青色）分布较为密集，表明 DW-Conv 对极端高频的收益不如其他层敏感。
*   **关键标记点**：**点 D** 是一个在 LA 和 LP 空间均表现优异的双优解（采用 HLS ST Inline，400 MHz）。

*   **高并行度潜力**：FC 层的计算模式最适合 ST 乘法器的 dot-product 特性。图中可见极大的 **IA (输入激活) 参数变化**（从 256 到 1024），大幅增加了 tile 大小从而减少了总执行次数。
*   **极端优化点**：右下角的点（如 `1024,32`）代表了 **极致的低延迟设计**，但代价是巨大的面积占用（接近 0.2 mm²）。
*   **权衡示例**：**点 E (1024,4)** 与 **点 F (HLS ST Inline)** 展示了在相似延迟下，不同乘法器实现带来的面积差异（约 18% 的功率/面积权衡空间）。


*   **ST 乘法器选型的决定性作用**：
    *   **主导地位**：**Booth [7]** 架构在 LA 帕累托前沿中占据绝对统治地位（约占 2D-Conv 前沿点的 75%，DW-Conv 的 93%，FC 的 80%）。这验证了其在 **宽频率范围内优异的面积效率**。
    *   **竞争者**：**BW-ADD**（改进型 Baugh-Wooley）和 **HLS ST**（高级综合生成）虽非绝对最优，但在特定频率或面积约束下非常接近前沿，提供了可行的替代方案。
    *   **淘汰者**：基于专用多路复用器架构的设计（如 [17], [21], [22]）因逻辑门冗余度高，**未能在任何一种加速器的前沿中出现**。

*   **并行度 (OC/OA) 的杠杆效应**：
    *   **OC/OA 是控制 PPA 权衡的最强旋钮**。增加 OC 会线性增加 MAC 单元数量和存储体分区，导致面积和功耗上升，但能显著降低延迟。
    *   **IC/IA 的次级效应**：增大 IC/IA 能减少 tile 划分次数，从而摊薄循环控制开销，对降低整体延迟有正向贡献（尤其在 FC 层中效果显著）。

*   **频率与面积的耦合关系**：
    *   图中 **颜色梯度清晰地揭示了“高频=高面积”的物理规律**。若追求 <200 MHz 的极低功耗/面积设计，应选择左下角蓝绿色区域的点；若追求高性能，则必须接受 >800 MHz 时因时序收敛困难而导致的面积膨胀（红/黑色区域）。

*   **跨维度最优解的不一致性**：
    *   论文特别指出，**LA 最优点（黑标签）往往不是 LP (Latency-Power) 最优点（红标签）**。例如，在 2D-Conv 中，点 B 是 LA 最优，但同等延迟下点 C 的功耗更低（尽管面积大 1.6 倍）。这要求 SoC 设计师必须根据具体的 **“面积预算”或“功耗墙”** 来做出最终决策。


| 评估维度 | 观察结论 |
| :--- | :--- |
| **最优 ST 乘法器** | **Radix-4 Booth [7]** 在绝大多数帕累托最优设计中胜出，兼具高频适应性与面积效率。 |
| **面积-延迟敏感性** | **极高**。面积增加 10 倍（0.02→0.2 mm²）可带来延迟降低 2-3 个数量级的收益。 |
| **层类型差异** | **FC 层** 从 ST 架构中获益最大（延迟下降最陡）；**DW-Conv 层** 受限于核大小，利用率存在理论瓶颈。 |
| **HLS 设计可行性** | **HLS ST (Inline 模式)** 成功进入 DW-Conv 和 FC 的帕累托前沿，证明 **自动化的高层综合流程** 可产生媲美手工 RTL 的竞争性设计。 |
| **实际应用建议** | 对于 **边缘设备 (TinyML)**，应优先考虑 **0.03-0.08 mm²** 区间的中低频设计（如点 A/P 附近），以平衡能效与实时性要求。 |

### c21e5282c2d3da8a8f7dbf0b1e7beaed145b88484bb67d575c1175cd9ed0530a.jpg

![c21e5282c2d3da8a8f7dbf0b1e7beaed145b88484bb67d575c1175cd9ed0530a.jpg](images/c21e5282c2d3da8a8f7dbf0b1e7beaed145b88484bb67d575c1175cd9ed0530a.jpg)

- **图表核心信息**
    - 图表类型：**Latency vs Power (LP) 空间散点图**，展示三类 DNN 加速器（2D-Convolution、Depth-wise Convolution、Fully-Connected）的设计空间探索（DSE）结果。
    - 坐标定义：X 轴为 **Power (mW)**（功耗），Y 轴为 **Normalized Latency**（归一化延迟，对数坐标）。
    - 颜色映射：数据点颜色代表 **Clock Frequency (MHz)**（目标时钟频率），色谱从蓝色（100 MHz）过渡至红色（1000 MHz）。
    - 符号图例：区分 **7 种 Sum-Together (ST) 乘法器架构**及 **Pareto front**（帕累托前沿）。

- **符号与架构对应关系**

| 符号 | 架构名称 | 来源/类型 |
| :--- | :--- | :--- |
| □ (方框) | BW [6] | Baugh-Wooley ST (SoA) |
| ○ (圆圈) | D&C [21] | Divide-and-Conquer ST (SoA) |
| ✕ (叉号) | Booth [7] | Radix-4 Booth ST (作者先前工作) |
| ◇ (菱形) | [17] | RISC-V SIMD 扩展单元 (SoA) |
| ▽ (倒三角) | [22] | 可配置定点乘法器 (SoA) |
| ⬡ (六边形) | BW-ADD | 改进型 BW-ST（本文提出） |
| △ (正三角) | HLS ST | HLS 推导的 ST（本文提出） |
| + (加号) | HLS ST Inline | HLS 内联模式 ST（本文提出） |
| — (实线) | Pareto front | LP 空间帕累托最优边界 |

- **分图层深度解析**

    * **1. 2D-Convolution 加速器（顶层子图）**
        - 设计参数：**(m, n = IC, OC)**，即输入/输出通道数。
        - 功耗范围：约 **0.8 mW 至 100+ mW**。
        - 延迟范围：跨越 **4 个数量级**（约 0.0001 至 1）。
        - **Pareto 最优解分布**（黑标签）：主要集中在右下角低延迟区。**Booth [7]**（✕）架构在 **中高频段（绿色-红色，500-1000 MHz）** 占据主导地位，构成帕累托前沿的主体；**BW [6]**（□）与 **BW-ADD**（⬡）在 **低频低功耗区（蓝色，≤200 MHz）** 表现优异。
        - 关键标记点：
            - **点 B**：(32, 8) 配置，~400 MHz，IP 模式 Booth [7]，属于 LP-optimal。
            - **点 C**：(32, 16) 配置，~100 MHz，IP 模式 BW [6]，属于 LP-optimal 但面积更大。
            - **点 P**：位于极低功耗端，代表极致节能方案。

    * **2. Depth-wise Convolution (DW-Conv) 加速器（中层子图）**
        - 设计参数：**(n = OC)**，即输出通道数。
        - 功耗范围：相对紧凑，约 **0.6 mW 至 60 mW**。
        - 特殊性：由于 DW-Conv 的计算特性（无通道维度累加），其 **ST 乘法器利用率低于 2D-Conv 和 FC**，导致整体能效曲线形态差异明显。
        - **Pareto 最优解分布**：**HLS ST Inline**（+）在 **高频段（>500 MHz）** 展现出竞争力（如点 D）；**Booth [7]** 与 **BW** 系列在中低频段交替占优。
        - 关键标记点：
            - **点 D**：OC=32，~400 MHz，采用 HLS ST Inline，是同时接近 LA 和 LP 前沿的平衡解。

    * **3. Fully-Connected (FC) 加速器（底层子图）**
        - 设计参数：**(m, n = IA, OA)**，即输入/输出激活值数量。
        - 功耗范围：最宽泛，约 **1 mW 至 >100 mW**。
        - 延迟动态范围：极大，受 IA（输入维度）影响显著。
        - **Pareto 最优解分布**：呈现明显的 **频率分层现象**。**低频区（蓝-绿）** 由 **BW** 系列和 **HLS ST** 主导；**高频区（黄-红）** 几乎被 **Booth [7]**（✕）垄断。
        - 关键标记点对比（论文重点讨论）：
            - **点 E** (LA-optimal)：(1024, 4)，~100 MHz，IP Booth [7]，延迟极低但功耗非最低。
            - **点 F** (LP-optimal)：邻近点 E，采用 HLS ST Inline，功耗比 E 低约 18%。
            - **点 G** (LP-optimal)：(1024, 32)，IP D&C [21]，功耗最优但面积较大。
            - **点 H** (LA-optimal)：邻近点 G，采用 HLS ST，面积比 G 小约 8%。

- **关键工程洞察**
    - **无单一赢家**：不存在一种 ST 乘法器在所有 PPA（性能-功耗-面积）约束下均最优。选择需依据具体场景：
        - **低功耗优先**：首选 **BW [6]** 或 **BW-ADD**（<400 MHz）。
        - **低延迟/高性能优先**：首选 **Radix-4 Booth [7]**（>600 MHz）。
        - **设计灵活性/快速原型**：**HLS ST / HLS ST Inline** 可达到接近手动 RTL 优化的效果，尤其在 LP 空间中多次跻身 Pareto 前沿，验证了 **High-Level Synthesis (HLS)** 在此类加速器设计中的可行性。
    - **频率与架构的强耦合**：高频约束下（>800 MHz），逻辑综合工具更倾向于优化 **行为级描述（如 [17]）** 或 **结构规整的 Booth 架构**；而低频下，**阵列式结构（BW）** 的面积效率优势得以显现。
    - **DW-Conv 的瓶颈**：该层加速器的 Pareto 前沿整体偏右上（相对更高功耗和延迟），印证了文中关于 ST 乘法器在 DW-Conv 中存在 **硬件利用率不足**（需填充零操作数）的论断，这也是作者未来计划引入 **Sum-Apart (SA)** 乘法器的原因。

### eb7208a1d82337c82cfc81280c7a26be2d5e5ebf17bd0da30ceedec999537334.jpg

![eb7208a1d82337c82cfc81280c7a26be2d5e5ebf17bd0da30ceedec999537334.jpg](images/eb7208a1d82337c82cfc81280c7a26be2d5e5ebf17bd0da30ceedec999537334.jpg)

针对论文中的 **Figure 9: Example of a complete DSE for 2D-Conv**（2D卷积加速器的完整设计空间探索示例图），以下是详细的专业分析：

*   **图表核心主旨**
    该图展示了针对 **2D-Convolution (2D-Conv)** 加速器进行的**全量设计空间探索（Full DSE）** 结果。与论文中 Figure 7 和 Figure 8 仅展示 **Pareto 最优解** 不同，此图呈现了在 **Latency-Area-Power (LAP)** 三维空间中投影得到的**所有采样设计点**，直观反映了硬件参数变化对性能、面积和功耗的连续影响。

*   **双轴坐标系解析**
    *   **左侧子图（Latency vs. Area）**：
        *   **X轴**：硬件面积 (**Area**)，单位为 $mm^2$，对数刻度，范围约 **0.02 至 0.2+**。
        *   **Y轴**：归一化延迟 (**Normalized Latency**)，对数刻度，范围跨越 **4个数量级**（0.0001 至 1）。
        *   **趋势**：呈现出典型的**负相关权衡**——随着芯片面积增大（增加并行度或提高频率），推理延迟显著下降。

    *   **右侧子图（Latency vs. Power）**：
        *   **X轴**：平均功耗 (**Power**)，单位为 mW，对数刻度，范围约 **1 至 100+**。
        *   **Y轴**：同样为归一化延迟。
        *   **趋势**：同样呈现**负相关**——更高的动态功耗换取了更低的计算延迟。

*   **多维变量编码体系**
    图例通过**几何形状**区分了加速器 MAC 阵列中所采用的 **Sum-Together (ST) 乘法器架构类型**：

| 符号标识 | 对应的 ST 乘法器架构 | 来源/性质 |
| :--- | :--- | :--- |
| **□ (方形)** | BW [6] | Baugh-Wooley 子字并行架构 |
| **○ (圆形)** | D&C [21] | Divide-and-Conquer (分治) 架构 |
| **✕ (叉号)** | Booth [7] | Radix-4 Booth 编码架构 |
| **✳ (星号)** | [17] | RISC-V SIMD 扩展单元架构 |
| **▽ (倒三角)** | [22] | 可配置定点乘法器架构 |
| **◇ (菱形)** | **BW-ADD (ours)** | 本文提出的改进型 BW 乘法器 |
| **+ (加号)** | **HLS ST (ours)** | 本文提出的 HLS 推导乘法器 (IP 模式) |
| **△ (正三角)** | **HLS ST Inline (ours)** | 本文提出的 HLS 推导乘法器 (内联模式) |

    此外，数据点的**颜色**编码了**目标时钟频率 (Clock freq.)**：
    *   **冷色调（蓝/青）**：代表 **低频段 (100–300 MHz)**，通常对应**低功耗、小面积但高延迟**的设计点（位于图表左上方）。
    *   **暖色调（红/橙）**：代表 **高频段 (700–1000 MHz)**，对应**高性能、大面积且高功耗**的设计点（密集分布于图表右下方）。

*   **关键设计洞察与分布特征**
    *   **设计空间的广度与密度**：图中密布的数据点证实了该 DSE 过程的高度系统性。它遍历了表 3 中定义的所有**硬件配置旋钮 (Hardware Configuration Knobs)**，包括输入/输出通道数 (IC/OC)、PSMAC 阵列并行度 (M)、内存分区策略以及多种 HLS 指令（如 `unroll`, `pipeline`, `interleave`）。
    *   **频率的主导作用**：颜色梯度清晰地表明，**时钟频率是决定设计点在 L/A/P 空间位置的最主要因素之一**。高频设计强制逻辑综合工具使用更大、更快的逻辑门，从而推高了面积和功耗，但大幅削减了延迟。
    *   **架构类型的渗透性**：尽管某些 ST 乘法器（如 Booth [7] 和 BW-ADD）在 Pareto 前沿上表现更优（如 Fig 6 所示），但在完整的 DSE 空间中，**所有架构类型都覆盖了从低端到高端的性能区间**。这说明通过调整外围电路的并行度和频率，次优的乘法器 IP 仍可满足特定的非极端约束。
    *   **非线性缩放效应**：在低频区（蓝色区域），面积和功耗的小幅增加能带来延迟的大幅下降；而在高频区（红色区域），要进一步降低延迟需要付出**指数级增长的面积和功耗代价**（边际效益递减）。

*   **工程实践意义**
    此图为 SoC 架构师提供了**全局视角的选型参考**：
    *   若目标应用处于**严苛的能量/面积预算**下（如超低功耗 IoT 节点），应聚焦于图表**左上角的冷色区域**（低频、低并行度设计）。
    *   若追求**极致吞吐量**（如实时边缘视觉处理），则需向**右下角的暖色区域**探索，并接受相应的硅片成本和散热挑战。
    *   该图也验证了 **High-Level Synthesis (HLS)** 驱动的自动化 DSE 流程能够有效覆盖离散的手动设计盲区，为**精度可扩展 (Precision-Scalable)** 加速器的快速定制提供了坚实的量化基础。

### 2008513bcf17bb5d49ff21e263ef7377a810afd5c38bd3bc83060e4691178af6.jpg

![2008513bcf17bb5d49ff21e263ef7377a810afd5c38bd3bc83060e4691178af6.jpg](images/2008513bcf17bb5d49ff21e263ef7377a810afd5c38bd3bc83060e4691178af6.jpg)

- **图像基本信息**
  - **文件标识**: 2008513bcf17bb5d49ff21e263ef7377a810afd5c38bd3bc83060e4691178af6.jpg
  - **图像类型**: 黑白/灰度肖像照
  - **拍摄风格**: 正式学术/专业头像照

- **人物特征分析**
  - **身份识别**: **LUCA URBINATI** (论文第一作者)
  - **外貌描述**:
    - 年轻男性，约25-35岁年龄段
    - 深色短发，发型整齐，略带蓬松感
    - 浓眉，深色眼睛，眼神直视镜头
    - 面部轮廓清晰，鼻梁挺直
    - 面带轻微微笑，表情自然自信
    - 面部有短胡茬/胡须痕迹
  - **着装**: 圆领T恤或休闲上衣

- **摄影技术参数**
  - **构图**: 胸像/头肩肖像构图
  - **背景**: 纯白色/浅灰色背景，无干扰元素
  - **光线**: 正面柔光照明，光线均匀柔和
  - **焦点**: 清晰对焦于面部，特别是眼部区域
  - **景深**: 较浅，背景完全虚化（纯色背景）

- **与论文的关联性**
  - **作者身份**: Graduate Student Member, IEEE
  - **所属机构**: Department of Electronics and Telecommunications, Politecnico di Torino, Italy
  - **研究方向**: 
    - Integrated architectures for artificial intelligence (AI) and machine learning
    - Implementation of AI algorithms on embedded edge devices through HW–SW co-design approach
  - **教育背景**:
    - B.Sc. in electronics and telecommunications from Università di Bologna (2017)
    - M.S. in electronic engineering from Politecnico di Torino (2019)
    - Currently pursuing Ph.D. degree at Politecnico di Torino

- **图像用途推断**
  - 学术论文作者简介配图
  - 专业学术档案/个人主页头像
  - 会议演讲者介绍材料
  - 研究团队网站成员展示

- **图像质量评估**
  - **分辨率**: 中等偏高，适合网络发布和印刷
  - **对比度**: 适中，灰阶层次分明
  - **噪点控制**: 良好，无明显噪点
  - **专业程度**: 符合学术界标准的专业形象照要求

### b9845cf4093befb742213c8a58a1d89d7ef7ea88d32db1107e5ea580a6fab92d.jpg

![b9845cf4093befb742213c8a58a1d89d7ef7ea88d32db1107e5ea580a6fab92d.jpg](images/b9845cf4093befb742213c8a58a1d89d7ef7ea88d32db1107e5ea580a6fab92d.jpg)

- **图像基本信息**
  - 图像类型：**灰度/黑白学术肖像照**
  - 拍摄对象：**MARIO R. CASU** (Senior Member, IEEE)
  - 图像特征：正面半身像，背景为纯白色/浅灰色，人物穿着**带领衬衫**（collared shirt），呈现专业学术形象

- **人物身份确认**
  - 当前职位：**Politecnico di Torino**（都灵理工大学）电子与电信系 **Associate Professor**（副教授）
  - IEEE 会籍等级：**Senior Member**（高级会员）

- **学术背景概览**

| 属性 | 详细信息 |
|------|----------|
| **最高学位** | Ph.D. in electronics and communications engineering |
| **授予机构** | Politecnico di Torino, Italy |
| **获得时间** | 2001年 |

- **核心研究领域**
  - **Systems-on-Chip (SoC)** 设计，特别是配备**专用加速器**（specialized accelerators）的 SoC
  - 面向 **FPGAs 和 ASICs** 的系统级设计（system-level design）与设计方法论（design methodology）
  - **嵌入式机器学习**（Embedded Machine Learning）
  - 工业应用领域的电路、系统与平台设计，涵盖：
    - **生物医学**（Biomedical）应用
    - **汽车电子**（Automotive）应用  
    - **食品科技**（Food）应用
  - 历史研究方向：**SoC 和片上网络**（Networks-on-Chip）的**延迟无关设计**（Latency-insensitive Design）

- **学术服务与贡献**
  - 定期担任国际顶级会议的技术程序委员会（Technical Program Committee）成员，包括：
    - **DAC** (Design Automation Conference)
    - **ICCAD** (International Conference on Computer-Aided Design)
    - **DATE** (Design, Automation & Test in Europe)

- **与本论文的关联**
  - 作为**通讯作者**（Corresponding Author）或资深合作者，指导了关于 **Sum-Together (ST) Multipliers** 在 **Precision-Scalable DNN Accelerators** 中应用的研究工作
  - 其在 **SoC 架构**、**硬件加速器设计**和 **HLS（High-Level Synthesis）方法论**方面的专业知识为本论文提供了重要的技术支撑
  - 研究方向与本论文主题高度契合：**边缘设备上的深度学习推理加速**、**混合精度量化**（Mixed-Precision Quantization）、**PPA（Performance-Power-Area）优化**

- **视觉特征细节**
  - 发型：**深色卷发**（dark curly hair），中等长度
  - 面部表情：**微笑**（smiling），展现亲和力与自信
  - 着装风格：**商务休闲**（business casual），穿着浅色系带领衬衫
  - 拍摄角度：**平视角度**（eye-level shot），标准学术肖像构图
  - 光影处理：采用**柔和照明**（soft lighting），面部轮廓清晰，无明显阴影

