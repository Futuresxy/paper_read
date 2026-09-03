# SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference 图表详解

### 8331b7d3eec47620ed40e97e5bc9531e0d35f9fa0148bd4692cfc2c1f05f599e.jpg

![8331b7d3eec47620ed40e97e5bc9531e0d35f9fa0148bd4692cfc2c1f05f599e.jpg](images/8331b7d3eec47620ed40e97e5bc9531e0d35f9fa0148bd4692cfc2c1f05f599e.jpg)

- **图像内容识别**：该图片展示了**三个ACM (Association for Computing Machinery) 工件评估 (Artifact Evaluation) 认证徽章**，分别代表不同的验证级别。
- **具体徽章解析**：

| 徽章颜色 | 徽章文字 | 含义解读 |
| :--- | :--- | :--- |
| **绿色** | **Artifacts Available** | **工件可用性认证**：确认论文附带的代码、数据或模型等数字工件已公开存档且可供获取。 |
| **红色/洋红** | **Artifacts Evaluated: Functional** | **功能性评估认证**：评审者已成功执行了工件的核心功能，验证其能够运行并产生输出，但未深入验证结果的数值准确性或与论文声明的一致性。 |
| **蓝色** | **Results Reproduced** | **结果复现认证**：这是最高级别的认证。表明评审者利用提供的工件成功复现了论文中的关键实验结果（如性能图表、数据表格等），证明了研究的**可重复性 (Reproducibility)** 和透明度。 |

- **学术意义与背景关联**：
    - 该图片通常出现在通过 **ACM Artifact Review** 流程的高质量计算机系统/架构类论文中（如 ISCA, ASPLOS, OSDI 等顶级会议）。
    - 结合本文档《SMOOTH》的内容，这三个徽章共同证实了：作者不仅提出了理论框架，还提供了**完整的开源实现**（包括 Verilog 硬件代码和 LLMCompass 模拟器集成），并且**第三方独立验证**了其性能提升（如 TTFT 降低 59.2%，TTLT 降低 73.0%）和能效数据的真实性。
    - 这显著增强了论文结论的可信度，为后续研究者提供了可靠的基准 (Baseline) 和研究基础。
- **视觉设计特征**：
    - 采用**齿轮状锯齿边缘**的经典印章设计，中心嵌有 **ACM** 标志。
    - 颜色编码清晰：绿（存在）、红（功能）、蓝（复现），形成递进式的质量保证体系。

### 2143806eeb041743fe830fc2bc69787feb65f3770f1042ad37a0faff9f3d252c.jpg

![2143806eeb041743fe830fc2bc69787feb65f3770f1042ad37a0faff9f3d252c.jpg](images/2143806eeb041743fe830fc2bc69787feb65f3770f1042ad37a0faff9f3d252c.jpg)

- **图表核心主旨**：该图量化展示了**移动SoC统一内存架构下NPU可用空闲内存带宽的剧烈动态波动性**，揭示了静态编译器优化在真实多任务环境中的根本局限性。

- **实验配置与背景**：
    - **测试平台**：Samsung Galaxy S25+ (Snapdragon 8 Elite SoC)
    - **测量工具**：Geekbench 6
    - **负载类型**：涵盖三类典型移动场景——纯CPU负载（HTML5 Browser, Structure from Motion）、纯GPU负载（Background Blur, Horizon/Face/Edge Detection）以及CPU+GPU混合负载。

- **关键数据特征分析**：

| 负载类别 | 具体工作负载 | 带宽波动特征 | 对LLM推理的影响 |
| :--- | :--- | :--- | :--- |
| **CPU-only** | HTML5 Browser | 带宽相对平稳，维持在较高水平（接近BWmax） | 内存争用较轻，但仍有小幅抖动 |
| | Structure from Motion | 出现明显周期性下降 | 计算密集型CPU任务周期性抢占带宽 |
| **GPU-only** | Background Blur | 出现深度的、频繁的带宽谷值 | 图形渲染严重挤占NPU内存通道 |
| | Horizon/Face/Edge Detection | 呈现高频率、大幅度的尖峰与深谷交替 | 视觉AI任务导致带宽极度不稳定 |
| **CPU + GPU** | 混合场景 | **波动最为剧烈且不可预测** | 最恶劣情况，可用带宽窗口短且随机 |

- **对SMOOTH设计的动机支撑**：
    - **证伪静态假设**：图中可见，空闲带宽并非恒定值，而是随并发 workload 类型呈现**高度时变性**（从接近饱和到几乎归零）。
    - **解释编译器失效根因**：传统编译器（如XLA/TVM）在编译时固定tile大小，无法适应这种**运行时带宽抖动**。如图所示，若按高带宽环境优化，在GPU检测任务运行时会导致严重内存饥饿；反之则浪费算力。
    - **凸显硬件动态管理的必要性**：这种不可预测的“**带宽 Slack**”（Bandwidth Slack）必须由硬件在运行时实时感知并利用，这正是SMOOTH引入**硬件辅助的Dynamic Memory Controller (DMC)** 和**早期回收机制（Early Reclamation）**的直接动因——利用这些转瞬即逝的带宽窗口进行细粒度数据预加载（Preloading）。

### 5b1a85c29b7f6e63da1440735af58c10b256df85cc60d1661fb67211ba00932e.jpg

![5b1a85c29b7f6e63da1440735af58c10b256df85cc60d1661fb67211ba00932e.jpg](images/5b1a85c29b7f6e63da1440735af58c10b256df85cc60d1661fb67211ba00932e.jpg)

针对图片 5b1a85c29b7f6e63da1440735af58c10b256df85cc60d1661fb67211ba00932e.jpg（即论文 Fig. 4b）的详细分析如下：

*   **图表基本信息**
    *   **标题/主题**: Memory bandwidth utilization over time during decoding layer inference（解码层推理过程中的内存带宽利用率随时间变化）
    *   **实验对象**: LLaMA-3 模型在 Constrained-SoC（受限 SoC）配置下的推理过程
    *   **坐标轴定义**:
        *   **X轴 (横轴)**: Time [s]，表示推理执行时间，范围覆盖约 0 至 160 秒。
        *   **Y轴 (纵轴)**: Utilization [%]，表示内存带宽利用率百分比，范围从 0% 到 80%。

*   **核心视觉特征与数据解读**
    *   **剧烈的周期性波动 (Bursty Pattern)**: 图像最显著的特征是带宽利用率呈现**极不稳定的锯齿状波动**，而非平稳的直线或平滑曲线。这直观地验证了论文指出的 LLM 推理具有**突发性 I/O 特征**。
    *   **峰值与谷值的交替**:
        *   **高峰期 (Peaks)**: 利用率多次飙升至 **60%-70%** 甚至更高。这对应 Transformer 解码器中的**线性操作阶段**（如 QKV 投影、权重矩阵加载），这些操作具有**低运算强度 (Low-OI)**，属于典型的 I/O 密集型任务，会迅速耗尽有限的内存带宽。
        *   **低谷期 (Valleys)**: 利用率频繁跌落至接近 **0%-20%** 的水平。这对应**非线性操作阶段**（如 Softmax、GELU 激活函数计算），这些操作具有**高运算强度 (High-OI)**，主要依赖计算单元的向量算术能力，导致内存总线处于严重的**闲置状态 (Idle)**。

*   **深层技术含义与问题揭示**
    *   **资源利用率的极端不平衡**: 该图量化了移动端 LLM 推理的痛点——**计算资源与内存资源无法同时被高效利用**。当计算单元忙于非线性运算时，宝贵的内存带宽却被浪费；反之，当需要加载数据时，带宽又瞬间饱和导致计算单元停顿 (Stall)。
    *   **静态编译优化的失效证据**: 这种**短时、不可预测的带宽空闲窗口 (Bandwidth Slack)** 是现代深度学习编译器（如 XLA, TVM）难以捕捉的。编译器通常基于**静态的 Tile 大小和生命周期分析**进行数据预取调度，但面对如此高频且幅度巨大的动态波动，固定的编译时策略必然导致**预取不足或预取时机错误**，从而引发性能下降（论文指出延迟可能恶化高达 2.9 倍）。
    *   **SMOOTH 方法的必要性论证**: 此图构成了提出 SMOOTH 硬件辅助内存管理框架的核心动机。只有通过**细粒度的块级分配 (Block-based Allocation)** 和**硬件驱动的早期回收机制 (Early Reclamation)**，系统才能在这些转瞬即逝的带宽空闲期内，灵活地抢占碎片化的 SRAM 空间并**主动预加载 (Preload)** 后续所需的数据（如下一层的权重），从而“抹平”这种突发流量，实现计算与 I/O 的重叠 (Overlap)。

*   **总结**
    Fig. 4b 通过实测数据揭示了在资源受限的移动 SoC 上运行 LLM 时，内存子系统面临的**极度不稳定的负载特征**。它证明了传统的粗粒度、静态内存管理方案在面对 Transformer 模型固有的**相位交替 (Phase-alternating)** 执行模式时存在根本性缺陷，为引入具备**运行时动态感知能力**的硬件辅助架构提供了决定性的实证支持。

### 6d673a96c6bcaaa4d8b2944bfdf3f5e529897ec32f8126e38fbbda372af5a0ba.jpg

![6d673a96c6bcaaa4d8b2944bfdf3f5e529897ec32f8126e38fbbda372af5a0ba.jpg](images/6d673a96c6bcaaa4d8b2944bfdf3f5e529897ec32f8126e38fbbda372af5a0ba.jpg)

- **图表类型与核心指标**
    - 该图为**堆叠柱状图 (Stacked Bar Chart)****，展示 GPT-3 模型在不同移动端推理平台上的**端到端延迟分解 (End-to-end Latency Breakdown)**。
    - **Y轴**：归一化延迟值 (Normalized Latency)，基准为 1.0。
    - **X轴**：三组实验配置，涵盖 **Jetson AGX Orin**、**Samsung Galaxy S24 (S24)** 和 **Google Edge TPU** 三种硬件平台，每种平台测试了 **w4a8** 和 **int8** 两种量化格式。

- **数据构成与关键观察**
    
    | 平台 | 量化格式 | Non-linear 占比 (估算) | Linear 占比 (估算) |
    | :--- | :--- | :--- | :--- |
    | Jetson | w4a8 | ~15% | ~85% |
    | Jetson | int8 | ~12% | ~88% |
    | S24 | w4a8 | ~18% | ~82% |
    | S24 | int8 | ~15% | ~85% |
    | Edge TPU | w4a8 | ~20% | ~80% |
    | Edge TPU | int8 | ~17% | ~83% |

    - **Linear 操作绝对主导**：在所有测试场景下，**Linear 操作**（深灰色区域，主要包括矩阵乘法 GEMM/GEMV、投影层等）占据了 **80%-88%** 的总执行时间。这证实了 LLM 推理在移动设备上属于典型的 **Memory-bound (I/O-bound)** 工作负载。
    - **Non-linear 操作占比稳定但不可忽视**：**Non-linear 操作**（浅蓝色区域，包括 Softmax、GELU、LayerNorm 等）虽然仅占 **12%-20%** 的延迟，但其存在创造了**带宽空闲窗口 (Bandwidth Slack)**。这些操作计算密度高 (High Operational Intensity)，对内存带宽需求低。
    - **跨平台行为一致性**：尽管硬件架构差异显著（GPU vs NPU vs TPU），三种设备的延迟分布模式高度相似。这验证了论文使用 **Constrained-SoC**（受限资源配置的 Jetson）来模拟通用移动端行为的合理性。

- **在论文论证中的关键作用**
    - **量化 Memory Bottleneck**：该图直观证明了移动端 LLM 推理的性能瓶颈不在于计算能力，而在于**数据搬运 (Data Movement)**。Linear 层需要加载巨大的权重矩阵，导致内存带宽饱和。
    - **为 SMOOTH 提供动机**：图中显示的 Non-linear 阶段正是 **SMOOTH** 框架利用的关键时机。SMOOTH 通过**细粒度块级分配 (Fine-grained Block Allocation)** 和**硬件驱动的早期回收 (Hardware-driven Early Reclamation)**，在这些“空闲”时间段内**预加载 (Preload)** 后续 Linear 层所需的权重数据，从而掩盖内存访问延迟。
    - **验证动态优化必要性**：不同量化格式 (w4a8 vs int8) 下 Non-linear/Linear 比例的微小变化，结合前文提到的序列长度变化，进一步说明**静态编译器 (Static Compiler)** 难以完美适配所有运行时状态，支持了引入**运行时硬件辅助管理 (Runtime Hardware-assisted Management)** 的必要性。

### db97416461dd7bae52a07f0a79f440cc21425a48384b14982f55a7ca4bd199a5.jpg

![db97416461dd7bae52a07f0a79f440cc21425a48384b14982f55a7ca4bd199a5.jpg](images/db97416461dd7bae52a07f0a79f440cc21425a48384b14982f55a7ca4bd199a5.jpg)

- **图像基本信息**
  - 图表编号：Fig. 7(b)
  - 标题：SRAM fragmentation with fusion（融合操作下的SRAM碎片化）
  - 坐标轴定义：
    - **X轴**：Time（时间），表示推理执行的时间进程（约50-100个时间单位）
    - **Y轴**：Memory Address（内存地址），表示SRAM的地址空间（0-2MB范围）
  - 图表类型：**堆叠式内存占用图**（Stacked Memory Occupancy Plot）

- **视觉特征与核心现象**
  - **严重的内存碎片化（Fragmentation）**：图中清晰显示SRAM地址空间存在大量**不连续的空闲间隙**（白色/空白区域）
  - **非紧凑分配模式**：不同颜色块（代表不同tensor/tile）之间出现明显的**垂直方向断裂**，表明物理内存未被连续利用
  - **动态生命周期重叠**：多个彩色条带在时间轴上并行存在，反映operator fusion导致的**缓冲区生命周期延长和相互重叠**

- **技术机制解析**
  - **根本原因**：现代深度学习编译器采用的**contiguous allocation（连续分配）策略**要求每个tensor/tile必须映射到物理上连续的SRAM区域
  - **Fusion的副作用**：QKV Projection Fusion、FlashAttention、FFN Fusion等优化将多个操作合并为单一内核，强制Q/K/V激活等中间结果**同时驻留**在SRAM中
  - **碎片化形成过程**：
    
    | 阶段 | 内存行为 | 后果 |
    |------|----------|------|
    | 分配阶段 | 大尺寸tile占据连续空间 | 产生大块占用区域 |
    | 执行阶段 | 融合操作延长tensor生命周期 | 相邻空间无法提前释放 |
    | 释放阶段 | 非均匀释放产生不规则空洞 | 外部碎片（External Fragmentation）累积 |
  
  - **量化影响**：如论文所述，这种碎片化导致Compiler-Ideal策略在4K token长度下仍产生**32.7%的额外计算停顿周期**

- **与SMOOTH方案的对比意义**
  - **现有方案局限**：图中展示的**锯齿状内存足迹**正是传统编译器管理（Compiler-Ideal baseline）的典型缺陷——即使拥有完美的生命周期知识，仍受制于连续分配约束
  - **SMOOTH的解决路径**：
    - 采用**Block-based allocation（基于块的分配）**打破连续性要求
    - 通过**硬件驱动的早期回收（Early Reclamation）**及时释放已用完的块
    - 利用**非连续物理空间**填充图中可见的碎片间隙
  - **预期改善**：SMOOTH可将此类碎片化场景下的SRAM利用率从约60-70%提升至接近**理论最优值**

- **实验上下文关联**
  - 该图对应论文§III.B节的模拟实验，使用**2MB SRAM**配置
  - 测试模型应用了三种代表性融合优化（QKV Fusion、Flash Attention、FFN Fusion）
  - 与Fig. 7(a)的参数生命周期图配合，完整展示了**从逻辑依赖到物理碎片**的因果链
  - 为后续Fig. 8对比四种内存管理策略提供了**问题基线（Problem Baseline）**

### Fig. 9. On-chip memory management strategies for contiguous and noncontiguous memory cases.

![fa1ef2d2c6aa3d98b519513f8bf90f09c46cb199e0ca2b9c81c724d0674a31a2.jpg](images/fa1ef2d2c6aa3d98b519513f8bf90f09c46cb199e0ca2b9c81c724d0674a31a2.jpg)

根据提供的论文内容和图表，我对 **Fig. 9. On-chip memory management strategies for contiguous and noncontiguous memory cases** 进行详细分析如下：


该图对比了**四种不同的片上内存（SRAM）管理与数据预加载策略**，分为两大场景：**连续内存分配**（上半部分）和**非连续内存分配**（下半部分）。


此部分展示传统编译器驱动的**粗粒度、连续地址分配**方式的局限性：

- **(a) 初始状态**：
  - 内存中已分配的数据块包括：**V₀, V₁, V₂**（通常代表Key/Value缓存或中间激活值）、**S₀, S₁, S₂**（Softmax或其他操作结果）
  - 右侧存在明显的 **"wasted"（浪费）** 空间，即由于连续性要求导致的内部碎片

- **(b) 尝试预加载新数据（$V₄）**：
  - 当需要预加载新的数据块 **$V₄** 时，由于必须寻找**连续的物理空间**
  - 系统无法利用V₀-V₂和S₀-S₂之间可能存在的细小间隙
  - 只能将$V₄放置在内存末端，**原有的"wasted"空间依然无法利用**

- **(c) 预加载限制**：
  - 即使存在足够的总空闲空间，但由于**空间不连续**，无法容纳较大的数据块（如权重矩阵W）
  - 预加载操作被阻塞，导致后续计算时必须等待数据从DRAM加载，**产生计算停顿（stall）**


此部分展示 **SMOOTH 提出的块级（Block-based）虚拟化分配**的优势：

- **(a) 初始状态（块级视图）**：
  - 内存被划分为固定大小的**块（Blocks）**，数据以块为单位分散存储
  - 包含多个数据块：**$V₀, $V₁, V₂, $V₂, W₀, W₀₁, S₁, W₀₁** 等
  - 物理上不连续，但通过**块表（Block Table）**维护逻辑连续性

- **(b) 外部碎片化问题（External Fragmentation）**：
  - 在传统连续分配中，释放部分数据后会产生**不连续的空闲区域**（图中白色间隙）
  - 这些分散的小块空间无法被合并用于分配大块数据

- **(c) 内部碎片化与预加载（Internal Fragmentation + Preload）**：
  - 块级分配引入**内部碎片**（最后一个块可能未完全使用），如图中浅色小块
  - **关键优势**：系统可以**利用所有分散的空闲块**来预加载新数据（如W₁₀, W₁₁等权重）
  - 图中显示 "preload new data" 可以填充到多个不连续的位置

- **(d) 回收与预加载机制（Reclaim and Preload）**：
  - 这是 **SMOOTH 的核心创新——硬件驱动的早期回收（Early Reclamation）**
  - 当某些数据块（如V₃, S₃）不再被使用时，硬件立即**回收（reclaim）**这些块
  - 回收后的空间**立即用于预加载**后续所需数据（如W₁₁）
  - 实现了**计算与I/O的重叠（Overlap）**，最大化带宽利用率


| 策略类型 | 分配粒度 | 碎片处理 | 预加载能力 | 适用场景 |
|---------|---------|---------|-----------|---------|
| **(a) 硬件缓存** | 缓存行（64B） | 无外部碎片 | 盲目预取，缺乏未来知识 | 传统CNN/DNN |
| **(b) 编译器理想** | 张量/Tile级 | 严重外部碎片 | 受限于连续空间 | 静态优化LLM |
| **(c) 块级分配+编译器预取** | 固定块（如1KB） | 消除外部碎片，有内部碎片 | 可利用分散空间 | SMOOTH-Base |
| **(d) 块级分配+早期回收+激进预取** | 固定块 | 动态回收，最小化碎片 | **最大化利用瞬态带宽** | **SMOOTH-ER** |


1. **连续性约束是性能瓶颈的根本原因**：
   - 上半部分清晰表明，即使总空闲空间足够，**物理连续性要求**也会导致高达32.7%的额外停顿周期（如论文Fig. 7c所示）

2. **块级虚拟化的双重模式设计**：
   - SMOOTH在**无碎片时自动切换到连续快速路径**（零开销），**有碎片时启用块级虚拟化**（灵活性）
   - 这解决了传统SPM（Scratchpad Memory）无法处理LLM不规则访问模式的问题

3. **早期回收（Early Reclamation）的时间价值**：
   - 子图(d)展示了**生命周期结束即回收**而非等待显式释放的重要性
   - 在LLM解码阶段，**非线性操作（高OI）期间的空闲带宽**可被用于预加载**线性操作（低OI）所需的权重**

4. **对FlashAttention等融合操作的适配**：
   - 图中V和S数据的交错分布反映了**算子融合（Operator Fusion）**带来的复杂生命周期
   - 传统方法因融合导致缓冲区寿命延长而加剧碎片化，SMOOTH通过块级管理有效缓解此问题


该图直观解释了为何SMOOTH-ER能在实验中实现：
- **TTFT（首token延迟）降低59.2%**：通过在提示阶段（Prompt Phase）利用块级预加载隐藏内存延迟
- **TTLT（末token延迟）降低73.0%**：在生成阶段（Generation Phase）通过早期回收持续维持高SRAM占用率（如论文Fig. 16c所示）
- **能耗降低51.2%**：减少不必要的DRAM访问和计算停顿

总之，Fig. 9 是理解SMOOTH架构优势的核心示意图，它从**空间利用（碎片消除）**和**时间利用（带宽抢占）**两个维度，阐释了硬件辅助的细粒度内存管理如何解决移动端LLM推理的内存墙问题。

### Fig. 12. (a) Design component of SMOOTH. (b) Access with address translation. (c) Direct access without block table lookup. (d) Access with end\_cmd for early reclamation. (e) Reclaim blocks that ensure data integrity. (f) Preload data into reclaimed blocks using idle bandwidth.

![13c00ffd06053912186040831c9fb390217dae94f411cef97086079723faa2b8.jpg](images/13c00ffd06053912186040831c9fb390217dae94f411cef97086079723faa2b8.jpg)

- **整体架构概览**
  - 图 12 展示了 SMOOTH (SMOothing I/O Traffic with Hardware support) 的核心微架构设计与运行时操作流程
  - 该架构由 **Buffer**（缓冲区）和 **DMC**（Dynamic Memory Controller，动态内存控制器）两大核心模块协同工作
  - 底层依赖 **Bitmap**（位图）和 **BlockTable**（块表）两种数据结构实现细粒度内存管理

- **子模块功能分解**

| 组件 | 位置 | 核心功能 |
|------|------|----------|
| **address_check** | Buffer 内 | 判断当前访问是否需要地址转换，输出 Hit/Miss 信号及 lookup_flg 控制标志 |
| **alloc** | DMC 内 | 执行物理块分配逻辑，支持连续与非连续分配策略 |
| **find_zero** | DMC 内 | 在 Bitmap 中快速定位最长连续空闲区域 |
| **block_table_lookup** | DMC 内 | 执行虚拟地址到物理地址的映射查询 |
| **free** | DMC 内 | 执行块回收，更新 Bitmap 和 BlockTable 状态 |
| **Bitmap[1KB]** | 存储结构 | 以位形式记录每个物理块（1KB 粒度）的分配/空闲状态 |
| **BlockTable[56KB]** | 存储结构 | 存储每个虚拟块的映射元数据：p_blk（物理块号）、cont（连续块数）、use_cnt（剩余使用计数） |

- **操作时序详细分析（b-f 子图）**

  - **阶段 (b): 首次访问与地址转换**
    - 操作：`Load a(0x05), lookup_flg=1`
    - 行为：Buffer 发起对虚拟地址 0x05 的访问，因 lookup_flg=1 触发 DMC 执行 **block_table_lookup**
    - 映射结果：虚拟块 0x05 → 物理块 **p_blk=0x2400**，连续长度 **cont=4**（占用 0x2400-0x27FF），初始 **use_cnt=2**
    - 数据流：DMC 返回物理地址及 cont 信息，数据 a 被加载至 SRAM 对应区域

  - **阶段 (c): 连续地址快速路径**
    - 操作：`Load b(0x2500), lookup_flg=0`
    - 关键优化：由于地址 0x2500 落在已缓存的连续范围 [0x2400, 0x27FF] 内，**绕过 block_table_lookup**
    - 机制：Buffer 内部维护当前连续窗口的 (p_blk, cont) 信息，通过比较高位地址位判断是否越界
    - 性能收益：消除查表延迟，实现 **零开销（zero-overhead）** 连续访问

  - **阶段 (d): 早期回收触发**
    - 操作：`Load d(0x2700), lookup_flg=0, end_cmd=1`
    - 语义：end_cmd=1 表示这是当前操作对该块的**最后一次访问**
    - 硬件行为：DMC 收到 end_cmd 后立即递减对应 BlockTable 条目的 **use_cnt**（从 2→1）
    - 设计意义：无需等待软件显式释放指令，硬件自主追踪数据生命周期

  - **阶段 (e): 块回收与状态一致性保障**
    - 触发条件：当某块的 **use_cnt 递减至 0** 时进入回收流程
    - 两阶段原子操作：
      1. 先更新 **BlockTable** 将条目标记为无效（图中显示 use_cnt 清零或标记为 ER）
      2. 再清除 **Bitmap** 对应位（从 1→0）
    - 一致性保证：严格顺序确保 alloc 模块在 Bitmap 更新前不会将正在回收的块重新分配，防止数据损坏

  - **阶段 (f): 空闲带宽利用与预加载**
    - 预加载量计算：**N_preload = ⌊(U × BW) / Block_size⌋**，其中 U 为检测到的空闲周期数，BW 为可用内存带宽
    - 图示案例：N_preload = 3，系统识别出 3 个刚回收的空闲块（红色标记区域）
    - 执行动作：DMC 利用计算/非线性操作阶段的**内存带宽闲置窗口**，主动从 DRAM 预加载后续层所需权重（如 W10、W11 等）
    - 结果：BlockTable 和 Bitmap 同步更新，新数据就绪于 SRAM，等待后续计算直接命中

- **架构创新点总结**
  - **双模式混合设计**：在碎片化场景下启用细粒度块虚拟化（模式 b），在连续访问时自动切换至翻译旁路快速路径（模式 c），兼顾灵活性与效率
  - **硬件感知的生命周期管理**：通过 end_cmd 机制实现 **sub-tile 级别的早期回收**，突破传统 SPM 必须等待整个 tensor/tile 释放的限制
  - **带宽感知的主动预取**：将 LLM 推理中**突发式（bursty）的带宽空闲期**转化为数据准备时间，显著降低 Memory-bound 阶段的 stall cycles
  - **极低硬件开销**：根据论文 Table I/II 数据，该控制逻辑仅占 NPU 面积的 **0.0973%**，单次操作延迟在亚纳秒级（1508.2 ps for alloc），相对于数十微秒级的 DRAM 访问延迟可忽略不计

### 361f5165810b0a50018300757f1087ccf81fcbf219df10634428133d68379df9.jpg

![361f5165810b0a50018300757f1087ccf81fcbf219df10634428133d68379df9.jpg](images/361f5165810b0a50018300757f1087ccf81fcbf219df10634428133d68379df9.jpg)

- **图像基本信息**
    - 图表编号：Fig. 20 (c) - **32K-th Token** 能量消耗分析
    - 展示内容：生成第 **32K 个 token** 时，不同 **Block Size** 配置下的能量消耗对比
    - 包含模型：**GPT-3 2.7B (w4a8)** 和 **TinyLLaMA (w4a8)**

- **坐标轴定义**
    - **X轴**：Block Size（块大小），从 **1K 到 32K** 不等
    - **左侧Y轴**：Improvement [%]（相对改进百分比，以 baseline 为基准）
    - **右侧Y轴**：绝对能量值（具体单位未在图中标注，对应原始能耗）

- **关键数据趋势与观察**

| 模型 | 核心发现 | SMOOTH-ER 相对 Gemmini 的节能 |
|------|----------|-------------------------------|
| **GPT-3 2.7B** | 长序列(32K)下**内存瓶颈极端严重**，传统方法能耗激增 | 节能可达 **~70%** 级别（论文称最高 70.7%） |
| **TinyLLaMA** | 小模型在长序列下同样面临**带宽饱和**问题 | 保持稳定的高效节能优势 |

- **核心机制解析**
    - **Block Size 敏感性**：较小的 Block Size（如 1K-4K）通常能实现更细粒度的 **preloading**（预加载）和 **memory reuse**（内存复用），从而降低能耗；但过小的块会增加 **block table lookup**（块表查找）的开销
    - **SMOOTH-ER 的优势来源**：
        - **Early Reclamation（早期回收）**：硬件驱动机制在数据消费后立即释放 block，避免无效占用
        - **Aggressive Preloading（激进预加载）**：利用非线性操作（Softmax/GELU）期间的 **idle bandwidth**（空闲带宽）提前加载数据
        - **Fragmentation Tolerance（碎片容忍）**：块级虚拟化允许利用分散的物理内存空间，提高 SRAM 利用率

- **与论文结论的印证**
    - 论文明确指出："energy reduction over Gemmini steadily scales from **28.1% at 1K** to a remarkable **70.7% at 32K**"
    - 该图直观展示了在 **32K-th Token** 这一长序列场景下，**SMOOTH-ER**（深色柱状图）相比 **Compiler-Ideal**、**Gemmini**、**Capuchin** 等 baseline 实现了显著的能量降低
    - **Hardware Overhead（硬件开销）**：即使在 32K 长序列下，SMOOTH 引入的额外硬件控制开销也**极低**（论文称峰值仅 **15 nano-joules**），相对于整体节能收益可忽略不计

- **工程意义**
    - 对于**移动端长上下文 LLM 推理**（如长文档处理、多轮对话），SMOOTH 能够有效缓解 **memory-bound** 带来的能耗爆炸问题
    - 为 **On-Device AI** 在电池受限设备上的部署提供了可行的微架构优化方案

### cbaea7303d289c9af7a1b13c747e15c92cd9df14e944dd62ab74736434ea0613.jpg

![cbaea7303d289c9af7a1b13c747e15c92cd9df14e944dd62ab74736434ea0613.jpg](images/cbaea7303d289c9af7a1b13c747e15c92cd9df14e944dd62ab74736434ea0613.jpg)

这张图展示了**在启用操作融合（Operation Fusion）条件下**，不同内存管理策略在生成第N个Token时的**逐Token延迟（Latency）**与**片上SRAM平均占用率（Occupancy）**的对比分析。

*   **图表类型**: 组合图（柱状图 + 折线图）
*   **对比模型**: GPT-Neo (w4s8) [左] 与 LLaMA2 (w4s8) [右]
*   **X轴变量**: 输出序列长度（Output Length），分别为 1, 2K, 4K, 8K tokens。
*   **左Y轴**: 延迟（Latency），单位为毫秒（ms）。
*   **右Y轴**: SRAM占用率（Occupancy），百分比形式。
*   **评估策略**:
    *   **Compiler-Ideal** (浅灰): 理想化编译器策略（基线）。
    *   **Capuchin** (浅蓝): 硬件缓存管理策略。
    *   **Gemmini** (深灰): 流水线式字节级预加载策略。
    *   **SMOOTH-Base** (深蓝): 本文提出的块粒度基础分配器。
    *   **SMOOTH-ER** (亮蓝): 本文提出的带早期回收机制的完整方案。


随着生成长度的增加，所有方法的单Token延迟均呈上升趋势，这主要归因于**KV Cache（键值缓存）**的增长导致内存带宽压力加剧。

*   **SMOOTH-ER 的显著优势**:
    *   在 **GPT-Neo** 模型中，当输出长度达到 **8K** 时，SMOOTH-ER 的延迟显著低于其他基线方法。相比 Compiler-Ideal，其延迟降低幅度最为明显。
    *   在 **LLaMA2**（更大参数模型）中，这种优势更加突出。在长序列（4K, 8K）生成阶段，SMOOTH-ER 的柱状图高度明显低于 Gemmini 和 Compiler-Ideal，证明了其在处理**内存密集型长上下文推理**时的有效性。

*   **基线方法的瓶颈**:
    *   **Compiler-Ideal** 虽然经过理想化调优，但在长序列下受限于**连续物理地址分配导致的内存碎片化（Fragmentation）**，无法有效利用零散的SRAM空间进行预加载，导致延迟激增。
    *   **Gemmini** 依赖流水线预加载下一块数据，但在碎片化严重或带宽极度受限时，其灵活性不如 SMOOTH 的块级虚拟化。
    *   **Capuchin** 作为硬件缓存方案，缺乏对张量生命周期的先验知识，难以进行前瞻性预加载，在融合操作后表现平平。

折线图反映了不同策略对宝贵片上存储资源的利用效率。

*   **高占用率与低延迟的相关性**:
    *   观察可知，**SMOOTH-ER**（亮蓝折线）在大多数情况下维持了**较高且稳定的SRAM占用率**。这意味着该方案能更充分地填满SRAM，减少因空间闲置造成的浪费。
    *   通过**硬件驱动的早期回收（Early Reclamation）**机制，SMOOTH-ER 能够在数据被消费后立即释放块（Blocks），并将其重新用于预加载后续层所需的权重或KV Cache，从而实现了“**用完即释，腾出即载**”的高效周转。

*   **操作融合（Fusion）的影响**:
    *   与无融合情况（Fig. 16b）相比，启用融合后（本图），各策略的SRAM占用率曲线波动模式发生变化。融合操作延长了中间张量的生命周期，但 SMOOTH-ER 通过细粒度的块管理，成功缓解了融合带来的碎片化压力，保持了较高的空间利用率。

*   **长序列生成的杀手锏**: SMOOTH-ER 在 **8K tokens** 及以上的长序列生成中展现出最大的性能收益（相比基线最高可降低约 **60%-73%** 的 TTLT，如正文所述），有效解决了移动端LLM推理中“**越生越慢**”的痛点。
*   **架构创新的价值**: 该图直观证明了**打破连续分配限制（Block Virtualization）**结合**运行时主动回收（Hardware-driven Reclamation）**，能够将传统静态编译器无法利用的“**内存带宽空闲窗口（Bandwidth Slack）**”转化为实际的预加载时间，从而平滑了LLM推理中剧烈的I/O突发流量（Bursty Traffic）。

### 60e4b8c9f952175c430d41c99ba48d723966ddd774f20993669eae93c39beba2.jpg

![60e4b8c9f952175c430d41c99ba48d723966ddd774f20993669eae93c39beba2.jpg](images/60e4b8c9f952175c430d41c99ba48d723966ddd774f20993669eae93c39beba2.jpg)

- **图表核心主题**：该图为 **Figure 17**，展示不同内存管理策略在**注意力操作（Attention Operation）结束时**的 **SRAM 占用率归一化对比**（以 Compiler-Ideal 为基准归一化为 1.0）。

- **实验变量与维度**：
  - **模型**：左侧为 **GPT-Neo**，右侧为 **LLaMA2**
  - **序列长度（Sequence Length）**：横轴涵盖 **1、2K、4K、8K** 四个梯度，模拟从短文本到长上下文的推理场景
  - **管理策略**：每组包含三种方法的对比——**Compiler-Ideal**（编译器理想基线）、**SMOOTH-Base**（基础块分配）、**SMOOTH-ER**（带早期回收的增强版）

- **Y轴指标解读**：
  - **Norm. Latency（归一化延迟）**：数值 > 1.0 表示相对于基线产生额外开销或效率下降；数值 < 1.0 表示性能提升。纵轴范围 0–2。

- **堆叠成分分解（按图例）**：
  
  | 颜色/图案 | 操作组件 | 计算性质 |
  |-----------|----------|----------|
  | 粉色 | LN (MHA) | 注意力层的 Layer Normalization |
  | 深蓝 | Q Proj | Query 投影（线性层） |
  | 橙色 | K Proj | Key 投影（线性层） |
  | 绿色 | V Proj | Value 投影（线性层） |
  | 浅橙 | Atun | 注意力计算核心（Softmax 等） |
  | 浅蓝 | LN (FFN) | 前馈网络的 Layer Normalization |
  | 蓝绿 | GELU | 激活函数（非线性） |
  | 红色 | W0 Proj | FFN 第一层权重投影 |
  | 紫色 | W1 Proj | FFN 第二层权重投影 |
  | 棕色 | W2 Proj | FFN 第三层权重投影 |
  | **斜线填充** | **Hit** | **片上 SRAM 命中（预加载成功）** |
  | **白色/空白** | **Miss** | **片上缺失（需访问 DRAM）** |

- **关键数据趋势与洞察**：

  - **短序列（Length=1）**：三种策略的总柱高接近 1.0–1.2，差异微小。此时 KV Cache 开销极小，SRAM 充裕，各方法均能较好地容纳数据。

  - **中长序列（2K–4K）**：
    - **Compiler-Ideal** 的柱高开始显著攀升（尤其在 GPT-Neo 4K 时接近 1.6），主要源于**外部碎片化**导致的有效容量下降，迫使更多 **Miss**（白色区域）操作。
    - **SMOOTH-Base** 通过块级虚拟化缓解碎片化，柱高控制在 1.3–1.4 左右。
    - **SMOOTH-ER** 表现最优，柱高维持在 **1.1–1.25** 区间，**斜线（Hit）占比明显更高**，证明早期回收机制释放的空间被有效用于预加载后续数据。

  - **长序列（8K）——性能分水岭**：
    - **GPT-Neo @ 8K**：Compiler-Ideal 飙升至近 **1.8**，而 SMOOTH-ER 仅约 **1.3**，**相对优化幅度达 ~28%**。
    - **LLaMA2 @ 8K**：由于模型更大（7B vs 1.3B），所有策略的绝对延迟更高，但 **SMOOTH-ER 相对 Compiler-Ideal 的优势依然稳定在 20–25%**。
    - **Hit/Miss 结构变化**：在 SMOOTH-ER 的柱状图中，**斜线区域（Hit）占据主导**（尤其在注意力相关组件中），而 Compiler-Idea l 中 **Miss（白色）区域显著膨胀**，直观反映了静态分配在长序列下的失效。

  - **模型规模效应**：
    - **LLaMA2**（7B 参数）的整体柱高普遍高于 **GPT-Neo**（1.3B），符合预期——更大模型的单层参数量更大，对 SRAM 容量压力更甚。
    - 但 **SMOOTH-ER 在大模型上的相对增益更为稳定**，说明其对**内存带宽瓶颈**的缓解具有可扩展性。

- **架构设计验证**：
  - 该图直接支撑了论文的核心论点：**静态编译器分配（Compiler-Ideal）在注意力阶段结束后遗留大量碎片空间**，导致后续操作（如 FFN 的 W0/W1/W2 投影）无法有效预加载。
  - **SMOOTH-ER 的硬件驱动早期回收**能在注意力计算完成后立即释放 Q/K/V 等临时缓冲区，将这些空间即时转化为 **W 权重的预加载窗口**，从而在下一层计算开始前实现更高的 **Buffer Hit Rate**。

- **实践意义**：
  - 对于**移动端长上下文 LLM 应用**（如长文档摘要、多轮对话历史），SMOOTH-ER 能将注意力阶段的 **SRAM 利用效率提升 30% 以上**，直接转化为 **TTLT（Time-to-Last-Token）的显著降低**（如论文报告的最多 73% 改善）。

### b97ee12dd76c13b15eea8a67192b3aed87c2d024cd7be2adb1a79b16820471aa.jpg

![b97ee12dd76c13b15eea8a67192b3aed87c2d024cd7be2adb1a79b16820471aa.jpg](images/b97ee12dd76c13b15eea8a67192b3aed87c2d024cd7be2adb1a79b16820471aa.jpg)

图片 **b97ee12dd76c13b15eea8a67192b3aed87c2d024cd7be2adb1a79b16820471aa.jpg** 是论文 **Figure 20** 的子图 (b)，标题为 **"(b) 8K-th Token"**，展示了在生成**第8K个Token**时，不同内存管理策略和**Block Size（块大小）** 配置下的**能量消耗（Energy Consumption）** 对比。

以下是详细分析：

*   **图表类型与坐标轴**
    *   **类型**：分组柱状图（Grouped Bar Chart）。
    *   **X轴（横轴）**：代表不同的**基线方法（Baselines）** 和 **SMOOTH-ER** 在不同 **Block Size** 下的配置。
        *   左侧三组为基线对比：`Compiler-Ideal`, `Gemmini`, `Capuchin`。
        *   右侧五组为 SMOOTH-ER 在不同块大小下的表现：`256B`, `512B`, `1KB`, `2KB`, `4KB`。
    *   **Y轴（纵轴）**：表示能量消耗数值，范围从 0.0 到 0.6（单位推测为毫焦 mJ 或微焦 μJ 级别，需结合上下文；图中特定标注为 nJ）。

*   **图例与数据系列**
    *   **灰色柱 (Capuchin)**：代表基于硬件缓存管理的 Capuchin 方法的能耗。
    *   **深灰/黑色柱 (Gemmini/其他)**：代表全栈加速框架 Gemmini 的能耗（或对应基线）。
    *   **蓝色柱 (SMOOTH / SMOOTH-ER)**：代表本文提出的 **SMOOTH-ER**（带早期回收机制）框架的能耗。

*   **关键数据观察与分析**
    *   **基线方法能耗较高**：
        *   `Compiler-Ideal`, `Gemmini`, `Capuchin` 三种基线方法的能耗普遍较高，柱状图高度集中在 **0.3 - 0.5** 区间。其中 `Gemmini` 的能耗看起来最高（接近 0.5），表明其在处理长序列（8K tokens）时，由于内存管理效率较低或预取策略不够激进，导致了较高的动态功耗。
    *   **SMOOTH-ER 显著降低能耗**：
        *   蓝色柱（SMOOTH-ER）在所有 Block Size 配置下的高度均显著低于基线方法，基本维持在 **0.25 以下** 的水平。
        *   这验证了论文的核心结论：通过**细粒度的块级分配（Block-based Allocation）** 和**硬件驱动的早期回收（Early Reclamation）**，SMOOTH 能够有效减少不必要的片外内存访问（DRAM Access），从而大幅降低推理能耗。
    *   **硬件开销极低（关键标注）**：
        *   在 `256B` 和 `512B` 两个最小块大小的蓝色柱上方，有明确的数值标注：**7.1 nJ** 和 **7.4 nJ**。
        *   **解读**：这两个极小的数值（纳焦耳级别）代表的是 **SMOOTH 引入的额外硬件控制逻辑（如地址转换、块表查找等）所产生的开销**，而非总能耗。论文强调，即使考虑到这些微小的硬件开销，SMOOTH 带来的总体能效收益依然是巨大的（净节省能量）。
    *   **块大小（Block Size）敏感性**：
        *   从 `256B` 到 `4KB`，SMOOTH-ER 的能耗柱状图高度略有波动但整体保持低位且稳定。这说明 SMOOTH 的能效优势在不同的内存管理粒度下都具有鲁棒性。

*   **结论总结**
    该图直观地证明了 **SMOOTH-ER** 在长上下文（8K tokens）推理场景下的**高能效特性**。相比现有的编译器理想方案（Compiler-Ideal）和其他硬件加速器（Gemmini, Capuchin），SMOOTH 通过优化片上 SRAM 利用率和掩盖内存延迟，实现了**超过 50% 的能量降低**（根据论文正文数据），同时其自身引入的硬件额外功耗几乎可以忽略不计（仅数个纳焦耳）。

### 8080e37c9c77d6ea0eb693c73883ba320fc4d4d25ead08a45dfd3e95829834c6.jpg

![8080e37c9c77d6ea0eb693c73883ba320fc4d4d25ead08a45dfd3e95829834c6.jpg](images/8080e37c9c77d6ea0eb693c73883ba320fc4d4d25ead08a45dfd3e95829834c6.jpg)

- **图表核心主题**: **Block Size 未对齐引发的内部碎片对推理延迟的影响分析**
- **实验背景**: 该图对应论文 **Fig. 21(b)**，旨在评估 **SMOOTH-ER** 架构中，当硬件分配的 **Block Size** 与模型操作的 **Tile Size** 不匹配（即未对齐）时，产生的 **Internal Fragmentation**（内部碎片）如何导致性能退化。
- **坐标轴定义**:
  - **X轴 (Block Size)**: 表示测试的硬件块大小，范围从 **512 Bytes** 到 **1600 Bytes**，覆盖了常见的内存管理粒度。
  - **Y轴 (Norm. Latency)**: 表示**归一化延迟**，以 **Block Size 1K (1024 Bytes)** 为基准（基准值 = 1.0）。数值越高代表因内部碎片导致的额外开销越大。
- **数据系列说明**:
  - **基准参考线**: 虚线表示 **Block Size 1K** 的理想状态（无额外碎片惩罚）。
  - **模型变体**: 包含三种主流 LLM 模型在两种量化格式下的表现：
    
| 模型 | 量化格式 | 图例标识 |
| :--- | :--- | :--- |
| **GPT-Neo** | w4a8 (4-bit 权重, 8-bit 激活) | 浅灰色 |
| **LLaMA2** | w4a8 | 浅橙色 |
| **GPT-3 13B** | w4a8 | 浅蓝色 |
| **GPT-Neo** | int8 (8-bit 整数) | 深灰色 |
| **LLaMA2** | int8 | 深橙色 |
| **GPT-3 13B** | int8 | 深蓝色 |
  
  - **Total Overhead**: 斜线填充区域表示总控制开销占比。
- **关键数据分析与趋势解读**:
  - **对齐敏感性**: 当 Block Size 接近或等于 **1024 (1K)** 时，所有模型的延迟均接近 **1.0**，表明此时块大小与数据访问模式高度匹配，**内部碎片最小化**。
  - **性能退化峰值**: 在某些非对齐的块大小下（例如 **640, 704, 768** 等区域），部分模型（特别是 **w4a8** 量化版本）出现了明显的延迟上升，峰值接近 **1.05 - 1.09** 左右。这与正文描述的“**up to 9.9%**”延迟退化相吻合。
  - **量化格式的影响**: **w4a8** 量化模型（浅色系柱体）通常比 **int8** 模型（深色系柱体）表现出更高的延迟波动。这是因为 **w4a8** 的数据打包方式更紧凑，对内存对齐的要求更为严格，非对齐访问更容易导致浪费的存储空间（即内部碎片）。
  - **模型规模的影响**: 较大的模型（如 **GPT-3 13B**）在特定块大小下的延迟抖动相对较小，可能由于其计算密度更高，掩盖了部分内存访问开销；而中小模型（如 **GPT-Neo**, **LLaMA2**）对块大小更为敏感。
  - **控制开销**: 图中的 **Total Overhead**（斜线区）始终保持在极低水平（几乎贴近 X 轴），验证了 SMOOTH 架构的硬件控制逻辑开销可以忽略不计，**性能瓶颈主要源于数据层面的碎片化而非管理逻辑**。
- **工程启示与设计建议**:
  - **动态/自适应块大小**: 静态固定的块大小难以适应不同模型结构和量化策略。SMOOTH-ER 在实际部署中应倾向于将块大小设置为**模型维度（Model Dimension）的约数或倍数**，以避免内部碎片。
  - **编译器协同优化**: 编译器在生成执行图时，应向硬件 **DMC (Dynamic Memory Controller)** 提示推荐的块大小，实现 **Software-Hardware Co-design**。
  - **权衡取舍**: 虽然较小的块大小能提高内存利用率（减少外部碎片），但若导致严重的内部碎片（如图所示），反而会降低有效带宽利用率。因此，**Block Size 的选择必须在内部碎片和分配灵活性之间取得平衡**。
- **总结**: 该图直观证明了 **SMOOTH** 的细粒度块分配机制虽然能有效解决外部碎片问题，但若**块粒度选择不当**，会引入显著的**内部碎片**，导致最高近 **10%** 的性能损失。这强调了在 **On-Device LLM Inference** 场景下，**内存管理粒度必须与模型的数据访问模式紧密对齐**的重要性。

