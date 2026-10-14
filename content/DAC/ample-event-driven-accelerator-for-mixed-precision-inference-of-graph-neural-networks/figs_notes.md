# AMPLE: Event-Driven Accelerator for Mixed-Precision Inference of Graph Neural Networks 图表详解

### Fig. 1. AMPLE Top Level Diagram. Packets propagate through dimensionorder routing in the Aggregation Engine’s Network-on-Chip (shown in green), and are driven diagonally into the Transformation Engine’s systolic array (shown in red). Dashed lines represent control flow interfaces, while solid lines represent data flow between units. Node embeddings are fetched through HBM, while instructions are stored in DRAM.

![84463d803a53ab0b6a76fd7aad3315d03156097b84731932812d5a48957f0e94.jpg](images/84463d803a53ab0b6a76fd7aad3315d03156097b84731932812d5a48957f0e94.jpg)

- **整体架构定位**
    - 该图为 **AMPLE (Accelerated Message Passing Logic Engine)** 的**顶层系统架构图 (Top Level Diagram)**。
    - 这是一个基于 **FPGA** 的专用硬件加速器，旨在解决 **Graph Neural Networks (GNNs)** 推理过程中的**不规则内存访问**和**非均匀节点分布**问题。

- **核心功能模块解析**

| 模块名称 | 英文缩写 | 功能描述 |
| :--- | :--- | :--- |
| **节点指令解码器** | Node Instruction Decoder (NID) | 包含 **64个 Nodeslots (0-63)**。作为主机 (**Host CPU**) 与加速器的桥梁，通过 **AXI-L** 接口接收指令，负责**异步调度**节点任务，实现**事件驱动编程 (Event-Driven Programming)**。 |
| **指令预取器** | Instruction Prefetcher | 从 **DRAM C0** 中提前读取计算指令，减少指令获取延迟。 |
| **聚合引擎** | Aggregation Engine (AGE) | 执行 GNN 的**聚合阶段 (Aggregation Phase)**。内部采用 **Network-on-Chip (NoC)** 架构（图中绿色网格），支持**维度有序路由 (Dimension-order Routing)**，处理邻居节点的特征汇总。 |
| **聚合缓冲区** | Aggregation Buffer (ABF) | 临时存储 AGE 计算出的聚合结果，作为 AGE 与 FTE 之间的数据桥梁。 |
| **特征变换引擎** | Transformation Engine (FTE) | 执行 GNN 的**变换阶段 (Transformation Phase)**。内部采用 **Systolic Array**（图中红色脉动阵列），执行高密度的矩阵乘法运算，更新节点特征。 |
| **预取器子系统** | Prefetcher | 包含 **Feature Bank** 和 **Weight Bank**。负责从片外 **High-Bandwidth Memory (HBM)** 提前获取**节点特征嵌入 (Node Embeddings)** 和**层权重 (Layer Weights)**，隐藏内存访问延迟。 |

- **数据流与控制流机制**
    - **控制流 (Control Flow, 虚线表示)**：
        - **Host CPU** 通过 AXI-L 总线向 **NID** 编程节点信息。
        - **NID** 协调 **Instruction Prefetcher**、**AGE**、**FTE** 及 **Prefetcher** 的工作时序。
        - 这种设计允许不同节点**独立分配资源**，避免了传统双缓冲机制中低度节点等待高度节点造成的**流水线间隙 (Pipeline Gaps)**。
    - **数据流 (Data Flow, 实线表示)**：
        - **HBM** $\rightarrow$ **Feature Bank/Weight Bank**：原始图数据和权重加载。
        - **Feature Bank** $\rightarrow$ **AGE**：邻居特征输入。
        - **AGE** $\rightarrow$ **ABF**：聚合结果暂存。
        - **ABF + Weight Bank** $\rightarrow$ **FTE**：执行线性变换。
        - **FTE** $\rightarrow$ **HBM**：写回更新后的特征向量。

- **关键设计创新点体现**
    - **异构计算资源**：图中 **AGE (绿色)** 与 **FTE (红色)** 的视觉区分，体现了针对聚合（不规则、稀疏）和变换（规则、密集）两种不同计算特性的**专用硬件架构**。
    - **混合精度支持**：虽然顶层图未直接画出精度位宽，但 NID 中的 Nodeslot 设计（参考文中 Table 2）包含了 **Precision 字段**，支持在运行时为不同度数 (Degree) 的节点动态分配 **Float** 或 **Int8** 等不同精度的计算资源。
    - **大规模图处理能力**：通过底层的 **Prefetcher** 与 **HBM** 的交互（标注了 32 位宽接口），系统无需将整个图的嵌入存储在片上缓冲区即可处理**超过 100K 节点**的大规模社交网络图谱。

- **性能优化的物理实现逻辑**
    - **并行性最大化**：64 个 Nodeslots 允许同时有数十个节点处于不同的计算阶段（如有的在聚合，有的在变换），极大地提高了**节点级并行度 (Node-level Parallelism)**。
    - **内存墙缓解**：**Feature Bank** 采用的 **Fetch Tag** 机制（结合文中第 3.3 节）允许在计算开始后继续预取剩余邻居数据，实现了**计算与访存的重叠 (Overlap)**。

### Fig. 2. Microarchitecture of AGE configured with three supported precisions. NID requests drive the Aggregation Managers (AGMs), which receive fetched embeddings from the Feature Bank (See Figure 1). These are then transferred to the Aggregation Cores (AGCs) through the network. Aggregation results are then bufered by the Bufering Managers (BMs).

![29cb96496a01f841cfda9f1ac6a7524f7044211e1b73f53fea4e29de9d5d2556.jpg](images/29cb96496a01f841cfda9f1ac6a7524f7044211e1b73f53fea4e29de9d5d2556.jpg)

- **架构定位与功能概述**
    - 该图为 **AGE (Aggregation Engine)** 的微架构示意图，是 AMPLE 加速器中执行 GNN **聚合阶段 (Aggregation Phase)** 的核心计算单元。
    - 架构采用 **异构多精度设计**，配置了 **三种独立的子网络 (Mesh)**，分别对应 **Float**、**Int8** 和 **Int4** 三种数值精度。
    - 整体数据流向为：**Node Scoreboard (NSB) → Aggregation Managers (AGMs) → Aggregation Cores (AGCs) → Buffering Managers (BMs)**。

- **核心组件详解**

| 组件 | 英文全称 | 功能描述 |
|------|----------|----------|
| **NSB** | Node Scoreboard | 位于架构最左侧，作为**全局调度器**。它维护节点状态机（如表2所示），根据节点的精度需求和度数，向对应的AGM组发送计算请求和控制信号。 |
| **AGM** | Aggregation Manager | 分为 **Float AGMs**、**Int8 AGMs** 和 **Int4 AGMs** 三组。每个AGM单元包含一个 **mask** 寄存器，用于**动态资源分配**和**事件驱动调度**。AGM负责从 **Feature Bank** 接收预取的邻居特征嵌入，并驱动数据进入对应的NoC网络。 |
| **AGC** | Aggregation Core | 分布在三个隔离的 **Mesh** 网络中（Float Mesh, Int8 Mesh, Int4 Mesh）。每个Mesh包含 $m \times n$ 个AGC（如 $AGC_0$ 到 $AGC_{m*n}$），实际执行**排列不变的聚合操作**（如求和、均值）。不同精度的AGC在物理上**相互隔离**，避免跨精度通信带来的拥塞。 |
| **BM** | Buffering Manager | 位于每个Mesh的输出端（$BM_0$ 到 $BM_m$），负责**暂存聚合结果**。这些缓冲区随后被 **FTE (Feature Transformation Engine)** 读取以执行变换阶段。 |
| **Feature Bank** | Feature Bank | 位于底部，为所有AGM提供**并发的特征数据供给**。它与HBM接口相连，支持大图处理时的部分响应机制（Partial Response Mechanism）。 |

- **关键设计创新与工作机制**
    - **精度隔离与异构计算**：如图所示，Float、Int8、Int4 形成三个完全独立的网格状 **Network-on-Chip (NoC)** 子网络。这种设计基于一个关键观察：**GNN量化误差主要来自高度数节点的聚合阶段**（引用 Degree-Quant 算法）。因此，系统可以将少数高度数节点映射到 Float Mesh 保持精度，而将大量低度数节点映射到 Int8/Int4 Mesh 以节省资源。
    - **事件驱动的动态分配**：NSB 通过 **mask** 信号独立控制每个 AGM。当一个节点完成计算后，其占用的 AGM 和 AGC 资源可立即释放并重新分配给新节点，无需等待批次内其他节点（克服了传统双缓冲机制的**流水线气泡问题**）。
    - **可扩展的资源配比**：图中显示各 Mesh 内部结构相似但精度不同。根据公式(6)，开发者可根据数据集的 **DQ Ratio**（如表5所示，通常<3%的节点需要Float）灵活配置 $m$ 和 $n$，实现**硬件资源与模型精度的最优映射**。
    - **并行访存优化**：多组 AGM 可同时从 Feature Bank 获取数据，配合论文第3.3节提到的 **Fetch Tag** 机制，隐藏了稀疏图数据访问的高延迟。

- **与系统其他模块的接口关系**
    - **上游接口**：接收来自 **NID (Node Instruction Decoder)** 的编程指令（通过 NSB）和来自 **Prefetcher/Feature Bank** 的邻居特征数据。
    - **下游接口**：聚合完成后，结果通过 **BMs** 写入 **ABF (Aggregation Buffer)**，随后触发 **FTE** 执行矩阵乘法变换（对应表3中的步骤3-4）。
    - **控制流**：NSB 维护的全局状态机确保了聚合与变换阶段的**流水线协同**，实现了论文所强调的**节点级细粒度并行**。

- **性能影响总结**
    - 该微架构直接支撑了 AMPLE 在实验中取得的 **243× (vs CPU)** 和 **7.2× (vs GPU)** 的平均加速比。
    - 通过将 **混合精度** 与 **事件驱动调度** 结合于 NoC 架构中，有效解决了 GNN 推理中**计算不规则性**（Irregularity）与**内存墙**（Memory Wall）两大瓶颈，特别是在处理 Reddit（23万节点）、Yelp（71万节点）等大规模社交网络图时表现出显著的吞吐量优势（见表6）。

### Fig. 3. Fetch Tags in the Feature Bank make concurrent memory access requests in a two-stage process; First, the list of neighbouring node IDs is stored in the Address Queue, and these are then used as pointers for the neighbouring feature embeddings, which are stored in the Message Queue.

![193fe8a78d412d11fc810e09ae7a67038e5111ed0d0aaeed058a8fce4f64a319.jpg](images/193fe8a78d412d11fc810e09ae7a67038e5111ed0d0aaeed058a8fce4f64a319.jpg)

基于提供的论文内容与图表（Fig. 3），以下是对 **AMPLE 加速器中 Feature Bank（特征库）微架构**的深入分析：

*   **核心功能定位：**
    *   该图展示了 **Prefetcher（预取器）** 内部的 **Feature Bank** 模块详细结构。
    *   其主要设计目的是解决 GNN 推理中的 **内存墙问题（Memory Wall Problem）**。由于图数据的稀疏性和不规则性，传统的缓存机制效率低下，Feature Bank 通过显式的 **Node-Centric（以节点为中心）** 预取机制，在计算单元处理当前节点时，提前将后续节点的邻域数据搬运至片上，从而 **隐藏内存访问延迟（Hide Memory Latency）**。

*   **模块化架构解析：**
    *   **Aggregation Manager (0-63)：**
        *   位于顶层，代表了 **Node Instruction Decoder (NID)** 中的 **Nodeslot（节点槽位）**。
        *   图中显示了 64 个 Manager（0 至 63），这与论文 Table 2 提到的 Nodeslot 数量一致。这意味着硬件在逻辑上支持 **64 个节点的并发调度与预处理**，提供了极高的并行度潜力。
    *   **Fetch Tag (获取标签) —— 核心缓冲单元：**
        *   每个 Aggregation Manager 对应一个独立的 Fetch Tag。
        *   **内部双队列结构：** 这是该设计的精髓所在，实现了一个 **Two-Stage Pipeline（两阶段流水线）** 的数据获取过程：
            1.  **Address Queue（地址队列）：** 存储目标节点的 **Neighbor IDs（邻居节点ID）**，即邻接表信息。这一步解决了图结构的 **间接寻址（Indirect Addressing）** 问题。
            2.  **Message Queue（消息队列）：** 存储根据上述 ID 取回的实际 **Feature Embeddings（特征嵌入向量）**，即计算所需的有效数据载荷。
    *   **REQ (Request) 逻辑：**
        *   连接地址队列与消息队列（或向下层发送请求），表示当地址就绪后，硬件自动生成读取请求去获取实际的特征数据。

*   **底层内存访问控制：**
    *   **Round-Robin Arbitrer（轮询仲裁器）：**
        *   位于底部，负责协调 **64 个 Fetch Tag** 对共享资源的竞争。
        *   由于所有 Tag 都需要通过底层的 **AXI Read Master** 访问片外内存（HBM/DRAM），必须有一个高效的仲裁机制。采用 **Round-Robin** 算法可以保证公平性（Fairness），防止某个高度数节点独占总线带宽，确保所有正在处理的节点都能均匀地获得数据服务。
    *   **AXI Read Master：**
        *   物理接口单元，负责遵循 AXI 协议向外部存储器（如 Alveo U280 上的 HBM）发起实际的读事务（Read Transactions）。

*   **设计优势与性能影响：**
    *   **支持大图处理（Large Graph Support）：**
        *   论文指出，对于度数（Degree）极高的节点，Fetch Tag 支持 Partial Response（部分响应）机制。即一旦 Message Queue 中积累了部分邻居数据，Aggregation Engine 就可以开始计算，同时 Fetch Tag 继续填充剩余数据。这种 **计算与访存的重叠（Overlap）** 极大地提高了流水线效率。
    *   **高并发带宽利用：**
        *   通过将 Fetch Tag 分组耦合到不同的 HBM Bank（论文提到最多 32 个并发访问），并结合这里的 Round-Robin 仲裁，系统能够最大化利用 FPGA 板卡的高带宽内存（HBM）吞吐率，缓解 GNN 访存密集型（Memory-Bound）的特性。
    *   **解耦计算与访存：**
        *   上层的 Aggregation Manager 只需要关注计算逻辑，下层的 Fetch Tag 和 Feature Bank 封装了复杂的非连续内存访问细节，简化了控制流的复杂度。

**总结：**
Fig. 3 揭示了 AMPLE 如何通过 **分布式 Fetch Tag** 和 **两级队列缓冲** 技术，将无序的图数据访问转化为可控的流水线操作。这种架构使得加速器在面对数十万节点（如 Yelp 数据集，716K 节点）的大规模稀疏图时，仍能保持高效率的 **Node Throughput（节点吞吐量）**。

### 74a50c8065173764b1c67d073eb6ac2c7754ddfde91c75b8ac03bbb394185a9d.jpg

![74a50c8065173764b1c67d073eb6ac2c7754ddfde91c75b8ac03bbb394185a9d.jpg](images/74a50c8065173764b1c67d073eb6ac2c7754ddfde91c75b8ac03bbb394185a9d.jpg)

基于提供的论文内容和图表（Figure 4），以下是对 AMPLE 加速器性能表现的深入分析：

*   **图表概览与实验设置**
    *   **对比基准**：该图表展示了 **GPU (RTX A6000)** 和 **AMPLE (FPGA 加速器)** 相对于 **CPU (Intel Xeon)** 基准线的推理速度提升倍数（Speed-up）。
    *   **评估模型**：实验覆盖了三种主流 GNN 模型：**GCN**、**GIN** 和 **GraphSAGE**。
    *   **数据集规模**：横坐标涵盖了从小型引用网络（**CR/CS/PB**，节点数 < 20K）到大型社交网络（**FL/RD/YL**，节点数最高达 ~716K）的六个数据集。
    *   **核心指标**：纵坐标采用对数刻度（$10^0$ 到 $10^3$），直观展示了不同量级的性能差异。

*   **GCN (Graph Convolutional Networks) 性能深度解析**
    *   **小图极致加速**：在 **Cora (CR)** 和 **CiteSeer (CS)** 等小规模数据集上，**AMPLE** 展现了压倒性优势，加速比分别高达 **994×** 和 **831×**。相比之下，GPU 仅实现了 **33×** 和 **24×** 的加速。这验证了论文中提到的 **Event-Driven（事件驱动）** 架构在处理稀疏图聚合（Aggregation）阶段时，能有效消除传统双缓冲机制带来的流水线气泡。
    *   **大规模下的性能收敛**：随着图规模的扩大（如 **Reddit (RD)** 和 **Yelp (YL)**），AMPLE 的加速比显著回落至 **38×** 和 **13×**。尽管仍优于 GPU（分别为 5× 和 6×），但边际效益递减明显。这反映了在处理超大规模图时，**off-chip memory access（片外内存访问）** 延迟和片上资源限制开始成为瓶颈。

*   **GIN (Graph Isomorphism Networks) 性能特征**
    *   **总体趋势与 GCN 一致**：GIN 在小图上同样表现优异（CR: **829×**），证明了 AMPLE 架构对不同聚合函数（Sum）的良好适应性。
    *   **临界点显现**：在最大的 **Yelp (YL)** 数据集上，出现了 **GPU (13×)** 反超 **AMPLE (9×)** 的现象。这可能是因为 GIN 使用了 MLP 进行转换，计算密度增加，导致 FPGA 的 **200MHz** 运行频率劣势凸显，无法像在小图那样通过并行度完全掩盖延迟。

*   **GraphSAGE 性能瓶颈分析**
    *   **计算密集型的挑战**：GraphSAGE 是 AMPLE 表现最弱的模型。虽然在 **Cora** 和 **CiteSeer** 上仍有 **235×** 和 **195×** 的惊人提速，但在中大型图上性能断崖式下跌。
    *   **GPU 的统治区**：在 **Flickr (FL)**、**Reddit (RD)** 和 **Yelp (YL)** 上，GPU 的性能全面超越 AMPLE。特别是在 Yelp 数据集上，GPU 达到 **16×** 而 AMPLE 仅为 **2×**。
    *   **原因推断**：GraphSAGE 涉及更复杂的邻居采样和特征变换（如公式 4 所示的 $W_3$ 变换），属于 **computation-bound（计算受限）** 场景。GPU 的高主频和针对密集矩阵乘法的 Tensor Core 优化在此类场景下更具优势，而 AMPLE 针对 **memory irregularity（内存不规则性）** 优化的架构难以发挥特长。

*   **综合性能总结与架构启示**
    *   **适用场景定位**：AMPLE 并非万能加速器，而是专为 **Sparse Aggregation（稀疏聚合）** 优化的专用硬件。它在节点数较少、度分布不均（High variance in node degree）的小中型图上效果最佳。
    *   **混合精度的价值**：结合论文 Table 5 的数据，大部分节点被量化为 **int8**，仅有极少数（<3%）高度节点保留 **float**。这种 **Mixed-Precision（混合精度）** 策略使得 AMPLE 能在有限的 FPGA 资源（LUT/DSP）内塞入更多的 **Aggregation Cores**，从而在小图上实现极高的并行度。
    *   **扩展性挑战**：图表清晰地揭示了 FPGA 加速器在处理 **Large-Scale Graphs（大规模图）** 时面临的“内存墙”问题。尽管引入了 **Prefetcher（预取器）** 和 **Partial Response Mechanism（部分响应机制）**，但在面对数十万节点的图时，HBM 带宽和片上 Buffer 容量依然是制约吞吐量的关键因素。

**关键数据对照表 (GCN Model Example):**

| Dataset | Nodes | GPU Speed-up | AMPLE Speed-up | AMPLE vs GPU Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Cora (CR)** | 2,708 | ~33× | **~994×** | **~30× faster than GPU** |
| **CiteSeer (CS)** | 3,327 | ~24× | **~831×** | **~34× faster than GPU** |
| **PubMed (PB)** | 19,717 | ~75× | **224×** | ~3× faster than GPU |
| **Reddit (RD)** | 232,965 | ~5× | **38×** | ~7× faster than GPU |
| **Yelp (YL)** | 716,847 | ~6× | **13×** | ~2× faster than GPU |

