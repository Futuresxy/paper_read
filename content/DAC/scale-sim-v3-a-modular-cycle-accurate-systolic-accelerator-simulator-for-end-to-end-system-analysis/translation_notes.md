# SCALE-Sim v3: A modular cycle-accurate systolic accelerator simulator for end-to-end system analysis 原文翻译

# SCALE-Sim v3：面向端到端系统分析的模块化周期级精确脉动阵列加速器模拟器

Ritik Raj<sup>∗</sup>, Sarbartha Banerjee<sup>†§</sup>, Nikhil Chandra<sup>∗§</sup>, Zishen Wan<sup>∗§</sup>, Jianming Tong

Ananda Samajdhar<sup>‡</sup>, Tushar Krishna<sup>∗</sup>

<sup>∗</sup>佐治亚理工学院

<sup>†</sup>德克萨斯大学奥斯汀分校

<sup>‡</sup>IBM 研究

摘要——AI、科学计算和高性能计算的快速进展，推动了对通用且高效的硬件加速器的需求。现有工具（如 SCALE-Sim v2）为基于脉动阵列的架构提供了有价值的周期级精确模拟，但在支持稀疏性、多核可扩展性以及全面的内存分析等关键现代特性方面存在不足。为了解决这些局限性，我们提出了 SCALE-Sim v3（GitHub 代码仓库），这是一个模块化、周期级精确的模拟器，扩展了其前代产品的能力。SCALE-Sim v3 引入了五项重大增强：支持时空划分与层次化内存结构的多核模拟、支持逐层与逐行稀疏性的稀疏矩阵乘法模拟、与 Ramuator 集成以进行详细的 DRAM 分析、精确的数据布局建模以减少内存停顿，以及通过 Accelergy 进行的能耗与功耗估算。这些改进使得针对现代 AI 加速器的更深入的端到端系统分析成为可能，能够适应多种多样的系统和工作负载，并就延迟、带宽和能效提供详细的全系统洞察。

对于 ViT-base，仅以延迟作为指标时，128×128 阵列比 32×32 阵列快 6.53 倍。然而，SCALE-Sim v3 发现，由于更好的利用率和更低的泄漏功耗，32×32 阵列在能效上高出 2.86 倍。就 EdP 而言，对于 ViT-base，64×64 阵列的表现优于 128×128 和 32×32 两种配置。SCALE-Sim v2 表明，对于六个 ResNet18 层，采用权重驻留相比输出驻留数据流可减少 21% 的计算周期。然而，在计入 DRAM 停顿后，OS 数据流的执行周期比 WS 低 30.1%，这凸显了详细 DRAM 分析的关键作用。

索引术语——cycle-accurate（周期级精确）、sparsity（稀疏性）、multi-core（多核）

## I. 引言

多年来，计算能力一直在显著提升，以推动人工智能、科学计算、高性能计算（HPC）、AR/VR 应用等领域算法的进步。根据文献 [1]，自 2015 年以来，训练算力需求（FLOPS）每九个月翻一番。目前已有大量加速器面向 AI [2]–[6]、科学计算 [7]–[10]、HPC [11]–[14]、AR/VR [15]–[17] 以及机器人 [18]–[21] 应用中的矩阵乘法进行优化。对于此类加速器而言，固定配置无法在所有种类的负载上都保持高效。这些差异催生了领域专用 AI 加速器的时代，加速器必须能够适配其特定负载。

设计高效的 AI 加速器是一个具有挑战性的问题，需要对计算硬件、领域专用负载、延迟、功耗和带宽等目标约束以及映射策略进行深入分析。为了缓解这一问题，现有多种工具 [22]–[26] 可用于对 AI 加速器的性能和能耗行为进行建模。SCALE-Sim（SystoliC AcceLErator SIMulator）v2 [24] 是目前广泛使用的、面向脉动阵列（systolic array）加速器的流行开源模拟器之一。从根本上说，它对卷积和 GEMM 算子的运行时间进行建模——这些算子是现代 AI 模型的关键构建模块。此外，它还支持灵活配置不同形状和尺寸的脉动阵列，以及输入激活、权重和输出激活的双缓冲片上存储器的大小。此外，它通过三种经典数据流 [27]——输入固定、权重固定和输出固定，支持将算子灵活映射到脉动阵列。SCALE-Sim v2 提供详细的输出报告，包括延迟、带宽需求以及 SRAM 和主存（DRAM 或 HBM）的周期级精确读写轨迹。

SCALE-Sim v2 是一个用于探索基于脉动阵列的 AI 加速器设计空间的有前景的工具，并已被多项近期工作 [30]、[35]–[38] 所采用。然而，在本工作中，我们识别出了 SCALE-Sim v2 的一些局限性，这些局限性使其无法对 AI 加速器领域的最新创新进行建模，也无法实现全系统分析和不同设计之间的比较，具体如下所述。

局限性 1：多 Tensor Core。鉴于现代 AI 负载中混合包含矩阵运算和向量运算，近期的加速器采用了矩阵乘法引擎（即脉动阵列）与向量引擎相结合的方式。此外，为了支持更灵活的划分与映射，近期的加速器采用了由共享便签式存储器支撑的多个 Tensor Core。相关实例包括 Google 的 TPUv4 [2]（具有两个 Tensor Core）、Meta 的 MTIA [4]（具有 64 个 PE，每个 PE 包含两个 RISC-V 核心）、Cerebras WSE-2 [3]（具有 85 万个稀疏 Tensor Core）。SCALE-Sim v2 无法对这类芯片进行建模，部分原因在于其提出之时，大多数加速器只有单个脉动阵列。

局限性 2：稀疏性。近年来，稀疏性已成为最先进 AI 模型的关键特性 [39]–[41]，催生了一系列软件优化 [42]、[43]、硬件加速器 [41]、[44]、[45]、[45]、[46] 以及软硬件协同设计 [47]、[48] 技术来提升计算效率。此外，人们也越来越关注将稀疏矩阵乘法加速器应用于 AI 之外的领域——例如本质上高度稀疏的科学计算和图分析。利用稀疏性可以减少不必要的计算和内存访问，从而提升计算效率并降低能耗。遗憾的是，SCALE-Sim v2 不支持稀疏矩阵乘法单元的建模，而这类单元如今已出现在 TPU v5 [49] 等现代 AI 加速器中。

局限性 3：主存接口。主存技术的具体实现 [50] 以及负载特定的访问模式可能导致内存读/写延迟、能耗和行冲突方面的显著差异。事实上，此类行为已被 row hammer [51]、[52] 等攻击所利用。因此，需要深入研究 DRAM/HBM 与片上存储器接口，以理解不同负载在 AI 加速器上运行时的内存行为。遗憾的是，SCALE-Sim v2 将主存建模为具有固定延迟和带宽的单体实体。SCALE-Sim v2 缺少主存接口建模，使其无法提供上述关键统计信息。

局限性 4：数据布局。片上存储（如 SRAM 或 BRAM）中数据的组织方式对机器学习加速器的性能起着至关重要的作用。数据布局与数据流之间的交互会显著影响延迟和计算效率 [34]，忽视实际数据布局可能导致数量级的性能偏差。最优数据流定义了数据在计算与存储资源间的处理和流动方式，但由于诸如 bank 冲突等问题——当多个数据元素同时从同一 memory bank 访问时会导致计算停顿——次优的数据布局可能阻碍最优数据流的实现。因此，对内存行为中数据布局的详细建模是必不可少的，而这正是 SCALE-Sim v2 所忽略的。
![](images/5345044d7001c31f8b81a2c1969a2d5bf1f6b266d03c83d6558ee0b4f5699c2f.jpg)

图 1：SCALE-Sim v3 概览，红色标注了相对 SCALE-Sim v2 的新特性
<table><tr><td rowspan=1 colspan=1>框架</td><td rowspan=1 colspan=1>计算</td><td rowspan=1 colspan=1>核心数</td><td rowspan=1 colspan=1>划分方式</td><td rowspan=1 colspan=1>SparseCore</td><td rowspan=1 colspan=1>功耗</td><td rowspan=1 colspan=1>能耗模型</td><td rowspan=1 colspan=1>数据布局</td><td rowspan=1 colspan=1>主存模型</td></tr><tr><td rowspan=1 colspan=1>MAESTRO [25], [28]</td><td rowspan=1 colspan=1>解析式</td><td rowspan=1 colspan=1>单核</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>Cacti-6.0 [29]</td><td rowspan=1 colspan=1>否</td><td rowspan=1 colspan=1>N/A</td></tr><tr><td rowspan=1 colspan=1>STONNE [23]</td><td rowspan=1 colspan=1>周期级精确</td><td rowspan=1 colspan=1>单核</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>类 SIGMA [30] 非结构化</td><td rowspan=1 colspan=1>平均值</td><td rowspan=1 colspan=1>基于综合与布局布线的表驱动模型</td><td rowspan=1 colspan=1>否</td><td rowspan=1 colspan=1>N/A</td></tr><tr><td rowspan=1 colspan=1>Timeloop v4 [22]</td><td rowspan=1 colspan=1>解析式</td><td rowspan=1 colspan=1>多核</td><td rowspan=1 colspan=1>时空划分</td><td rowspan=1 colspan=1>基于分布</td><td rowspan=1 colspan=1>平均值</td><td rowspan=1 colspan=1>Accelergy [31]</td><td rowspan=1 colspan=1>否</td><td rowspan=1 colspan=1>CactiDRAM [32]</td></tr><tr><td rowspan=1 colspan=1>SCALE-Sim v2 [24]</td><td rowspan=1 colspan=1>周期级精确</td><td rowspan=1 colspan=1>多核</td><td rowspan=1 colspan=1>空间划分</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>否</td><td rowspan=1 colspan=1>N/A</td></tr><tr><td rowspan=1 colspan=1>SCALE-Sim v3</td><td rowspan=1 colspan=1>周期级精确</td><td rowspan=1 colspan=1>多核</td><td rowspan=1 colspan=1>时空划分</td><td rowspan=1 colspan=1>层与行级 N:M</td><td rowspan=1 colspan=1>瞬时值 + 平均值</td><td rowspan=1 colspan=1>Accelergy [31]</td><td rowspan=1 colspan=1>是</td><td rowspan=1 colspan=1>Ramulator [33]</td></tr></table>

表 I：相关工作

<sup>\*</sup> FEATHER [34] 论文为 Timeloop 增加了数据布局建模支持，但尚未包含在 Timeloop 的公开代码仓库中。

局限 5：能耗。近年来，硬件加速器已成为提升能效的关键手段，其核心在于利用特定应用领域的独特特性。然而，精确预测能耗通常需要完成物理设计布局，由于开销高昂且仿真速度缓慢，这严重阻碍了设计空间探索 [31]。为了实现更快、更高效的设计探索，能耗估计必须在设计流程的早期阶段进行，而无需完整详细的硬件描述。此外，数据搬运能耗构成了加速器总能耗的重要组成部分，尤其是在数据依赖的场景下。遗憾的是，SCALE-Sim v2 缺乏对能耗和功耗建模的支持，限制了其为设计高能效加速器提供有价值洞察的能力。

在本工作中，我们解决了 SCALE-Sim v2 的上述局限性，并通过如图 1 所示的五项新特性对其进行增强。1 对多核特性的建模，包括时空划分（spatio-temporal partitioning）、L2 共享内存、异构 Tensor Core 以及非均匀工作负载划分。2 支持基于 systolic array 的稀疏加速器，面向逐层（layer-wise）和逐行（row-wise）的 SpMM 工作负载。3 通过集成 Ramulator [33] 对 DRAM 和片上内存接口进行建模。4 对数据布局（data layout）的建模，以实现更精确的片上内存停顿分析。5 通过集成 Accelergy [31] 对能耗与功耗进行分析。

## II. 背景

## A. Systolic array 与多核加速器

Systolic array 是一种专用硬件架构，旨在高效执行重复性计算，尤其是线性代数运算，例如在数字信号处理和机器学习工作负载中常见的矩阵乘法和卷积 [53]–[59]。

研究表明，自 2012 年以来，AI 模型的计算需求每 3-4 个月翻一番 [60]。这些需求源自日益复杂的模型开发，以及 AI 应用在自然语言处理（NLP）[61]–[63]、自动驾驶 [64]–[66]、医疗健康 [67]–[69] 等各个领域的扩展。这种指数级增长催生了对专用硬件加速器的需求。许多 AI 加速器，包括 Google TPU v5 [49]、Meta MTIA [4]、Cerebras WSE-2 [3]、Nvidia DGX GH200 [70] 和 Tesla Dojo [71]，都采用了多芯片和多核设计以满足日益增长的需求。

## B. 稀疏加速器

由于稀疏性在提升计算效率方面的潜力，它已被广泛应用于商用加速器中。例如，NVIDIA 的 Ampere 架构 [45] 引入了 Sparse Tensor Core，利用具有 2:4 稀疏率的结构化稀疏，即每四个张量元素中有两个为非零元素。类似地，Google TPU [49] 利用稀疏加速器高效处理稀疏张量，引入了稀疏权重剪枝和稀疏矩阵运算等机制，从而在大规模 AI 工作负载下实现有效扩展。VerSA [72] 和 VEGETA [46] 等加速器则提出了一种通用 systolic array，能够同时加速稠密和稀疏矩阵乘法，以满足现代深度神经网络应用的多样化需求。

## C. 内存仿真器

Ramulator [33]、[73] 或 DRAM-Sim3 [74] 等内存仿真器可仿真 DDR 及其他内存技术的内部行为。此类仿真器通常提供详细的指标，例如总内存请求数、读/写操作数、平均延迟、带宽利用率、行缓冲命中率以及功耗估计，从而助力全面的内存系统性能分析。具体而言，我们将 Ramulator [33] 集成为 SCALE-Sim v3 的内存模型。Ramulator 是一款高性能、周期精确的 DRAM 仿真器，便于内存系统的评估与设计。其模块化架构支持对多种 DRAM 标准的直接集成，包括 DDR3、DDR4、LPDDR4、GDDR5、WIO1、WIO2 和 HBM，使研究人员能够在统一框架内评估和比较不同的内存技术。

## D. Accelergy

Accelergy [31] 是一款架构级能耗估计工具，旨在为加速器设计提供精确且灵活的能耗分析。为应对高能效计算领域的挑战而开发的 Accelergy，允许设计者对加速器架构中的基本组件以及复杂的用户自定义复合组件进行建模。它系统性地捕获各种构建模块的能耗，无需详细的物理布局即可实现快速设计空间探索，对 Eyeriss [27] 等知名深度神经网络加速器的能耗估计准确率可达 95%。

## E. SCALE-Sim v2

SCALE-Sim v2 是一款周期精确的仿真器，旨在为卷积和 GEMM 建模基于 systolic array 的加速器。该仿真器接受用户自定义配置，允许指定多项参数，包括 systolic array 尺寸、片上双缓冲 SRAM 容量、基于复用方式（input stationary、weight stationary 或 output stationary）的数据流，以及 DRAM 带宽。它处理以 CSV 格式提供的神经网络拓扑，通过逐层规格说明来精确建模工作负载。SCALE-Sim v2 运行后会生成详细的报告，包括计算周期数、带宽利用率以及周期精确的 SRAM 和 DRAM 访问轨迹，为所仿真架构的性能特征提供洞察。

在本文的其余部分，我们将介绍为 SCALE-Sim v2 新增的五项增强功能——即对多 Tensor Core、稀疏性、主存、数据布局以及能耗/功耗的支持。

## III. 多 Tensor Core

本节介绍 SCALE-Sim v3 的多 Tensor Core 仿真特性，其与 v2 的区别体现在以下四个方面：1 SCALE-Sim v3 将 v2 中提到的空间工作负载划分扩展为时空划分（第 III-A 节），如图 2 所示。2 片上内存采用分层结构建模，多个核共享一个 L2 便签式存储器（第 III-B 节），如图 4 所示。3 异构 Tensor Core（第 III-C 节），体现在 systolic array 尺寸和 SIMD 长度上的差异。4 非均匀工作负载划分（第 III-D 节），面向具有不同延迟特征的核 [75]。

## A. 时空分区

<table><tr><td rowspan=2 colspan=1>Dataflow</td><td rowspan=1 colspan=3>Mapping</td></tr><tr><td rowspan=1 colspan=1>Sr</td><td rowspan=1 colspan=1>Sc</td><td rowspan=1 colspan=1>T</td></tr><tr><td rowspan=1 colspan=1>Input Stationary</td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>M</td></tr><tr><td rowspan=1 colspan=1>Weight Stationary</td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>N</td></tr><tr><td rowspan=1 colspan=1>Output Stationary</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>K</td></tr></table>

TABLE II: 不同数据流下的 GEMM 映射
![](images/56d1ba8168f7694c9ac9a8592ba11b3d6f3ce93f919a6f45ce7031d069fe0073.jpg)  
Fig. 2: 空间分区与时空分区

我们引入时空分区作为一项新特性，对 SCALE-Sim v2 论文 [24] 中引入的空间分区进行了增强，如图 2 所示。它指的是沿着某个空间维度（Sr/Sc）和时间维度（T）（表 II）进行分区，而不仅仅是沿着空间维度分区。假设 R 和 C 为脉动阵列的维度；Sr、Sc 和 T 为空间与时间映射维度（表 II）；Pr 和 Pc 为行与列的分区数（图 2），即核心数量 = Pr×Pc，下列公式给出了不同分区的运行时间：

$$
(2R+C+T-2)\left\lceil\frac{S_r}{P_r/R}\right\rceil\left\lceil\frac{S_c}{P_c/C}\right\rceil\ [24]\tag{1}
$$

$$
(2R+C+\lceil T/P_c\rceil-2)\left\lceil\frac{S_r}{P_r/R}\right\rceil\left\lceil\frac{S_c}{P_c}\right\rceil\tag{2}
$$

$$
(2R+C+\lceil T/P_r\rceil-2)\left\lceil\frac{S_r}{R}\right\rceil\left\lceil\frac{S_c}{P_c/C}\right\rceil\tag{3}
$$

其中公式 1 表示空间分区 [24]，而公式 2 和公式 3 表示时空分区。

![](images/081471efeb480cb9af4656168608f7d4ee1e68ca013dfe4805e9c103e566022d.jpg)

(a) 计算周期优化的分区（Pr、Pc）

![](images/1d616243c08147d5e84a6898dccf768d875be4b56b35c0f72f01e7c1062256fd.jpg)

(b) 内存占用优化的分区（Pr、Pc）

图3：横向扩展条件下，空间与时空分区的计算周期和内存占用权衡。  

图 3 展示了空间分区、时空分区 1（公式 2）和时空分区 2（公式 3）这三种分区方案之间的内存占用与计算周期权衡。GEMM 维度 M、N 和 K 从 [1000, 5000, 10000] 中选取，因此总共有 27 个工作负载。脉动阵列维度（行数与列数）从 [8, 16, 32] 中选取，而 scale-out 核心数量则从 [16, 32, 64] 中选取。Pr 和 Pc 的选取分别基于计算周期优化（图 3a）和内存占用优化（图 3b）。折线连接表示这些点具有相同的工作负载、阵列规模和核心数量配置。在图 3a（计算优化）中，三个相连点中的最优分区是内存占用最小的那一个。在图 3b（内存优化）中，最优分区是计算周期最少的那一个。图 3a 中有多个例子表明时空分区方案优于空间分区。然而，在图 3b 中，空间分区在大多数情况下表现更优。

## B. 具有共享 L2 的层次化内存

由于采用空间分区，每个核心处理大小为 Pr × T 的输入分区和大小为 Pc × T 的权重分区，如图 4 所示。同一行中的每个核心接收相同的输入矩阵分区，而同一列中的每个核心接收相同的权重矩阵分区。如果只有 L1 SRAM，同一行的多个核心之间（输入矩阵的重复）或同一列的多个核心之间（权重矩阵的重复）将产生大量数据复制。为了缓解数据复制问题，我们采用共享 L2 SRAM，如图 4 所示。为了确保不出现停顿，L2 SRAM 的容量应足以容纳输入/权重分区。可以灵活配置 L1 SRAM 的大小，以在 L1 容量与 L1 未命中之间取得良好的平衡。类似地，在时空分区的情况下，共享 L2 将接收具有公共维度 Sr 和 Sc 的输入/权重分区。

![](images/12f326f1d862d6e4386bd179de95127ebb540db242c9259500aea2dc05d97a40.jpg)  
Fig. 4: 共享的输入与权重 L2 SRAM

## C. 异构 Tensor Core

我们增加了对 tensor core 的支持，沿用 Google TPU [2] 的命名约定，即每个 TensorCore 由一个或多个矩阵乘单元（MXU）和一个向量单元组成。向量/SIMD 单元用于通用计算，例如激活和 softmax。类似地，Meta 的 MTIA [4] 中的 SIMD 单元使用查找表和浮点单元来近似 exp、sigmoid 和 tanh 等函数，以处理量化/反量化以及非线性函数。因此，SCALE-Sim 支持在 SIMD 单元的长度与类型和/或脉动阵列维度上具有异构性的任意 tensor core，如 [76]–[78] 中所采用的做法。此外，SIMD 单元的延迟可根据具体用例进行定制。

## D. 非均匀工作负载分区

包括 Simba [75] 在内的基于多芯片模块 [79]–[82] 的加速器，根据各个核心所在位置的不同，呈现出非均匀的延迟特性。距离主存较远的核心/chiplet 在传输输入或输出矩阵时需要更多的跳数或更高的封装内网络延迟。另一方面，距离主存较近的核心/chiplet 在输入或输出通信上的 NoP 延迟较低。这种 NoP 延迟上的差异催生了非均匀工作负载分区，即较远的核心/chiplet 接收较少的工作负载（较低的 Pr 和 Pc），而较近的核心/chiplet 接收较多的工作负载（较高的 Pr 和 Pc）。SCALE-Sim 支持非均匀分区，以有效模拟具有不同计算需求的核心/chiplet。

## IV. 稀疏性

稀疏性是许多深度学习模型中观察到的一项基本特征，其中相当大比例的张量元素（例如权重或激活值）为零或数值极小。稀疏性可以形式化地量化为一个块中非零元素的数量（N）与该块中元素总数（M）之比，即稀疏率为 N:M。

## A. SCALE-Sim v3 中的稀疏性

在此基础上，SCALE-Sim v3 通过为脉动阵列架构引入对层级稀疏性和行级稀疏性的支持，对其前身进行了扩展。

1) 层级稀疏性：SCALE-Sim v3 允许稀疏性配置在神经网络的不同层之间变化。这种灵活性契合了不同层表现出不同程度稀疏性的观察结果。例如，CNN 的初始层通常具有更稠密的连接，而更深的层则更适合更高的稀疏度。

2) 行级稀疏性：行级稀疏性以行为粒度强制固定非零元素的数量 [46]，从而对工作负载的稀疏性提供更精细的控制。在 SCALE-Sim v3 中，行级稀疏性针对各种 N:M 稀疏率实现，即一行中每 M 个元素组成的一组恰好包含 N 个非零值。该特性捕捉了真实 CNN 模型中存在的现实稀疏模式。SCALE-Sim v3 将稀疏率限制为 $N \ \leq \ M / 2 ,$，以确保稀疏性在计算上仍然是有利的。当 $N > M / 2 ,$ 时密度增加，稀疏性的收益被抵消，趋近于稠密配置。

## B. SCALE-Sim v3 + 稀疏性集成

步骤 1：架构与工作负载输入：在拓扑文件中，新增了一个 SparsitySupport 列，用于表示 N : M 格式的稀疏率。配置文件中新增了一个 "sparsity" 段，其中包含与稀疏性相关的架构旋钮。对于层级稀疏性，SparsitySupport 旋钮设为 true，OptimizedMapping 旋钮设为 false。对于本文中呈现的所有仿真，SparseRep 均设置为 ellpack block。对于行级稀疏性，OptimizedMapping 旋钮设为 true，BlockSize 的值即为 N : M 比率中的 M。非零元素的数量 (N) 在不同行之间是随机化的，并保持 $\leq M / 2$。所有稀疏性仿真的数据流均设置为权重驻留。

![](images/880ce62d25be7959c3a78de0ff0068c674bd0c754c65a113d553847ee0847de8.jpg)

图 5：ResNet-18 在 1:4、2:4 和 4:4 稀疏率下，总计算周期（含内存停顿）与片上内存的关系
![](images/d16e9659fb5b216d18b0cc065302b60e0e04bca07b4dfbc15469ade10b9630d8.jpg)
图 6：(a) 原始矩阵 分块 ELLPACK 格式

步骤 2：生成操作数与需求矩阵：从拓扑文件和配置文件中读取稀疏性架构旋钮和详细信息后，将对操作数矩阵进行修改，以反映模型的稀疏性。为简单起见，我们假设前 N 行含有非零元素，其余 N − M 行为零元素。对于行级稀疏性，每一行被分配一个随机的稀疏率，使得 $N \le M / 2$。我们不再向脉动阵列逐行流入单个输入元素，而是需要流入一个输入元素块。这一点在 IFMAP 需求矩阵的生成中通过从 IFMAP SRAM 中读取相应地址得以体现。

步骤 3：收集统计信息：新构建的 SCALE-Sim v3 仿真器会输出一个 SPARSE REPORT.csv，其中包含诸如稀疏表示、原始滤波器存储、新滤波器存储（由压缩后的滤波器矩阵和元数据组成）等信息与指标。

## C. 稀疏性案例研究

图 5 展示了 ResNet-18 的总周期（含内存停顿）与片上内存之间的关系。该关系在不同稀疏率下（即 1 : 4、2 : 4 和 4 : 4）进行了观察。随着片上内存规模的增大，单次可载入 SRAM 的数据量增多，从而减少了停顿周期。对于给定的片上内存，稀疏模型由于稀疏度更高，所需的计算周期更少。

SCALE-Sim v3 支持 Compressed Sparse Row (CSR)、Compressed Sparse Column (CSC) 和 Blocked ELLPACK 格式。对于所有与稀疏性相关的仿真，均采用 Blocked ELLPACK 格式。对于图 6a 所示的矩阵以及等于 4 的块大小，分块 ELLPACK 表示如图 6b 所示；其中绿色单元格表示非零滤波器矩阵的值，淡紫色单元格表示相应的元数据。在一般情况下，单个元数据条目所需的位数由 log (Block Size) 给出。

![](images/57a3cf1b920877d33bef42cfceec5d956ce43b9deafd59e91d5ef3e1de146cda.jpg)

图 7：内存图显示了 ResNet-18 在稠密、1:4、2:4 和 3:4 稀疏率下的内存存储对比
![](images/94f5a28e95a6d07db5262bd30d4d2b5959535a56d33bb3c65a89d81ce005a797.jpg)
图 8：ViT 的 Feed Forward 层在不同阵列尺寸、稀疏率和块大小下的计算周期变化。

图 7 比较了 ResNet-18 各层在稠密、1:4、2:4 和 3:4 稀疏率下的内存存储需求。对于每种稀疏率，存储内容由稀疏滤波器数据及相关元数据构成。随着稀疏度的增加，内存中存储的滤波器元素数量及其对应的元数据随之减少，凸显了稀疏性对内存需求的影响。

为了研究不同块大小对性能的影响，进行了如图 8 所示的研究。第一组运行采用不同的脉动阵列尺寸（4x4、8x8、16x16、32x32），且块大小始终等于脉动阵列的维度。因此，稀疏率的范围仅从 1 : M 到 M : M。这组结果与第二组运行进行了对比，后者采用固定的阵列尺寸（32x32），并改变块大小（M = 4、8、16 和 32）。对于每个 M，我们将有 M 个不同的 N 值，从而得到 N 个不同的稀疏率。通过增大块大小，我们可以对模型进行更细粒度的控制，而利用 N : M 值中较低的范围/谱段则能带来更好的性能。

## V. 主内存集成

SCALE-Sim v2 的一个关键局限在于缺乏与脉动计算单元配套的内存模型。早期版本假设数据会以某种方式随时可用地存在于片上便笺存储器中。这一假设忽略了内存延迟的估计、由 DRAM bank 冲突引起的停顿以及数据通路中的其他流水线气泡，而这些对于访存密集型工作负载至关重要。

SCALE-Sim v3 通过将 Ramulator [33] 内存模型与脉动阵列集成来弥补这一局限。它通过配置 load/store 队列大小、DRAM 通道和内存技术（即 DDR4、HBM、LPDDR4 等），实现了内存数据通路的建模。它还将内存延迟以及流水线停顿计入 ML 推理延迟中。

我们首先在 V-A 节介绍主内存集成所需的原语，随后在 V-B 节介绍使用 Ramulator 仿真 ML 推理延迟的步骤，最后在 V-C 节给出一些结果，包括内存吞吐量和整体执行延迟。

## A. 内存集成原语

1) 主内存模型：一个精细的内存模型会仿真每次 load 和 store 内存事务的往返延迟。现有的主内存模型（如 Ramulator [33]、[73]、DRAMSim3 [74] 等）可仿真不同的 DRAM 技术，同时允许用户配置内存控制器。该模型以内存需求请求 trace 作为输入。每条 trace 条目由请求周期、内存地址和事务类型（读或写）组成。主内存模型会给出每个事务的往返延迟，以及整体运行统计信息，包括内存吞吐量、row buffer 命中/未命中、bank 冲突等。这些统计信息可用于研究各工作负载数据流、内存控制器配置和主内存技术的执行瓶颈。

2) 有限内存请求队列：SCALE-Sim v3 增加了可配置的内存请求队列，用于保存待处理的内存事务。请求队列中记录的需求请求会在事务完成后被清除。读请求在接收到数据值后清除，而写请求在被记录到内存控制器后清除。这些请求队列的有限容量会在待处理队列已满时使加速器停顿。这些队列有助于建模影响访存密集型工作负载执行运行时间的内存停顿。

3) 内存延迟建模：脉动阵列处理多个数据位，这些数据由输入和权重便笺 SRAM 缓冲区提供。SCALE-Sim v3 会考虑内存加载延迟，并且仅当每个数据都已在便笺存储器中就绪时才触发脉动阵列。这在数据流式传输过程中增加了运行时延迟和流水线气泡，降低了某些数据流的计算性能。数据平面组件的精确建模使用户能够评估数据流、稀疏性以及其他工作负载优化方案。

## B. 内存仿真运行时

主内存仿真流程如下：

第一步：加速器内存请求生成：使用脉动阵列模拟器根据执行工作负载所需的数据序列生成内存需求事务。ifmap、filter 和 ofmap 地址与周期时间戳一起生成，并被顺序化以形成单一的内存 trace。每个条目包含请求周期、内存地址和事务类型。

第二步：评估内存往返延迟：我们将生成的内存 trace 输入到内存模拟器中。我们在评估中使用 Ramulator [33]。技术、rank 数量与通道数、频率等内存参数在配置文件中设置。内存模拟器报告每个单独请求的往返延迟。该内存模型还捕获吞吐量、rowbuffer 命中与未命中等内存统计数据。往返延迟包括内存 bank 冲突带来的延迟以及控制器延迟。

第三步：带内存延迟的 SCALE-Sim v3 仿真：再次运行脉动阵列模拟器，但使用有限的请求队列和实际的内存 load/store 延迟。来自内存模拟器的数据作为输入发送给 SCALE-Sim v3。执行停顿（stall）通过启用有限内存请求队列以及第 V-A 节中描述的内存延迟建模来计算。总体而言，上述流程基于所仿真的 ML 加速器内存模块，提供了切合实际的脉动阵列效率。

![](images/2276a45d7fcadda849b435a390ac8e269406b7cabb97c13029b6043037bfae1e.jpg)  
Fig. 9: DRAM 通道数对 Resnet18 内存吞吐量的影响

## C. 内存模型评估

1) SCALE-Sim v3 与内存配置：SCALE-Sim v3 采用 Google TPU 配置运行。读和写请求队列分别配置为 128 个条目的大小。Ramulator 仿真每个通道容量为 4Gb、速率为 2400 MHz 的 DDR4 内存。

2) DRAM 通道数对内存吞吐量的影响：ML 推理执行已知可以从多个 DRAM 通道中获益，尤其是对于内存密集型层。我们通过将 DDR 通道数从 1 调整到 8 来评估 Resnet18 各层的内存吞吐量影响，如图 9 所示。我们观察到，对于初始的层，增加更多内存通道时内存吞吐量成比例地提升，而最后几层的性能在通道数为 2 时即达到饱和。初始层较大的 ifmap 和 filter 尺寸从更多数量的通道中获益，某些层达到了超过 2000 MB/s 的带宽。

1 × 1 的 filter 和较小的 ifmap 降低了后续卷积层和全连接层的内存吞吐量。虽然内存密集型层和流式访问模式可以从多通道中获益，但计算密集型或较小的层无法高效利用多个通道。每个内存通道还会为内存控制器带来额外的面积成本，以及并行数据加载带来的功耗成本。

3) 内存建模造成的停顿：以往的 ML 推理模拟器并未对内存事务引起的停顿进行建模。我们将停顿周期数定义为脉动阵列在 scratchpad 上等待数据时的执行周期。图 10 展示了不同工作负载的停顿周期数占总执行周期的比例。三根柱子分别代表需求请求队列大小为 32、128 和 512 个条目的情况。32 个条目的请求队列无疑会增加大量停顿周期，因为脉动阵列很容易用需求请求填满一个小队列。然而，我们看到停顿周期的百分比和总执行周期随着条目数的增加而降低。当请求队列大小从 32 变为 128 时，平均总周期数减少了 3.76×，而 512 个条目时又进一步改善了 38%。总体而言，内存仿真带来的停顿周期可能使整体推理延迟从 1 万个周期延长到数百万个周期，具体取决于工作负载，在估算整体推理延迟时必须加以考虑。图 10 中的停顿周期展示了 SCALE-Sim-v2（其假设数据传输延迟为零）中未计入的执行低效部分。DRAM 建模提升了总执行周期估算的精度，并优化了整体脉动阵列利用率指标。

![](images/0638662673c9006219a7b5d299946bb9c427dfb840e1eee60c55657b5fccc36c.jpg)  
Fig. 10: 内存停顿对整体 ML 推理延迟的影响。三根柱子分别配置了大小为 32、128 和 512 的读/写请求队列。

## VI. 内存中的数据布局

在多核加速器中，多个计算单元共享片上公共内存，并并行处理不同的工作负载。当位于同一 bank 不同行的多个计算单元请求数据时，需要多个周期的延迟才能将所需数据从片外取出，这被称为 bank 冲突减速。这种减速会显著降低实际性能 [34]。为了对此建模，我们将精确的布局建模集成到 SCALE-Sim v3 中。

![](images/3b35ee9c19ca29e586470a5c80e532d7a40ba82f3be26fbeed455ffbbcf02e54.jpg)  
Fig. 11: 多 bank 片上内存中数据布局建模的概览。我们使用嵌套循环对数据布局进行建模，其中每一行是所有 bank 中相同索引行的聚合。每个 bank 是一个二维数组，提供有限数量的端口用于并发的读/写访问。

一般来说，布局建模包括两个组件：布局建模：支持多 bank 片上缓冲区中的各种数据组织方式。

考虑布局的内存延迟评估：评估访问给定数据的实际 bank 冲突。

## A. 数据布局建模

在现代多 bank 片上内存系统中，全局带宽均匀地分布在各个内存 bank 之间。我们将多 bank 片上内存建模为一个二维数组，其中每一行聚合了所有 bank 中相同索引的行。这种抽象确保了单行的带宽与片上总带宽相匹配。每个 bank 拥有自己独立的数据访问端口，从而实现细粒度的数据访问并减少 bank 冲突。

我们将数据布局定义为数据在由多 bank 内存导出的二维数组中的组织方式。这包括确定每个数据元素的行列位置。我们将行列位置的顺序分别称为行间维度顺序和行内维度顺序。每个维度顺序都表示为嵌套循环，如图 11 所示。

## B. 内存延迟评估

SCALE-Sim v3 以周期为粒度分析计算阵列数据请求的延迟。以图 11 中一个 $C \times H \times W$ 张量为例，当请求元素 $( c , h , w )$ 时，其行号、列号和 bank 索引可由以下公式推导得出。

$$
\begin{aligned}\mathrm{line}_{id}={}&\lfloor c/c1\_step\rfloor\lfloor H/h1\_step\rfloor\lfloor W/w1\_step\rfloor\\&+\lfloor h/h1\_step\rfloor\lfloor W/w1\_step\rfloor+\lfloor w/w1\_step\rfloor\end{aligned}
$$

![](images/d89ed35d92d88543e826ac8e95548f6ae0ec6363030c0478059d9151dbf5c6ec.jpg)

图 12：不同（片上带宽、bank 数量）组合下不同数据流的性能减速。阵列尺寸：$1 2 8 \times 1 2 8$ 工作负载：ResNet18。
![](images/d28fb1edc66732bbf73ab717616f22fe7daa501ca861605fb9c3c086eb42773c.jpg)
图 13：不同（片上带宽、bank 数量）组合下不同数据流的性能减速。阵列尺寸：$1 2 8 \times 1 2 8$ 工作负载：ViT。

$$
\begin{aligned}\mathrm{col}_{id}={}&(w\bmod w1\_step)\,h1\_step\,c1\_step\\&+(h\bmod h1\_step)\,c2\_step+(c\bmod c1\_step)\end{aligned}
$$

$$
\mathrm{bank}_{id}=\left\lfloor\frac{\mathrm{col}_{id}}{\mathrm{bandwidth\_per\_bank}}\right\rfloor
$$

其中 bandwidth per bank 表示单个 bank 中一行内可容纳的最大元素数量。

给定某一数据流所需全部数据的 $( l i n e _ { i d } , c o l _ { i d } , b a n k _ { i d } )$，我们可以计算每个 bank 每周期需要并发访问的总行数，进而得到如下总体的数据访问延迟，其中假设每个 bank 可用于并发读取的总端口数为 num ports。

$$
\mathrm{slowdown}=\max_{i=0}^{\mathrm{num\_bank}}\left\lceil\frac{\mathrm{total\_rows\_bank}_i}{\mathrm{num\_ports\_bank}_i}\right\rceil
$$

在片上存储的所有 bank 中，bank 冲突最严重的 bank 决定了从片上存储提供请求数据的关键延迟。

图 12 和图 13 展示了真实片上 bank 模型与 SCALE-Sim v2 原始理想带宽建模之间的性能差异。图中将真实布局模型下的延迟相对于 SCALE-Sim v2 的纯带宽建模进行了归一化，突出显示了实际的减速情况。值得注意的是，在片上带宽相同的情况下，增加 bank 数量能够持续提升性能，这从 bank 数量更多时减速的降低可以得到印证。这一提升归因于更多的 bank 提供了更细粒度的片上数据访问灵活性，从而有效缓解了 bank 冲突。

![](images/623a9849efc53d46a773a0491aec307574fafe20a255d72d979b62322f2e2781.jpg)
图 14：Accelergy 中动作类型的转换表，以及生成的内存动作计数 YAML 文件片段。

## VII. 能耗与功耗建模

本节介绍 SCALE-Sim v3 的能耗与功耗建模，该建模通过集成 Accelergy [31] 实现。内容涵盖框架概述（第 VII-A 节）以及集成方法的详细介绍（第 VII-B 节至第 VII-E 节）。

## A. SCALE-Sim v3 能耗建模框架概述

SCALE-Sim v3 能耗建模框架包含三个关键组成部分：(1) 统一的用户输入，(2) trace 文件与动作计数，(3) 性能与能耗报告。以下是详细说明：

步骤 1：架构与工作负载输入：神经网络描述在 SCALE-Sim v3 与 Accelergy 之间保持一致，作为独立输入用于生成 trace 文件和周期级性能数据。SCALE-Sim v3 自动生成一个架构输入的 YAML 文件，将用户输入与预定义的脉动阵列组件描述及能耗估算插件相结合，以生成能耗参考表（ERT）。用户也可以自定义组件描述，以获得更高的灵活性。

步骤 2：trace 文件与动作计数：Accelergy 会区分不同动作类型（例如重复访问与随机内存访问）的能耗。为了利用这一特性，SCALE-Sim v3 在生成 trace 文件之外，还加入了动作类型统计功能。动作计数文件与性能指标（延迟、阵列利用率和内存访问次数）由 SCALE-Sim 生成，并作为 Accelergy 能耗分析的输入。

步骤 3：性能与能耗输出：Accelergy 仿真完成后，会生成一个将能耗估算与 SCALE-Sim v3 周期级性能摘要相结合的统一输出。这一组合输出同时提供了时序可预测性与能效方面的洞察，使用户能够优化其加速器设计，以获得更优的能耗延迟积（energy-delay product）。

## B. SCALE-Sim v3 与 Accelergy 的输入集成

SCALE-Sim v3 与 Accelergy 的输入文件在抽象层级上有所不同。SCALE-Sim v3 的配置输入文件指定的是高层级配置，例如阵列形状、SRAM 大小和数据流，而 Accelergy 的架构 YAML 文件描述的是更低层级的细节，如硬件组件类型、内存 bank 数量和数据宽度。

为确保两个框架之间的一致性，我们引入了一个 YAML 文件生成器，它提取高层级的脉动阵列配置，并使用基准 YAML 模板外推出详细的架构描述，最终输出用于能耗建模的 architecture.yaml 文件。基准模板包含默认组件，例如每个 PE 的三个寄存器文件和一个整数 MAC 单元，以及用于输入激活值、权重和部分和的三个智能缓冲 SRAM。用户可以通过提供相应的 Accelergy 兼容描述来自定义组件类型。

## C. 重复访问查找

使 Accelergy 能够提供精确能耗估算的关键特性之一，是其区分不同动作类型的能力，例如重复内存访问和随机内存访问，二者的能耗差异可能超过两倍。为了支持这一特性，SCALE-Sim v3 会生成动作计数文件作为 Accelergy 的输入。

SCALE-Sim v3 引入了两个可调参数：'row size' 和 'bank size'。'row size' 参数定义了每次内存访问所取回的数据块，使得在访问同一数据块内的连续地址时能够进行高能效的重复读取。'bank size' 参数则用于建模一个内存 bank 中的多个行缓冲区，从而支持跨周期的数据复用而无需重新加载。

## D. 内存动作计数

如第 VII-C 节所述，动作计数摘要文件使 Accelergy 能够提供详细的架构级能耗估算。SCALE-Sim v3 与 Accelergy 之间的数据流分为两个阶段。

第一阶段，SCALE-Sim v3 通过所实现的重复计数函数，为 ifmap、ofmap 和 filter 数据生成内存事务计数（idle、random-read/write 和 repeated-read/write）。例如：

$$
\mathrm{ifmap}_{\mathrm{SRAM\text{-}idle}}=\mathrm{cycles}\times\mathrm{arraysize}-\mathrm{counts}
$$

$$
\mathrm{ifmap}_{\mathrm{SRAM\text{-}random}}=\mathrm{counts}-\mathrm{repeat\_counts}
$$

这些计数会被导出到一个摘要文件中。

在第二阶段，这些数据经过后处理，生成用于 Accelergy 能耗估算的 YAML 文件。附加参数（例如 data∆ 和 address∆）用于指示相应的连线开关是否被激活。详细的参数配置如图 14 所示。生成的 YAML 文件在 SCALE-Sim v3 仿真完成后，作为 Accelergy 的主要输入使用。

## E. 处理单元动作计数

MAC 动作计数。在 SCALE-Sim v3 中，我们定义了两种基本的 MAC 动作：MAC random（带有新数据的正常操作）和 MAC constant（无数据变化或计算）。它们的计算方式如下：

$$
\mathrm{MAC\_random}=\#\mathrm{PEs}\times\mathrm{cycles}\times\mathrm{utilization}
$$

![](images/400753242517abbab8a3de40366253422dbdbeadfeb9dea08691eee83d87811a.jpg)  
Fig. 15: 三种工作负载在不同数据流类型和脉动阵列尺寸（128×128、64×64、32×32、16×16 和 8×8）下的能耗。

$$
\mathrm{MAC\_constant}=\#\mathrm{PEs}\times\mathrm{cycles}\times(1-\mathrm{utilization})
$$

未使用的 MAC 单元可以利用时钟门控（clock gating），在 Accelergy 中被建模为 MAC gated，此时仅消耗静态能量，从而降低动态时钟能耗。我们的能耗评估中已纳入这一优化。

Scratchpad 动作计数。每个 PE 内部有三个 scratchpad 存储器，分别为 ifmap scratch pad、weights scratch pad 和 psum scratch pad。基于 Accelergy，每个 scratchpad 有两种动作类型：写（write）和读（read）。我们为每个 scratchpad 生成动作计数。对于 weight spad，我们将写计数建模为 SRAM filter 读取次数，读计数建模为 MAC 操作次数。对于 ifmap spad，我们将写计数建模为 SRAM IFMAP 读取次数，读计数建模为 MAC 操作次数。对于 psum spad，我们将其读和写均建模为 MAC 操作次数。每个 scratchpad 的读写次数与数据流相关。输入固定（input stationary）会使 ifmap spad 的写动作次数更少。输出固定（output stationary）使每个 MAC 单元负责一个 OFMAP 像素所需的全部计算。权重固定（weight stationary）会使 weight spad 的写动作次数更少。

综合以上分析，图 15 展示了 SCALE-Sim v3 在不同尺寸阵列采用不同数据流执行工作负载时的能耗分析示例。OS 在几乎所有情况下都优于另外两种数据流。在 WS 和 IS 之间，WS 在较小阵列尺寸下更优，而 IS 在较大阵列尺寸下更优。这一分析为设计多样化加速器的能效提供了宝贵的洞察。此外，它可以与其他特性（例如 Ramulator、稀疏性、布局等）无缝集成，以获得更深入的设计洞察。

## VIII. 验证

SCALE-Sim v3 构建在 SCALE-Sim v2、Accelergy 和 Ramulator 之上，这三者已分别通过脉动阵列的 RTL（周期精确）、Eyeriss 芯片（误差在 5% 以内）以及 Micron DDR4 的 RTL [83]（周期精确）完成了验证。我们在下面提供一些额外的验证信息：

稀疏性。我们支持两类稀疏加速器：

Ampere 固定 2:4 稀疏性。SCALE-Sim v3 相比 Sparse Tensor Core 的报告 [84] 达到了 100% 的准确率。虽然 Sparseloop [85] 声称计算周期是确定性的，但内存停顿（memory stall）并非如此。SCALE-Sim v3 对周期精确的内存停顿进行建模，并通过 RTL 进行了验证，因而具有更高的硬件准确性。

<table><tr><td rowspan=1 colspan=1>系统状态</td><td rowspan=1 colspan=1>PnR 能耗</td><td rowspan=1 colspan=1>SCALE-Sim v3+Accelergy</td><td rowspan=1 colspan=1>误差</td></tr><tr><td rowspan=1 colspan=1>空闲（时钟门控）</td><td rowspan=1 colspan=1>12.3</td><td rowspan=1 colspan=1>12.6</td><td rowspan=1 colspan=1>+2.4%</td></tr><tr><td rowspan=1 colspan=1>活跃</td><td rowspan=1 colspan=1>315.8</td><td rowspan=1 colspan=1>308.5</td><td rowspan=1 colspan=1>-2.3%</td></tr><tr><td rowspan=1 colspan=1>电源门控</td><td rowspan=1 colspan=1>4.7</td><td rowspan=1 colspan=1>4.9</td><td rowspan=1 colspan=1>+4.3%</td></tr></table>

TABLE III: 不同系统状态（空闲、活跃、电源门控）下 Accelergy 集成的验证结果

<table><tr><td rowspan=1 colspan=1>工作负载</td><td rowspan=1 colspan=1>多核</td><td rowspan=1 colspan=1>稀疏2:4</td><td rowspan=1 colspan=1>稀疏1:4</td><td rowspan=1 colspan=1>Acce-lergy</td><td rowspan=1 colspan=1>Ramu-lator</td><td rowspan=1 colspan=1>布局</td></tr><tr><td rowspan=1 colspan=1>Alexnet</td><td rowspan=1 colspan=1>1.83x</td><td rowspan=1 colspan=1>0.29x</td><td rowspan=1 colspan=1>0.14x</td><td rowspan=1 colspan=1>1.44x</td><td rowspan=1 colspan=1>1.19x</td><td rowspan=1 colspan=1>22.62x</td></tr><tr><td rowspan=1 colspan=1>Resnet-18</td><td rowspan=1 colspan=1>2.06x</td><td rowspan=1 colspan=1>0.30x</td><td rowspan=1 colspan=1>0.30x</td><td rowspan=1 colspan=1>1.09x</td><td rowspan=1 colspan=1>1.74x</td><td rowspan=1 colspan=1>3.6x</td></tr><tr><td rowspan=1 colspan=1>ViT-L</td><td rowspan=1 colspan=1>2.59x</td><td rowspan=1 colspan=1>0.47x</td><td rowspan=1 colspan=1>0.32x</td><td rowspan=1 colspan=1>1.06x</td><td rowspan=1 colspan=1>2.03x</td><td rowspan=1 colspan=1>18.88x</td></tr><tr><td rowspan=1 colspan=1>ViT-S</td><td rowspan=1 colspan=1>2.67x</td><td rowspan=1 colspan=1>0.62x</td><td rowspan=1 colspan=1>0.39x</td><td rowspan=1 colspan=1>1.17x</td><td rowspan=1 colspan=1>3.57x</td><td rowspan=1 colspan=1>19.01x</td></tr><tr><td rowspan=1 colspan=1>均值</td><td rowspan=1 colspan=1>2.29x</td><td rowspan=1 colspan=1>0.42x</td><td rowspan=1 colspan=1>0.29x</td><td rowspan=1 colspan=1>1.19x</td><td rowspan=1 colspan=1>2.13x</td><td rowspan=1 colspan=1>16.03x</td></tr></table>

TABLE IV: 在类 TPU-v2 配置上的仿真时间开销

Vegeta [46] 灵活 N:M 稀疏性。我们在 2:4 和 1:4 配置下与 Vegeta RTL 进行了验证，误差 ≤ 5%。

Accelergy 集成：使用 AlexNet 和 ResNet-18 的 16 位量化权重以及 CIFAR-10 输入数据，我们比较了 PnR 结果（65nm）与 SCALE-Sim v3 之间的总能耗及相对能耗分布（GLB、NoC、PE 阵列），观察到 Eyeriss 的偏差为 4.6%，通用脉动阵列（8x8 阵列、OS 数据流）的偏差为 4.8%。Accelergy 通过经 RTL、综合和 PnR 验证的单位能量值，为各种系统状态（例如 idle、read rand、read repeat、write rand、write repeat、write cst data）提供精确的能耗估算。SCALE-Sim v3 生成周期精确的工作负载轨迹（trace），由 Accelergy 处理以得出每种状态的动作计数，从而实现精确的能耗和功耗计算。这一集成确保了对系统行为的精确刻画，如表 III 所示：

Ramulator 集成：读写请求队列的建模模拟了 Micron DDR4 Verilog 模型 [83] 中 AXI 总线上在途事务（in-flight transaction）的数量。

## IX. 与 SCALE-SIM V2 的比较

## A. 仿真时间对比

SCALE-Sim v3 相较于 v2 的仿真时间开销见表 IV。多核、2:4 稀疏、1:4 稀疏、Accelergy、Ramulator 和 Layout 特性的平均开销分别为 2.29×、0.42×、0.29×、1.19×、2.13× 和 16.03×。Layout 仿真开销较大，这是由于其对 SRAM bank 中的数据布局和 bank 冲突进行了细致而精确的建模。

## B. SCALE-Sim v3 相较于 v2 所发现的不同设计

本节展示了 SCALE-Sim v3 增强功能的必要性，展示了其相较于 v2 得出不同设计结论的情况。

能耗。SCALE-Sim v2 仅以延迟为指标时，显示对于 ViT-base，128×128 阵列比 32×32 快 6.53×。

<table><tr><td rowspan=1 colspan=1>指标</td><td rowspan=1 colspan=3>Resnet-50</td><td rowspan=1 colspan=3>RCNN</td><td rowspan=1 colspan=3>ViT-base</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>32x32</td><td rowspan=1 colspan=1>64x64</td><td rowspan=1 colspan=1>128x128</td><td rowspan=1 colspan=1>32x32</td><td rowspan=1 colspan=1>64x64</td><td rowspan=1 colspan=1>128x128</td><td rowspan=1 colspan=1>32x32</td><td rowspan=1 colspan=1>64x64</td><td rowspan=1 colspan=1>128x128</td></tr><tr><td rowspan=1 colspan=1>延迟（周期/层）</td><td rowspan=1 colspan=1>98721</td><td rowspan=1 colspan=1>35838</td><td rowspan=1 colspan=1>19501</td><td rowspan=1 colspan=1>126830</td><td rowspan=1 colspan=1>52243</td><td rowspan=1 colspan=1>29581</td><td rowspan=1 colspan=1>444970</td><td rowspan=1 colspan=1>130601</td><td rowspan=1 colspan=1>68160</td></tr><tr><td rowspan=1 colspan=1>能耗（mJ）</td><td rowspan=1 colspan=1>6.91</td><td rowspan=1 colspan=1>10.6</td><td rowspan=1 colspan=1>15.2</td><td rowspan=1 colspan=1>12.61</td><td rowspan=1 colspan=1>28.36</td><td rowspan=1 colspan=1>29.25</td><td rowspan=1 colspan=1>11.02</td><td rowspan=1 colspan=1>16.31</td><td rowspan=1 colspan=1>31.49</td></tr><tr><td rowspan=1 colspan=1>EdP（周期 x mJ/层）</td><td rowspan=1 colspan=1>682162.11</td><td rowspan=1 colspan=1>379882.8</td><td rowspan=1 colspan=1>296415.2</td><td rowspan=1 colspan=1>159932</td><td rowspan=1 colspan=1>148161</td><td rowspan=1 colspan=1>865244.25</td><td rowspan=1 colspan=1>490356.9</td><td rowspan=1 colspan=1>2130102.31</td><td rowspan=1 colspan=1>2146358.4</td></tr></table>

表 V：32x32、64x64 和 128x128 阵列的延迟、能耗和 EdP 对比

> 译注：表V原文部分EDP值与“延迟×能量”不一致，本文保留原表数值。例如ViT-base的32×32应按444970×11.02得到约4,903,569.4，而原表写490356.9。RCNN的32×32、64×64也有类似数量级问题。作者关于ViT的64×64 EDP最优结论与按乘积计算的结果一致；引用定量值时应核对原始数据。



<table><tr><td rowspan=1 colspan=1>数据流</td><td rowspan=1 colspan=2>单核 128x128</td><td rowspan=1 colspan=2>16 个核，每个含 32x32 PE</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>延迟</td><td rowspan=1 colspan=1>能耗</td><td rowspan=1 colspan=1>延迟</td><td rowspan=1 colspan=1>能耗</td></tr><tr><td rowspan=1 colspan=1>比值 ws/is</td><td rowspan=1 colspan=1>1.87</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>1.14</td><td rowspan=1 colspan=1>0.70</td></tr></table>

表 VI：多核的延迟和能耗开销

然而，v3 发现 32×32 由于利用率更高、漏电更低，在能效上高出 2.86×。如表 V 所示，对于 ViT-base 的 EdP 而言，64×64 优于 128×128 和 32×32。

DRAM。SCALE-Sim v2 显示，对于 ResNet18 的六层，采用权重固定（WS）数据流相较于输出固定（OS）可减少 21% 的计算周期。然而，在考虑 DRAM 停顿后，OS 数据流的执行周期反而比 WS 低 30.1%，这凸显了详细 DRAM 分析的关键作用。我们在访存密集型层以及请求队列规模较小的设计中观察到类似的结果。我们将在修订版中补充相关分析。

稀疏性。在图 5 中，一个延迟受限（250,000 周期）的设计若采用稠密核，需要 3.00 MB 的片上存储。假设采用 2:4 稀疏，SCALE-Sim v3 可使稀疏核仅需 768 kB，显著减小了面积。

多核。对于 ViT-base，在等计算量设计（单核 128×128 与 16 个 32×32 多核）的对比中，权重固定在延迟上分别比输入固定高出 1.87×（单核）和 1.14×（多核）。SCALE-Sim v2 倾向于选择权重固定，但 v3 使得在排除输入固定之前能够进行更深入的分析，因为其在多核情况下的 EdP 表现好 1.31×，如表 VI 所示。

## X. 相关工作

周期精确仿真器。周期精确仿真器是评估 AI 加速器（尤其是基于脉动阵列的加速器）性能与效率的重要工具。如表 I 所示，若干现有仿真器涵盖了加速器设计的各个方面。uSystolic-SIM [26] 构建于 SCALE-Sim [24] 之上，通过利用一元编码来简化计算，引入了一种独特的周期级仿真方法，不过它主要面向字节爬取架构，且不支持多核或稀疏加速器。

多核仿真器。多核脉动加速器因其能在数据并行工作负载中提升性能而备受关注。Timeloop [22] 等工具为多核加速器提供了分析式建模，并通过 Accelergy [31] 提供基本的能耗估计，但它们缺乏周期精确仿真和数据布局支持。MuchiSim [86] 是另一款值得关注的工具，可仿真大规模多核和多芯片加速器，并对核间通信和存储层次进行了详细建模。然而，它缺乏计算建模，需依赖第三方工具。

稀疏加速器仿真器。稀疏加速器近年来日益受到重视，已有多款仿真器被开发出来，用于探索此类加速器的设计与性能。Sparseloop [85] 扩展了 Timeloop 的能力，增加了对稀疏数据流和映射的显式支持，从而能够详细探索稀疏模式及其对加速器性能的影响。它对稀疏相关的硬件优化（包括门控、跳过和压缩）进行了建模。然而，它是分析式的（将稀疏性建模为一种分布），缺乏周期精确的洞察能力。STONNE [23] 提供了对特定非结构化稀疏加速器 SIGMA [30] 的周期精确仿真。然而，它不支持多核架构、数据布局或详细的存储建模。

## XI. 结论

在本工作中，我们提出了 SCALE-Sim v3，这是一款模块化、周期精确的仿真器，旨在满足现代 AI 加速器设计与分析不断演进的需求。我们在 SCALE-Sim v2 基础上的增强带来了实质性的改进，包括对多核特性、时空分区、稀疏矩阵运算、通过 Ramulator 进行的存储接口建模、通过 Accelergy 进行的能耗/功率估计以及片上数据布局建模的支持。这些特性共同实现了全面的端到端系统分析，从而能够更深入地洞察性能、效率和架构权衡。我们相信，SCALE-Sim v3 将成为旨在探索和优化面向领域的 AI 加速器设计空间的研究者和开发者的宝贵工具。

## 致谢

本工作部分得到 ACE（JUMP 2.0 的七个中心之一，一个由 DARPA 资助的半导体研究公司（SRC）项目）的支持。

## REFERENCES

[1] J. Sevilla, L. Heim, A. Ho, T. Besiroglu, M. Hobbhahn, and P. Villalobos, “Compute trends across three eras of machine learning,” in 2022 International Joint Conference on Neural Networks (IJCNN), pp. 1–8, IEEE, 2022.

[2] N. Jouppi, G. Kurian, S. Li, P. Ma, R. Nagarajan, L. Nai, N. Patil, S. Subramanian, A. Swing, B. Towles, et al., “Tpu v4: An optically reconfigurable supercomputer for machine learning with hardware sup port for embeddings,” in Proceedings of the 50th Annual International Symposium on Computer Architecture (ISCA), pp. 1–14, 2023.

[3] Cerebras, “Wafer Scale Engine (WSE-2) Dataset.” https://f. hubspotusercontent30.net/hubfs/8968533/WSE-2%20Datasheet.pdf.

[4] A. Firoozshahian, J. Coburn, R. Levenstein, R. Nattoji, A. Kamath, O. Wu, G. Grewal, H. Aepala, B. Jakka, B. Dreyer, et al., “Mtia: First generation silicon targeting meta’s recommendation systems,” in Proceedings of the 50th Annual International Symposium on Computer Architecture, pp. 1–13, 2023.

[5] C. Lichtenau, A. Buyuktosunoglu, R. Bertran, P. Figuli, C. Jacobi, N. Papandreou, H. Pozidis, A. Saporito, A. Sica, and E. Tzortzatos, “Ai accelerator on ibm telum processor: Industrial product,” in Proceedings of the 49th Annual International Symposium on Computer Architecture, pp. 1012–1028, 2022.

[6] A. Reuther, P. Michaleas, M. Jones, V. Gadepally, S. Samsi, and J. Kepner, “Survey of machine learning accelerators,” in 2020 IEEE high performance extreme computing conference (HPEC), pp. 1–12, IEEE, 2020.

[7] B. Feinberg, U. K. R. Vengalam, N. Whitehair, S. Wang, and E. Ipek, “Enabling scientific computing on memristive accelerators,” in 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), pp. 367–382, IEEE, 2018.

[8] G. Sun, S. Kang, and S.-W. Jun, “Burstz+: Eliminating the communication bottleneck of scientific computing accelerators via accelerated compression,” ACM Transactions on Reconfigurable Technology and Systems (TRETS), vol. 15, no. 2, pp. 1–34, 2022.

[9] A. Bocco, Y. Durand, and F. De Dinechin, “Smurf: Scalar multipleprecision unum risc-v floating-point accelerator for scientific computing,” in Proceedings of the Conference for Next Generation Arithmetic 2019, pp. 1–8, 2019.

[10] R. Weber, A. Gothandaraman, R. J. Hinde, and G. D. Peterson, “Comparing hardware accelerators in scientific applications: A case study,” IEEE Transactions on Parallel and Distributed Systems, vol. 22, no. 1, pp. 58–68, 2010.

[11] M. Ujaldon, “Hpc accelerators with 3d memory,” in´ 2016 IEEE Intl Conference on Computational Science and Engineering (CSE) and IEEE Intl Conference on Embedded and Ubiquitous Computing (EUC) and 15th Intl Symposium on Distributed Computing and Applications for Business Engineering (DCABES), pp. 320–328, IEEE, 2016.

[12] O. Temam, “A defect-tolerant accelerator for emerging high-performance applications,” ACM SIGARCH Computer Architecture News, vol. 40, no. 3, pp. 356–367, 2012.

[13] K. A. Britt, F. A. Mohiyaddin, and T. S. Humble, “Quantum accelerators for high-performance computing systems,” in 2017 IEEE International Conference on Rebooting Computing (ICRC), pp. 1–7, IEEE, 2017.

[14] K. Li, W. Mao, J. Zhou, B. Li, Z. Yang, S. Yang, L. Du, S. Huang, and H. Yu, “A vector systolic accelerator for multi-precision floatingpoint high-performance computing,” IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 69, no. 10, pp. 4123–4127, 2022.

[15] L. Yang, R. M. Radway, Y.-H. Chen, T. F. Wu, H. Liu, E. Ansari, V. Chandra, S. Mitra, and E. Beigne, “Three-dimensional stacked neural´ network accelerator architectures for ar/vr applications,” IEEE Micro, vol. 42, no. 6, pp. 116–124, 2022.

[16] H. E. Sumbul, J.-s. Seo, D. H. Morris, and E. Beigne, “A fully-digital and row-pipelined compute-in-memory neural network accelerator with soc-level benchmarking for ar/vr applications,” IEEE Micro, 2023.

[17] S. Li, Y. Zhao, C. Li, B. Guo, J. Zhang, W. Zhu, Z. Ye, C. Wan, and Y. C. Lin, “Fusion-3d: Integrated acceleration for instant 3d reconstruction and real-time rendering,” in 2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 78–91, IEEE, 2024.

[18] S. Krishnan, Z. Wan, K. Bhardwaj, P. Whatmough, A. Faust, S. Neuman, G.-Y. Wei, D. Brooks, and V. J. Reddi, “Automatic domain-specific soc design for autonomous unmanned aerial vehicles,” in 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 300–317, IEEE, 2022.

[19] Y. Hao, Y. Gan, B. Yu, Q. Liu, Y. Han, Z. Wan, and S. Liu, “Orianna: An accelerator generation framework for optimization-based robotic applications,” in Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, pp. 813–829, 2024.

[20] Q. Liu, Z. Wan, B. Yu, W. Liu, S. Liu, and A. Raychowdhury, “An energy-efficient and runtime-reconfigurable fpga-based accelerator for robotic localization systems,” in 2022 IEEE Custom Integrated Circuits Conference (CICC), pp. 01–02, IEEE, 2022.

[21] S. Liu, Z. Wan, B. Yu, and Y. Wang, Robotic computing on fpgas. Springer, 2021.

[22] A. Parashar, P. Raina, Y. S. Shao, Y.-H. Chen, V. A. Ying, A. Mukkara, R. Venkatesan, B. Khailany, S. W. Keckler, and J. Emer, “Timeloop: A

systematic approach to dnn accelerator evaluation,” in 2019 IEEE international symposium on performance analysis of systems and software (ISPASS), pp. 304–315, IEEE, 2019.

[23] F. Munoz-Mart˜ ´ınez, J. L. Abellan, M. E. Acacio, and T. Krishna,´ “Stonne: Enabling cycle-level microarchitectural simulation for dnn inference accelerators,” in 2021 IEEE International Symposium on Workload Characterization (IISWC), pp. 201–213, IEEE, 2021.

[24] A. Samajdar, Y. Zhu, P. Whatmough, M. Mattina, and T. Krishna, “Scale-sim: Systolic cnn accelerator simulator,” arXiv preprint arXiv:1811.02883, 2018.

[25] H. Kwon, P. Chatarasi, V. Sarkar, T. Krishna, M. Pellauer, and A. Parashar, “Maestro: A data-centric approach to understand reuse, performance, and hardware cost of dnn mappings,” IEEE micro, vol. 40, no. 3, pp. 20–29, 2020.

[26] D. Wu and J. San Miguel, “usystolic: Byte-crawling unary systolic array,” in 2022 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pp. 12–24, IEEE, 2022.

[27] Y.-H. Chen, T. Krishna, J. S. Emer, and V. Sze, “Eyeriss: An energyefficient reconfigurable accelerator for deep convolutional neural networks,” IEEE journal of solid-state circuits, vol. 52, no. 1, pp. 127–138, 2016.

[28] H. Kwon, P. Chatarasi, M. Pellauer, A. Parashar, V. Sarkar, and T. Krishna, “Understanding reuse, performance, and hardware cost of dnn dataflow: A data-centric approach,” in Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pp. 754–768, 2019.

[29] N. Muralimanohar, R. Balasubramonian, and N. P. Jouppi, “Cacti 6.0: A tool to model large caches,” HP laboratories, vol. 27, p. 28, 2009.

[30] E. Qin, A. Samajdar, H. Kwon, V. Nadella, S. Srinivasan, D. Das, B. Kaul, and T. Krishna, “Sigma: A sparse and irregular gemm accelerator with flexible interconnects for dnn training,” in 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA), pp. 58–70, IEEE, 2020.

[31] Y. N. Wu, J. S. Emer, and V. Sze, “Accelergy: An architecturelevel energy estimation methodology for accelerator designs,” in 2019 IEEE/ACM International Conference on Computer-Aided Design (IC-CAD), pp. 1–8, IEEE, 2019.

[32] R. Balasubramonian, A. B. Kahng, N. Muralimanohar, A. Shafiee, and V. Srinivas, “Cacti 7: New tools for interconnect exploration in innovative off-chip memories,” ACM Transactions on Architecture and Code Optimization (TACO), vol. 14, no. 2, pp. 1–25, 2017.

[33] Y. Kim, W. Yang, and O. Mutlu, “Ramulator: A fast and extensible dram simulator,” IEEE Computer architecture letters, vol. 15, no. 1, pp. 45– 49, 2015.

[34] J. Tong, A. Itagi, P. Chatarasi, and T. Krishna, “Feather: A reconfigurable accelerator with data reordering support for low-cost on-chip dataflow switching,” 2024.

[35] J. Kim, G. Lee, S. Kim, G. Sohn, M. Rhu, J. Kim, and J. H. Ahn, “Ark: Fully homomorphic encryption accelerator with runtime data generation and inter-operation key reuse,” in 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 1237–1254, IEEE, 2022.

[36] T. Ma, Y. Feng, X. Zhang, and Y. Zhu, “Camj: Enabling systemlevel energy modeling and architectural exploration for in-sensor visual computing,” in Proceedings ofthe 50th Annual International Symposium on Computer Architecture, pp. 1–14, 2023.

[37] H. Guo, Y. Zhao, Z. Li, Y. Hao, C. Liu, X. Song, X. Li, Z. Du, R. Zhang, Q. Guo, et al., “Cambricon-u: A systolic random increment memory architecture for unary computing,” in Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture, pp. 424–437, 2023.

[38] A. Samajdar, E. Qin, M. Pellauer, and T. Krishna, “Self adaptive reconfigurable arrays (sara) learning flexible gemm accelerator configuration and mapping-space using ml,” in Proceedings of the 59th ACM/IEEE Design Automation Conference, pp. 583–588, 2022.

[39] Z. Liu, J. Wang, T. Dao, T. Zhou, B. Yuan, Z. Song, A. Shrivastava, C. Zhang, Y. Tian, C. Re, et al., “Deja vu: Contextual sparsity for efficient llms at inference time,” in International Conference on Machine Learning, pp. 22137–22176, PMLR, 2023.

[40] T. Gale, E. Elsen, and S. Hooker, “The state of sparsity in deep neural networks,” arXiv preprint arXiv:1902.09574, 2019.

[41] B. Liu, M. Wang, H. Foroosh, M. Tappen, and M. Pensky, “Sparse convolutional neural networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 806–814, 2015.

[42] T. Gale, D. Narayanan, C. Young, and M. Zaharia, “Megablocks: Efficient sparse training with mixture-of-experts,” Proceedings of Machine Learning and Systems, vol. 5, pp. 288–304, 2023.

[43] H. Shen, H. Meng, B. Dong, Z. Wang, O. Zafrir, Y. Ding, Y. Luo, H. Chang, Q. Gao, Z. Wang, et al., “An efficient sparse inference software accelerator for transformer-based language models on cpus,” arXiv preprint arXiv:2306.16601, 2023.

[44] C. Giannoula, I. Fernandez, J. Gomez-Luna, N. Koziris, G. Goumas, and´ O. Mutlu, “Towards efficient sparse matrix vector multiplication on real processing-in-memory architectures,” ACM SIGMETRICS Performance Evaluation Review, vol. 50, no. 1, pp. 33–34, 2022.

[45] “NVIDIA Ampere Architecture.” https://www.nvidia.com/en-in/datacenter/ampere-architecture/, 2020.

[46] G. Jeong, S. Damani, A. R. Bambhaniya, E. Qin, C. J. Hughes, S. Subramoney, H. Kim, and T. Krishna, “Vegeta: Vertically-integrated extensions for sparse/dense gemm tile acceleration on cpus,” in 2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pp. 259–272, IEEE, 2023.

[47] K. Kanellopoulos, N. Vijaykumar, C. Giannoula, R. Azizi, S. Koppula, N. M. Ghiasi, T. Shahroodi, J. G. Luna, and O. Mutlu, “Smash: Codesigning software compression and hardware-accelerated indexing for efficient sparse matrix operations,” in Proceedings of the 52nd annual IEEE/ACM international symposium on microarchitecture, pp. 600–614, 2019.

[48] X. Zhou, Z. Du, Q. Guo, S. Liu, C. Liu, C. Wang, X. Zhou, L. Li, T. Chen, and Y. Chen, “Cambricon-s: Addressing irregularity in sparse neural networks through a cooperative software/hardware approach,” in 2018 51st Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 15–28, IEEE, 2018.

[49] Google, “TPUv5e.” https://cloud.google.com/tpu/docs/v5e.

[50] H. Jun, J. Cho, K. Lee, H.-Y. Son, K. Kim, H. Jin, and K. Kim, “Hbm (high bandwidth memory) dram technology and architecture,” in 2017 IEEE International Memory Workshop (IMW), pp. 1–4, IEEE, 2017.

[51] O. Mutlu, “The rowhammer problem and other issues we may face as memory becomes denser,” in Design, Automation & Test in Europe Conference & Exhibition (DATE), 2017, pp. 1116–1121, IEEE, 2017.

[52] O. Mutlu and J. S. Kim, “Rowhammer: A retrospective,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems vol. 39, no. 8, pp. 1555–1571, 2019.

[53] H. T. Kung and C. E. Leiserson, “Systolic arrays (for vlsi),” in Sparse Matrix Proceedings 1978, vol. 1, pp. 256–282, Society for industrial and applied mathematics Philadelphia, PA, USA, 1979.

[54] H.-T. Kung, Why systolic architecture? Design Research Center, Carnegie-Mellon University, 1982.

[55] C. Mead and L. Conway, “Introduction to vlsi systems,” 1980.

[56] B. K. Saptalakar, D. Kale, M. Rachannavar, and M. Pavankumar, “Design and implementation of vlsi systolic array multiplier for dsp applications,” International Journal of Scientific Engineering and Technology, vol. 2, no. 3, pp. 156–159, 2013.

[57] D.-F. Chiper, M. Swamy, M. O. Ahmad, and T. Stouraitis, “A systolic array architecture for the discrete sine transform,” IEEE transactions on signal processing, vol. 50, no. 9, pp. 2347–2354, 2002.

[58] P. Quinton, The systematic design of systolic arrays. PhD thesis, INRIA, 1983.

[59] K. T. Johnson, A. R. Hurson, and B. Shirazi, “General-purpose systolic arrays,” Computer, vol. 26, no. 11, pp. 20–31, 1993.

[60] OpenAI, “AI and compute.” https://openai.com/index/ai-and-compute/.

[61] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan, et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[62] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al., “Gemini: a family of highly capable multimodal models,” arXiv preprint arXiv:2312.11805 2023.

[63] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[64] W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen, “Genad: Generative end-to-end autonomous driving,” in European Conference on Computer Vision, pp. 87–104, Springer, 2025.

[65] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, et al., “Planning-oriented autonomous driving,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17853–17862, 2023.

[66] S. Atakishiyev, M. Salameh, H. Yao, and R. Goebel, “Explainable artificial intelligence for autonomous driving: A comprehensive overview and field guide for future research directions,” IEEE Access, 2024.

[67] A. Rahman, J. M. J. Valanarasu, I. Hacihaliloglu, and V. M. Patel, “Ambiguous medical image segmentation using diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11536–11546, 2023.

[68] Y. Zhang, H. Jiang, Y. Miura, C. D. Manning, and C. P. Langlotz, “Contrastive learning of medical visual representations from paired images and text,” in Machine Learning for Healthcare Conference, pp. 2–25, PMLR, 2022.

[69] Y. Gao, X. Xiang, N. Xiong, B. Huang, H. J. Lee, R. Alrifai, X. Jiang, and Z. Fang, “Human action monitoring for healthcare based on deep learning,” Ieee Access, vol. 6, pp. 52277–52285, 2018.

[70] Nvidia, “DGX GH200 for Large Memory AI Supercomputer.” https: //www.nvidia.com/en-in/data-center/dgx-gh200/.

[71] E. Talpes, D. Williams, and D. D. Sarma, “Dojo: The microarchitecture of tesla’s exa-scale computer,” in 2022 IEEE Hot Chips 34 Symposium (HCS), pp. 1–28, IEEE Computer Society, 2022.

[72] J. Seo and J. Kong, “Versa: Versatile systolic array architecture for sparse and dense matrix multiplications,” Electronics, vol. 13, no. 8, p. 1500, 2024.

[73] H. Luo, Y. C. Tugrul, F. N. Bostancı, A. Olgun, A. G. Ya˘ glıkc¸ı, and˘ O. Mutlu, “Ramulator 2.0: A modern, modular, and extensible dram simulator,” IEEE Computer Architecture Letters, vol. 23, no. 1, pp. 112– 116, 2023.

[74] S. Li, Z. Yang, D. Reddy, A. Srivastava, and B. Jacob, “Dramsim3: A cycle-accurate, thermal-capable dram simulator,” IEEE Computer Architecture Letters, vol. 19, no. 2, pp. 106–109, 2020.

[75] Y. S. Shao, J. Clemons, R. Venkatesan, B. Zimmer, M. Fojtik, N. Jiang, B. Keller, A. Klinefelter, N. Pinckney, P. Raina, et al., “Simba: Scaling deep-learning inference with multi-chip-module-based architecture,” in Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pp. 14–27, 2019.

[76] A. Symons, L. Mei, S. Colleman, P. Houshmand, S. Karl, and M. Verhelst, “Towards heterogeneous multi-core accelerators exploiting finegrained scheduling of layer-fused deep neural networks,” arXiv preprint arXiv:2212.10612, 2022.

[77] A. Nandakumar, S. Shao, and B. Nikolic, “Accelerating deep learning on heterogenous architectures,” 2022.

[78] O. Spantidi, G. Zervakis, S. Alsalamin, I. Roman-Ballesteros, J. Henkel, H. Amrouch, and I. Anagnostopoulos, “Targeting dnn inference via efficient utilization of heterogeneous precision dnn accelerators,” IEEE Transactions on Emerging Topics in Computing, vol. 11, no. 1, pp. 112– 125, 2022.

[79] J. H. Lau, Chip on board: technology for multichip modules. Springer Science & Business Media, 1994.

[80] D. A. Doane and P. Franzon, Multichip module technologies and alternatives: the basics. Springer Science & Business Media, 2013.

[81] A. Arunkumar, E. Bolotin, B. Cho, U. Milic, E. Ebrahimi, O. Villa, A. Jaleel, C.-J. Wu, and D. Nellans, “Mcm-gpu: Multi-chip-module gpus for continued performance scalability,” ACM SIGARCH Computer Architecture News, vol. 45, no. 2, pp. 320–332, 2017.

[82] V. N. Sekhar, M. D. Kumar, S. K. Tippabhotla, B. C. Rao, I. C. Daniel, S. C. Chong, and V. S. Rao, “Multi-chip stacked memory module development using chip to wafer (c2w) hybrid bonding for heterogeneous integration applications,” in 2024 IEEE 74th Electronic Components and Technology Conference (ECTC), IEEE, 2024.

[83] Micron Technology, “Micron DDR4 Verilog Model,” 2018.

[84] “Nvidia a100 tensor core gpu architecture.” https://images.nvidia. com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecturewhitepaper.pdf.

[85] Y. N. Wu, P.-A. Tsai, A. Parashar, V. Sze, and J. S. Emer, “Sparseloop: An analytical approach to sparse tensor accelerator modeling,” in 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 1377–1395, IEEE, 2022.

[86] M. Orenes-Vera, E. Tureci, M. Martonosi, and D. Wentzlaff, “Muchisim: A simulation framework for design exploration of multi-chip manycore systems,” in 2024 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), pp. 48–60, IEEE, 2024.



## 附录

## A. 摘要

本 artifact 附录部分描述了如何访问图 1 中介绍的 SCALE-Sim v3 模拟器的 artifacts。该 artifact 还描述了如何复现通过 SCALE-Sim v3 框架开展的第三节（图 3）、第四节（图 5、图 7 和图 8）、第五节（图 9 和图 10）、第六节（图 12 和图 13）中的实验。

## B. Artifact 核对清单（元信息）

• 程序：Python3 和 C++11 编译器（例如 clang++、g++-5）

• 硬件：基于 Linux 的系统

• 指标：延迟、内存、吞吐量和能耗

• 输出：计算与带宽 csv 报告，此外还有稀疏性报告、Ramulator 报告和 Accelergy 报告 • 大约需要多少磁盘空间？：10 GB

• 准备工作流程大约需要多少时间？：5 分钟

• 完成实验大约需要多少时间？：15-20 小时

• 是否公开可用？：是

• 代码许可（如果公开可用）？：MIT 许可证（GitHub 仓库中的 LICENCE 文件）

## C. 描述

1) 访问方式：该 artifact 已上传至 Zenodo（DOI: 10.5281/zenodo.15170910）

2) 硬件依赖：该 artifact 已在基于 Linux 的系统上完成测试

3) 软件依赖：Python3 和 C++11 编译器（例如 clang++、g++-5）

## D. 安装

若要进行 artifact 评估，请遵循以下步骤：

```shell
$ git clone
https://github.com/scalesim-project/scale-sim-v2.git
$ cd scale-sim-v2
$ git checkout v3_artifact
$ pip install -r requirements.txt
$ pip install -e .
$ export PYTHONPATH=.:$PYTHONPATH
```

## E. 实验工作流程

环境搭建完毕后，我们将运行 SCALE-Sim v3 artifact 并评估结果。第一步是编写配置文件和拓扑文件。实验所用的配置文件和拓扑文件分别位于 configs 和 topologies 文件夹下。此外，布局文件位于 layouts 文件夹下。

有关针对稀疏性的配置文件和拓扑文件自定义，请参阅“实验自定义”部分。

为了更方便地评估实验，我们为每个部分提供了 python 和 shell 脚本，将在下一节中介绍。

## F. 评估与预期结果

1) 多核（图 3）：

\$ python3 multi-core/analytical.py   
\$ python3 multi-core/plot\_scatter.py

生成的图片将位于 multi-core 目录中，文件名为 compute optimized tradeoff.pdf 和 memory optimized tradeoff.pdf，分别复现图 3a 和图 3b。

2) 稀疏性（图 5）：

```shell
$ python3 sparsity/compute_cycles_plot_128x128.py
```

生成的图片将位于 sparsity 目录中，文件名为 resnet18 compute cycles.pdf，复现图 5。

请注意，该 python 文件基于 sparsity/sparsity results 128x128 文件夹中收集的实验数据进行绘图。若要复现实验数据，请遵循 sparsity 文件夹中的 README.md 文件。完成所有运行预计需要约 10-12 小时。下一小节也将使用相同的实验数据。

3) 稀疏性（图 7）：

```shell
$ python3 sparsity/memoryplot.py
```

生成的图片将位于 sparsity 目录中，文件名为 memoryplot.pdf，复现图 7。

4) 稀疏性（图 8）：

```shell
$ bash sparsity/run_vit_compute.sh
```

生成的图片将位于 sparsity 目录中，文件名为 vit compute cycles scatterplot.pdf，复现图 8。

5) 布局（图 12、图 13）：

```shell
$ bash layout_plots/run_layout.sh
```

生成的图片将位于 layout plots 目录中，文件名为 multi bank vs bandwidth model resnet18.pdf 和 multi bank vs bandwidth model vit.pdf，分别复现图 12 和图 13。

6) Ramulator：请遵循以下步骤：

```shell
$ git checkout dev-ramulator-merge
$ pip install -r requirements.txt
$ git submodule update --init --recursive
$ cd submodules
$ cp -r ../scripts/ramulator_patch/<sub>*</sub> ./ramulator/
$ cd ramulator
$ make -j <num_jobs>
```

若要复现图 9，请使用以下命令：

```shell
$ cd ../../
$ source generate_fig9_ramulator_mem_bw_plot.sh
```

若要复现图 10，请使用以下命令：

```shell
$ source generate_fig10_ramulator_stall_plot.sh
```

## G. 实验自定义

若要运行自定义实验，用户可以编写自己的配置文件和拓扑文件并将其提供给程序。以稀疏性为例。配置文件中完整的稀疏性参数列表包括：

SparsitySupport : 启用或禁用稀疏性支持   
SparseRep: 稀疏性表示（csr、csc、   
ellpack\_block)   
OptimizedMapping : 启用行级稀疏性，   
默认为层级稀疏性   
BlockSize : 行级稀疏性中 N:M 比率的 M

稀疏性拓扑文件如下所示：

Layer Name, M, N, K, Sparsity,   
GEMM\_1, 3, 5, 16, 3:4,   
GEMM\_2, 1, 5, 16, 1:4,

其中，M、N 和 K 是常规 GEMM 的维度。Sparsity 中的条目采用 N:M 格式，对应于 N:M 行级稀疏性。

## H. 方法论

提交、评审和徽章授予方法论：

• https://www.acm.org/publications/policies/artifactreview-and-badging-current

• https://cTuning.org/ae