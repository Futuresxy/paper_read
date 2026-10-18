# Taxonomy and Benchmarking of Precision-Scalable MAC Arrays under Enhanced DNN Dataflow Representation 原文翻译

# 增强 DNN 数据流表示下精度可扩展 MAC 阵列的分类与基准测试

Ehab M. Ibrahim、Linyan Mei（IEEE 学生会员）、Marian Verhelst（IEEE 高级会员）。前两位作者对本文贡献相同。

作者单位：比利时鲁汶大学电气工程系（ESAT）MICAS 实验室。邮箱：ehab.ibrahim@magics.tech；linyan.mei@kuleuven.be；marian.verhelst@kuleuven.be。

原文版本：arXiv:2108.04773v2 [cs.DC]，2022 年 1 月 17 日。

## 摘要

低精度和可变精度乘累加（MAC）运算能够在不损失或只有限损失算法性能的情况下，显著提升 DNN 加速器的能效和吞吐量，为在资源受限的边缘设备上部署 AI 应用创造条件。因此，近年来提出了多种精度可扩展 MAC 阵列（PSMA）架构。然而，由于不同 PSMA 分别在不同系统和工艺中展示，很难对这些方案进行公平比较。

本文旨在清晰呈现 PSMA 的设计空间，并帮助设计者根据自身需求选择最优架构。首先，我们为 DNN 数据流引入包含精度信息的 for 循环表示。随后，利用这一表示建立全面的 PSMA 分类体系，既能系统覆盖大多数有代表性的先进 PSMA，也能发现新的 PSMA 架构。在此基础上，我们构建高度参数化的 PSMA 模板，可在设计时配置为分类体系所覆盖设计空间中的大量方案，由此公平、全面地对 72 种不同 PSMA 架构开展基准测试。

研究采用 28nm 工艺，目标是在运行时支持从 8 比特到 2 比特的精度伸缩，分别工作于 200MHz 和 1GHz。对所得能量与面积分解结果的分析，揭示了 PSMA 架构的关键设计准则。

**关键词：** 深度神经网络、加速器、ASIC、综合、精度可扩展、乘累加、MAC 阵列。

## I. 引言

边缘设备上的深度学习推理正在迅速发展。然而，阻碍嵌入式推理广泛部署的主要因素，是边缘端有限的能量预算。因此，研究者致力于改善嵌入式深度神经网络（DNN）推理能效的技术。一种常用方法是降低 DNN 操作数——权重、输入及输出激活——的精度，通常称为网络量化。由于其计算特性，DNN 对近似误差具有较强容忍能力 [1]、[2]，因而能够降低参数精度，而不会付出很大的推理准确率代价 [3]–[5]。

支持说明：本研究部分受佛兰德政府 AI 研究计划、佛兰德科学研究基金 FWO-Vlaanderen（G006718N），以及欧盟 ERC-2016-STG-715037 项目资助。Ehab M. Ibrahim 参与本研究时，于 2020–2021 年在鲁汶大学 ESAT-MICAS 担任研究助理。

DNN 量化有两类方法：（1）训练后量化，先以全精度训练网络，再量化预训练参数，可以选择是否在量化后微调 [6]；（2）量化感知训练，直接以较低精度训练网络 [7]。这些方法推动 DNN 从 16 比特定点量化 [8]，发展到 8 比特定点 [9]，乃至二值精度 [10]。不过，DNN 参数的最佳精度因网络模型而异，甚至在同一模型的不同层之间也不同 [6]、[11]。

这促使 DNN 加速器从传统固定精度 MAC 阵列转向运行时精度可扩展 MAC 阵列。PSMA 可以在运行时切换到所需精度模式，从而有效地将低精度机会转化为能量和性能收益。

近年来出现了许多精度可扩展加速器，其中一些仅支持权重精度伸缩，即一维伸缩 [12]–[14]；另一些支持输入与权重同时伸缩，即二维伸缩 [15]–[19]。这些精度可扩展 MAC 使用不同 RTL 编码风格、不同硅工艺、不同 MAC 阵列配置和不同 DNN 工作负载进行展示，因此很难比较各自优劣。

文献 [20] 在 MAC 层面提供了全面的精度可扩展 MAC 单元综述与比较。然而，它只关注单个 MAC，没有研究将这些 MAC 放入阵列后的影响，因而忽略了两项重要的阵列级效率因素。

第一，根据精度可扩展 MAC 架构的不同，精度伸缩相关开销可能能够、也可能无法在整个阵列的 MAC 之间摊销，这会显著影响每次运算的能量和面积开销。第二，MAC 架构会影响低精度模式下可以在阵列中利用哪些空间展开维度，同样会改变阵列级能效。

为了克服这些不足，并在 MAC 阵列层面公平比较精度可扩展架构，本文作出以下贡献：

- 在传统卷积神经网络（CNN）层的七重 for 循环表示上，增加两个包含精度信息的位组循环，并讨论新增循环对底层硬件及其空间、时间展开选择的影响，详见第 II 节。


![图 1：简单的四比特精度可扩展乘法器](images/psma-taxonomy-benchmark-figure-01.png)

**图 1.** 一个简单的 4 比特精度可扩展乘法器。图中以 2 比特为一个位组（BG），展示 4×4、4×2 与 2×2 比特乘法的组成方式。

本文的其他贡献包括：

- 基于新提出的九重循环 CNN 表示，建立新的 PSMA 分类体系；将多种先进 PSMA 架构映射到该体系，并识别新的拓扑，详见第 III 节。
- 设计统一且高度参数化的 PSMA 模板，可在设计时配置为这一分类体系所覆盖设计空间中的大量方案，详见第 IV 节。
- 从面积和能量方面对该设计空间进行基准测试，形成 PSMA 设计见解，详见第 V 节。

源代码与补充资料：[PSMA-benchmark](https://github.com/KULeuven-MICAS/PSMA-benchmark)。

## II. 包含精度信息的 CNN 循环表示

本节对传统七重 CNN for 循环形式作精度驱动的扩展。传统表示已广泛用于描述不同加速器的数据流。我们先说明扩展动机，再依据这些循环对底层硬件的影响进行分类。

### A. 动机

为了理解扩展七重循环表示的动机，首先解释位组（BG）的概念。精度可扩展加速器通常利用低精度子乘法器组成底层 MAC 单元 [21]。当执行全精度运算时，将中间部分积移位并相加；而在低精度模式中，这些部分积可直接作为独立结果。

图 1 展示了一个简单的 4 比特精度可扩展乘法器，架构类似 BitFusion [15] 中的 BitBrick。该 4×4 比特乘法器由四个 2×2 比特子乘法器构成。在完整的 4 比特模式下，两个输入分别划分为两个 2 比特 BG，合起来表示完整的四比特输入。不同子单元计算这些 BG 的乘积，再移位、聚合，得到最终 4×4 比特乘法结果。

在低精度 2 比特模式下，四个 2 比特输入可以独立使用，每个输入只含一个 BG。此时每个子单元都像独立乘法器一样，产生独立结果。从这个意义上说，乘法器可以被视作微型阵列：全精度时在 BG 层面进行空间展开，低精度时则释放出额外的空间展开能力。

![图 2：新增两个位组循环](images/psma-taxonomy-benchmark-figure-02.png)

**图 2.** 在传统七重 CNN 循环之外增加两个 BG 循环。传统循环分别对应批次 b、输出通道/权重卷积核 k、输入/权重通道 c、输出行 oy、输出列 ox、卷积核行 fy 和列 fx；新增 bi、bw 分别遍历输入与权重位组。图中假定每组 2 比特，取出相应位段后分别乘以 $2^{2b_i}$、$2^{2b_w}$，还原各位组的数值权重。图例区分权重共享、输入共享、输出共享以及需要移位相加的 BG 循环。

将这一概念加入 CNN 数据流表示，便能统一描述各类精度伸缩技术。CNN 加速器的数据流包含空间展开，即如何将 CNN 循环映射到二维 MAC 阵列，以及时间映射，即如何在加速器上调度这些循环 [22]。传统上，数据流使用七重嵌套 for 循环表示。为了包含精度伸缩的影响，如图 2 所示，我们加入两个 BG 循环。

在图 1 的例子中，高精度时，两个 BG 循环在 MAC 单元内部空间展开，因此单元没有余地再空间展开其他循环。降低精度后，工作负载的 BG 循环规模缩小，于是可以在 MAC 内部空间展开其他循环。

将 BG 循环视作普通 CNN 循环，也就可以灵活选择空间展开或时间展开；后者正是位串行（BS）架构的做法。由此可以统一刻画多种 PSMA 的行为。除卷积层外，全连接、深度卷积和逐点卷积等 DNN 层也可以用同样方式扩展。

### B. 对 MAC 阵列硬件的影响

除 BG 循环之外，还必须理解各循环如何映射到 MAC 阵列。如图 2 所示，传统七个 CNN 循环根据输入、权重或输出共享方式分类。例如，K（输出通道维度）是输入共享循环，因为无论按空间还是时间遍历 K，都需要相同的输入元素，因此这些输入可以共享。权重共享和输出共享循环遵循相同思路。


![图 3：分层阵列模板](images/psma-taxonomy-benchmark-figure-03.png)

**图 3.** L4 由 4×4 个 L3 单元组成，L3 由 4×4 个 L2 单元组成，L2 由 4×4 个 L1 单元组成；L1 是基本 2×2 比特乘法器，输出宽度为 4 比特。

![图 4：某层的输入、混合和输出共享](images/psma-taxonomy-benchmark-figure-04.png)

**图 4.** 第 n 层（Ln）的输入共享、混合共享和输出共享。输入广播线分配操作数；同一累加岛中的结果在空间上相加，形成一个输出。（a）两个维度分别共享输入和权重；（b）一个维度共享输入或权重，另一个维度共享输出；（c）两个维度都共享输出。

当输入共享或权重共享循环沿 MAC 阵列某一维度空间展开时，输入或权重可以广播给该维度上的全部 MAC。相反，如果空间展开的是输出共享循环，各 MAC 的结果就沿该维度相加。此外，BG 循环的空间或时间展开也会影响硬件：前者需要部分结果经过移位加法树，后者则需要在移位累加寄存器中聚合。

## III. 精度可扩展 MAC 阵列分类体系

利用新的循环表示，我们引入可映射各种先进精度伸缩技术的分类体系。它用于定义 PSMA 的完整设计空间，第 V 节将对其进行广泛探索和基准测试。

### A. 高度参数化的 PSMA 模板

统一且高度参数化的 MAC 阵列模板，以基本 2 比特乘法器为构件，称为第一级 L1 单元，并与前文的精度伸缩概念兼容。L1 按不同层级组合成完整阵列。本文每级组合 4×4 个单元：16 个 L1 构成一个 L2，16 个 L2 构成一个 L3，16 个 L3 构成一个 L4，如图 3 所示。

采用分层架构的主要动机是灵活性。较小阵列可以止于 L3，标准阵列使用 L4，更大的阵列也可以继续向上扩展。较小的 L1 使 MAC 单元能够伸缩到 2 比特，这是大多数精度可扩展加速器所支持的最低精度 [15]、[16]、[18]、[23]。同时，多层结构清晰展示了每个 CNN 循环在何处展开。

### B. 空间展开

前节的权重共享 WS、输入共享 IS、输出共享 OS 和 BG 循环分类，现在可以与分层阵列模板关联，以分析硬件后果。我们定义每层 Ln 沿水平、垂直方向各支持一种空间展开，两者可以相同也可以不同。

具体来说，若 IS/WS 循环沿某层的一个维度空间展开，说明输入或权重在该维度的不同 Ln−1 单元间广播；这些单元的结果彼此不能相加，需要独立累加器。若展开的是 OS 循环，则该维度上 Ln−1 单元的部分和由加法树相加，同时意味着各单元所用的输入与权重不同。

由此形成图 4 的三种展开方式：（a）两个维度都共享操作数，即共享输入和权重，产生 16 个独立输出；（b）一个维度共享操作数，另一维度共享输出，产生 4 个聚合输出；（c）两个维度都共享输出，产生一个共同的和。

### C. 位组展开

除 WS/IS/OS 循环外，BG 循环也可以在阵列上空间展开，并决定移位加法树放在何处。在本文分类中，BG 循环可以在阵列最底部两个层级之一，即 L2 或 L3，进行空间展开；注意 L1 本身不是阵列层。另一选择是在时间上展开，即位串行处理 BS。

时间展开 BG 循环时，需要内部寄存器保存中间结果，从而在多个周期执行移位相加。BS 设计还可按内部寄存器位于 PSMA 的哪个层级进一步分类。

需要注意，只有高于最低精度时才需要展开 BG 循环；在最低精度下，没有需要展开的 BG 循环，也就是该循环维度的大小等于 1。

### D. 精度伸缩模式

现有先进 PSMA 支持不同的精度伸缩模式。在新分类中，如果只有权重精度可变，而激活精度固定在最高精度，则称为一维精度伸缩，例如 8b×8b、8b×4b 和 8b×2b。如果权重与激活精度均可变，则称为二维精度伸缩。


![图 5：完全展开设计的 L2](images/psma-taxonomy-benchmark-figure-05.png)

**图 5.** BG 在 L2 展开时，完全展开（FU）设计的 L2。子图分别为 L2 输入共享、混合共享和输出共享，展示 8、4、2 比特模式。绿色表示运行时可配置的移位加法逻辑，红色边框表示累加岛。

二维伸缩还可以细分：如果权重和激活精度相同，例如 8b×8b、4b×4b 和 2b×2b，称为二维对称伸缩（2D-S）；如果允许权重与激活使用不同精度，例如 8b×8b、8b×4b、8b×2b、4b×4b 和 2b×2b，则称为二维非对称伸缩（2D-A）。

### E. 完全展开与子字展开设计

降低精度会带来存储带宽与硬件利用率之间的取舍。用其他循环展开替代 BG 循环展开时，若要维持充分的硬件利用率，阵列通常需要增加输入/权重或输出带宽。例如，BitFusion [15] 与 BitBlade [16] 从 8 比特降到 4 比特、2 比特时，输入/权重带宽分别增加到 2 倍、4 倍。图 5 展示不同 L2 展开方式在各精度下的行为。为简化说明，图中只画出 BG 位于 L2 的情形；同样的概念与取舍也适用于其他 BG 配置。

在完整精度下，图 5 的所有展开方式行为相同，因为 BG 循环已经直接占用该层，没有余地展开其他循环。精度降低后，BG 循环的展开规模缩小，使其他 for 循环获得更多展开空间。

如果额外展开输入共享循环，权重和激活可沿水平与垂直方向广播，因此输入带宽保持不变。但输出不能相加，输出带宽因此增加。如果改为展开输出共享循环，全部部分结果可以相加，输出带宽维持不变；但各单元需要不同输入，输入带宽就必须增加。混合共享 HS 则处于二者之间。因此，对所有设计而言，低精度运算都会增加某一侧的 I/O 带宽需求。

![图 6：子字展开设计的 L2](images/psma-taxonomy-benchmark-figure-06.png)

**图 6.** BG 在 L2 展开时，子字展开（SWU）设计的 L2。（a）L2 不共享；（b）L2 输出共享。图中比较 8、4、2 比特模式，绿色为固定移位加法逻辑，红色表示累加岛，灰色区域为被门控的硬件，移位量以硬连线方式实现。

另一类 PSMA 拓扑选择在不同精度下保持固定 I/O 带宽，以简化存储接口设计。代价是低精度硬件利用率降低：精度下降时，对部分 L1 单元实施门控，如图 6 所示。这类 MAC 最早见于 [24]、[25]，之后在 [20]、[26] 中扩展为更广的设计集合。本文称其为子字展开 SWU，与前文的完全展开 FU 拓扑区分。

### F. 完整分类与先进设计映射

基于上述概念，可以建立完整的 PSMA 分类。每种架构由六个参数完整定义，分为以下几组：

1. **L4/L3/L2**：每一层的空间展开方式，从 IS、HS、OS 中选择。
2. **BG**：BG 循环展开的位置，可在 L2/L3 空间展开，也可按位串行 BS 在时间上展开。对于 BS，还用 BS-Lx 表示内部寄存器所处层级。
3. **Config**：I/O 带宽与硬件利用率的取舍配置，选择 FU 或 SWU。


![表 I：先进 PSMA 的分类映射](images/psma-taxonomy-benchmark-table-01.png)

**表 I. 先进设计映射。** 高、低分别表示高精度与最低精度；三层配置的顺序均为 L4/L3/L2。IS 表示输入与权重共享，OS 为输出共享，HS 为混合共享，No 表示不共享。BS-Lx 表示内部移位累加寄存器位于 Lx 的位串行设计；SWU/FU 分别表示子字展开/完全展开；1D、2D-A、2D-S 分别表示一维、二维非对称、二维对称伸缩。

| 架构 | 高精度 L4/L3/L2 | 低精度 L4/L3/L2 | BG | 配置 | 模式 |
| --- | --- | --- | --- | --- | --- |
| DNPU [12] | IS / OS / BG | IS / OS / IS | L2 | FU | 1D |
| BitFusion [15] | IS / OS / BG | IS / OS / OS | L2 | FU | 2D-A |
| BitBlade [16] | IS / BG / OS | IS / OS / OS | L3 | FU | 2D-A |
| Ghodrati [18] | IS / BG / OS | IS / OS / OS | L3 | FU | 2D-A |
| Stripes [27] | IS / IS / OS | IS / IS / OS | BS-L2 | FU | 1D |
| UNPU [13] | IS / OS / IS | IS / OS / IS | BS-L1 | FU | 1D |
| Loom [17] | IS / IS / OS | IS / IS / OS | BS-L2 | FU | 2D-A |
| Envision [23] | IS / IS / BG | IS / IS / No | L2 | SWU | 2D-S |
| ST [25] | IS / OS / BG | IS / OS / OS | L2 | SWU | 2D-S |

4. **Mode**：精度伸缩方式，可以是一维、二维非对称或二维对称。

这一分类体系现在能够映射文献中的各种 PSMA 技术，如表 I。由于 BG 展开只在高于最低精度时才相关，表中为每种设计分别列出高、低精度两行。

对 DNPU [12] 与 BitFusion [15]，每个 L2 单元每周期产生一个完整乘积，因此 BG 在 L2 空间展开。两者差别是 BitFusion 支持二维非对称伸缩，DNPU 仅支持一维伸缩；低精度下，BitFusion 的 L2 是输出共享，而 DNPU 的 L2 是输入/权重共享。BitBlade [16] 与 Ghodrati [18] 的 PSMA 非常相似，因此分类映射相同：移位器在 L3 由多个 L2 共享，所以 BG 在 L3 空间展开。

Stripes [27]、UNPU [13] 和 Loom [17] 均为位串行设计，BG 循环在时间而非空间上展开。Stripes 和 UNPU 支持一维伸缩，Loom 支持二维伸缩。UNPU 的 L2 为 IS，其内部移位累加寄存器位于 L1；Stripes 和 Loom 的 L2 为 OS，内部寄存器位于 L2。

最后一组 Envision [23] 与 ST [25] 均为 SWU。它们的根本区别在于低精度 L2 行为：Envision 不共享，ST 则为 OS；此外，Envision 在 L3 选择 IS，ST 选择 OS。

新的分类为在相同工作条件和工艺下，对完整设计空间开展基准比较铺平了道路，有助于理解各 PSMA 的优缺点。为此，我们构建统一且高度参数化的 MAC 阵列模板，高效实现分类体系覆盖的不同设计点。据作者所知，这是首次尝试对不同 PSMA 架构进行这种定性比较。

## IV. 统一且参数化的 PSMA 模板

表 I 中的已有实现并非唯一可能的 PSMA 实例。分类体系可以识别文献尚未覆盖的更多架构。为了快速实现并测试设计空间中的不同配置，我们开发统一且高度参数化的 PSMA 模板。它按照用户定义的参数，在设计时配置为分类体系子空间中的任意阵列架构，支持不同的 L2、L3、L4、BG 和 FU/SWU 设置。

### A. 位组配置

统一模板支持不同 BG 展开，其硬件影响在图 7 中作了简化展示。全精度时，若 BG 在 L2 空间展开，每个 L2 都产生完整乘积；移位加法树也位于每个 L2 内部，因此移位器不能像 L3 或 BS BG 展开那样，在多个 L1 结果之间摊销。

若 BG 在 L3 空间展开，移位加法树就位于 L3，由聚合后的 L1 结果共享。为保证功能正确，每个 L2 只计算数值权重相同的位组乘积，因为该 L2 的所有乘积将采用相同移位量。图 7 中，相同颜色的 BG 级运算需要相同移位量，所以组合在同一个 L2 中。

位串行设计则在时间上展开 BG。这意味着加法器和移位器解耦，移位运算按照精度跨多个周期执行。为了正确工作，BS 还需要调度计时逻辑、内部寄存器等额外电路。默认情况下，只有 BS 设计在阵列层级之间包含内部寄存器。

图 8 有助于进一步理解位串行阵列内部架构及其调度。假设 L2 为 OS，每个 L2 将 16 个 2b×2b 结果相加，产生一个 8 比特结果。


![图 7：不同位组展开位置的循环与硬件影响](images/psma-taxonomy-benchmark-figure-07.png)

**图 7.** BG 在 L2、L3 空间展开，或在 L3 时间展开（BS-L3）的比较。为简化，假定 L2、L3 均为 OS，且每个 Ln 仅含 2×2 个 Ln−1。示例工作负载为 $O=\sum_{n=0}^{15}I[n]W[n]$，输入和权重均为 4 比特、每 BG 为 2 比特，输出 12 比特。图中将 n 拆成 n2、n1，并比较循环映射、硬件结构及数据通路精度。

| BG 位置 | 时间展开 | L3 空间展开 | L2 空间展开 | 数据通路 |
| --- | --- | --- | --- | --- |
| L2 | n2 | n1 | bi、bw | L1：2b×2b；L2 移位加法：4b+6b+6b+8b；L3 加法：4 个 8b；时间累加：10b+12b |
| L3 | n2 | bi、bw | n1 | L1：2b×2b；L2 加法：4 个 4b；L3 移位加法：6b+8b+8b+10b；时间累加：10b+12b |
| BS-L3 | bi、bw | n2 | n1 | L1：2b×2b；L2 加法：4 个 4b；L3 加法：4 个 6b；时间移位累加：8/10/10/12b+12b |

调度分成两个阶段。阶段 1 中，权重保持不变，输入每周期右移 2 比特，中间结果保存于第一个内部寄存器。当输入的全部位处理完后，权重右移 2 比特，第一个寄存器的值转移至第二个寄存器，这构成阶段 2，之后再次执行阶段 1。重复这一过程，直到全部权重位处理结束；最后，阵列末端累加器（图 8 未画出）被激活，累加完整乘积。

从图 7 可见，位串行设计不再需要可配置移位器，因此能降低 L2、L3 加法树复杂度。但它也存在局限：首先，需要在算法中找到足够多的其他循环，而不是 BG 循环，在阵列上进行空间展开，才能保证较高利用率。其次，每个部分输出都需要额外调度控制与内部电路。因此，采用这种模式时，减少生成的输出数量是有利的，尤其是在阵列较低层级。基于这一认识，完整设计空间中许多低效设计可以提前剪枝。

### B. 设计空间约束

第 III 节分类中的参数组合会产生非常多的阵列配置。为了让探索可管理，同时保留有意义的区域，我们施加若干约束。下文列出约束及理由，表 II 汇总受限设计空间。

![表 II：受限设计空间](images/psma-taxonomy-benchmark-table-02.png)

**表 II.** L4 和 L3 均可选 IS、HS、OS。其余配置共有八种组合：

| 配置 | BG | 允许的 L2 |
| --- | --- | --- |
| FU | L2 | IS、HS、OS |
| FU | L3 | HS、OS |
| FU | BS-L2 | OS |
| SWU | L2 | IS（无共享子字布局）、OS |

共有 $3\ (L4)\times3\ (L3)\times8\ (\text{Config/BG/L2})=72$ 种设计。

1. **BG 在 L3 展开时，L2 不能为 IS。** L3 BG 的好处是 L2 内全部 L1 可以共享移位逻辑。如果 L2 不共享输出，移位器数量将与 BG 在 L2 展开时相同。
2. **BG 在时间展开时，使用 L2 为 OS 的 BS-L2。** BS 需要内部寄存器来支持每个输出的时间移位。


![图 8：位串行计算的共享与调度](images/psma-taxonomy-benchmark-figure-08.png)

**图 8.** BG 在时间上展开、且 L2 为 OS 时，不同精度下 L3 的输入共享、混合共享和输出共享。（a）L3 为 IS，对应 BS-L2；（b）L3 为 HS，对应 BS-L3；（c）L3 为 OS，对应 BS-L3；（d）8b×8b、8b×4b、8b×2b、4b×4b、2b×2b 模式的数据访问时间线。阶段 1 遍历输入位组，阶段 2 切换权重位组并转移中间累加值。图中保留各级数据位宽、移位与寄存器连接。

接续约束 2：本文固定 BS 内部寄存器位于 L2，以降低模板复杂度并平衡关键路径。为尽可能降低开销，最好每个 L2 单元只需要一个内部寄存单元，总计 256 个。对 BS-L3/L4 进行基准研究也很有意义，但会导致更多输入寄存器，并要求工作负载提供较大的输出共享展开规模；这留待未来研究。

3. **SWU 的 L2 只能是 OS 或不共享。** 由于 SWU 的性质，L2 必须对称，因此输出要么共享，要么不共享。

在这些约束下，表 I 已有加速器中，只有 UNPU [13] 无法被统一模板支持，因为它是 L2 为 IS 的位串行设计，即 BS-L1。

![图 9：输出累加寄存器容量计算](images/psma-taxonomy-benchmark-figure-09.png)

**图 9.** 一个输出累加寄存器容量计算及其全精度/低精度运行时配置的例子。示例 PSMA 采用 L4:IS、L3:OS、FU、BG:L2、L2:OS。

| 位置 | 8b×8b 输出与解释 | 2b×2b 输出与解释 |
| --- | --- | --- |
| L2（OS） | 16b，一个 8b×8b 结果 | 8b，16 个 2b×2b 结果求和 |
| L3（OS） | 20b，16 个 16b 结果求和 | 12b，16 个 8b 结果求和 |
| L4（IS） | 320b，16 个 20b 结果 | 192b，16 个 12b 结果 |
| 所需输出累加寄存器 | 384b，即 $16\times(20+4)$，每输出 4b 累加余量 | 256b，即 $16\times(12+4)$，每输出 4b 累加余量 |
| 实际寄存器使用 | 384b 全部使用 | 同一 384b 寄存器部分门控 |

### C. 寄存器布局

为保证统一模板评估的各设计具有正确时序，我们在阵列外围放置输入、输出寄存器。只有 BS 设计在阵列内部额外配置寄存器，用于周期性的移位相加。根据 L4/L3/L2 的 IS/OS 配置和 BG 展开方式，各设计每周期所需的最大输入/输出带宽不同，阵列接口所需寄存器数量也不同。IS/WS 通常需要较少输入寄存器，但更多输出寄存器；OS 恰好相反。

实际计算中，在 L2/L3/L4 的任一层级，每增加一个 OS 或 IS 共享维度，所需输入或输出寄存器字数就相应乘以 4，因为每个 Ln 是由 4×4 个 Ln−1 构成的二维阵列。每个输入字的位数固定，全精度时为 8 比特；每个输出字的位数则同时取决于输入精度和预期最大时间累加长度。

这里按所有精度模式的最坏情况，计算所需最大位宽，并为每个预期输出增加 4 比特的时间累加余量。图 9 给出具体 PSMA 中从 L2、L3、L4 到累加器的输出精度与最终寄存器容量计算。

最后，BS 设计需要两阶段累加寄存器。由于第 IV-B 节只保留 BS-L2，BS 寄存器布局始终相同。如图 8a，BS 的 L2 对 16 个 2×2 比特部分积求和，始终产生 8 比特结果。全精度时，内部寄存器每四次迭代清空，其保存值需要右移 2 比特共三次，因此每个 BS 累加寄存器需要 6 比特余量。

## V. 实验设置

本节分别在低频 200MHz 和高频 1GHz 条件下，评估设计空间中每种方案的每次运算能量与面积。同时分解各硬件部分，理解不同设计时配置与运行时配置的取舍。


![图 10：每次运算能量热图](images/psma-taxonomy-benchmark-figure-10.png)

**图 10.** （a）200MHz 和（b）1GHz 下每次运算的能量，单位 fJ。每组包含 8b×8b、8b×4b、4b×4b、2b×2b 四个热图。行表示九种 L4/L3 组合，列表示八种 Config/BG/L2 组合；完整数值保留于原图。非对称模式中标注“不支持”的位置属于 SWU。

### A. 方法

为保证公平，全部 72 种设计都由相同参数化 HDL 模板综合，采用相同编码风格与优化集合。在精度配置上，大多数设计同时支持对称伸缩和仅权重伸缩；例外是 SWU，只支持对称精度伸缩。输入激活与权重支持 8、4、2 比特。

所有设计采用 SystemVerilog，通过 Cadence Genus v19.11 高强度编译，分别以 200MHz 和 1GHz 为目标进行综合。工艺为 TSMC 28nm，标称电压 1V。功耗估计通过综合后仿真完成：使用 Questa Sim v10.6c 获取开关活动，使用 Cadence Genus 提取功耗。

仿真执行 4096 个周期，使用能够保证所有设计在全部精度下都充分利用硬件的理想工作负载。本节基准中的 BS 设计全部为 BS-L2，为简洁起见直接记作 BS。

### B. 工作负载

理想工作负载可以用 CNN 循环描述。在最低精度 2b×2b 下，PSMA 最多包含 4096 个功能 PE。因此，为保证所有设计在各种精度下充分利用硬件，工作负载至少应提供 4096 规模的 OS 循环与 4096 规模的 IS/WS 循环。表 III 列出各循环类别的取值，覆盖所有层均为 IS、所有层均为 OS 这两个极端。

全 IS 设计空间展开工作负载的 IS/WS 循环，OS 循环则在时间上展开，因而提供更多输出驻留；全 OS 设计恰好相反。最终，所有设计运行相同工作负载，使用随机生成的输入与权重，但采用不同数据流和映射。

![表 III：理想工作负载的最小规模](images/psma-taxonomy-benchmark-table-03.png)

**表 III.** 理想工作负载的最小规模，保证全部设计在各种精度下充分利用硬件。

| 2b×2b 下的极端配置 | IS 循环规模 | WS 循环规模 | OS 循环规模 |
| --- | --- | --- | --- |
| L4、L3、L2 均为 IS | 64 | 64 | 1 |
| L4、L3、L2 均为 OS | 1 | 1 | 4096 |
| 工作负载总体要求 | 64 | 64 | 4096 |

### C. 吞吐量

全部仿真采用经优化的工作负载，使各设计在标称精度下达到 100% 空间利用率。因此，全精度时 FU 和 SWU 的吞吐量，即每周期运算数，相同。精度降低后，FU 继续使用完整硬件；SWU 则门控部分 PSMA，以在所有精度下保持固定输入带宽。结果是 SWU 吞吐量低于 FU：在 4 比特和 2 比特时，分别只有 FU 的一半与四分之一。

### D. 能效

所有受支持设计均按每次运算所需能量比较。一次运算定义为一次完整乘法或一次加法，即一次完整 MAC 计为两次运算。200MHz、1GHz 的能效热图分别见图 10a、10b。

除 8b×8b、4b×4b、2b×2b 对称精度外，还比较一种非对称精度 8b×4b。SWU 不支持非对称伸缩。每行对应一种 L4/L3 模式，每列对应一种 Config/BG/L2 组合。

在 200MHz 下，各精度中最节能的几列是 FU/BG:L3/L2:OS、FU/BG:BS/L2:OS，以及 SWU/BG:L2/L2:OS。


![图 11：每次运算能量与面积的关系](images/psma-taxonomy-benchmark-figure-11.png)

**图 11.** （a）200MHz、（b）1GHz 下，每次运算能量（fJ）与面积（mm²）的关系；横纵轴均采用对数坐标。每组分别比较四种精度。颜色表示 Config/BG/L2 组合，形状表示 L4/L3 模式，标注 DNPU、BitFusion（BF）、BitBlade（BB）、Loom、Envision（Env）与 ST 的位置。左下角更理想。

全精度时，SWU/BG:L2/L2:OS 的设计更节能，这是因为其结构更简单，移位器无需配置，同时保持与 FU 相同的吞吐量。较低精度下，SWU 运算数量较少，因此其表现更接近 FU。直观上，FU/BG:L3 和 FU/BG:BS 应优于 FU/BG:L2，因为前两者的移位器开销可在多个 L2 之间共享。BS 内部寄存器还带来额外开关开销，而当前频率较低，因此回报不大。

良好能效的关键是令 L2 为 OS，因为无论精度如何，它只产生一个部分乘积结果，从而简化 L3 和 L4。200MHz 时，只要 L2 为 OS，L3/L4 展开方式的影响就相对有限。此时最佳 L4/L3 配置为 IS/IS、IS/OS、OS/IS。若 L3、L4 同时偏向 OS，加法树会延长关键路径，对能效产生负面影响；但低频下影响不算很大。

1GHz 的图 10b 呈现类似趋势，也有一些差别。L2:OS 仍是良好能效的关键，BG:L3、BG:BS 仍优于 BG:L2。不过，BG:BS 因关键路径较短而略优于 BG:L3：高频时，内部寄存器的优势开始显现。与 200MHz 相同，最佳 L4/L3 组合为 IS/IS、IS/OS、OS/IS。所有层都采用 OS，会因关键路径加长而难以满足严格时序要求，对能效产生明显不利影响。

本次基准中 BS 表现突出，与文献 [20] 将 BS MAC 判为最差的结果形成鲜明对比。主要原因是，本文 PSMA 模板中的全部 BS 设计都假设 L2 为 OS，内部寄存器开销可在同一 L2 的各 L1 之间摊销，也就是 BS-L2，从而降低每次运算能量。而 [20] 研究单个 MAC，忽略了这种摊销机会，即 BS-L1。这个结果强调了阵列级位串行精度可扩展 MAC 引擎中硬件资源共享的重要性。

### E. 能量与面积

为了更好判断阵列效率，图 11 在不同对称、非对称精度下，比较 200MHz、1GHz 的能量/运算与面积。每种颜色表示一组 Config/BG/L2，每种形状表示一组 L4/L3；最佳设计位于左下角。

200MHz 时，紫色和橙色方案最优，分别代表 SWU/L2:OS 与 FU/BG:L3/L2:OS。1GHz 时，青色 BS 标记具有更好能效，而紫色、橙色方案面积更小。后文将说明，相对 BS，后两类没有内部寄存器，因此面积更小。

多数情况下，蓝色 FU/BG:L2/L2:IS 位于图的右上方，是最低效的配置之一。原因在于，最低层采用 IS 会在低精度下产生许多不能相加的独立部分积。图的左上角、右下角通常空缺，说明本基准中多数设计的面积与每次运算能量具有较强相关性。


![图 12：分解图的硬件与图例](images/psma-taxonomy-benchmark-figure-12.png)

**图 12.** 图 13、14 的图例与硬件位置。输入寄存器包括输入激活和权重；分解项依次包括 L1 乘法器、L2/L3/L4 加法树、BS 专有内部寄存器、输出累加器和输出寄存器。每组九根柱从左至右为 L4/L3：IS/IS、IS/HS、IS/OS、HS/IS、HS/HS、HS/OS、OS/IS、OS/HS、OS/OS。

![图 13：200MHz 能量分解](images/psma-taxonomy-benchmark-figure-13.png)

**图 13.** 200MHz 下每次运算能量分解，单位 fJ。（a）8b×8b；（b）8b×4b；（c）4b×4b；（d）2b×2b。横轴按八类 Config/BG/L2 分组，每组九根柱对应图 12 的 L4/L3 顺序。SWU 的非对称位置不支持。

![图 14：1GHz 能量分解](images/psma-taxonomy-benchmark-figure-14.png)

**图 14.** 1GHz 下每次运算能量分解。四种精度、分组方式与颜色含义同图 13。

![图 15：面积分解](images/psma-taxonomy-benchmark-figure-15.png)

**图 15.** （a）200MHz、（b）1GHz 下各阵列配置的面积分解，单位 mm²。柱形区分组合逻辑与时序逻辑，横轴按相同的八类配置及每组九种层级共享方式排列。


### F. 组成分解

为了理解不同阵列配置表现差异的原因，图 13–15 汇总全部 PSMA 在 200MHz、1GHz 下的每次运算能量与面积分解，图 12 给出图 13、14 的图例。能量分解包括 L1 乘法器、L2/L3/L4 加法树、最终累加加法器，以及输入、输出和 BS 内部寄存器。

所有情况下，L1 的 2 比特乘法器能耗几乎相同，而且只占总能耗的小部分。若 BG 在 L2 展开，L2 加法树同时包含加法器与移位器，因此能耗更高。SWU 的移位器不可配置，所以其 L2 加法树能量略低于 FU。将 BG 移到 L3 或采用 BS 后，L2 加法树复杂度明显降低。

若 L3 或 L4 为 IS，意味着前一级部分结果彼此不相加，该层就不存在相应加法树。此外，如果 L2 为 IS，精度降低时，L3 加法树（若存在）会消耗更多能量。

当更多层采用 IS 时，输出寄存器和累加加法器逐渐主导能耗，而输入寄存器能量较低；反向改变共享方式时，情况也反过来。内部寄存器只存在于 BS，且占其总能耗的相当部分。

1GHz 时，多数层为 OS 的阵列比其他设计消耗更多能量。原因是部分结果必须经过多层复杂加法树，关键路径较长；为了满足时序约束，能耗会受到不利影响。

图 15 显示面积主要由组合逻辑占据，时序逻辑只占较小比例。此外，L2 为 OS 的设计面积均小于对应的 L2:IS 或 L2:HS 设计。这是因为 L2 产生更多部分积时，L3/L4 加法树需要更复杂的结构，以保证不同的 L2 部分结果不被错误地相加。

总体上，除 BS 外，1GHz 设计面积更大，可归因于为满足架构时序约束而使用更大标准单元。

## VI. 结论

本文提出新的循环表示，在传统七重 CNN 循环上增加支持精度伸缩的 BG 循环。同时，将原有循环分为 IS、WS、OS 三类。这为新的分类体系提供基础：利用循环类别带来的灵活性，BG 可以在 MAC 阵列不同层级空间展开，也可以时间展开，即位串行。随后，我们将已有先进 PSMA 映射到该分类体系，并详细讨论各分类参数对底层硬件及 PSMA 输入、输出带宽的影响。

本文还提出统一且高度参数化的 PSMA 模板，覆盖分类体系中的大量设计点，支持输入与权重的对称、非对称精度伸缩。

受限设计空间中的全部架构均使用 TSMC 28nm 工艺综合，并从能效与面积方面进行广泛基准测试。结果显示，BG 在 L2 展开通常最不理想，例如 BitFusion [15]、DNPU [12]；较低频率时更适合在 L3 展开，如 BitBlade [16]、Ghodrati [18]，较高频率时更适合时间展开，如 Loom [17]、Stripes [27]。

BG 在 L3 或时间上展开的好处，是移位器开销可以在不同 L2 之间摊销，从而降低加法树复杂度。总体上，在不同层级混合使用 IS/WS 与 OS 循环是较好的选择。FU 更适合低精度常见的工作负载；SWU 更适合高精度占主导的工作负载。

本基准显示，L2:OS 是高能效 PSMA 的关键。200MHz 时，BG:L3 略优于 BG:BS，因为 BS 内部寄存器在低频下增加开销，却没有额外收益。1GHz 时，BS 的能效略优于 BG:L3，而 BG:L3 面积更好。两种频率下，最佳 L4/L3 组合均为 IS/IS（类似 Loom [17]）、IS/OS（类似 BitBlade [16]）和 OS/IS。跨 200MHz、1GHz 表现最好的 SWU 配置为 L4:IS、L3:OS、L2:OS，即类似 ST 的架构 [25]。

需要强调，本研究根据理想工作负载选择最佳 PSMA，该负载保证所有设计在全部精度下充分利用硬件。但不同工作负载会改变被测设计的利用率，进而形成不同的性能格局。评估不同非理想工作负载对各 PSMA 的影响，留待未来工作。

深度学习模型已经显示出对低精度的容忍能力，推动了 DNN 加速器的 PSMA 研究。本文帮助更好地分类现有先进 PSMA，清晰理解设计内部和设计之间的取舍，并为不同条件下寻找最合适的 PSMA 提供依据。

## 致谢

作者感谢 Vincent Camus 对自动化基准测试流程提供的见解。


## References

[1] V. K. Chippa, S. T. Chakradhar, K. Roy, and A. Raghunathan, “Analysis
and characterization of inherent application resilience for approximate
computing,” in Proceedings of the 50th Annual Design Automation
Conference, 2013, pp. 1–9.
[2] G. Abich, J. Gava, R. Garibotti, R. Reis, and L. Ost, “Applying
lightweight soft error mitigation techniques to embedded mixed prec-
ision deep neural networks,” IEEE Transactions on Circuits and Systems
I: Regular Papers, vol. 68, no. 11, pp. 4772–4782, 2021.

[3] B. Moons, B. De Brabandere, L. Van Gool, and M. Verhelst, “Energy-
efﬁcient convnets through approximate computing,” in 2016 IEEE
Winter Conference on Applications of Computer Vision (WACV), 2016,
pp. 1–8.
[4] I. Hubara, M. Courbariaux, D. Soudry, R. El-Yaniv, and Y. Ben-
gio, “Quantized neural networks: Training neural networks with low
precision weights and activations,” The Journal of Machine Learning
Research, vol. 18, no. 1, pp. 6869–6898, 2017.
[5] L. Cavigelli and L. Benini, “Origami: A 803-gop/s/w convolutional
network accelerator,” IEEE Transactions on Circuits and Systems for
Video Technology, vol. 27, no. 11, pp. 2461–2475, 2016.
[6] P. Judd, J. Albericio, T. Hetherington, T. Aamodt, N. E. Jerger, R. Ur-
tasun, and A. Moshovos, “Proteus: Exploiting precision variability in
deep neural networks,” Parallel Computing, vol. 73, pp. 40–51, 2018.
[7] M. Courbariaux, Y. Bengio, and J.-P. David, “Training deep neu-
ral networks with low precision multiplications,” arXiv preprint
arXiv:1412.7024, 2014.
[8] S. Gupta, A. Agrawal, K. Gopalakrishnan, and P. Narayanan, “Deep
learning with limited numerical precision,” in International conference
on machine learning.
PMLR, 2015, pp. 1737–1746.
[9] R. Banner, I. Hubara, E. Hoffer, and D. Soudry, “Scalable methods
for 8-bit training of neural networks,” arXiv preprint arXiv:1805.11046,
2018.
[10] M. Courbariaux, Y. Bengio, and J.-P. David, “Binaryconnect: Training
deep neural networks with binary weights during propagations,” arXiv
preprint arXiv:1511.00363, 2015.
[11] Y. Zhang, R. Zhao, W. Hua, N. Xu, G. E. Suh, and Z. Zhang, “Precision
gating: Improving neural network efﬁciency with dynamic dual-precision
activations,” in International Conference on Learning Representations,
2020.
[12] D. Shin, J. Lee, J. Lee, and H.-J. Yoo, “14.2 dnpu: An 8.1 tops/w recon-
ﬁgurable cnn-rnn processor for general-purpose deep neural networks,”
in 2017 IEEE International Solid-State Circuits Conference (ISSCC).
IEEE, 2017, pp. 240–241.
[13] J. Lee, C. Kim, S. Kang, D. Shin, S. Kim, and H.-J. Yoo, “Unpu:
An energy-efﬁcient deep neural network accelerator with fully variable
weight bit precision,” IEEE Journal of Solid-State Circuits, vol. 54,
no. 1, pp. 173–185, 2018.
[14] E. Lee, T. Han, D. Seo, G. Shin, J. Kim, S. Kim, S. Jeong, J. Rhe,
J. Park, J. H. Ko, and Y. Lee, “A charge-domain scalable-weight in-
memory computing macro with dual-sram architecture for precision-
scalable dnn accelerators,” IEEE Transactions on Circuits and Systems
I: Regular Papers, vol. 68, no. 8, pp. 3305–3316, 2021.
[15] H. Sharma, J. Park, N. Suda, L. Lai, B. Chau, V. Chandra, and H. Es-
maeilzadeh, “Bit fusion: Bit-level dynamically composable architecture
for accelerating deep neural network,” in 2018 ACM/IEEE 45th Annual
International Symposium on Computer Architecture (ISCA).
IEEE,
2018, pp. 764–775.
[16] S. Ryu, H. Kim, W. Yi, and J.-J. Kim, “Bitblade: Area and energy-
efﬁcient precision-scalable neural network accelerator with bitwise
summation,” in Proceedings of the 56th Annual Design Automation
Conference 2019, 2019, pp. 1–6.
[17] S. Sharify, A. D. Lascorz, K. Siu, P. Judd, and A. Moshovos, “Loom:
Exploiting weight and activation precisions to accelerate convolutional
neural networks,” in 2018 55th ACM/ESDA/IEEE Design Automation
Conference (DAC).
IEEE, 2018, pp. 1–6.
[18] S. Ghodrati, H. Sharma, C. Young, N. S. Kim, and H. Esmaeilzadeh,
“Bit-parallel vector composability for neural acceleration,” in 2020 57th
ACM/IEEE Design Automation Conference (DAC).
IEEE, 2020, pp.
1–6.
[19] Q. Yang and H. Li, “Bitsystolic: A 26.7 tops/w 2b 8b npu with
conﬁgurable data ﬂows for edge devices,” IEEE Transactions on Circuits
and Systems I: Regular Papers, vol. 68, no. 3, pp. 1134–1145, 2021.
[20] V. Camus, L. Mei, C. Enz, and M. Verhelst, “Review and bench-
marking of precision-scalable multiply-accumulate unit architectures for
embedded neural-network processing,” IEEE Journal on Emerging and
Selected Topics in Circuits and Systems, vol. 9, no. 4, pp. 697–711,
2019.
[21] Z. Shun, O. A. Pfander, H.-J. Pﬂeiderer, and A. Bermak, “A vlsi arch-
itecture for a run-time multi-precision reconﬁgurable booth multiplier,”
in 2007 14th IEEE International Conference on Electronics, Circuits
and Systems.
IEEE, 2007, pp. 975–978.
[22] L. Mei, P. Houshmand, V. Jain, S. Giraldo, and M. Verhelst, “ZigZag:
Enlarging joint architecture-mapping design space exploration for dnn
accelerators,” IEEE Transactions on Computers, vol. 70, no. 8, pp. 1160–
1174, 2021.
[23] B. Moons, R. Uytterhoeven, W. Dehaene, and M. Verhelst, “14.5 en-
vision: A 0.26-to-10tops/w subword-parallel dynamic-voltage-accuracy-
frequency-scalable convolutional neural network processor in 28nm
fdsoi,” in 2017 IEEE International Solid-State Circuits Conference
(ISSCC).
IEEE, 2017, pp. 246–247.
[24] B. Moons, R. Uytterhoeven, W. Dehaene, and M. Verhelst, “Dvafs:
Trading computational accuracy for energy through dynamic-voltage-
accuracy-frequency-scaling,” in Design, Automation & Test in Europe
Conference & Exhibition (DATE), 2017.
IEEE, 2017, pp. 488–493.
[25] L. Mei, M. Dandekar, D. Rodopoulos, J. Constantin, P. Debacker,
R. Lauwereins, and M. Verhelst, “Sub-word parallel precision-scalable
mac engines for efﬁcient embedded dnn inference,” in 2019 IEEE
International Conference on Artiﬁcial Intelligence Circuits and Systems
(AICAS).
IEEE, 2019, pp. 6–10.
[26] V. Camus, C. Enz, and M. Verhelst, “Survey of precision-scalable
multiply-accumulate units for neural-network processing,” in 2019 IEEE
International Conference on Artiﬁcial Intelligence Circuits and Systems
(AICAS).
IEEE, 2019, pp. 57–61.
[27] P. Judd, J. Albericio, T. Hetherington, T. M. Aamodt, and A. Moshovos,
“Stripes: Bit-serial deep neural network computing,” in 2016 49th
Annual IEEE/ACM International Symposium on Microarchitecture (MI-
CRO).
IEEE, 2016, pp. 1–12.




## 作者简介

以下经历与“目前”等时间表述均对应原文撰写时。

![Ehab M. Ibrahim 的原文作者照片](images/psma-taxonomy-benchmark-author-01.png)

**Ehab M. Ibrahim** 于 2012 年获埃及扎加齐格大学电子与通信工程学士学位，2019 年获埃及日本科技大学（E-JUST）电子与通信工程硕士学位。2019 年春，他在日本东京国立情报学研究所实习。原文撰写时任比利时 Magics Technologies 数字集成电路设计工程师。本研究完成于他 2020–2021 年在鲁汶大学 MICAS 担任研究助理期间。其研究兴趣包括高能效深度学习加速、软硬件协同设计和精度可扩展计算。

![Linyan Mei 的原文作者照片](images/psma-taxonomy-benchmark-author-02.png)

**Linyan Mei** 于 2016 年获中国北京理工大学电子科学与技术学士学位，2018 年获比利时鲁汶大学电气工程硕士学位。原文撰写时在鲁汶大学 MICAS 实验室攻读博士学位，研究嵌入式机器学习处理器的数字设计。她于 2017–2018 年在比利时鲁汶 IMEC 实习，2021 年冬在美国门洛帕克 Facebook 实习。其研究兴趣包括高能效 DNN 加速器的设计空间探索、算法—硬件—映射协同优化，以及精度可扩展计算。

![Marian Verhelst 的原文作者照片](images/psma-taxonomy-benchmark-author-03.png)

**Marian Verhelst** 于 2008 年获鲁汶大学博士学位。2005 年夏任加州大学伯克利分校 Berkeley Wireless Research Center 访问学者，2008–2011 年任美国俄勒冈州希尔斯伯勒 Intel Labs 研究科学家。原文撰写时为鲁汶大学电气工程系 MICAS 实验室正教授。研究方向包括嵌入式机器学习、硬件加速器、自适应电路与系统、传感器融合，以及低功耗边缘处理。

她是 DATE 与 ISSCC 执行委员会成员，担任 AICAS 2020 和 tinyML 2020 技术程序委员会联合主席，并为 DATE、ESSCIRC 技术程序委员会成员。她是 SSCS 杰出讲师，曾是比利时青年科学院成员、佛兰德政府 STEM 顾问委员会成员，并担任 TVLSI、TCAS-II、JSSC 副编辑。原文撰写时，她持有欧盟 ERC Starting Grant，并于 2016 年获比利时皇家科学院表彰。

[原文与版本信息](https://arxiv.org/abs/2108.04773v2)。技术总结、通俗讲解与图表说明属于阅读解读，可结合全文译文对照原文。
