# AMPLE: Event-Driven Accelerator for Mixed-Precision Inference of Graph Neural Networks 原文翻译

# AMPLE：面向图神经网络混合精度推理的事件驱动加速器

PEDRO GIMENES, Imperial College London, United Kingdom AARON ZHAO, Imperial College London, United Kingdom GEORGE CONSTANTINIDES, Imperial College London, United Kingdom

图神经网络（Graph Neural Networks，GNNs）近年来因其在非欧几里得数据上的出色表现而备受关注。由于图的稀疏结构导致的不规则内存访问模式，定制硬件架构对 GNNs 尤为有利。然而，现有的 FPGA 加速器受限于其双重缓冲（double buffering）机制，该机制未能考虑典型图数据集中节点分布的不规则性。为解决这一问题，我们提出了 AMPLE（Accelerated Message Passing Logic Engine），一种采用全新事件驱动编程流的 FPGA 加速器。我们设计了混合算术架构，使 GNN 推理能够在节点级粒度上进行量化。最后，我们实现了针对数据与指令的预取器，以优化片外内存访问并最大化节点并行度。在规模从 2K 到 700K 节点的引文与社交媒体图数据集上的评估表明，相较于 CPU 和 GPU 对应方案，平均加速比分别达到 243× 和 7.2×。

CCS Concepts: • Hardware → Hardware accelerators; Reconfigurable logic applications.

Additional Key Words and Phrases: FPGA, Graph Neural Networks, Graph Convolutional Networks, Neural Network Quantization, Mixed-Precision Neural Networks, Graph Processing, Network-on-Chip, Hardware Architecture

## ACM Reference Format:

Pedro Gimenes, Aaron Zhao, and George Constantinides. 2025. AMPLE: Event-Driven Accelerator for Mixed-Precision Inference of Graph Neural Networks. In The 5th Workshop on Machine Learning and Systems (EuroMLSys ’25), March 30-April 3, 2025, Rotterdam, Netherlands. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3721146.3721963

# 1 引言

图（Graph）是一种强大的表示方法，用于捕获实体之间的关系，实体以节点的形式表示，并通过边相互连接。这种结构使得我们能够对各种复杂系统进行建模，包括社交网络 [Alamsyah et al. 2021]、生物交互 [Wu et al. 2021] 以及推荐系统 [Wang et al. 2021]。图神经网络（Graph Neural Networks, GNNs）已成为处理图数据的变革性方法，旨在通过利用图内部的相互连接来学习复杂的关系信息 [Kipf and Welling 2016; Veličković et al. 2017]。

GNN 模型的推理可以分为两个主要的计算阶段：（1）聚合（Aggregation）和（2）变换（Transformation）[Gilmer et al. 2017]。在聚合阶段，对节点邻居的特征嵌入应用置换不变函数（例如求和或求平均）。该阶段的结果随后被用于变换阶段，变换阶段由一个全连接层构成，用于为每个节点生成更新后的特征嵌入。变换阶段呈现出高度规则的计算模式，可以在诸如 GPU 之类的并行设备上高效加速；而聚合阶段由于典型图数据的随机性和稀疏性，涉及大量不规则的内存访问。此外，聚合延迟是节点度数的函数，而节点度数遵循高度非均匀的分布。因此，一个设计高效的 GNN 加速器需要缓解聚合阶段的计算不规则性，同时充分利用变换阶段的规律性 [Yan et al. 2020]。

尽管 CPU 内存系统是一项成熟且高度优化的技术，但图数据的稀疏结构使得传统缓存系统的效果大打折扣，因为节点聚合会产生大量对非连续内存范围的访问。GPU 上的推理由于深层次的并行性而提供更高的性能，然而，这些设备受到高延迟内存管理机制的限制。此外，GPU 不支持阶段间的流水线化，这意味着聚合结果必须先存储到片外存储器中，然后才能被重新取回用于变换阶段。最后，现代设备对低精度数值格式计算的支持有限。

这些考虑促成了若干 GNN 加速器的设计。HyGCN 利用一组单指令多数据（Single Instruction Multiple Data, SIMD）核心进行聚合，并使用一个脉动阵列（systolic array）进行节点变换 [Yan et al. 2020]。与此同时，GenGNN 被提出作为一个模型无关的 GNN 加速框架，通过高层次综合（High-Level Synthesis）工具来解决 GNN 模型的发展速度与定制加速器之间的差距 [Abi-Karam et al. 2022]。表 1 总结了现有 GNN 硬件平台的特性。尽管先前提出的 GNN 加速器具有诸多优势，仍然存在以下不足： HyGCN 中部署的双缓冲机制并不适合图计算，因为节点度数的分布是非均匀的。在这种范式下，低度数节点必须等待高度数节点才能继续计算，从而导致大量的流水线空隙。这凸显了对事件驱动编程流的需求，即节点被独立地分配资源并被调度到加速器上。 这两种加速器均不提供对模型量化的硬件支持。正如 Tailor 等人 [Tailor et al. 2020] 所观察到的，GNN 中量化的精度损失主要发生在聚合阶段，且与节点的度数直接相关。因此，将低度数节点转换为低精度格式，同时将高度数节点保持在高精度，可以在对模型精度影响较小的情况下降低内存开销和资源占用。 现有加速器需要对整个输入图的节点嵌入进行片上缓冲。因此，这些加速器在大型图（> 100k 节点）上的推理适用性有限，因为此时嵌入无法被实际地存储在片上，这凸显了对一种以节点为中心的预取系统的需求，以便在加速器繁忙时隐藏内存访问延迟。

表 1. 各硬件平台图处理特性总结。尽管 CPU 可以进行并行化，但多线程 CPU 与并行化加速器相比核心数量有限。GPU 上可以实现事件驱动编程，但其计算并不遵循以节点为中心的流程。尽管 GenGNN 可以进行预取，但所有处于处理中的节点的传入消息都需要存储在片上。最后，没有现有的加速器支持任意的多精度计算。

<table><tr><td>硬件平台</td><td>并行化</td><td>事件驱动编程</td><td>节点预取</td><td>多精度</td></tr><tr><td>CPU (Intel Xeon)</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>GPU (RTX A6000)</td><td>√</td><td>√</td><td>X</td><td>X</td></tr><tr><td>HyGCN [Yan et al. 2020]</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>GenGNN [Abi-Karam et al. 2022]</td><td>√</td><td>X</td><td>√</td><td>X</td></tr><tr><td>AMPLE</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

我们通过引入一种新颖的 GNN 加速器 AMPLE 来解决这些不足，并做出以下贡献：

• 我们展示了一种用于 GNN 加速的事件驱动编程模型，使主机能够通过内存映射寄存器异步地对节点进行编程。

• 我们提出了一种架构，其特点是包含一个由多精度聚合核心组成的异构池，这些核心通过网络-on-chip（Network-on-Chip）互连，并根据节点的度数和精度动态地分配给节点。

• 我们在规模从 2K 到 700K 节点不等的大规模图数据集上评估了 AMPLE，与 CPU 和 GPU 基线相比，分别实现了平均 243× 和 7.2× 的加速。

本文正文结构如下。第 2 节介绍 GNN 和神经网络量化的背景。第 3 节解释 AMPLE 加速器的架构，包括每个高层特性如何在电路层面实现。最后，第 4 节说明测试方法以及相对于 CPU/GPU 基线的实验结果。

# 2 背景

## 2.1 图的表示

一个图 $G \ : = \ : ( \mathcal { V } , \mathcal { E } )$ 是节点/顶点集合 V 与边集合 $\varepsilon .$ 第 $l$ 层的特征表示集合记为矩阵 $X ^ { ( l ) } \in \mathcal { R } ^ { N \times D }$，其中 $N = | \mathcal { V } |$ 是节点数量，$D$ 是特征维度。若元素 $e _ { i , j } = ( v _ { i } , v _ { j } )$ 存在于 E 中，则表示节点 $v_i$ 与 $v _ { j }$ 之间存在连接，即节点 $v _ { j }$ 包含在 $i$ 的邻居集合 $N _ { i : }$ 中，且 $v _ { i }$ 包含在 $N _ { j } .$ 中。在无向图中，边元素 $e _ { i , j }$ 对应于 $e _ { j , i } .$。图中的连接可以使用一个 $N \times N$ 邻接矩阵来表示，其中每个元素 $A _ { i , j }$ 表示节点 $i$ 与 $j$ 之间的一条边。

## 2.2 图神经网络（GNN）

在 GNN 中，图数据经过多层变换以对整个图或单个节点/边执行分类和/或回归任务。GNN 可以通过消息传递机制（Message Passing Mechanism）[Gilmer et al. 2017] 来表示，该机制将节点更新规则推广如下。

$$
\mathbf { x } _ { i } ^ { l + 1 } = \gamma ( \mathbf { x } _ { i } ^ { l } , \mathcal { \mathcal { A } } _ { j \in N ( i ) } ( \phi ( \mathbf { x } _ { i } ^ { l } , \mathbf { x } _ { j } , e _ { i , j } ^ { l } ) ) )\tag{1}
$$

可以看出，在一般情形下，每个节点聚合以任意函数 $\phi$ 表示的传入消息，当 $\phi = \mathbf { x } _ { j } ^ { l }$ 时，这等价于聚合邻居 Embedding。消息通过在节点 $i$ 的邻域上施加的任意置换不变聚合函数 $\mathscr { A } _ { j \in N ( i ) }$ 进行聚合，并通过任意变换函数 $\gamma ( \mathbf { x } _ { i } ^ { l } , \mathbf { m } _ { i } ^ { l } )$ 进行变换，其中 $\mathbf { m } _ { i } ^ { l }$ 是聚合的结果（即 $\mathbf { m } _ { i } = \mathcal { A } _ { j \in N ( i ) } \phi ( \mathbf { x } _ { i } ^ { l } , \mathbf { x } _ { j } ^ { l } , e _ { i , j } ^ { l } ) )$）

2.2.1 图卷积网络（GCN）。GCN 是作为计算机视觉领域中卷积神经网络（Convolutional Neural Networks）的类比方案而出现的 [Kipf and Welling 2016]。单个 GCN 层的逐元素节点更新规则如公式 2 所示。

$$
\mathbf { x } _ { i } ^ { l + 1 } = W \left( \sum _ { j \in N _ { i } \cup \{ i \} } \frac { e _ { j , i } } { \sqrt { \hat { d } _ { j } \hat { d } _ { i } } } \mathbf { x } _ { j } ^ { l } \right)\tag{2}
$$

归一化因子由 $\begin{array} { r } { \hat { d } _ { i } = 1 + \sum _ { j \in N ( i ) } e _ { j , i } . } \end{array}$ 给出。可以看出，A 被取为求和 $\begin{array} { r } { \mathcal { A } = \sum _ { j \in N _ { i } } \phi ( \mathbf { x } _ { j } , e _ { i , j } ) } \end{array}$，且 $\gamma ( \mathbf { x } _ { i } , \mathbf { m } _ { i } ) = W \mathbf { m } _ { i }$

2.2.2 图同构网络（GIN）。GIN 被提出作为一种模型，能够可证明地为两个可通过 Weisfeiler-Lehman 测试判定为非同构的图生成不同的特征更新 [Leman 2018]，从而最大化其表示能力 [Xu et al. 2018]。其更新规则由下式给出，其中 $\epsilon$ 是用于数值稳定性的一个小标量。

$$
\mathbf { x } _ { i } ^ { l + 1 } = M L P \left( \left( 1 + \epsilon \right) \cdot \mathbf { x } _ { i } ^ { l } + \sum _ { j \in N ( i ) } \mathbf { x } _ { j } ^ { l } \right)\tag{3}
$$

这里使用了与 GCN 相同的聚合 $\mathcal { A }$。与 GCN 不同的是，GIN 在聚合中不使用归一化因子（即 $\boldsymbol { \phi } = \mathbf { x } _ { j } )$，并且在聚合之后加入了残差连接，这等价于在图的邻接矩阵中添加一个自连接。

表 2. 运行时 Node Scoreboard 的示例状态，其中各 nodeslot 处于不同状态。邻接表指针和更新后特征指针分别指示从哪个内存地址获取邻居节点 ID 列表以及将更新后的特征 Embedding 写入何处。精度字段决定了运行时分配哪些算术核心。
<table><tr><td>槽位</td><td>节点 ID</td><td>精度</td><td>状态</td><td>邻居数</td><td>邻接表指针</td><td>更新后特征指针</td></tr><tr><td>0</td><td>267</td><td>float</td><td>变换（Transformation）</td><td>32</td><td>0x3BC90188</td><td>0x4FE8B774</td></tr><tr><td>1</td><td>268</td><td>float</td><td>聚合（Aggregation）</td><td>8</td><td>0xCAF5C03F</td><td>0xE672109F</td></tr><tr><td>…</td><td>.</td><td></td><td></td><td>…</td><td></td><td></td></tr><tr><td>63</td><td>330</td><td>int4</td><td>预取（Prefetch）</td><td>1</td><td>0x78E26A27</td><td>0xA4D89ED9</td></tr></table>

2.2.3 GraphSAGE。GraphSAGE 被提出作为一种归纳式框架，能够为未见过的节点和/或子图生成具有高表示能力的特征 Embedding [Hamilton et al. 2017]。

$$
\mathbf { x } _ { i } ^ { l + 1 } = W _ { 1 } \mathbf { x } _ { i } + W _ { 2 } \cdot \left( \underset { j \in N ( i ) } { \operatorname* { m e a n } } \sigma ( W _ { 3 } \mathbf { x } _ { j } ^ { l } + \mathbf { b } ) \right)\tag{4}
$$

可以看出，消息传递函数 $\phi$ 被取为对邻居 Embedding $\mathbf { x } _ { j }$ 施加激活函数 $\sigma$ 的全连接层，$\mathcal { A }$ 被取为均值，而变换为 $\gamma ( \mathbf { x } _ { i } , \mathbf { m } _ { i } ) = W _ { 1 } \mathbf { x } _ { i } + W _ { 2 } \mathbf { m } _ { i }$，其中 $W _ { 1 } , W _ { 2 }$ 为线性投影矩阵。由 $W _ { 1 }$ 参数化的投影可以看作是一种缩放残差连接。

## 2.3 神经网络量化

量化作为一种降低神经网络模型复杂度与计算延迟的方法，已被广泛研究。量化感知训练（Quantization-Aware Training, QAT）通过在前向传播中对激活值进行量化，从而在低精度表示下最小化精度损失，并在反向传播中利用直通估计器（Straight-Through Estimator, STE）来估计不可微分的量化梯度。通常，激活值按照公式 5 进行量化，其中 $q _ { m i n } , q _ { m a x }$ 构成所选的可表示值范围，$s$ 是用于将 $x$ 置于所需范围内的缩放因子，$z$ 是零点（量化空间中数值 0 所对应的浮点等价值），方括号表示取整操作。

$$
x _ { q } = \operatorname* { m i n } ( q _ { m a x } , \operatorname* { m a x } ( q _ { m i n } , \left\lfloor \frac { x } { s } + z \right\rceil ) )\tag{5}
$$

其中的 min 与 max 函数表明，任何超出指定范围的值都将取该范围的定点值极限。随后，激活值可通过 $\begin{array} { r } { \hat { x } = ( x _ { q } - z ) s , } \end{array}$ 进行反量化，其中 $\hat{x}$ 是原始浮点值的近似。

2.3.1 面向图神经网络的量化感知训练。由 Tailor 等人提出的 Degree-Quant 是最早将量化感知训练应用于图神经网络的方法之一 [Tailor et al. 2020]。首先，Tailor 等人指出，GNN 推理中的聚合阶段是量化误差的主要来源。这种效应在入度较高的节点上表现得更为严重，这可以直观地理解为：聚合结果的绝对幅度会随着邻居数量的增加而增大。高度数节点聚合值的增长会影响 $q_{max}$ 与 $q _ { m i n }$ 的取值，由于聚合结果分布中存在这些离群值，量化分辨率随之降低。

Degree-Quant 的作者通过在每个层级遵循伯努利分布随机地施加保护掩码来解决量化误差问题 [Tailor et al. 2020]。受保护的节点以浮点形式运算，而未受保护的节点则以定点形式运算。一个节点受到保护的概率是其度数的函数，并在一个可参数化的范围 $[ p _ { m i n } , p _ { m a x } ]$ 内插值得到，其中具有最小/最大邻居数的图节点被赋予边界概率。

![](images/84463d803a53ab0b6a76fd7aad3315d03156097b84731932812d5a48957f0e94.jpg)  
Fig. 1. AMPLE 顶层架构图。数据包在 Aggregation Engine 的片上网络（Network-on-Chip，以绿色显示）中沿维度序路由（dimension-order routing）传播，并被对角地馈入 Transformation Engine 的脉动阵列（以红色显示）。虚线表示控制流接口，实线表示单元之间的数据流。节点 Embedding 通过 HBM 获取，而指令则存储在 DRAM 中。

## 3 架构

如图 1 所示，AMPLE 由以下功能单元组成，各自承担相应的功能。

• Node Instruction Decoder（节点指令解码器，NID）：与主机设备通信，并驱动其他功能单元，将工作调度到加速器上。

• Prefetcher（预取器）：将层权重与邻居特征 Embedding 读取并存储到本地存储器中（分别对应 Weight Bank 与 Feature Bank）。

• Aggregation Engine（聚合引擎，AGE）：通过片上网络架构，对一个节点的所有邻居执行置换不变（permutation-invariant）聚合函数。

• Aggregation Buffer（聚合缓冲区，ABF）：包含由 AGE 生成的聚合特征 Embedding 的存储单元。

• Feature Transformation Engine（特征变换引擎，FTE）：通过将 Weight Bank 中的权重与 Aggregation Buffer 中的聚合结果进行矩阵乘法运算，为每个节点计算更新后的特征 Embedding。

## 3.1 通过 Node Instruction Decoder 实现事件驱动编程

AMPLE 与主机之间的通信由 Node Instruction Decoder（NID）处理，它是一个由可配置数量的 nodeslot 组成的内存映射寄存器组。如表 2 所示，每个 nodeslot 包含执行一个节点的聚合与变换步骤所需的信息，并维护一个指示各节点状态的状态机。算法 1 展示了主机如何卸载工作，其与加速器并发运行。首先，NID 被写入若干全局与层级参数，包括节点/特征数量以及聚合函数。随后，主机对 nodeslot 进行编程，并更新掩码 $\mathrm{available\_nodeslots} \in \{0,1\}^{n}$ 中的值，其中 $n$ 为 nodeslot 的数量。当某个节点正在被编程时，加速器会对先前已编程的节点执行聚合与变换。当计算完成后，加速器会独立地撤销 $\mathrm{available\_nodeslots}$ 掩码的置位。因此，该加速器支持一种节点级、事件驱动的计算范式。注意，<sup>′</sup>1 与 <sup>′</sup>0 分别表示全 1 与全 0 掩码。

Algorithm 1 主机编程伪代码   
Require: 全局参数 P、层 L、节点 V   
nid\_register\_bank.global\_parameters ← P   
available\_nodeslots ← ’1   
for layer in L do   
nid\_register\_bank.layer\_config ← layer   
layer.prefetch\_layer\_weights()   
while V ≠ ∅ do   
if available\_nodeslots != ’0 then   
chosen\_nodeslot ← choose(available\_nodeslots)   
chosen\_nodeslot.programming ← V.pop\_head()   
available\_nodeslots [chosen\_nodeslot] ← 0   
end if   
end while   
end for

在某个 nodeslot 完成编程后，NID 随即按照表 3 所述驱动 Prefetcher、AGE 与 FTE 执行计算，并在每个功能步骤完成后更新该节点的内部状态机。此后无需主机进一步干预，并在第 7 步之后发送中断，以指示该 nodeslot 可以被复用。需要指出的是，节点在 nodeslot 中被编程的顺序并不意味着任何优先级或时间上的相关性。典型的图数据集在逐节点的执行时间上往往表现出较高的方差，具体取决于邻居数量与数值精度。每当一个 nodeslot 完成其计算时，主机即可立即用下一个节点对其进行重新编程。这种事件驱动的控制流要求主机与加速器并发运行，以监控其状态并在资源可用时驱动后续工作。在 NID 内部，并发运行的节点通过轮询仲裁（round-robin arbitration）获得 Aggregation Engine 与 Transformation Engine 中共享资源的访问权。

Table 3. nodeslot 编程完成后，NID 按如下顺序驱动加速器中的其他单元，以执行聚合与变换。
<table><tr><td></td><td>来自</td><td>发往</td><td>数据流请求</td></tr><tr><td>1</td><td>NID</td><td>Prefetcher</td><td>读取节点的邻接表及邻居特征 Embedding</td></tr><tr><td>2</td><td>NID</td><td>AGE</td><td>对邻居特征进行聚合</td></tr><tr><td>3</td><td>AGE</td><td>ABF</td><td>将聚合结果存入缓冲区</td></tr><tr><td>4</td><td>NID</td><td>FTE</td><td>将聚合结果与层权重相乘，计算更新后的特征</td></tr><tr><td>5</td><td>FTE</td><td>Prefetcher</td><td>请求存储在 Weight Bank 中的层权重。</td></tr><tr><td>6</td><td>FTE</td><td>DRAM</td><td>变换完成后，将更新后的特征存入片外存储器。</td></tr><tr><td>7</td><td>FTE</td><td>NID</td><td>返回信号，表明变换已完成，触发该 nodeslot 释放</td></tr></table>

## 3.2 混合精度算术

AGE 和 FTE 内的计算单元在局部是同构的，即每个处理单元仅支持单一数值精度。在 Aggregation Engine 内部，这些计算单元以片上网络结构组织，构成一个由处理单元组成的异构网格，其中分配给每种精度的 PE 比例可以在编译时根据应用需求进行配置。每个 PE 耦合到一个路由器，负责在网络中传输数据包。每个数据包由携带路由负载的头微片、携带任意数量数据的数据体微片以及尾微片组成。由于不同精度的 PE 之间没有通信需求，它们被放置在相互隔离的子网络中，如图 2 所示，这有助于降低数据包拥塞。

如第 1 节所述，通过双缓冲机制实现的静态流水线在处理节点度方差较大的图时会导致流水线空隙，因为低度节点必须等待高度节点释放资源。AGE 通过在每种聚合子网络内根据节点的特征数量和精度动态分配处理单元来缓解这一问题。因此，nodeslot 的资源分配独立于任何其他正在进行的负载，并且这些资源在任务完成后可以立即被复用，从而形成一种事件驱动的编程模型。

## 3.3 大图处理

大图上的推理由 Prefetcher 中的 Feature Bank 支持，后者为 NID 中的每个 nodeslot 包含一个名为 "Fetch Tag" 的存储单元。每组包含可参数化数量的 Fetch Tags，并与 Alveo U280 卡上的每个 HBM bank 相耦合，这意味着最多可以有 32 个 Fetch Tags 并发访问内存资源，从而缓解稀疏图数据固有的内存受限问题。在每个 Fetch Tag 组内，对 HBM bank 的访问通过轮询仲裁方式授予。

![](images/29cb96496a01f841cfda9f1ac6a7524f7044211e1b73f53fea4e29de9d5d2556.jpg)  
Fig. 2. 配置为支持三种精度的 AGE 微架构。NID 请求驱动 Aggregation Managers (AGMs)，后者接收从 Feature Bank 获取的 Embedding（见图 1）。这些随后通过网络传输到 Aggregation Cores (AGCs)。聚合结果随后由 Bufering Managers (BMs) 进行缓冲。

![](images/193fe8a78d412d11fc810e09ae7a67038e5111ed0d0aaeed058a8fce4f64a319.jpg)  
Fig. 3. Feature Bank 中的 Fetch Tags 通过两阶段过程发起并发内存访问请求；首先，邻居节点 ID 列表被存储在 Address Queue 中，随后它们被用作指向邻居特征 Embedding 的指针，这些 Embedding 被存储在 Message Queue 中。

Feature Bank 通过其部分响应机制支持大图应用场景。对于节点度高于 Fetch Tag 容量的节点，Fetch Tag 会填满 Message Queue 并直接解除 AGE 的阻塞以开始聚合过程。聚合开始后，Fetch Tag 被重新启用并继续获取剩余的邻居节点，从而隐藏内存访问延迟。该机制降低了每个 nodeslot 的存储需求，允许 Feature Bank 中容纳更多的 Fetch Tags，即更深的节点并行度。

## 4 实验结果

本文部署了三个基础 GNN 模型来评估该加速器，其架构各不相同，如表 4 所示。各模型的更新法则见第 2 节。

此外，选取了 6 个图数据集，前三个是小型引文网络，后三个是较大的社交媒体图。表 5 展示了每个评估数据集的节点数和平均节点度——后者作为图稀疏性的指标，与稀疏性呈负相关关系，即平均度越高稀疏性越低。

Table 4. 用于对加速器进行基准测试的 GNN 模型。残差连接表示在聚合或变换步骤之后加上节点的原始 Embedding。
<table><tr><td>Model</td><td>Aggregation</td><td>Residual</td><td>Normalization</td></tr><tr><td>GCN</td><td>sum</td><td>X</td><td>aggregation</td></tr><tr><td>GIN</td><td>sum</td><td>aggregation</td><td>X</td></tr><tr><td>GraphSAGE</td><td>mean</td><td>transformation</td><td>transformation</td></tr></table>

Table 5. 用于基准测试的数据集。DQ ratio 表示通过 DegreeQuant 算法映射为 float 精度的节点比例，其余节点以 int8 运行。
<table><tr><td></td><td>Name</td><td>Nodes</td><td>Mean Degree</td><td>Features</td><td>DQ Ratio</td></tr><tr><td>CR</td><td>Cora</td><td>2,708</td><td>3.9</td><td>1,433</td><td>2.1 %</td></tr><tr><td>CS</td><td>CiteSeer</td><td>3,327</td><td>2.7</td><td>3,703</td><td>2.7 %</td></tr><tr><td>PB</td><td>PubMed</td><td>19,717</td><td>4.5</td><td>500</td><td>2.9 %</td></tr><tr><td>FL</td><td>Flickr</td><td>89,250</td><td>10.0</td><td>500</td><td>0.2 %</td></tr><tr><td>RD</td><td>Reddit</td><td>232,965</td><td>99.6</td><td>602</td><td>2.7 %</td></tr><tr><td>YL</td><td>Yelp</td><td>716,847</td><td>19.5</td><td>300</td><td>0.4 %</td></tr></table>

## 4.1 混合精度算术

采用 DegreeQuant 算法为各图数据集中的每个节点分配精度，即根据节点度对其进行随机保护（见第 2 节）。如表 5 所示，所有数据集中受保护节点的比例均低于 3%，这表明加速器上应分配给 float 的资源比例与之相似。随后按如下方式选择配置参数；给定分别用于 float 和 int8 节点的两个节点组，使用 DegreeQuant 得到的比例为每种精度组分配资源预算 $R _ { p } ^ { m a x , r }$（其中 $p$ ∈ [float, int8] 表示数值格式，$r$ ∈ [LUT, FF, BRAM, DSP] 表示资源类型）。同时，分别针对 float 和 int8 综合了单算术变体，并估算了每种精度和资源类型下每个 nodeslot 的资源利用率 $C _ { p } ^ { r }$。最后，每种精度的 nodeslot 数量 $N _ { p }$ 由公式 6 确定，其中方括号表示向上取整至最近的整数。

![](images/74a50c8065173764b1c67d073eb6ac2c7754ddfde91c75b8ac03bbb394185a9d.jpg)  
Fig. 4. 在 RTX A6000 GPU 和 AMPLE 仿真上获得的相对于 Intel Xeon CPU 基准的推理加速比。GPU 在所有数据集上对 GCN、GIN 和 GraphSAGE 分别取得了 29.8×、37.8× 和 26.7× 的平均加速比。在 AMPLE 上的相应加速比分别为 361.3×、285.8× 和 81.7×。

$$
N _ { p } = \left\lceil \operatorname* { m i n } _ { r } { \frac { R _ { p } ^ { m a x , r } } { C _ { p } ^ { r t } } } \right\rceil\tag{6}
$$

预期在受保护节点比例较低时，资源可以分布到更多数量的 nodeslot 上，这是由于定点核心的资源占用较低。事实上，研究发现为浮点分配单个 nodeslot 通常足以满足任务精度的要求，同时最大化硬件节点并行度。

## 4.2 性能分析

每个模型首先在 Intel Xeon CPU 和 RTX A6000 GPU 上对所有数据集进行了基准测试，节点特征和层权重均采用随机初始化。在每种情况下，均对 100 次试验取平均延迟，以消除非确定性过程导致的运行时抖动。在每次预测步骤之前都会清空 GPU 缓存，从而使延迟读数包含特征和权重的片外内存访问。GPU 预热时间未被计入，即推理时间是在驱动初始化完成后开始测量的。最后，AMPLE 上的推理延迟来自 Modelsim 19.2 的仿真结果，工作频率为 200MHz，并使用 Vivado 23.1 工具流针对 Alveo U280 卡获得。如图 4 所示，在所有模型上，AMPLE 相比 CPU/GPU 基线均取得了更优的平均推理时间。表 6 展示了 GCN 的延迟和节点吞吐量数值。

## 5 结论

本工作提出了 AMPLE，一种用于大型图上 GNN 推理的 FPGA 加速器。我们引入了一种事件驱动的编程流程，并结合通过片上网络通信实现的动态资源分配机制，克服了节点度分布不均匀的图中节点批处理（node batching）带来的性能瓶颈。通过以节点为中心的数据预取器，我们减轻了对权重和激活值片上存储的需求，从而能够在社交媒体图数据集上实现 GNN 加速。这些因素使得与 CPU 和 GPU 基线相比分别取得了平均 243× 和 7.2× 的加速。最后，我们提供了首个支持以节点粒度量化图加速的平台，展示了一种最优的资源映射方案，能够以较低的模型精度代价最大化节点并行度。

表 6. 使用单层 GCN 模型在所评估数据集上的推理时间。平均延迟为 100 次迭代的统计结果。
<table><tr><td rowspan="2"></td><td colspan="2">CPU (Intel Xeon)</td><td colspan="2">GPU (RTX A6000)</td><td colspan="4">AMPLE @200MHz</td></tr><tr><td>平均延迟 [ms]</td><td>吞吐量 [nodes/ms]</td><td>平均延迟 [ms]</td><td>吞吐量 [nodes/ms]</td><td>平均延迟 [ms]</td><td>吞吐量 [nodes/ms]</td><td>延迟增益 (CPU)</td><td>延迟增益 (GPU)</td></tr><tr><td>Cora</td><td>244.4</td><td>11.1</td><td>7.2</td><td>376.3</td><td>0.246</td><td>11,022.0</td><td>994.8×</td><td>29.3×</td></tr><tr><td>CiteSeer</td><td>244.3</td><td>13.6</td><td>10.1</td><td>330.0</td><td>0.294</td><td>11,318.6</td><td>831.2×</td><td>34.3X</td></tr><tr><td>PubMed</td><td>362.4</td><td>54.4</td><td>4.8</td><td>4,099.5</td><td>1.617</td><td>12,193.2</td><td>224.1×</td><td>3.0×</td></tr><tr><td>Flickr</td><td>475.4</td><td>187.8</td><td>14.5</td><td>6,146.2</td><td>7.227</td><td>12,350.0</td><td>65.8×</td><td>2.0X</td></tr><tr><td>Reddit</td><td>953.3</td><td>244.4</td><td>171.0</td><td>1,362.0</td><td>24.6</td><td>9,463.6</td><td>38.7×</td><td>6.9X</td></tr><tr><td>Yelp</td><td>760.8</td><td>942.2</td><td>110.9</td><td>6461.6</td><td>57.5</td><td>12,471.7</td><td>13.2×</td><td>1.9X</td></tr><tr><td>平均</td><td>506.8</td><td>242.2</td><td>53.1</td><td>3,129.3</td><td>15.2</td><td>11,469.9</td><td>361.1×</td><td>12.9×</td></tr></table>

EuroMLSys ’25, March 30-April 3, 2025, Rotterdam, Netherlands.

## References

Stefan Abi-Karam, Yuqi He, Rishov Sarkar, Lakshmi Sathidevi, Zihang Qiao, and Cong Hao. 2022. GenGNN: A Generic FPGA Framework for Graph Neural Network Acceleration. (1 2022). doi:10.48550/arxiv.2201.08475

Andry Alamsyah, Budi Rahardjo, and Kuspriyanto. 2021. Social Network Analysis Taxonomy Based on Graph Representation. (2 2021). https://arxiv.org/abs/2102. 08888v1

Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. 2017. Neural Message Passing for Quantum Chemistry. arXiv:1704.01212 [cs.LG]

William L. Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive Representation Learning on Large Graphs. In Neural Information Processing Systems. https://api. semanticscholar.org/CorpusID:4755450

Thomas N. Kipf and Max Welling. 2016. Semi-Supervised Classification with Graph Convolutional Networks. 5th International Conference on Learning Representations, ICLR 2017 - Conference Track Proceedings (9 2016). doi:10.48550/arxiv.1609.02907

Adrien Leman. 2018. THE REDUCTION OF A GRAPH TO CANONICAL FORM AND THE ALGEBRA WHICH APPEARS THEREIN. https://api.semanticscholar.org CorpusID:49579538

Shyam A. Tailor, Javier Fernandez-Marques, and Nicholas D. Lane. 2020. Degree-Quant: Quantization-Aware Training for Graph Neural Networks. (8 2020). doi:10.48550/

arxiv.2008.0500

Petar Veličković, Arantxa Casanova, Pietro Liò, Guillem Cucurull, Adriana Romero, and Yoshua Bengio. 2017. Graph Attention Networks. 6th International Conference on Learning Representations, ICLR 2018 - Conference Track Proceedings (10 2017). doi:10.48550/arxiv.1710.10903

Shoujin Wang, Liang Hu, Yan Wang, Xiangnan He, Quan Z. Sheng, Mehmet A. Orgun, Longbing Cao, Francesco Ricci, and Philip S. Yu. 2021. Graph Learning based Recommender Systems: A Review. IJCAI International Joint Conference on Artificial Intelligence (5 2021), 4644–4652. doi:10.24963/ijcai.2021/630

Hsiang-Yun Wu, Martin Nöllenburg, and Ivan Viola. 2021. Graph Models for Biological Pathway Visualization: State of the Art and Future Challenges. (10 2021). https: //arxiv.org/abs/2110.04808v1

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. 2018. How Powerful are Graph Neural Networks? ArXiv abs/1810.00826 (2018). https://api.semanticscholar. org/CorpusID:52895589

Mingyu Yan, Lei Deng, Xing Hu, Ling Liang, Yujing Feng, Xiaochun Ye, Zhimin Zhang, Dongrui Fan, and Yuan Xie. 2020. HyGCN: A GCN Accelerator with Hybrid Architecture. Proceedings - 2020 IEEE International Symposium on High Performance Computer Architecture, HPCA 2020 (1 2020), 15–29. doi:10.48550/arxiv.2001.02514

accepted 11 March 2025