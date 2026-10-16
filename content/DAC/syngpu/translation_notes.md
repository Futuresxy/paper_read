# SynGPU: Synergizing CUDA and Bit-Serial Tensor Cores for Vision Transformer Acceleration on GPU 原文翻译

# SynGPU：协同 CUDA 核心与位串行 Tensor 核心，加速 GPU 上的视觉 Transformer

Yuanzheng Yao<sup>1</sup>, Chen Zhang<sup>1</sup>\*, Chunyu Qi<sup>1</sup>, Ruiyang Chen<sup>1</sup>, Jun Wang<sup>2</sup>, Zhihui Fu<sup>2</sup>,

Naifeng Jing<sup>1</sup>, Xiaoyao Liang<sup>1</sup>, and Zhuoran Song<sup>1</sup>\*

<sup>1</sup>上海交通大学，中国上海

<sup>2</sup>OPPO 研究院

<sup>1</sup>{yzh yao, songzhuoran}@sjtu.edu.cn

摘要—视觉 Transformer（ViT）通过有效提取全局特征，在计算机视觉任务中展现了卓越的性能。然而，随着图像分辨率或视频时长的增加，其自注意力机制的时间和内存复杂度呈二次增长，导致其在 GPU 上的执行效率低下。为加速 ViT，现有工作主要关注基于数值级稀疏性剪除词元。然而，这些工作忽略了位级稀疏性，因而错失了实现峰值性能的机会。为此，我们提出词元间位稀疏性感知（Inter-token Bit-sparsity Awareness，IBA）算法，通过挖掘相似词元中的位稀疏性来加速 ViT。随后，我们通过解决两个问题，在 GPU 上协同 CUDA 核心与 Tensor 核心来实现 IBA：首先，寄存器文件的带宽拥塞限制了 CUDA 核心与 Tensor 核心的并行能力。其次，由于浮点向量中各元素的指数不同，难以通过基于定点的位级电路来加速 Tensor 核心中的位稀疏矩阵乘加（MMA）运算。因此，我们提出了用于加速 ViT 的算法与硬件协同设计框架 SynGPU。SynGPU 通过一种新颖的数据映射方式增强数据复用，使 CUDA 核心与 Tensor 核心能够完全并行执行。此外，该框架引入了位串行 Tensor 核心（Bit-Serial Tensor Core，BSTC），通过结合定点位串行点积（Bit-Serial Dot Product，BSDP）与指数对齐技术，支持定点和浮点 MMA 运算。大量实验表明，相比 A100 GPU，SynGPU 平均可实现 2.15× ∼ 3.95× 的加速比和 2.49× ∼ 3.81× 的计算密度。

## I. 引言

近年来，以视觉 Transformer（ViT）[1] 为代表的基于注意力的模型，在图像分类 [2]、目标检测 [3] 和视频识别 [4] 等计算机视觉（CV）任务中取得了卓越的性能。与传统卷积神经网络（CNN）模型相比，ViT 凭借有效提取全局特征的能力，在各种应用中兼具更高的性能与更强的灵活性。

然而，ViT 的这些成就是以低效率为代价的。众所周知，ViT 中自注意力机制的时间和内存复杂度相对于输入词元数量呈二次增长。因此，随着图像分辨率或视频时长的增加，ViT 在 GPU 上的计算性能会显著下降。为解决这一局限，已有多种方法被提出 [5]–[8]，重点是利用图像块之间或视频帧之间的词元相似性，减少数值级冗余。除数值稀疏性之外，我们的分析表明，位级冗余也为进一步提高 ViT 的计算效率提供了机会。

如图 1 所示，相邻图像像素的数值往往相近，因此其差值中包含大量 0 比特。基于这一观察，我们提出了词元间位稀疏性感知（IBA）算法，该算法计算输入词元之间的距离，并对相似词元对作减法以获得词元差值。这些词元差值表现出更高的位级稀疏性，可被有效利用以加速计算。为在 GPU 上实现这一策略，我们利用 GPU 内部原本彼此独立的两类计算核心：CUDA 核心与 Tensor 核心，发挥它们的互补功能，通过利用位级稀疏性提高 ViT 的效率。

![](images/2642217be05a1008fc1928236ac3b073b4b591683c7bc764bb66d01c83d43e2d.jpg)  
图 1. SynGPU 的协同机制。

这一方案的关键洞察有两点：第一，利用 CUDA 核心的高精度计算能力，在不损害准确率的前提下挖掘相似词元之间的位级稀疏性；第二，利用针对高吞吐量矩阵乘加（MMA）运算优化的 Tensor 核心，实现位稀疏加速。然而，实现 CUDA 核心与 Tensor 核心之间的高效协同面临两个关键挑战。首先，CUDA 核心与 Tensor 核心会争用寄存器文件（RF），这可能导致 RF 带宽拥塞，限制二者完全并行执行的能力。其次，Tensor 核心通过并行执行点积（DP）运算来实现浮点 MMA。由于向量内各浮点数的指数不同，设计同时支持定点和浮点 DP 的位级电路并非易事。

![](images/2740947ab78193e390b237d0c2a57ab5be0f952d728d21b53c47978c0e0595a2.jpg)  
(a) FP16 中的位级稀疏性

![](images/2da93dac921cab7198ce274701f84335294722a625c133d05446a5ec34ed0eee.jpg)  
(b) INT8 中的位级稀疏性  
图 2. 词元差值与未作差词元在 FP16/INT8 格式下的位级稀疏性

本文提出了算法与硬件协同框架 SynGPU，通过协同 CUDA 核心与 Tensor 核心，加速 GPU 上的 ViT。具体而言，为应对第一个挑战，SynGPU 包含一种新颖的数据映射方案，增强了 Tensor 核心中的数据复用。该方法显著降低了 Tensor 核心访问 RF 的频率，从而为 CUDA 核心释放 RF 资源。此外，为应对第二个挑战，我们开发了位串行点积（BSDP）算法及位串行 Tensor 核心（BSTC），对每个向量内各浮点数的指数进行对齐，并重新封装 DP 运算的结果。这一设计使 SynGPU 能够有效支持浮点和定点位稀疏运算。

## 总结而言，我们的贡献如下：

• 我们提出了 SynGPU，这是一种算法与硬件协同设计框架，利用 GPU 上 CUDA 核心与 Tensor 核心之间的协同来加速位稀疏 ViT。SynGPU 利用 CUDA 核心挖掘由词元相似性产生的位级稀疏性，并协调 Tensor 核心执行位稀疏 MMA，从而显著提升 ViT 的性能。据我们所知，SynGPU 是首个专门针对 GPU 上 ViT 位级加速而设计的框架。

• 在算法层面，我们提出了 IBA 和 BSDP 两种算法。IBA 旨在提取相似词元之间的位级稀疏性，而 BSDP 则着重于通过对齐浮点向量的指数，将定点与浮点 MMA 相结合，以实现高效的位级稀疏浮点 MMA。

• 在硬件层面，我们作出了两项贡献。首先，我们设计了一种新的数据映射方式，解决了带宽拥塞问题，使 CUDA 核心与 Tensor 核心能够完全并行执行。其次，我们开发了 BSTC，能够加速定点和浮点格式的位稀疏 MMA 运算。

## II. 背景与动机

## A. 视觉 Transformer

受 Transformer [9] 在 NLP 任务中出色表现的启发，研究人员开展了大量工作，将 Transformer 应用于视觉领域。其中，ViT 凭借其简洁的设计以及在图像和视频任务中的强大能力脱颖而出。ViT 将图像或视频帧划分为图像块，再通过图像块嵌入层将其转换为词元，从而能够直接输入 Transformer。

![](images/7b91e9a2a1f1f359af4e324ad1c5387a2c2fd75c57e60a94cc9f6f2a48b765ab.jpg)  
图 3. 在 A100 上执行 ViT 时，CUDA 核心和张量核心的 RF 带宽占用率 (a) 和活跃时间 (b)。

目前已有多种加速器架构 [5]–[7] 被提出用于优化 ViT。例如，DynamicViT [6] 设计了专用的动态预测模块，用于预测并剪除冗余词元。ViTCoD [5] 使用自编码器模块对注意力图进行重排，生成更稠密或更稀疏的固定模式，以平衡 ViT 的工作负载。然而，这些进展主要关注数值级稀疏性的优化，忽视了相似词元之间潜在的比特级稀疏性。

## B. Ampere GPU

近年来，神经网络的快速发展及其巨大的计算需求推动了 GPU 结构和性能的显著演进。Nvidia A100 [10] 就是一个典型例子，它包含多个流式多处理器（SM）。SM 的每个子核心由若干 CUDA 核心和一个张量核心组成，共享同一个 RF。具体而言，CUDA 核心用于高精度通用计算，而张量核心则面向高吞吐量 MMA 运算设计，内部包含大量支持各种定点和浮点格式的 DP 单元。

## C. 研究动机

当前关于 ViT 加速器的研究主要关注数值级稀疏性，而我们的实验结果表明，相似词元之间存在一种更具潜力的比特级稀疏性。如图 2 所示，我们运行了多种具有代表性的 ViT 模型实现，并对注意力图中的相似词元进行相减。结果发现，FP16 和 INT8 中的 0 比特数量均显著增加，比特稀疏率分别从 50.19% 提高到 65.98%，以及从 50.48% 提高到 75.82%。

![](images/ab64244549f69e8b905cae2895fb7f6645c6e42b4e983674ef9ab98cfcf1fb15.jpg)  
图 4. 词元间比特稀疏性感知算法概览。

基于上述观察，我们计划提取并利用这种比特级稀疏性，通过 CUDA 核心与张量核心的协作来加速 ViT。然而，如图 3 所示，当张量核心满负荷运行、RF 带宽占用率达到 91.6% 时，由于对 RF 的竞争访问，CUDA 核心大部分时间处于空闲状态。此外，浮点向量中不同的指数阻碍了使用定点比特级设计 [11]、[12] 来加速张量核心中的 MMA 运算。尽管 Bitlet [13] 致力于在比特级支持浮点格式，但它在较低比特宽度的定点计算中会导致硬件利用率不足。为解决这些低效问题，我们提出了 SynGPU，这是一种算法与硬件协同设计框架，能够促进 CUDA 核心和张量核心的并发利用，以加速 ViT。

![](images/5e0c840f5658021c98001f0cbb0495c03443c66ab5c762982331c61c3207c4af.jpg)  
图 5. 详细说明：(a) 并行点积。(b) 比特串行点积。(c) 指数对齐算法。

## III. 算法

## A. 词元间比特稀疏性感知

本节介绍词元间比特稀疏性感知（IBA），这是一种旨在利用 ViT 内部比特级稀疏性的算法。如图 4 所示，在步骤 1 中，我们以固定间隔选取一部分词元作为关键词元，其余词元则为非关键词元。随后，我们计算最小曼哈顿距离，以此为准则找到与每个非关键词元最相似的关键词元，并将其记录在关键词元映射中。在所给示例中，我们以 2 为间隔选择关键词元，并发现与 $T _ { 2 }$ 和 $T _ { 4 }$ 最相似的词元分别为 $T _ { 1 }$ 和 $T _ { 3 }$。接着，步骤 2 对相似词元对进行相减，得到由大量比特稀疏差分词元组成的比特稀疏矩阵。蓝色高亮部分展示了差分词元 $\Delta T _ { 2 } = T _ { 2 } - T _ { 1 }$ 中各数值的二进制表示，表明词元之间的相似性带来了更多冗余的 0 比特。随后，步骤 3 执行比特稀疏 MMA 运算，高效地得到输出 ∆ 矩阵。最后，基于 MMA 运算的线性性质，步骤 4 利用关键词元映射中的信息，将输出 $O _ { 1 } , O _ { 3 }$ 分别加到 $\Delta O _ { 2 }$ 和 $\Delta O _ { 4 }$ 上，从而无损地恢复输出矩阵。

## B. 比特串行点积

执行比特稀疏 MMA 的主要挑战在于同时支持浮点和定点格式，尤其是浮点向量中的指数各不相同。由于浮点乘法包含指数相加和尾数相乘这两个主要运算，而后者遵循与定点乘法相同的机制，因此我们提出了比特串行点积（BSDP），这是一种将定点比特串行乘法与浮点指数对齐相结合的算法，用于应对这一挑战。

定点 BSDP。图 5(a) 所示的传统 DP 运算并行计算所有比特。为跳过不必要的 0 比特，如图 5(b) 所示，我们需要将矩阵元素编码为比特索引格式，从而将乘法过程转换为比特串行输入模式。这种方法将乘法转换为一系列移位累加运算，有效地跳过了 0 比特。

浮点 BSDP。浮点操作数表示为 $X \ = \ 2 ^ { E } \times M$，其中 M 为尾数，E 为指数（为简便起见，我们省略了符号位）。浮点 BSDP 包含三个步骤：指数对齐、尾数相乘和重新封装。如图 5(c) 所示，我们首先通过对尾数进行右移，将向量中所有元素的指数对齐到最大的 $E _ { m }$。此外，由于右移可能造成溢出并进而导致精度损失，我们将 10 比特尾数扩展为 16 比特：

$$
X _ {i} = 2 ^ {E _ {i}} \times M _ {i} = 2 ^ {E _ {m}} \times (M _ {i} \gg \Delta E _ {i})\tag{1}
$$

其中，$\Delta E _ { i } = E _ { m } - E _ { i }$ 为尾数的右移位数。对于浮点 BSDP ${ \vec { a } } \times { \vec { b } } ,$，我们分别对两个向量的指数进行对齐，然后按照与定点 BSDP 相同的方式将尾数相乘。随后，我们根据式 (2) 将统一后的指数相加，将结果重新封装为标准浮点格式：

$$
\begin{array}{l} {d p = \sum_ {i = 0} ^ {m} [ 2 ^ {E _ {i} ^ {a}} \times M _ {i} ^ {a} ] \times [ 2 ^ {E _ {i} ^ {b}} \times M _ {i} ^ {b} ]} \\ {\qquad = \sum_ {i = 0} ^ {k} 2 ^ {E _ {m} ^ {a} + E _ {m} ^ {b}} \times (M _ {i} ^ {a} \gg \Delta E ^ {a}) \times (M _ {i} ^ {b} \gg \Delta E ^ {b})} \end{array}\tag{2}
$$

![](images/5394efffb3f9628dc041ebffac602d1587e18bb666e27df0b94b93ce849a918e.jpg)  
图 6. 比特串行张量核心的架构：(a) BSTC 概览。(b) BSTC 细节。

为减少冗余计算，我们进一步将对齐运算中的两次右移与定点 BSDP 中的左移合并，得到浮点 BSDP 运算的最终结果，如式 (3) 所示：

$$
d p = \sum_ {i = 0} ^ {k} \sum_ {j = 0} ^ {l} 2 ^ {E _ {m} ^ {a} + E _ {m} ^ {b}} \times [ M _ {i} ^ {b} \ll (I _ {j} - \Delta E ^ {a} - \Delta E ^ {b}) ]\tag{3}
$$

其中，$I _ { j }$ 表示 ⃗a 的元素中值为 1 的比特位置的索引。

## IV. 架构

## A. 概述

SynGPU 架构包含两项关键设计。首先，我们引入一种数据映射方案，以解决 CUDA 核心与 Tensor Core 之间的 RF 带宽拥塞问题，使两类核心能够并行处理。其次，我们提出位串行 Tensor Core（BSTC），它能够计算定点和浮点 MMA。此外，为最大限度地提高 BSTC 加速位稀疏 MMA 的效率，我们设计了一个重排模块，以平衡矩阵的位稀疏性。

![](images/7c928401db72f24450de26bebde89f8eadf936ba3b79ffd197162e78adb84b58.jpg)  
图 7. 数据映射的细节。

## B. 数据映射

在原始 Tensor Core 中执行 $1 6 \times 1 6 \times 8$ 矩阵运算时，该运算被划分为 8 个步骤，每个步骤执行一次 $8 \times 8 \times 4$ MMA 过程（图 7 中仅给出了 4 个步骤），并以流水线方式运行，每个过程包含 3 个部分：从 RF 读取、计算以及回写至 RF。如图所示，在步骤 1 中，缓冲区读取 $A _ { 0 }$ 和 $B _ { 0 } ,$，而计算和回写分别对应步骤 2 和步骤 3。遗憾的是，上述执行过程占用了大部分 $\mathrm { R F s }$ 存储带宽，导致 CUDA 核心在 Tensor Core 运算期间保持空闲。此外，Tensor Core 的这种数据流还会导致重复的数据读写操作，例如步骤 2 和步骤 4 中的 $B _ { 1 }$，从而产生不必要的开销。

为解决这一问题，我们提出增大 Tensor Core 数据缓冲区的容量，使其能够在步骤 1 中一次性读取当前计算所需的全部数据。通过在 BSDP 运算中将元素各比特的处理方式从并行转换为串行，RF 在后续的 DP 计算步骤中便处于空闲状态，从而可供 CUDA 核心使用。

此外，由于串行处理降低了 Tensor Core 的并行度，我们扩展 B 矩阵的维度，以确保吞吐量不会下降。具体而言，由于 III-B 节中详述的 16 位扩展尾数，在最坏情况下，计算 MMA 所需的周期数可达到 Ampere Tensor Core 的 16×。因此，将 B 矩阵的扩展尺寸设为 $1 6 ~ \times ~ 3 2$。此外，为处理重复读写操作，我们将 A 矩阵分成两组，每组大小为 $1 6 \ \times \ 8 .$，以同时计算 $A _ { 0 } \times B _ { 0 }$ 和 $A _ { 0 } \times B _ { 1 }$。

## C. 位串行 Tensor Core 设计

如 III-B 节所述，为实现位稀疏 MMA，需要对浮点数进行预处理以对齐指数，而定点数则可以直接执行 BSDP 运算。因此，如图 6 所示，BSTC 架构主要由两个模块组成：BSDP 阵列和浮点预处理器。

BSDP 阵列。为实现位稀疏的矩阵 A × 矩阵 B 运算，我们将 Tensor Core 内部的 DP 单元替换为 BSDP 单元，并将它们组织成网格阵列，以并发执行多个 BSDP 运算，其中第 (x, y) 个单元执行 A 的第 x 行与 B 的第 y 列之间的 BSDP 运算。每个 BSDP 单元由多个移位器、一个加法器和一个重打包模块组成，以完成 III-B 节中描述的定点 BSDP。在每个周期内，从 B 缓冲区获取一个向量，并从 A 缓冲区获取矩阵 A 的一串位索引编码，以执行移位累加操作；重复这一过程，直到完成整个向量的 DP 运算。

浮点预处理器。浮点预处理器的主要功能是对矩阵中的浮点元素进行解包，并以向量为单位对齐指数。首先，解包过程从浮点数据中分离并提取指数和尾数，然后找出最大的指数作为 $E _ { m } .$，并将各部分分别存储到 $E _ { i }$ 缓冲区、$M _ { A }$ 缓冲区和 $E _ { m }$ 缓冲区中。值得注意的是，由于 B 的尾数不需要进行位串行编码，因此可以直接存入 B 缓冲区，用于后续的 DP 计算。接下来，Exp-Align 模块执行式 (3) 中提到的减法 $\Delta E _ { i } ~ = ~ E _ { m } - E _ { i }$。随后，根据 III-B 节中讨论的原理，位串行编码器结合 $\Delta E _ { i }$ 和 $M _ { A }$，将其编码为位串行形式，并存储到 A 缓冲区中，以与 B 的尾数执行 BSDP 运算。最后，在每个单元末尾添加一个与 $E _ { m }$ 缓冲区对接的重打包模块，将 DP 结果规格化为标准浮点格式，将 $E _ { m } ^ { a } + E _ { m } ^ { b }$ 合入最终指数，并在回写之前将结果存储到累加缓冲区中。

## D. 重排模块设计

尽管 IV-C 节中的 BSTC 能够加速定点和浮点位稀疏 MMA 运算，我们发现，限制计算性能的一个主要瓶颈是矩阵 A 各元素之间的位不均衡。显然，图 5(b) 所示的 BSDP 单元执行 BSDP 运算的速度受到向量中含有最多 1 比特的元素 $a _ { 0 }$ 的制约。同样，在 BSDP 阵列中，整个矩阵 A 范围内也存在相同的障碍。此外，由于矩阵的随机分布，矩阵中通常至少存在一个位稠密元素，显著降低了 BSDP 阵列的速度。

![](images/c669effa518a68cb64e17bcc50c1702e5e5c64d25e77863b3fdad11f4cb461a9.jpg)  
图 8. 细节：(a) 重排方案。(b) 加速比。

为解决这种不均衡问题，如图 8 所示，我们以 BSDP 单元向量长度的两倍为一个单位，将矩阵各行组织为位稀疏和位稠密两个部分，它们分别对应 IV-B 节所述的映射大小。具体而言，我们从矩阵 A 中采样少量元素，以粗略预测矩阵的稀疏度，并将其作为数据重排的阈值 θ。接下来，我们将每个值的稀疏度与 θ 进行比较，从而构建一个位稀疏性均衡的矩阵。在给定示例中，我们将元素从 $a _ { 0 } , a _ { 1 } , a _ { 2 } , a _ { 3 }$ 重排为 $a _ { 1 } , a _ { 2 } , a _ { 4 } , a _ { 6 } .$。因此，重排模块将所需周期数从 6 减少到 3。此外，为基于重排后的 A 矩阵正确实现 MMA，我们维护一份存储在 Idx 缓冲区中的索引列表，作为对 B 矩阵重排的参考。

## V. 评估

## A. 方法

软件实现。为验证 SynGPU 框架的有效性，我们选择采用 ImageNet2021 [14] 数据集的 DeiT [2] 作为图像任务中 ViT 的代表，并选择采用 Kinetics-400（K400）、Kinetics-600（K600）[17] 和 SSv2 [18] 数据集的 TimeSformer [4]、Motionformer [15] 和 Xvit [16] 作为视频任务中 ViT 的代表。所有模型均使用官方源代码和预训练模型实现，未经微调。此外，在下述实验中，IBA 算法采用的间隔为 80，批大小设为 8。

硬件实现。对于内核级评估，我们将 NVIDIA A100 GPU 设为基线。我们采用集成了 SynGPU 模型的开源周期精确模拟器 Accel-Sim [19]，模拟 A100 和 SynGPU 在 FP16 和 INT8 格式下的计算性能。我们使用 Verilog 实现所提出的 SynGPU 架构，并通过 Synopsys Design Compiler 进行综合，以获得 28nm 工艺下的面积和功耗。鉴于缺乏公开的 A100 Tensor Core 面积数据，并考虑到 A100 采用 7nm 工艺，而我们的研究基于 28nm 工艺，我们遵循文献 [20] 的方法，估算 A100 在 28nm 工艺和 1.41GHz 频率下的面积和功耗。对于基于 SRAM 的片上缓冲区，我们使用 CACTI 7 [21] 对其在 32nm 工艺下的面积和功耗进行建模，并使用 DeepScaleTool [22] 将其缩放至 28nm。

## B. 实验

加速比。我们分别在 FP16 和 INT8 格式下，对比了 A100 Tensor Core 与 SynGPU BSTC 在多种模型和数据集上的性能，如图 9(a) 和图 9(b) 所示。由于 A100 Tensor Core 不会跳过任何 0 位，因此在处理数据维度相同的张量时，其消耗的周期数相同。另一方面，SynGPU BSTC 具备加速位稀疏 MMA 的能力，因此对于不同的输入数据，其相对于 A100 的加速比会随注意力图的位稀疏性而变化。SynGPU-FP16 和 SynGPU-INT8-w/o-rearrangement 实现的平均加速比分别为 2.2× 和 2.15×，峰值加速比分别为 2.89× 和 2.20×，均未达到根据图 2 所示位稀疏率预期的理想加速比。原因在于，没有重排时，BSTC 无法以最高效率利用位稀疏性。因此，我们基于 INT8 格式进行了消融实验，证明重排能够有效发挥 BSTC 的位稀疏加速能力，平均实现 3.38× 的加速比。

能效。图 9(c) 和图 9(d) 展示了能效结果。与 A100 GPU 相比，Syn-GPU 在 FP16、无重排的 INT8 和 INT8 情况下，平均分别实现了 1.84×、1.85× 和 2.68× 的能效提升。FP16 和无重排的 INT8（INT8-w/o-rearrangement）中的优化归因于 SynGPU 跳过了不必要的 0 位计算，从而节省了片上动态能耗。对于采用重排的 INT8，提升来自利用重排模块缩短计算周期，进而减少片上静态能耗。

![](images/7b62590d98a98ba155354c08c61e875ee967eab013e0bbbc7c2aafb626f493cf.jpg)  
(a) FP16 下的加速比

![](images/7a7931f9cd633b87d9ab816ab8c97582b3b4a2719192740866fee407074684a5.jpg)

![](images/16d4827d37ce03f46d75c77a2f4e8be2d01686127f905ffb428a98c5f361588e.jpg)  
(b) INT8 下的加速比

(c) FP16 下的能效  
![](images/4fb5ba0aaf0f85b1be8030d318a9206f1c35b0e4edbef5aeddaec9b8b2b3f19d.jpg)  
(d) INT8 下的能效  
图 9. FP16 和 INT8 格式下 SynGPU 相对于 A100 的加速比与能效

表 I  
SYNGPU 架构的面积与功耗。

<table><tr><td>参数</td><td>面积( $mm^2$ )</td><td>面积占比</td><td>功耗(W)</td></tr><tr><td>16x16 位串行 DP 单元</td><td>0.9306</td><td></td><td>0.3000</td></tr><tr><td>1.25 KB A 缓冲区</td><td>0.0516</td><td></td><td>0.0358</td></tr><tr><td>1KB B 缓冲区</td><td>0.0233</td><td>96.45%</td><td>0.0269</td></tr><tr><td>2KB Acc 缓冲区</td><td>0.0857</td><td></td><td>0.1835</td></tr><tr><td>BSDP 阵列</td><td>1.0913</td><td></td><td>0.5462</td></tr><tr><td>最大指数查找</td><td>0.0006</td><td></td><td>0.0013</td></tr><tr><td>指数对齐</td><td>0.0003</td><td></td><td>0.0008</td></tr><tr><td>0.3KB Ma 缓冲区</td><td>0.0138</td><td>1.91%</td><td>0.1813</td></tr><tr><td>0.15KB Exp 缓冲区</td><td>0.0069</td><td></td><td>0.1813</td></tr><tr><td>浮点预处理器</td><td>0.0216</td><td></td><td>0.3647</td></tr><tr><td>位串行编码器</td><td>0.0019</td><td>0.16%</td><td>0.0430</td></tr><tr><td>比较器与选择器</td><td>0.0098</td><td></td><td>0.0225</td></tr><tr><td>0.15KB IdxList 缓冲区</td><td>0.0069</td><td>1.48%</td><td>0.1813</td></tr><tr><td>重排模块</td><td>0.0167</td><td></td><td>0.2039</td></tr></table>

硬件开销与面积。表 I 概述了 SynGPU 每个 SM 中 BSTC 的硬件配置、面积和功耗。BSTC 的主要开销来自 BSDP 阵列，其面积为 $1 . 0 9 1 3 \mathrm { m m } ^ { 2 } .$。其他模块，即浮点预处理器、位串行编码器和重排模块的面积分别为 0.0216mm<sup>2</sup>、0.0019mm<sup>2</sup> 和 $0 . 0 1 6 7 \mathrm { m m } ^ { 2 }$，分别带来约占整个 Tensor Core 的 1.91%、0.16% 和 1.48% 的开销，与占芯片面积 96.45% 的 BSDP 阵列相比，这些开销可以忽略不计。值得注意的是，重排模块以很小的硬件开销带来了可观的加速效果。

计算性能。表 II 展示了 A100 Tensor Core 和 SynGPU BSTC 在 FP16 和 INT8 格式下的计算性能。尽管 SynGPU 中 BSTC 的面积相较于 A100 中的 Tensor Core 略有增加（1.13 mm<sup>2</sup> 对比 0.98 mm<sup>2</sup>），但 SynGPU 的峰值计算性能具有显著优势，在 FP16 下达到 901 TFLOPs，在 INT8 下达到 2761 TOPs，而 A100 在 FP16 和 INT8 下分别为 312 TFLOPs 和 624 TOPs。因此，在相同面积预算下，SynGPU 在 FP16 和 INT8 下的计算密度相较于 A100 分别提升了 2.49× 和 3.81×。此外，根据表 I 给出的计算模块功耗仿真结果，我们得到了 SynGPU 在 FP16 和 INT8 格式下的功耗效率，分别达到 7.69 TFLOPs/W 和 23.57 TOPs/W，相较于 A100 分别提升了 2.58× 和 3.95×。得益于位级稀疏性带来的加速和能耗降低，SynGPU 在计算密度和计算功耗效率方面均取得了显著进步。

表 II  
A100 和 SYNGPU 的计算性能。

<table><tr><td>设备</td><td>峰值性能</td><td>每个 SM 的面积</td><td>计算密度</td><td>功耗效率</td></tr><tr><td>A100-FP</td><td>312 TFLOPs</td><td> $0.98 \text{ mm}^{2}$ </td><td> $2.96 \text{ TFLOPs/mm}^{2}$ </td><td> $2.98 \text{ TFLOPs/W}$ </td></tr><tr><td>A100-INT8</td><td>624 TOPs</td><td> $0.98 \text{ mm}^{2}$ </td><td> $5.93 \text{ TOPs/mm}^{2}$ </td><td> $5.96 \text{ TOPs/W}$ </td></tr><tr><td>SynGPU-FP</td><td>901 TFLOPs</td><td> $1.13 \text{ mm}^{2}$ </td><td> $7.37 \text{ TFLOPs/mm}^{2}$ </td><td> $7.69 \text{ TFLOPs/W}$ </td></tr><tr><td>SynGPU-INT8</td><td>2761 TOPs</td><td> $1.13 \text{ mm}^{2}$ </td><td> $22.60 \text{ TOPs/mm}^{2}$ </td><td> $23.57 \text{ TOPs/W}$ </td></tr></table>

## VI. 结论

本文提出了 SynGPU，这是一种创新的算法与硬件协同设计框架，通过协同 GPU 上的 CUDA Cores 和 Tensor Cores，挖掘并利用相似 token 之间的位级稀疏性，显著提升了 ViT 的性能。SynGPU 解决了两个关键挑战：第一，CUDA Cores 和 Tensor Cores 竞争访问 RF 所导致的低并行度；第二，浮点向量中的指数不同，导致 Tensor Core 无法加速位级稀疏 MMA。我们的创新方案充分利用了 ViT 的位稀疏性来实现加速，在不产生精度损失的情况下取得了显著成效。

[1] A. Dosovitskiy, “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[2] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou, “Training data-efficient image transformers & distillation´ through attention,” in International conference on machine learning. PMLR, 2021, pp. 10 347–10 357.

[3] D. Wodajo, “Deepfake video detection using convolutional vision transformer,” arXiv: Computer Vision and Pattern Recognition,arXiv: Computer Vision and Pattern Recognition, Nov 2020.

[4] G. Bertasius, H. Wang, and L. Torresani, “Is space-time attention all you need for video understanding?” in ICML, vol. 2, no. 3, 2021, p. 4.

[5] H. You, Z. Sun, H. Shi, Z. Yu, Y. Zhao, Y. Zhang, C. Li, B. Li, and Y. Lin, “Vitcod: Vision transformer acceleration via dedicated algorithm and accelerator co-design,” in 2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA). IEEE, 2023, pp. 273–286.

[6] Y. Rao, W. Zhao, B. Liu, J. Lu, J. Zhou, and C.-J. Hsieh, “Dynamicvit: Efficient vision transformers with dynamic token sparsification,” Advances in neural information processing systems, vol. 34, pp. 13 937– 13 949, 2021.

[7] P. Dong, M. Sun, A. Lu, Y. Xie, K. Liu, Z. Kong, X. Meng, Z. Li, X. Lin, Z. Fang et al., “Heatvit: Hardware-efficient adaptive token pruning for vision transformers,” in 2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA). IEEE, 2023, pp. 442– 455.

[8] Z. Li, M. Sun, A. Lu, H. Ma, G. Yuan, Y. Xie, H. Tang, Y. Li, M. Leeser, Z. Wang et al., “Auto-vit-acc: An fpga-aware automatic acceleration framework for vision transformer with mixed-scheme quantization,” in 2022 32nd International Conference on Field-Programmable Logic and Applications (FPL). IEEE, 2022, pp. 109–116.

[9] A. Vaswani, “Attention is all you need,” Advances in Neural Information Processing Systems, 2017.

[10] J. Choquette, W. Gandhi, O. Giroux, N. Stam, and R. Krashinsky, “Nvidia a100 tensor core gpu: Performance and innovation,” IEEE Micro, vol. 41, no. 2, pp. 29–35, 2021.

[11] J. Albericio, A. Delmas, P. Judd, S. Sharify, G. O’Leary, R. Genov,´ and A. Moshovos, “Bit-pragmatic deep neural network computing,” in Proceedings of the 50th annual IEEE/ACM international symposium on microarchitecture, 2017, pp. 382–394.

[12] A. Delmas Lascorz, P. Judd, D. M. Stuart, Z. Poulos, M. Mahmoud, S. Sharify, M. Nikolic, K. Siu, and A. Moshovos, “Bit-tactical: A software/hardware approach to exploiting value and bit sparsity in neural networks,” in Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems, 2019, pp. 749–763.

[13] H. Lu, L. Chang, C. Li, Z. Zhu, S. Lu, Y. Liu, and M. Zhang, “Distilling bit-level sparsity parallelism for general purpose deep learning acceleration,” in MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture, 2021, pp. 963–976.

[14] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009, pp. 248–255.

[15] M. Patrick, D. Campbell, Y. Asano, I. Misra, F. Metze, C. Feichtenhofer, A. Vedaldi, and J. F. Henriques, “Keeping your eye on the ball: Trajectory attention in video transformers,” Advances in neural information processing systems, vol. 34, pp. 12 493–12 506, 2021.

[16] A. Bulat, J. M. Perez Rua, S. Sudhakaran, B. Martinez, and G. Tzimiropoulos, “Space-time mixing attention for video transformer,” Advances in neural information processing systems, vol. 34, pp. 19 594– 19 607, 2021.

[17] W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev et al., “The kinetics human action video dataset,” arXiv preprint arXiv:1705.06950, 2017.

[18] R. Goyal, S. Ebrahimi Kahou, V. Michalski, J. Materzynska, S. Westphal, H. Kim, V. Haenel, I. Fruend, P. Yianilos, M. Mueller-Freitag et al., “The” something something” video database for learning and evaluating visual common sense,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 5842–5850.

[19] M. Khairy, Z. Shen, T. M. Aamodt, and T. G. Rogers, “Accel-sim: An extensible simulation framework for validated gpu modeling,” in 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA). IEEE, 2020, pp. 473–486.

[20] Z. Mo, L. Wang, J. Wei, Z. Zeng, S. Cao, L. Ma, N. Jing, T. Cao, J. Xue, F. Yang et al., “Lut tensor core: Lookup table enables efficient low-bit llm inference acceleration,” arXiv preprint arXiv:2408.06003, 2024.

[21] R. Balasubramonian, A. B. Kahng, N. Muralimanohar, A. Shafiee, and V. Srinivas, “Cacti 7: New tools for interconnect exploration in innovative off-chip memories,” ACM Transactions on Architecture and Code Optimization (TACO), vol. 14, no. 2, pp. 1–25, 2017.

[22] S. Sarangi and B. Baas, “Deepscaletool: A tool for the accurate estimation of technology scaling in the deep-submicron era,” in 2021 IEEE International Symposium on Circuits and Systems (ISCAS), May 2021, p. 1–5. [Online]. Available: http://dx.doi.org/10.1109/iscas51556. 2021.9401196

> 本文由 Codex/GPT 根据论文 PDF 生成中文译文与解读；PDF 由 MinerU 解析。全文译文保留原始参考文献，解读与原文结论请结合阅读。
