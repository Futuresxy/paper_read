# MXDOTP: A RISC-V ISA Extension for Enabling Microscaling (MX) Floating-Point Dot Products 原文翻译

# MXDOTP：一种支持微缩放（MX）浮点点积的 RISC-V ISA 扩展

Gamze <sup>˙</sup>Islamoglu˘ <sup>∗</sup>, Luca Bertaccini<sup>∗</sup>, Arpan Suravi Prasad<sup>∗</sup>, Francesco Conti<sup>†</sup>, Angelo Garofalo<sup>†</sup>, Luca Benini<sup>∗†</sup> <sup>∗</sup>IIS，苏黎世联邦理工学院，瑞士；<sup>†</sup>DEI，博洛尼亚大学，意大利

{gislamoglu, lbertaccini, prasadar, lbenini}@iis.ee.ethz.ch, {f.conti, angelo.garofalo}@unibo.it

摘要——快速且节能的低位宽浮点（FP）运算对人工智能（AI）系统至关重要。微缩放（MX）标准化格式最近成为基础低位宽 FP 格式的一种颇具前景的替代方案，通过将块级共享的指数缩放因子与逐元素数值相结合，提高了精度。然而，要以 MX 格式高效执行 AI 应用中的关键线性代数原语，需要为缩放点积等基本运算提供专门的硬件支持。在本工作中，我们提出了 MXDOTP，这是首个面向 MX 点积的 RISC-V ISA 扩展，重点支持 8 位 MXFP8 浮点格式。我们为开源 Snitch RISC-V 核心扩展了一个专用的 MXFP8 点积累加单元，该单元可完整处理打包到 64 位输入中的、由八个 8 位操作数组成的数据块。为了每周期向 MXDOTP 提供包括块缩放因子在内的四个操作数，使其满负荷运行，我们利用了 Snitch 的流语义寄存器（SSR），实现了高达 80 % 的利用率，同时对 Snitch 核心架构的影响极小，且无需修改寄存器文件。采用 12 nm FinFET 工艺实现的集群包含八个经 MXDOTP 扩展的核心，在 0.8 V、1 GHz 下计算 MXFP8 矩阵乘法时，能效最高达到 356 GFLOPS/W。软件基准方案通过将 FP8 输入类型转换为 FP32 来获得更高的累加精度，并显式执行块缩放，以此计算 MX 点积；与该方案相比，该集群实现了 25 倍加速和 12.5 倍能效提升，而面积仅增加 5.1 %。

## I. 引言

紧凑浮点（FP）运算在节能计算领域受到了广泛关注，尤其是在机器学习工作负载中 [1]–[4]。大型语言模型（LLM）等深度学习模型需要海量的计算和内存资源，因此计算效率和内存带宽都成为关键问题。在 FP8 等低位宽格式中，扩大指数范围可以改善动态范围，但会降低精度；而增大尾数则可以提高精度，却要以缩小动态范围为代价。块格式通过共享指数缓解了这种难以兼顾的权衡，同时保留了动态范围和精度。

尽管块浮点（BFP）格式已经得到了广泛研究 [1]，微缩放（MX）格式通过两级指数表示提供了更大的灵活性：一组 FP 元素共享一个宽位宽指数，而每个低位宽元素仍保留一个较窄的独立指数，以实现更细粒度的缩放。这在保留低位宽 FP 优势的同时，减少了精度损失。MX 格式已在卷积神经网络（CNN）、LLM 和视觉 Transformer（ViT）等工作负载中展现出很高的精度 [2]，因此被开放计算项目（OCP）制定的一项标准规范提案所采用 [5]。

本研究得到 NeuroSoC 项目的部分资助，该项目由 Horizon Europe 资助协议 n° 101070634 提供经费。

低位宽格式使用更少的位来表示模型参数，从而显著降低内存和带宽需求。然而，要充分实现其效率收益，需要对缩放点积提供原生硬件支持，因为这类运算占据了深度学习模型的大部分执行时间。在大型数据中心中，通常会优先减少内存占用和带宽需求，即使这会带来解压缩开销。相比之下，嵌入式人工智能（AI）系统受到严格的功耗和计算资源约束，格式转换的开销可能超过其带来的收益。因此，必须直接以 MX 格式执行运算，才能最大限度地提高效率。

NVIDIA 的 Blackwell Tensor Core 引入了微张量缩放，纳入了社区定义的 MX 格式 [6]。类似地，AMD 的 Versal AI Edge Series Gen 2 在其 AI 引擎中采用了 MX6 和 MX9 格式 [7]。在 RISC-V 生态系统中，Tenstorrent 的 Tensix 核心支持 BFP 格式 [8]，但与 MX 不同，它们仅依赖每个块的共享指数，缺乏 MX 所具有的更细粒度缩放能力（MXINT8 除外）。所有这些解决方案都是专有的，并且与专用张量处理单元紧密耦合。相比之下，我们的方法无需专用张量加速器，就能在 RISC-V 核心上灵活、高效地执行 MX 计算。

现有 RISC-V 架构缺乏对高效执行混合精度 MX 点积的原生支持，因为这需要并发访问多种类型的数据——缩放因子、向量元素和累加器。在软件中处理这些依赖关系，需要频繁进行格式转换和显式缩放操作，从而造成显著的延迟和能耗开销。为解决这些局限，我们提出了 MXDOTP，这是一种专用 ISA 扩展，可将缩放原生集成到点积累加操作中。通过消除冗余的数据搬移和格式转换，MXDOTP 能够以高吞吐量和高能效执行 MX 矩阵乘法。本文的贡献如下：

• 我们设计了一个点积累加单元，支持 MXFP8（E5M2、E4M3）数据格式，采用 8 位缩放因子和 FP32 累加器，以尽量减少溢出和精度损失。

• 我们提出了 MXDOTP，这是一种 RISC-V ISA 扩展，将缩放和点积融合为一条四操作数指令。为持续实现每周期读取四个操作数，我们利用 Snitch 核心 [9] 中的流语义寄存器（SSR）提供块缩放因子，无需修改通常仅有三个读端口的寄存器文件。

• 我们将 MXDOTP 集成到一个 8 核 Snitch 集群中，实现了高达 80 % 的利用率。采用 12 nm FinFET 工艺实现后，其核心级和集群级面积开销分别仅为 11 % 和 5.1 %，空闲时的功耗开销为 1.9 %。

• 我们证明，与基准集群上基于软件的 MX 矩阵乘法相比，该方案实现了 25 倍加速和 12.5 倍能效提升。

## II. 背景

## A. 微缩放（MX）格式

MX 格式是 OCP 提出的一项标准，是针对 AI 优化的低位宽数据格式，由 AMD、Arm、Intel、Meta、Microsoft、NVIDIA 和 Qualcomm 等主要业界厂商共同开发。这些格式旨在提高计算效率、减少内存占用，同时保持深度学习模型的精度。传统的逐张量缩放或 BFP 在大块数据中共享单个指数，而 MX 则允许单个元素或更小的块保留独立的缩放因子，从而提供更细粒度的缩放。这在保持内存效率的同时减少了数值误差。MX 可在推理中直接替代 FP32，并支持大型生成式模型进行低于 8 位的训练，且精度损失可忽略不计 [2]。

符合 MX 规范的格式由三个部分组成：缩放因子、私有元素和块大小。该规范定义了四种具体格式：MXFP8、MXFP6、MXFP4 和 MXINT8。每个包含 k 个元素的块共享一个缩放因子，而元素则采用 FP8（E5M2/E4M3）、FP6（E3M2/E2M3）、FP4（E2M1）或 INT8 编码。缩放因子采用 E8M0 格式编码，块大小固定为 32。

MX 规范将点积运算定义为：

$$
C = D o t (A, B) = X ^ {A} X ^ {B} \sum_ {i = 1} ^ {k} (P _ {i} ^ {A} \times P _ {i} ^ {B})\tag{1}
$$

其中，$X ^ { A }$ 和 $X ^ { B }$ 为缩放因子，$P _ { i } ^ { A }$ 和 $P _ { i } ^ { B }$ 为向量 A 和 B 的私有元素。点积的内部精度和运算顺序由具体实现定义。两个符合 MX 规范的向量的一般点积定义为：

$$
C = \operatorname{DotGeneral} (A, B) = \sum_ {j = 1} ^ {n} \operatorname{Dot} \left(A _ {j}, B _ {j}\right)\tag{2}
$$

其中，A 和 B 均由 n 个大小为 k 的 MX 块组成，按照 MX 规范的建议，结果 C 应为 FP32 数。

在本工作中，我们重点研究 MXFP8 一般点积，以评估 MX 格式在低位宽算术运算中的效率。具体而言，我们采用 E5M2 和 E4M3 这两种 MXFP8 格式，分析它们对功耗、性能和面积的影响，因为对应的 FP8 格式已在深度学习中得到广泛采用，并受到现有 FPU 的支持。

## B. Snitch 集群

Snitch 集群是一种针对浮点工作负载优化的高性能、高能效 RISC-V 多核架构 [10]。它包含八个 RISC-V RV32IMAFD 核，每个核均配备一个 64 位浮点单元（FPU），支持从 FP64 到 FP8 的格式。一个由 32 个存储体组成的 128 KiB 共享 L1 暂存存储器（SPM）通过单周期对数互连连接，提供高带宽、低延迟的访问。一个专用 DMA 控制核负责 SPM 与外部存储器之间的内存传输，通过 512 位交叉开关为数据和指令访问提供 512 位的峰值带宽，而另一个独立的 64 位交叉开关负责外设通信。

该集群具有两项关键的 ISA 扩展。FREP（浮点重复）扩展 [10] 执行预定义的浮点指令序列，消除紧凑算术循环中的分支开销。SSR（流语义寄存器）扩展 [9] 通过支持硬件管理的内存流，消除显式加载/存储指令。SSR 无需发出单独的内存指令，而是根据可配置的仿射访问模式自主读取和存储数据，从而减少地址计算开销和指令数量。每个核具有三个 SSR，可通过编程设置基地址、步长以及最多四个维度的循环边界。配置完成后，SSR 逻辑管理直接传入核寄存器的内存传输，而核则专注于计算。这提高了利用率和整体效率。

## III. 架构

## A. MXDOTP 数据通路及其在 FPU 中的集成

为高效实现 MXFP8 格式的一般点积运算，我们采用了 [3] 中提出的提前累加原理。图 1a 展示了 MXDOTP 数据通路的简化框图。为兼容 E5M2 和 E4M3 这两种 FP8 格式，我们采用 FP9（E5M3）中间格式。提前累加并不将乘积之和转换为 FP32 后再通过专用 FP32 加法器将其加到 FP32 累加器中，而是在最终转换为 FP32 之前，直接以定点形式将移位后的累加器值加到乘积之和上。各元素乘积与累加器值之和采用锚点位于 34 的 95 位定点格式表示，确保其能够容纳八个乘积之和以及移位后的累加器值的完整范围，包括符号位和舍入位。由于 MX 规范将内部精度留给具体实现定义，我们保守地选择了保证结果精确所需的最小位宽。我们实现了 roundTiesToEven（RNE），这是 AI 工作负载中最常用的舍入模式，从而确保符合规范要求；该规范将其他模式列为可选项。

MXDOTP 数据通路作为一个附加运算组集成到 Snitch 集群的 FPU 中<sup>1</sup>。由于 FPU 具有用于支持双精度浮点数的 64 位数据接口，MXDOTP 数据通路能够以每周期一个结果的吞吐量高效计算两个各含八个元素的向量之间的点积，而其延迟取决于参数化的流水线级数。

## B. 指令及其在 Snitch 中的集成

我们将带有 MXDOTP 的 FPU 集成到 Snitch 核中，并通过 MXDOTP 指令扩展 ISA。该指令接收五个输入并产生一个累加输出，详见表 I。与传统的三操作数融合乘加（FMA）指令不同，MXDOTP 需要块缩放因子作为额外输入。然而，Snitch 核及其 FPU 的设计最多支持三个输入操作数，浮点寄存器堆（FP RF）具有三个读端口和一个写端口，符合 RISC-V ISA。虽然增加一个读端口是一种可行的解决方案，但这会引入额外的面积开销（约为 FP RF 面积的 12 %）。因此，我们利用 Snitch 的 SSR 扩展以流式方式传输缩放因子，在不增加面积的同时，还消除了用于加载缩放因子的显式加载指令。

![](images/6bbfbc428ac8c7fac565fdfeba496def914c4bb29c6598b7734438b40c9dc5bf.jpg)  
图 1：（a）简化的 MXDOTP 数据通路（省略符号位）。（b）MXDOTP 在 Snitch 核中的集成。（c）采用三级流水线、跨多个周期执行 MXDOTP。SSR 支持高效的操作数流式传输，确保在无停顿的情况下实现满利用率。

表 I：MXDOTP 指令的操作数。

<table><tr><td>符号</td><td>操作数</td><td>作用</td><td>数据类型</td></tr><tr><td> $P^{A} / P^{B}$ </td><td> $A / B$ 的元素</td><td>输入</td><td>8 × FP8</td></tr><tr><td> $X^{A} / X^{B}$ </td><td> $A / B$ 的缩放因子</td><td>输入</td><td>INT8</td></tr><tr><td> $C$ </td><td>累加器</td><td>输入/输出</td><td>FP32</td></tr></table>

表 II：MXDOTP 指令的编码格式。

<table><tr><td>位</td><td>31-27</td><td>26-25</td><td>24-20</td><td>19-15</td><td>14-12</td><td>11-7</td><td>6-0</td></tr><tr><td>字段</td><td>rs3</td><td>s1</td><td>rs2</td><td>rs1</td><td></td><td>rd</td><td>操作码</td></tr><tr><td>MXDOTP</td><td> $X^{A}$ &amp; $X^{B}$ </td><td>选择</td><td> $P^{B}$ </td><td> $P^{A}$ </td><td></td><td>C</td><td>1110111</td></tr></table>

图 1b 和 c 展示了如何向 FPU 和 MXDOTP 数据通路提供操作数。为在遵守寄存器堆约束的同时高效管理操作数获取，我们使用了三个 SSR。其中两个 SSR 通过 FPU 的两个 64 位输入接口，将 A 和 B 向量直接流式传输到 FPU，第三个 SSR 则流式传输 A 和 B 的两个缩放因子。由于缩放因子需要一个超出标准三操作数 FPU 设计的额外操作数，我们将块缩放因子与 FP32 累加器合并，并通过 FPU 的第三个 64 位输入接口传输它们。最终的 FP32 累加结果随后被写回 FP RF。为支持 MXDOTP，Snitch 核中唯一需要的硬件修改就是将缩放因子与累加器合并。

表 II 详细展示了指令 mxdotp rd, rs1, rs2, rs3, sl 的编码。由于寄存器堆只有三个端口，该指令要求至少有一个操作数通过 SSR 提供。位 26-25 从寄存器 rs3 中选择相应的缩放因子，因为每个 64 位寄存器存储四组缩放因子。为区分 E5M2 和 E4M3 FP8 格式，我们引入了一个专用的控制状态寄存器（CSR），允许在计算前配置格式。

## IV. 结果

## A. 实现设置与面积分析

我们使用 Synopsys Fusion Compiler，在 GLOBALFOUNDRIES 的 12 nm FinFET 工艺下，对采用 MXDOTP 扩展的 Snitch 集群进行综合、布局和布线，最差条件（SS/0.72 V/125 °C）下的目标频率为 0.95 GHz。为维持频率，MXDOTP 单元采用三级流水线寄存器实现。该集群在典型条件（TT/0.8 V/25 °C）下达到了 1.09 GHz，且未引入任何新的关键路径。我们使用 Synopsys 的 Prime-Time 估算典型条件下的功耗，所用的开关活动从 1 GHz 下的布局后门级仿真中提取。我们对从 DeiT-Tiny [11] 中提取的十个样本的功耗取平均值，这些样本使用微软的 MX PyTorch 仿真库<sup>2</sup>量化为 MXFP8。

采用 MXDOTP 扩展核的集群总面积为 4.89 MGE，相比基线 Snitch 集群仅增加 5.1 %。Snitch 核复合体由核本身、其私有指令缓存、SSR 和浮点子系统组成，其中浮点子系统包括 FPU、FREP 逻辑和 FP RF。图 3 展示了这些组件的面积构成。MXDOTP 占 FPU 面积的 17 %，占核复合体面积的 9.5 %。此外，MXDOTP 在空闲时仅给集群带来 1.9 % 的额外功耗。

## B. 软件基准测试设置

为评估性能和能效，我们比较了三种矩阵乘法（MM）内核：（1）FP32，使用标准 32 位浮点乘加（MAC）运算的基线内核；（2）FP8-to-FP32，一种基于软件的 MX 实现，将 FP8 输入转换为 FP32，执行 MAC 运算，并在累加后进行块缩放因子乘法；（3）MXFP8，利用所提出的 MXDOTP ISA 扩展，直接在硬件中计算 MX 点积，块大小仍可通过软件配置。

如图 2 所示，FP32 内核使用双路 SIMD MAC 对 FP32 输入和累加器进行运算，而 FP8-to-FP32 内核则由于 FP8 到 FP32 的格式转换和显式缩放操作而引入了显著开销。相比之下，MXFP8 内核通过在单条指令中执行整个缩放点积，消除了这些开销，并将缩放操作高效地集成到硬件流水线中。

## C. 性能与能效

如图 4 所示，MXDOTP 相较于 FP32 实现了 3.1 至 3.4 倍的加速，相较于 FP8-to-FP32 软件基线实现了 20.9 至 25.0 倍的加速；其能效也分别提升了 3.0 至 3.2 倍和 10.4 至 12.5 倍。MXFP8 内核实现了最高 102 GFLOPS 的吞吐量和 356 GFLOPS/W 的能效，在计入 SSR 和 FREP 配置、循环开销、累加器初始化以及最终结果存储的开销后，达到了理想吞吐量的 79.7 %。

尽管即使没有专用点积单元，MX 也能减少内存占用，但我们的结果表明，如果没有针对通用点积算子的硬件支持，就无法充分实现 MX 在性能与能效方面的优势。与 FP32 相比，MXDOTP 利用较低的位宽，使每个周期能够获取并处理更多数据，从而实现更高的吞吐量。虽然 FP8-to-FP32 同样以这一低位宽运行，但 MXDOTP 通过将缩放和格式转换融合为单个操作，提高了效率。此外，FP8-to-FP32 中频繁的 FP8 到 FP32 格式转换和缩放操作引入了显著的计算开销，使其能效甚至低于 FP32 基线。这些发现表明，专用的 MX 点积硬件对于充分释放 MX 格式在 AI 工作负载中的性能与能效优势至关重要。通过将其实现为 ISA 扩展，我们提供了充分的软件灵活性，支持无缝的混合精度执行。

```txt
FP32
float *A, *B, *C; // FP32 输入/输出矩阵
// 配置 SSR，以流方式传输 A（ft0）、B（ft1）
for (m = 0; m < M; m++):
    for (n = 0; n < N; n+=unroll): // unroll=8
    // 将 SIMD 向量初始化为零
    // 8 次（c0-c7）
    "vfcpka.s.s %[c0], %[zero], %[zero]"
    // 配置 FREP 硬件循环
    "frep.o %[n_frep], %[unroll], 0, 0"
    // 执行乘累加，8 次（c0-c7）
    "vfmac.s %[c0], ft0, ft1"
```

```asm
FP8-to-FP32
char *A, *B; float *C; // FP8 输入、FP32 输出矩阵 uint8_t *Sa, *Sb; // 缩放因子
for (m = 0; m < M; m++):
for (n = 0; n < N; n=unroll): // unroll=8
    // 初始化累加器 fa1、fa2、ft8-ft11（6 条指令）
    "1bu t1, 0(%[Sa])"    // 加载缩放因子 A
    "1bu t2, 0(%[Sb])"    // 加载缩放因子 B
    "add t3, t1, t2"    // 将缩放因子相加
    "addi t3, t3, -127"    // 减去偏置
    "flh ft0, 0(%[A])"    // 加载 2 个 FP8，4 次（ft0-ft3）
    "vfcvt.s.b ft0, ft0"    // 转换为 FP32（A），4 次
    "flh ft4, 0(%[B])"    // 加载 2 个 FP8，4 次（ft4-ft7）
    "vfovt.s.b ft4, ft4"    // 转换为 FP32（B），4 次
    "vfmac.s ft8, ft0, ft1"    // 4 次（ft8-ft11）
    "alli t3, t3, 23" // 将缩放因子移位至指数位
    "fmv.w.x ft0, t3" // 将缩放因子（Sa + Sb）移入浮点寄存器
    // 将结果归约至 fa2（4 条指令）
    "fmadd.s fa1, fa2, ft0, fa1" // 缩放结果

MXFP8 (MXDOTP)
char *A, *B; // FP8 输入矩阵 float *C; // FP32 输出矩阵 uint8_t *S; // 缩放因子
// 重排缩放因子（将 Sa 和 Sb 重排为 S），以供 SSR 流式传输
// 配置 SSR，以流方式传输 A（ft0）、B（ft1）、S（ft2）
for (m = 0; m < M; m++):
for (n = 0; n < N; n=unroll): // unroll=8
// 将 SIMD 向量初始化为零
// 8 次（c0-c7）
    "vfcpka.s.s %[c0], %[zero], %[zero]"
    // 配置 FREP 硬件循环
    "frep.o %[n_frep], %[unroll], 0, 0"
    // 执行 MXDOTP 指令，8 次（c0-c7）
    "mxdotp %[c0], ft0, ft1, ft2, 0"
```  
图 2：FP32、FP8-to-FP32 和 MXFP8 MM 内核的比较。M 和 N 分别表示输出矩阵的行维度和列维度。核心复合体 浮点子系统 Snitch SSRs 其他 表 III：FP8 点积单元（前四行）与计算 273 集群（最后两行）的比较。

<table><tr><td rowspan="62"></td><td rowspan="62">浮点子系统</td><td colspan="5">273</td></tr><tr><td colspan="2">197</td><td>27</td><td>34</td><td>15 kGE %</td></tr><tr><td colspan="2">浮点单元</td><td>10</td><td>12</td><td>5 其他</td></tr><tr><td colspan="2"></td><td>FREP</td><td colspan="2">浮点寄存器堆</td></tr><tr><td colspan="2"></td><td colspan="2"></td><td>其他</td></tr>
<tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr>
<tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr>
<tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr>
<tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr>
<tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr>
<tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr>
<tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td>18</td><td>27</td><td>3 kGE %</td></tr><tr><td colspan="2"></td><td>12</td><td>18</td><td>2 kGE %</td></tr>
<tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr>
<tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr>
<tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr>
<tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr>
<tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td></tr></table>

图 3：扩展了 MXDOTP 的 Snitch 核心的面积构成。

![](images/22e4f0cbd96f9c5a344df4dab8f10c1d4fc0e595da0a7a7da5a43416b7c1306e.jpg)  
图 4：在行数和列数固定为 64、内维度变化时，FP32、FP8-to-FP32 和 MXFP8 MM 内核的 (a) 吞吐量和 (b) 能效。当内维度为 256 时，FP32 无法装入 L1。

## D. 与现有最先进技术的比较

我们在表 III 中将本工作与先前的 FP8 点积单元和计算集群进行比较。首先，我们分析独立的点积单元，对应于表中的前四行。ExSdotp [4] 执行采用 FP16 累加的 FP8 点积，而 Desrentes 等人 [12] 支持精确累加和 FP32 转换。然而，两者均不支持缩放，而 Lutz 等人 [3] 引入了一个 7 位缩放因子。我们的设计在这些方法的基础上进一步引入了两个独立的 8 位缩放因子，全面支持 MX 通用点积。其能效达到 2 TFLOPS/W，优于 ExSdotp。[12] 的能效更高，主要源于三个因素：（1）我们的设计包含额外的缩放操作，而这些操作未计入操作数（OPs）；（2）[12] 报告的是不同频率下的综合估计结果，而我们提供的是布局后的面积和功耗结果；（3）对于 [12]，我们仅报告 E5M2 的结果，因为其设计与 MXDOTP 不同，不支持将 E5M2 和 E4M3 合并在同一单元中。

在集群层面（最后两行），与将 ExSdotp 集成为 ISA 扩展的 MiniFloat-NN [4] 相比，我们的实现具有竞争力的吞吐量和能效。然而，MiniFloat-NN 不包含处理块缩放的专用机制，因此需要额外的软件阶段。相比之下，我们的方法在点积操作中原生集成了块缩放，同时实现了与 MiniFloat-NN 相同的按频率归一化的吞吐量。

<table><tr><td>设计</td><td>工艺 nm</td><td>电压 V</td><td>频率 GHz</td><td>面积 mm $^{2}$ </td><td>缩放支持</td><td>累加格式</td><td>吞吐量* GFLOPS</td><td>能效* GFLOPS/W</td></tr><tr><td>ExSdotp $^{\dagger}$  [4]</td><td>12</td><td>0.8</td><td>1.26</td><td> $5.13 \times 10^{-3}$ </td><td>✗</td><td>FP16</td><td>20.2</td><td>1631</td></tr><tr><td>Desrentes et al. $^{\ddagger}$  [12]</td><td>16</td><td>—</td><td>1.25</td><td> $9.81 \times 10^{-3}$ </td><td>✗</td><td>FP32</td><td>80.0</td><td>11300</td></tr><tr><td>Lutz et al. [3]</td><td>5</td><td>—</td><td>3.6</td><td> $6.74 \times 10^{-4}$ </td><td>√ (1 × 7b)</td><td>FP32</td><td>28.8</td><td>—</td></tr><tr><td>本工作</td><td>12</td><td>0.8</td><td>1.09</td><td> $3.15 \times 10^{-3}$ </td><td>√ (2 × 8b)</td><td>FP32</td><td>17.4</td><td>2035</td></tr><tr><td>MiniFloat-NN [4]</td><td>12</td><td>0.8</td><td>1.26</td><td>0.52</td><td>✗</td><td>FP16</td><td>128</td><td>575</td></tr><tr><td>本工作</td><td>12</td><td>0.8</td><td>1.00</td><td>0.59</td><td>√ (2 × 8b)</td><td>FP32</td><td>102</td><td>356</td></tr></table>

<sup>\*</sup> 1 FLOP = 1 次浮点乘法或 1 次浮点加法。  
<sup>†</sup> 报告的是采用 FP16 累加的 FP8 结果，但该单元也支持采用 FP32 累加的 FP16。<sup>‡</sup> 我们报告的是 E5M2 格式下点积单元（以 250 MHz 综合，假设其可采用四级流水线）与 FP32 压缩单元（以 1.25 GHz 综合）的合并结果。

## V. 结论

我们提出了 MXDOTP，这是一种支持高效执行微缩放（MX）点积的 RISC-V ISA 扩展。通过将专用点积累加单元集成到 Snitch 核中，并利用 SSRs 处理缩放因子，我们实现了对基于 MX 的矩阵乘法的硬件加速。我们的设计采用 12 nm FinFET 工艺实现，达到 102 GFLOPS 和 356 GFLOPS/W，相比基于软件的 MX 矩阵乘法内核，实现了 25 倍加速和 12.5 倍的能效提升。这些结果凸显了对于采用激进块量化的 AI 工作负载，提供专用 MX ISA 和微体系结构支持的必要性。

## REFERENCES

[1] B. D. Rouhani et al., “Pushing the Limits of Narrow Precision Inferencing at Cloud Scale with Microsoft Floating Point,” in NIPS, 2020.

[2] B. D. Rouhani et al., “Microscaling Data Formats for Deep Learning,” 2023. [Online]. Available: http://arxiv.org/abs/2310.10537

[3] D. R. Lutz et al., “Fused FP8 4-Way Dot Product With Scaling and FP32 Accumulation,” in ARITH, 2024.

[4] L. Bertaccini et al., “MiniFloats on RISC-V Cores: ISA Extensions With Mixed-Precision Short Dot Products,” IEEE Trans. on Emerging Topics in Computing, vol. 12, no. 4, pp. 1040–1055, 2024.

[5] B. D. Rouhani et al., “OCP Microscaling Formats (MX) Specification,” 2023. [Online]. Available: https://www.opencompute.org/documents/ ocp-microscaling-formats-mx-v1-0-spec-final-pdf

[6] “NVIDIA Tensor Cores: Versatility for HPC & AI.” [Online]. Available: https://www.nvidia.com/en-us/data-center/tensor-cores/

[7] “AMD Versal™ AI Edge Series Gen 2 Product Brief.” [Online]. Available: https://www.amd.com/content/dam/amd/en/documents/products/ adaptive-socs-and-fpgas/versal/versal-ai-edge-gen2-product-brief.pdf

[8] “Data Formats and Math Fidelity — TT Buda documentation.” [Online]. Available: https://docs.tenstorrent.com/pybuda/latest/dataformats.html

[9] F. Schuiki et al., “Stream Semantic Registers: A Lightweight RISC-V ISA Extension Achieving Full Compute Utilization in Single-Issue Cores,” IEEE Trans. Comput., vol. 70, no. 2, pp. 212–227, 2021.

[10] F. Zaruba et al., “Snitch: A Tiny Pseudo Dual-Issue Processor for Area and Energy Efficient Execution of Floating-Point Intensive Workloads,” IEEE Trans. Comput., vol. 70, no. 11, pp. 1845–1860, 2021.

[11] H. Touvron et al., “Training data-efficient image transformers & distillation through attention,” in ICML, 2021.

[12] O. Desrentes et al., “Exact Dot Product Accumulate Operators for 8-bit Floating-Point Deep Learning,” in DSD, 2023.

> 来源与版本：使用用户提供的 arXiv:2505.13159v1 作者稿，正式会议为 ASAP 2025；实际实现以 MXFP8 为重点，不将标准的所有 MX 类型都视作已实现。

> 本文根据用户提供的 PDF 经 MinerU 解析，由 Codex/GPT 翻译和解读，未调用 GLM 翻译 API。参考文献保留原文；技术总结、通俗讲解和图表说明属于解读。[原文入口](https://doi.org/10.1109/ASAP65064.2025.00021)。
