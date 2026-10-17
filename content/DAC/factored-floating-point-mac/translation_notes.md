# Factored Floating-Point MAC: Eliminating Redundant Normalization for Energy-Efficient Systolic Arrays 原文翻译

# 分解式浮点 MAC：消除冗余规格化以实现高能效脉动阵列

Hao Xiong<sup>∗</sup> 国防科技大学计算机科学与技术学院 中国长沙

Zhichao Rui<sup>∗</sup> 国防科技大学计算机科学与技术学院 中国长沙

Rui Gong<sup>†</sup> 国防科技大学计算机科学与技术学院 中国长沙

Zhijie Yang 军事科学院国防科技创新研究院 启元实验室 中国北京

Lei Wang<sup>†</sup> 军事科学院国防科技创新研究院 启元实验室 中国北京

## 摘要

浮点乘累加（FPMAC）在科学计算和人工智能中至关重要，但仍是主要的性能与能耗瓶颈。传统的 IEEE-754 浮点融合乘加（FP-FMA）方案针对单次乘加融合进行优化，却忽略了 GEMM 中的长链 MAC 模式，导致冗余的舍入和规格化、数值误差累积，以及宽位进位传播加法器（CPA）带来的高延迟和高功耗开销。我们提出分解式 FPMAC，在脉动阵列（SA）内部使用 4:2 进位保存加法器（CSA）和基于 CSA 的冗余表示，从处理单元中消除 CPA，并将规格化推迟到统一的后处理阶段。为防止延迟规格化情况下的累加中间值溢出（AIVO），我们引入了一种轻量级分层风险感知边界保护机制。为进一步降低冗余表示带来的寄存器开销，我们以阵列共享缓冲池替代每个 PE 的双缓冲。实验表明，F-FPMAC 将关键路径延迟降低了 61.9%，功耗降低了 22.6%，能效提升了 3×，误差降低了两个数量级，溢出次数减少了 44.5%。

## CCS 概念

• 计算机系统组织 → 脉动阵列。

## 关键词

乘累加，浮点，规格化，脉动阵列

## ACM 参考文献格式：

Hao Xiong, Zhichao Rui, Zhijie Yang, Rui Gong, and Lei Wang. 2026. Factored Floating-Point MAC: Eliminating Redundant Normalization for Energy-Eficient Systolic Arrays. In 63rd ACM/IEEE Design Automation Conference (DAC ’26), July 26–29, 2026, Long Beach, CA, USA. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3770743.3804226

## 1 引言

浮点通用矩阵乘法（GEMM）是科学计算和人工智能中的关键算子[3, 4, 17]，不断增长的计算需求使效率变得至关重要。然而，当前的浮点 GEMM 加速器依赖经典 FMA 设计[9, 15, 16]，在一次计算中完成乘法、加法、舍入和规格化，以确保数值精度，却忽略了累加层面的优化。然而，这种即时规格化策略在长 MAC 链中带来了三个瓶颈：（1）宽位 CPA 的关键路径延迟随位宽线性增加，限制了吞吐量和频率；（2）频繁的舍入和规格化会累积误差，并引发过早规格化诱发的溢出（PNIO）；（3）规格化逻辑带来了显著的面积和功耗开销[1]。

现有工作大多关注单个 MAC 的电路级改进，缺乏针对连续 FPMAC 运算的系统级协同设计。Ullah [7] 将基于 Radix-8 Booth 编码的乘法器应用于脉动阵列。Zhang [28] 通过将累加操作数编码为部分积以进行统一处理，优化了整数 MAC，从而消除了传统 MAC 中的传统加法器开销。Inayat 和 Chung[6] 提出了一种 CPA 分离式脉动阵列（CPA-FSA）架构，使用 CSA 替代 CPA，探索了架构与 MAC 之间的整体优化。然而，他们仅讨论了整数精度。Li[13] 提出了一种向量浮点脉动阵列，使用加法器树优化累加，但其本质上仍是一维阵列，适用场景有限。Liu[14] 提出了一种三维可扩展浮点脉动阵列，通过允许在不进行中间规格化的情况下执行有限次数的连续 FPMAC 运算，减少冗余的舍入和规格化，其中次数上限由硬件的物理层数决定。然而，这种方法既没有完全消除规格化，也没有提供一种机制，以保证在极端情况下长时间不进行规格化的 FPMAC 运算的正确性。

为应对这些挑战，我们提出了分解式 FPMAC（F-FPMAC），这是一种针对连续 FPMAC 运算进行协同优化的阵列与 MAC 架构。我们的主要贡献可概括如下：

• 我们采用 CSA 冗余表示来表示中间结果，并在每个 PE 中使用一个 4:2 CSA 来压缩乘积和累加操作数，从而消除脉动阵列中的 CPA。这消除了累加过程中的进位传播，显著缩短了关键路径。

![](images/3e5676e21ebb5cab2150fd8d764e8eec34701ebf8e324aff5537ac67134c7193.jpg)  
图 1：（a）朴素 FPMAC（N-FPMAC）架构（即双 CPA），（b）常规 FPMAC（C-FPMAC）架构（即单 CPA），（c）分解式 FPMAC（F-FPMAC）架构，以及（d）SA 架构。

• 我们提出后置规格化，将冗余的舍入和规格化操作移至阵列边缘的专用后处理列。这缩短了关键路径，同时保留更多有效位，以尽量减小累积舍入误差。

• 我们引入一种低开销的分层风险感知边界保护机制，确保结果在后置规格化引入 AIVO 的情况下仍然正确。

• 为缓解 CSA 冗余表示和后置规格化带来的寄存器开销增加，我们采用阵列共享缓冲池和优化的数据流来替代每个 PE 的双缓冲，从而降低寄存器成本。

## 2 背景与动机

## 2.1 脉动阵列架构

脉动阵列由规则排列、通过最近邻互连连接的处理单元（PE）组成，凭借其并行计算能力、高数据复用率和极少的片上通信，成为现代 TPU 的首选[26]。脉动阵列通常采用三种数据流方案：输入驻留（IS）、输出驻留（OS）和权重驻留（WS）[10, 19]。在 OS 中，部分和保持驻留，而输入和权重在阵列中传播。在 WS/IS 中，权重或输入被预加载为驻留操作数，而其他操作数和部分和则穿过阵列进行累加。

![](images/d010032251e0f8aea4df5f0a377e8ffb91ef13edba5cd7bbee4dce58e338df19.jpg)  
图 2：（a）C-FPMAC 和（b）F-FPMAC 中操作数尾数的有效位。

## 2.2 浮点乘加融合

标准浮点融合乘加（FP-FMA）单元包含三个阶段：预处理、计算和后处理。预处理通过提取符号、指数和尾数字段、恢复隐含位并进行基于指数的对齐来对齐操作数。计算阶段执行尾数乘法和加法。后处理通过前导零检测、指数调整和舍入对结果进行规格化，然后将其重新组装为 IEEE-754 格式。

图 1(a) 展示了一种朴素 FPMAC（N-FPMAC）设计。a 和 b 的尾数通过 Booth 编码生成部分积（PP），并压缩为和项与进位项。进位传播加法器（CPA）将这些项合并为乘积，随后通过第二个 CPA 将该乘积与对齐后的 c 的尾数进行累加。由于 CPA 的延迟随位宽增长，因此它主导了 MAC 单元的最大逻辑传播延迟（t <sub>pd</sub>）[2]。为缓解这一问题，图 1(b) 中的传统 FPMAC（C-FPMAC）设计采用一个 3:2 CSA，将乘法器的和项、进位项以及 c 的尾数压缩为两个冗余中间值，再在输出阶段通过一个 CPA 将其合并。然而，剩余的宽位宽 CPA 仍然是主要的延迟瓶颈[2]。

图 2(a) 表明，在连续 FPMAC 运算中，FMA 在每次乘加步骤后执行的多余舍入和规格化会显著降低中间结果的精度。这一架构局限不仅会引入过早规格化引发的溢出（PNIO）——一种可避免的伪溢出，即尽管最终结果仍然有效，中间值却暂时超出了可表示范围——还会放大累积误差。这种不必要的精度损失和可避免的异常凸显了传统逐步规格化架构固有的低效性。

## 2.3 采用流水线 MAC 的 OS 中的冒险

对于采用 n 级流水线 FPMAC 的 m × m SA，WS/IS 和 OS 的输出延迟分别由式 (1) 和式 (2) 给出。OS 数据流的端到端延迟更短，具有明显的时序优势。

![](images/6ef76de089f4f3441c43dfc53ec0abf93e34dc9d1d0d3e562e6834bf084cf6cc.jpg)  
图 3：采用 FP-FMA 的 (a) WS 和 (b) OS SA 中 PE 内的流水线操作。

SCALE-Sim[20, 21] 仿真证实，在阵列规模相同且采用非流水线 MAC 单元的理想条件下，OS 通常能在多种工作负载下实现更高的能效。

$$
(m \times n) + (m - 1)\tag{1}
$$

$$
m + (m - 1) + (n - 1)\tag{2}
$$

然而，在实际硬件中，FP-MAC 的关键路径较长，因此必须采用多级流水线才能满足时序约束。标准 IEEE-754 FMA 架构仅针对独立的乘加运算进行了优化，而非长链连续 MAC。因此，使用流水线 FPMAC 执行 OS 数据流会引入写后读（RAW）冒险。如图 3 所示，这些冒险的产生是因为 OS 执行本地乘累加，而 WS 和 IS 通过执行相互独立的乘加运算，并在 PE 之间传递中间结果以进行累加，从而避免这些冒险。因此，WS 和 IS 模式对 MAC 流水线深度不敏感，并且能够避免此类 RAW 冒险。

一种直接的缓解方法是在每个 PE 中配置额外的累加器寄存器，通过交替写入来解决 RAW 冲突，随后使用专用加法器合并中间结果。然而，这种方案会产生大量硬件开销，削弱 OS 数据流的能效。因此，必须对 FPMAC 单元和脉动阵列架构进行协同优化，才能高效支持长链 MAC 运算。

## 3 提出的分解式 FPMAC 方案

本节介绍所提出的 F-FPMAC 架构，该架构面向长链 MAC，并通过架构分解和数据流重构提高效率。我们的核心思路是：(1) 重新设计移位对齐策略，以防止位宽过度扩展；(2) 以冗余表示保留中间结果，并消除 SA 内部的宽位宽 CPA；(3) 在阵列外部设置统一的后处理阶段，先合并 CSA 冗余表示中的和与进位，再执行舍入和规格化；(4) 采用一种轻量机制，确保在不进行规格化的连续 FPMAC 运算中保持正确性。

## 3.1 串行尾数移位策略

在 C-FPMAC 中，通常通过对 c 的尾数进行移位（左移或右移）来完成指数对齐。这可以尽量减少移位器的数量，并使移位与乘法并行进行，以减轻关键路径压力。然而，左移操作可能显著增加 c 的尾数位宽，从而需要额外硬件来处理扩展的高位。此外，$( a \times b ) _ { s u m } , ( a \times b ) _ { c a r r y } ,$ 与 c 的尾数之间的位宽不匹配也会使后续处理更加复杂。

为解决这一问题，F-FPMAC 对指数较小的操作数进行右移——该操作数可能是 c 的尾数，也可能是乘积（$( ( a \times b ) _ { s u m }$ 和 $( a \times b ) _ { c a r r y } )$）。这一设计实现了三项关键优化。首先，尽管增加了移位器，但它消除了处理左移所扩展高位的需要，并减少了规格化解耦后所存储中间结果的位宽，从而缓解过大的寄存器开销。其次，尽管移位对齐不再与乘法并行进行，但修改后的计算流程整合了对操作数 c 的处理，防止其过早进入流水线，如图 1(b) 和 (c) 中的红色路径所示。最后，它统一了 $( a \times b ) _ { s u m } , ( a \times b ) _ { c a r r y } ,$ 与 c 的尾数的位宽，通过一致的数据格式为后续优化阶段提供便利。

## 3.2 中间结果的冗余表示

3:2 CSA 在不进行进位传播的情况下，将三个输入压缩为两个冗余输出（和、进位）[5, 24]。由于没有进位链，CSA 的延迟与操作数位宽无关，而硬件成本则随位宽线性增长[25]。

在 FP32 中，尾数包含 25 位（23 位显式位、1 位隐含位、1 位符号位）。为适应 Booth 编码乘法器对偶数位宽的要求，尾数被扩展为 26 位。因此，两个尾数的乘积采用定点 6.46 格式表示，其中有 6 位带符号整数和 46 位小数，而 C-FPMAC 中剩余的 52 位 CPA 主导了关键路径，成为主要的频率瓶颈。

为消除这一瓶颈，我们改变了传统 FMA 的执行范式，即先将 $( a \times b ) _ { s u m }$ 与 $( a \times b ) _ { c a r r y }$ 相加，再与 c 进行累加。中间值在每个 PE 内以 CSA 冗余格式保留，将和与进位的合并推迟到阵列外部的专用 CPA 中进行。一旦 $( a \times b ) _ { s u m } , ( a \times b ) _ { c a r r y } .$ 与 c 的尾数的操作数位宽得到统一，就可以用一个 4:2 CSA 替代 C-FPMAC 中的一个 3:2 CSA 与一个 CPA 的组合。它接收四个位宽相同的操作数——乘法器的两个输出（$( ( a \times b ) _ { s u m } , ( a \times b ) _ { c a r r y } )$）以及反馈寄存器（c<sub>in\_sum</sub>、c<sub>in\_carry</sub>）——并生成更新后的冗余对（$c_{\mathrm{out\_sum}}$、$c_{\mathrm{out\_carry}}$），如图 1(c) 所示。这消除了 PE 内部的宽位宽 CPA，并且尽管 4:2 CSA 在逻辑上等效于两个级联的 3:2 CSA，其关键路径延迟却仅增至 1.5×，而非 2×，从而显著降低了 $t _ { p d }$

## 3.3 后置规格化与分层风险感知边界保护

如图 4 所示，规格化模块占 C-FPMAC 面积的 31% 和功耗的 7%。C-FPMAC 在每次 MAC 后执行规格化，导致冗余的硬件开销和功耗。因此，如图 1(c) 和 (d) 所示，F-FPMAC 将规格化与 MAC 解耦，并将其移至阵列外部，仅在最终输出时执行一次。这不仅显著降低了片上面积和动态功耗，还充分利用了图 2 中被 FP-FMA 忽略的中间结果有效位，从而提高了整体计算精度。

![](images/82fc91fc4f6b0130120afc8a5686eaad6af6e57a7249975f26edfc0dc30a0407.jpg)  
图 4：C-FPMAC 各子模块的 LUT 和功耗。

然而，在指数和尾数相近的连续 FPMAC 运算中，F-FPMAC 的中间结果可能持续增长，最终超出有符号 6.46 定点格式的可表示范围。这一数值问题称为累积中间值溢出（AIVO），而在 C-FPMAC 中不会出现，因为其每一步计算都包含带指数缩放的规格化操作。为保留规格化解耦带来的硬件简化优势，同时确保数值安全、避免 AIVO，我们提出了算法 1 所示的分层风险感知边界保护机制。该机制通过一个对高风险小指数差作出快速响应的主计数器，以及一个捕捉中等指数差长期累积效应的从计数器，动态调整累积结果的指数。我们使用指数差 $\Delta = E x p ( c _ { i n } ) - E x p ( a \times b )$ 作为风险判据：

• 高风险：当 $- 1 \leq \Delta \leq 1$ 时，几乎相等的指数和相互重叠的尾数会增大扩展风险。在最坏情况下（a、b 和 c 的尾数各位均为 1，且指数相同），$a \times b \approx 4$，$c \approx 2 .$。在 6.46 格式约为 $( - 3 3 , 3 3 )$ 的范围内，连续七次 MAC 得到的结果约为 30，而第八次将导致 AIVO。因此，使用阈值为 7 的主计数器来跟踪这种情况。

• 中风险：当 $2 \leq \Delta \leq 4 ,$ 较小的操作数会移位 2-4 位，其对结果的贡献仅为 6.25%-25%，这意味着在极端情况下，四次中风险事件相当于一次高风险事件。虽然单次更新的影响有限，但反复出现这种情况仍可能使尾数逐渐增长。在此范围内，从计数器递增；当其达到 4 时，将主计数器递增一次，并将自身清零。

• 低风险：当 $\Delta > 4 ,$ 对齐后的较小操作数的贡献低于 1/16，因此可以忽略，无需更新计数器。

• 风险重置条件：当 $\Delta < - 1$ 时，新生成的 $( a \times b ) _ { s u m }$ 和 $( a \times b ) _ { c a r r y }$ 主导累积结果的量级，而先前累积的 $c _ { i n \_ s u m }$ 和 $c _ { i n \_ c a r r y }$ 被右移并衰减至低有效位。因此，其增长趋势被有效终止，不再对潜在的 AIVO 产生贡献。故将此条件视为风险重置，并将两个计数器均清零。

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
算法 1 对齐、4:2-CSA 累加与分层风险感知边界保护。

输入： $(a \times b)_{\text{sum}}$ ,  $(a \times b)_{\text{carry}}$ ,  $c_{in\_sum}$ ,  $c_{in\_carry}$ , Exp $(a \times b)$ , Exp $(c_{in})$ 

输出： $c_{out\_sum}$ ,  $c_{out\_carry}$ , Exp $(c_{out})$ 

1: 静态变量（跨周期保持）：
MasterBoundCnt, SlaveBoundCnt ← 0

2:  $\Delta \leftarrow \text{Exp}(c_{in}) - \text{Exp}(a \times b)$ 

3: 若 $\Delta \leq 0$ 则

4:  $E_{shift} \leftarrow -\Delta$ 

5: Max_exp ← Exp $(a \times b)$ 

6:  $Sum_{to\_shift}$ ,  $Carry_{to\_shift} \leftarrow c_{in\_sum}$ ,  $c_{in\_carry}$ 

7:  $Sum_{unshifted}$ ,  $Carry_{unshifted} \leftarrow (a \times b)_{\text{sum}}$ ,  $(a \times b)_{\text{carry}}$ 

8: 否则

9:  $E_{shift} \leftarrow \Delta$ 

10: Max_exp ← Exp $(c_{in})$ 

11:  $Sum_{to\_shift}$ ,  $Carry_{to\_shift} \leftarrow (a \times b)_{\text{sum}}$ ,  $(a \times b)_{\text{carry}}$ 

12:  $Sum_{unshifted}$ ,  $Carry_{unshifted} \leftarrow c_{in\_sum}$ ,  $c_{in\_carry}$ 

13: 结束条件判断

14:  $Sum_{shifted} \leftarrow Sum_{to\_shift} \gg E_{shift}$ 

15:  $Carry_{shifted} \leftarrow Carry_{to\_shift} \gg E_{shift}$ 

(Sumout,  $Carry_{out}$ ) ← 4:2CSA(Sumshifted,  $Carry_{shifted}$ ,

16:  $Sum_{unshifted}$ ,  $Carry_{unshifted}$ )

17: 若 $-1 \leq \Delta \leq 1$ 则

18: MasterBoundCnt ← MasterBoundCnt + 1

19: 否则，若 $2 \leq \Delta \leq 4$ 则

20: SlaveBoundCnt ← SlaveBoundCnt + 1

21: 若 SlaveBoundCnt == 4 则

22: MasterBoundCnt ← MasterBoundCnt + 1

23: SlaveBoundCnt ← 0

24: 结束条件判断

25: 否则，若 $\Delta < -1$ 则

26: MasterBoundCnt, SlaveBoundCnt ← 0

27: 结束条件判断

28: 若 MasterBoundCnt == 7 则

29:  $c_{out\_sum} \leftarrow Sum_{out} \gg 4$ 

30:  $c_{out\_carry} \leftarrow Carry_{out} \gg 4$ 

31: Exp $(c_{out}) \leftarrow Max\_exp + 4$ 

32: MasterBoundCnt, SlaveBoundCnt ← 0

33: 否则

34:  $c_{out\_sum} \leftarrow Sum_{out}$ 

35:  $c_{out\_carry} \leftarrow Carry_{out}$ 

36: Exp $(c_{out}) \leftarrow Max\_exp$ 

37: 结束条件判断
</div>

• 保护触发：当主计数器达到 7 时，认为累积风险过高。为确保前文分析的最坏情况下的尾数（≈ 30）以最小精度损失返回规格化区间 [1,2)，我们执行保护性右移：将 $c _ { o u t }$ 的和与进位右移 4 位，将指数增加 4，并重置两个计数器。

在不进行规格化的情况下，该机制可在极端情况下防止 6.46 格式中的 AIVO，并避免导致精度损失的过度补偿右移。总体上仅增加了两个计数器和一些比较逻辑，无需复杂的可变移位器。该机制以极小的硬件开销，确保了规格化解耦后的数值正确性。

## 4 架构

## 4.1 流水线计算阵列

图 1(d) 展示了所提出的阵列架构。作为基本计算单元，F-FPMAC 由六个功能组件构成：提取单元、PPs\_Generator、压缩树、

![](images/fd9aef0d7f88b3859298ce8b4bffc4b31e8f4f82be681ae3fa7dbd9844bb01f0.jpg)  
图 5：GEMM 在 (a) WS 阵列和 (b) OS 阵列中的 M、N、K 维度。

对齐器、4:2 CSA 和异常处理单元。与图 1(b) 所示的四级流水线 C-FPMAC 相比，图 1(c) 所示的 F-FPMAC 在达到相同频率的同时，将流水线深度减至仅两级。

该设计采用 OS 数据流。在两级流水线中，c 的输入与最终累积结果的输出位于同一级，从而有效消除了 RAW 冒险。此外，如图 5 所示，OS 数据流能够在不增加物理阵列尺寸的情况下沿 K 维无缝扩展，支持任意次数的连续 MAC，且仅需一次最终的后置规格化。这充分利用了 F-FPMAC 的低动态功耗特性，显著提高了能效。相比之下，WS 的 K 维受到由阵列尺寸决定的硬件限制，因而从根本上限制了可扩展性。

## 4.2 后处理模块

该阶段包含两个依次执行的操作：(1) 一个专用的 CPA 列，用于合并冗余进位保存表示中的和与进位；(2) 一列归一化模块，执行舍入和归一化，以生成最终的 IEEE-754 输出。需要强调的是，该阶段的硬件成本由整个处理阵列分摊。随着阵列规模增大，其边际开销变得可以忽略。因此，所提出的设计不仅最大限度地减小了面积占用，还提高了功耗效率。

## 4.3 阵列共享缓冲池

图 6 展示了脉动阵列在周期 n、n+1 和 n+2 的计算状态分布：橙色区域表示已经计算完成但需要暂存的部分和，蓝色区域表示当前周期计算得到的结果，黄色区域表示当前的有效输出，灰色区域表示仍在计算中的结果。利用脉动阵列的对角线特性，只需存储橙色三角形区域，即可保证连续运行。

传统 PE 通常采用双缓冲架构，并依靠乒乓调度将数据搬移与计算解耦。然而，这不仅引入了硬件冗余，而且由于引入 CSA 的冗余表示，并为后归一化保留完整的 52 位中间结果，还增加了寄存器数量，导致面积和功耗上升。为抑制这种增长，我们使用单组 PE 内部寄存器，以及一个用于存储橙色区域的阵列共享缓冲池。此外，通过将缓冲池置于后处理阶段之后，中间值在存储前就从 CSA 冗余格式转换为归一化的标准 IEEE-754 结果，从而最大限度地减少寄存器需求。

![](images/6198838875fa2d97cce2df2d1e5aa6459ad2a633e318f3f3c6bfa164af805d1b.jpg)  
图 6：SA 中的结果在周期 (a) n、(b) n+1 和 (c) n+2 的状态

## 5 评估

## 5.1 精度

为评估所提出的 F-FPMAC 设计在连续 MAC 运算中的精度与鲁棒性，我们采用了开源并行超低功耗（PULP）平台[18]中基于 [12, 22, 23] 的单精度 FPMAC，将其作为 C-FPMAC 的参考，并据此开发了 F-FPMAC。在表 1 和表 3 中，这两种设计分别记为 C 和 F。我们使用两个数据集评估精度：

• 正常范围测试：从 LLaMA-7B 和 ResNet-50 的真实中间数据中提取 200,000 个样本，范围为 $1 0 ^ { - 6 }$ 至 $1 0 ^ { 6 } ;$，在此范围内不会发生溢出。我们以双精度（FP64）结果作为真值参考，评估四种设计在 FP32 执行下的数值精度：(1) 不使用 FMA 的 x86，使用 GCC -O2 编译；(2) 使用 FMA 的 x86，使用 GCC -O2 -mfma 编译；(3) C-FPMAC；(4) F-FPMAC。我们使用绝对误差和相对误差分析计算结果，两者分别反映计算结果与参考结果之间的实际差值，以及绝对误差与参考结果的比值。

• 极端范围测试：随机生成 200,000 个样本，覆盖 $1 0 ^ { - 1 9 }$ 至 $1 0 ^ { 1 9 }$ 的范围，用于评估抗溢出能力。我们测量了 C-FPMAC 和 F-FPMAC 的溢出比例。

在正常范围测试中（图 7），不使用 FMA 的 x86 误差最高。相比之下，使用 FMA 的 x86 和 C-FPMAC 均严格遵循 IEEE-754 FMA 标准，达到了相同且更低的误差。然而，由于舍入和归一化误差的累积，x86（FP32）和 C-FPMAC 的误差均随着 MAC 次数增加而增大。相反，F-FPMAC 通过消除中间舍入并保留有效位，使误差保持稳定，且低近两个数量级。

![](images/f53531c3cddc40d60825cd56ccc4676241c01f078f19c495c37bf9ee2705f00c.jpg)

![](images/33da20300e64026679f219fbac77710bee05597e9e17b90e1543370c23906fa7.jpg)  
图 7：x86、x86-fma、C-FPMAC 和 F-FPMAC 在不同 MAC 次数下的 (a) 绝对误差和 (b) 相对误差。

表 1：不同 MAC 次数下的溢出统计。

<table><tr><td>迭代次数</td><td>两者均溢出(%)</td><td>仅 C 溢出(%)</td><td>仅 F 溢出(%)</td><td> $\frac{onlyC}{Both+onlyC+onlyF}$ (%)</td></tr><tr><td>10</td><td>0.14</td><td>0.03</td><td>0</td><td>17.65</td></tr><tr><td>20</td><td>2.31</td><td>0.98</td><td>0</td><td>29.79</td></tr><tr><td>30</td><td>6.22</td><td>3.65</td><td>0</td><td>36.98</td></tr><tr><td>50</td><td>14.82</td><td>10.22</td><td>0</td><td>40.81</td></tr><tr><td>100</td><td>30.96</td><td>24.8</td><td>0</td><td>44.48</td></tr></table>

表 在极端范围测试中（表 1），随着 MAC 运算次数增加，F-FPMAC 和 C-FPMAC 的溢出发生次数均有所增加。然而，没有出现仅 F-FPMAC 溢出的情况。随着累加次数增加，F-FPMAC 减少的溢出所占比例从 17.7% 上升到 44.5%，这可归因于后归一化减少了 PNIO。总体而言，与 C-FPMAC 相比，F-FPMAC 在持续 MAC 工作负载中实现了显著更高的精度和更强的抗溢出能力。

## 5.2 性能

我们在 Xilinx Zynq UltraScale+ VCU118 FPGA 上实现了 C-FPMAC 和 F-FPMAC 两种架构。如表 2 所示，与 C-FPMAC 相比，所提出的 F-FPMAC 将关键路径延迟降低了 61.9%，功耗降低了 22.6%，但在非流水线配置下寄存器数量有所增加。然而，由于浮点乘加（FP MAC）固有的关键路径较长，通常需要采用流水线才能达到实际应用所需的工作频率。相比之下，F-FPMAC 只需显著更少的流水级即可达到相同的目标频率，从而减少流水线实现中的寄存器总数。这表明其在能效和资源利用方面均具有明显优势。

表 3 将所提出的架构与以往基于 FPGA 的设计 [8, 11, 13, 14, 16, 27] 进行了比较。所有设计均支持四路并行 FP32 运算。为保证公平，实例化四个非流水线 C-FPMAC 单元组成阵列，而四个非流水线 F-FPMAC 单元与一个后处理单元组成一个 1 × 4 脉动阵列。

在 FP32 工作负载下，针对低功耗进行优化且不支持多精度的基线 C-FPMAC 实现了较高的能效，仅次于 [13]。在此基线基础上，所提出的 F-FPMAC 将能效提升至约 3×。尽管 [14] 认识到了冗余规格化导致的动态功耗开销，但其缓解措施依赖于大深度堆叠才能接近峰值能效（50 层时为 5.537 GFLOPS/W）。相比之下，我们的设计完全消除了阵列内部的冗余规格化，从设计本身实现了能效提升。重要的是，这些改进无需修改内部乘法器或加法器即可实现，因此我们的设计与现有的多精度和脉动阵列优化技术相互正交且互为补充。 s

表 2：资源与性能比较

<table><tr><td>设计</td><td>C-FPMAC</td><td>F-FPMAC</td><td>后处理</td></tr><tr><td>查找表（LUT）</td><td>2440</td><td>1724</td><td>433</td></tr><tr><td>触发器（FF）</td><td>36</td><td>117</td><td>36</td></tr><tr><td>延迟 (ns)</td><td>12.075</td><td>4.593</td><td>3.597</td></tr><tr><td>功耗 (W)</td><td>0.062</td><td>0.048</td><td>0.025</td></tr></table>

表 3：与以往基于 FPGA 的工作的比较

<table><tr><td>设计</td><td>[8]</td><td colspan="2">[27]</td><td>[16]</td><td>[13]</td><td>[11]</td><td>[14]</td><td>C</td><td>F</td></tr><tr><td>Xilinx 系列</td><td>Virtex5</td><td>Virtex5</td><td>Virtex7</td><td>US+</td><td>US+</td><td>US+</td><td>US+</td><td>US+</td><td>US+</td></tr><tr><td>频率 (MHz)</td><td>336.0</td><td>348.2</td><td>427.4</td><td>179.9</td><td>164.5</td><td>216.5</td><td>173.2</td><td>82.8</td><td>217.7</td></tr><tr><td>功耗 (W)</td><td>0.250</td><td>0.248</td><td>0.189</td><td>0.378</td><td>0.483</td><td>0.296</td><td>0.346</td><td>0.248</td><td>0.217</td></tr><tr><td colspan="10">吞吐量 (GFLOPS)</td></tr><tr><td>INT8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>4.114</td><td>12.47</td><td>-</td><td>-</td></tr><tr><td>FP16</td><td>-</td><td>-</td><td>-</td><td>3.597</td><td>5.260</td><td>1.949</td><td>4.157</td><td>-</td><td>-</td></tr><tr><td>FP32</td><td>0.192</td><td>0.232</td><td>0.284</td><td>0.899</td><td>1.314</td><td>0.433</td><td>1.386</td><td>0.662</td><td>1.742</td></tr><tr><td>FP64</td><td>0.056</td><td>0.077</td><td>0.094</td><td>0.180</td><td>0.328</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="10">能效 (GFLOPS/W)</td></tr><tr><td>INT8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>13.90</td><td>36.04</td><td>-</td><td>-</td></tr><tr><td>FP16</td><td>-</td><td>-</td><td>-</td><td>9.516</td><td>10.89</td><td>6.584</td><td>12.01</td><td>-</td><td>-</td></tr><tr><td>FP32</td><td>0.768</td><td>0.936</td><td>1.507</td><td>2.379</td><td>2.72</td><td>1.463</td><td>4.01</td><td>2.671</td><td>8.026</td></tr><tr><td>FP64</td><td>0.224</td><td>0.312</td><td>0.502</td><td>0.476</td><td>0.68</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

![](images/63fb05208fe6cbe4ebc521752a94dcec33c77787ad24133ca34d1fbc5767913b.jpg)  
图 8：8 × 8 大小阵列中各类 FPMAC 和缓冲区的 LUT 与 FF 资源使用情况。“-OS”表示 OS 数据流，“-pool”表示使用缓冲池。

如图 8 所示，所提出的 F-FPMAC-OS 的硬件占用显著小于 N-FPMAC-OS，这主要得益于其更紧凑的计算结构，以及对额外冒险管理逻辑的消除。此外，相较于 F-FPMAC-OS 设计，F-FPMAC-OS-pool 通过以阵列共享的缓冲池架构替代 PE 中的双缓冲机制，进一步减少了寄存器用量，从而降低了总体面积和存储开销。

## 6 结论

Factored-FPMAC 通过以冗余形式保留中间结果，并将规格化推迟到专用后处理阶段，消除了脉动阵列中的冗余规格化。借助分层边界保护和阵列共享缓冲，我们的架构将关键路径延迟降低了 61.9%，功耗降低了 22.6%，能效提升至 3×，同时显著提高了数值精度。这种协同设计方法能够为科学计算和 AI 工作负载提供高能效的浮点加速。

## 致谢

本研究的部分工作得到了国家自然科学基金（资助编号：62406335 和 62372461）、国家国防科技工业局国防重点实验室项目（资助编号：WDZC20235250112）以及先进微处理器芯片与系统重点实验室的支持。

## References

[1] Kosmas Alexandridis, Christodoulos Peltekis, Dionysios Filippas, and Gior gos Dimitrakopoulos. 2024. Floating-Point Multiply-Add with Approximate Normalization for Low-Cost Matrix Engines. arXiv:2408.11997 [cs.AR] https: //arxiv.org/abs/2408.11997

[2] Yaniv Blumenfeld, Itay Hubara, and Daniel Soudry. 2024. Towards Cheaper Inference in Deep Networks with Lower Bit-Width Accumulators. arXiv:2401.14110 [cs.LG] https://arxiv.org/abs/2401.14110

[3] Yasong Cao, Mei Wen, Zhongdi Luo, Xin Ju, Haolan Huang, Junzhong Shen, and Haiyan Chen. 2024. ABS: Accumulation Bit-Width Scaling Method for Designing Low-Precision Tensor Core. IEEE Transactions on Very Large Scale Integration (VLSI) Systems 32, 9 (2024), 1590–1601. doi:10.1109/TVLSI.2024.3414260

[4] Jack Choquette. 2023. NVIDIA Hopper H100 GPU: Scaling Performance. IEEE Micro 43, 3 (2023), 9–17. doi:10.1109/MM.2023.3256796

[5] O. Gustafsson, A.G. Dempster, and L. Wanhammar. 2004. Multiplier blocks using carry-save adders. In 2004 IEEE International Symposium on Circuits and Systems (IEEE Cat. No.04CH37512), Vol. 2. II–473. doi:10.1109/ISCAS.2004.1329311

[6] Kashif Inayat and Jaeyong Chung. 2021. Carry-Propagation-Adder-Factored Gemmini Systolic Array for Machine Learning Acceleration. Electronics 10, 6 (2021). doi:10.3390/electronics10060652

[7] Kashif Inayat, Inayat Ullah, and Jaeyong Chung. 2024. Factored Systolic Arrays Based on Radix-8 Multiplication for Machine Learning Acceleration. IEEE Transactions on Very Large Scale Integration (VLSI) Systems 32, 7 (2024), 1205–1215. doi:10.1109/TVLSI.2024.3378197

[8] Manish Kumar Jaiswal and Ray C.C. Cheung. 2013. Area-eficient architectures for double precision multiplier on FPGA, with run-time-reconfigurable dual single precision support. Microelectronics Journal 44, 5 (2013), 421–430. doi:10. 1016/j.mejo.2013.02.021

[9] Shiann-Rong Kuang, Chih-Yuan Liang, and Ming-Fong Chang. 2016. A multifunctional multi-precision 4D dot product unit with SIMD architecture. Arabian Journal for Science and Engineering 41, 8 (2016), 3139–3151. doi:10.1007/s13369- 016-2117-3

[10] Kung. 1982. Why systolic architectures? Computer 15, 1 (1982), 37–46. doi:10. 1109/MC.1982.1653825

[11] Boyu Li, Kai Li, Jiajun Zhou, Yuan Ren, Wei Mao, Hao Yu, and Ngai Wong. 2024. A Reconfigurable Processing Element for Multiple-Precision Floating/Fixed-Point HPC. IEEE Transactions on Circuits and Systems II: Express Briefs 71, 3 (2024), 1401–1405. doi:10.1109/TCSII.2023.3322259

[12] Gongqiong Li and Zhaolin Li. 2007. Design of A Fully Pipelined Single-Precision Multiply-Add-Fused Unit. In 20th International Conference on VLSI Design held jointly with 6th International Conference on Embedded Systems (VLSID’07). 318–323. doi:10.1109/VLSID.2007.64

[13] Kai Li, Wei Mao,Junzhuo Zhou, Boyu Li, Zhengke Yang, Shuxing Yang, Laimin Du, Sixiao Huang, and Hao Yu. 2022. A Vector Systolic Accelerator for Multi-Precision Floating-Point High-Performance Computing. IEEE Transactions on Circuits and Systems II: Express Briefs 69, 10 (2022), 4123–4127. doi:10.1109/TCSII.2022.3183007

[14] Haotian Liu, Xicheng Lu, Xiaoyu Yu, Kai Li, Kaiyuan Yang, Haihang Xia, Sizhao Li, and Tiantai Deng. 2025. A 3-D Multi-Precision Scalable Systolic FMA Architecture. IEEE Transactions on Circuits and Systems I: Regular Papers 72, 1 (2025), 265–276. doi:10.1109/TCSI.2024.3497724

[15] K. Manolopoulos, D. Reisis, and V.A. Chouliaras. 2010. An eficient dual-mode floating-point Multiply-Add Fused Unit. In 2010 17th IEEE International Conference on Electronics, Circuits and Systems. 5–8. doi:10.1109/ICECS.2010.5724440

[16] Wei Mao, Kai Li, Quan Cheng, Liuyao Dai, Boyu Li, Xinang Xie, He Li, Longyang Lin, and Hao Yu. 2022. A Configurable Floating-Point Multiple-Precision Processing Element for HPC and AI Converged Computing. IEEE Transactions on Very Large Scale Integration (VLSI) Systems 30, 2 (2022), 213–226. doi:10.1109/TVLSI.2021.3128435

[17] Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu, and Hao Wu. 2022. FP8 Formats for Deep Learning. arXiv:2209.05433 [cs.LG] https://arxiv.org/abs/2209.05433

[18] pulp platform. [n. d.]. pulp-platform/fpu. https://github.com/pulp-platform/fpu [19] Md Mizanur Rahaman Nayan, Ritik Raj, Gouse Basha Shaik, Tushar Krishna, and Azad J Naeemi. 2025. Axon: A Novel Systolic Array Architecture for Improved Run Time and Energy Eficient GeMM and Conv Operation with On-Chip im2col. In 2025 Design, Automation & Test in Europe Conference (DATE). 1–7. doi:10.23919/ DATE64628.2025.10992860

[20] Ritik Raj, Sarbartha Banerjee, Nikhil Chandra, Zishen Wan, Jianming Tong, Ananda Samajdhar, and Tushar Krishna. 2025. SCALE-Sim V3: a Modular Cycle Accurate Systolic Accelerator Simulator for End-To-End System Analysis. In 2025 IEEE International Symposium on Performance Analysis ofSystems and Software (ISPASS). 186–200. doi:10.1109/ISPASS64960.2025.00026

[21] Ananda Samajdar, Yuhao Zhu, Paul Whatmough, Matthew Mattina, and Tushar Krishna. 2019. SCALE-Sim: Systolic CNN Accelerator Simulator. arXiv:1811.02883 [cs.DC] https://arxiv.org/abs/1811.02883

[22] M.S. Schmookler and K.J. Nowka. 2001. Leading zero anticipation and detectiona comparison of methods. In Proceedings 15th IEEE Symposium on Computer Arithmetic. ARITH-15 2001. 7–12. doi:10.1109/ARITH.2001.93009

[23] Eric M. Schwarz. 2006. Binary Floating-Point Unit Design. Springer US, Boston, MA, 189–208. doi:10.1007/978-0-387-34047-0\_8

[24] Sreehari Veeramachaneni, Kirthi M Krishna, Lingamneni Avinash, Sreekanth Reddy Puppala, and M.B. Srinivas. 2007. Novel Architectures for High-Speed and Low-Power 3-2, 4-2 and 5-2 Compressors. In 20th International Conference on VLSI Design held jointly with 6th International Conference on Embedded Systems (VLSID’07). 324–329. doi:10.1109/VLSID.2007.116

[25] Qizhe Wu, Huawen Liang, Yuchen Gui, Zhichen Zeng, Zerong He, Linfeng Tao, Xiaotian Wang, Letian Zhao, Zhaoxi Zeng, Wei Yuan, Wei Wu, and Xi Jin. 2025. Exploring the Performance Improvement of Tensor Processing Engines through Transformation in the Bit-weight Dimension of MACs. In 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA). 685–700. doi:10. 1109/HPCA61900.2025.00058

[26] Rui Xu, Sheng Ma, Yang Guo, and Dongsheng Li. 2023. A Survey of Design and Optimization for Systolic Array-based DNN Accelerators. ACM Comput. Surv. 56, 1, Article 20 (Aug. 2023), 37 pages. doi:10.1145/3604802

[27] Hao Zhang, Dongdong Chen, and Seok-Bum Ko. 2017. Area- and power-eficient iterative single/double-precision merged floating-point multiplier on FPGA. IET Computers & Digital Techniques 11 (2017), 149–158. Issue 4. arXiv:https://digitallibrary.theiet.org/doi/pdf/10.1049/iet-cdt.2016.0100 doi:10.1049/iet-cdt.2016.0100

[28] Song Zhang, Jiangyuan Gu, Shouyi Yin, Leibo Liu, and Shaojun Wei. 2021. A Multiple-Precision Multiply and Accumulation Design with Multiply-Add Merged Strategy for AI Accelerating. In 2021 26th Asia andSouth Pacific Design Automation Conference (ASP-DAC). 229–234.

> 中文翻译与解读由 Codex/GPT 根据所提供的论文 PDF 生成。PDF 使用 MinerU 解析，乱码数学符号已对照原文校正；参考文献保留原文。技术总结和通俗讲解属于解读，请与全文译文结合阅读。[原文 DOI](https://doi.org/10.1145/3770743.3804226)。
