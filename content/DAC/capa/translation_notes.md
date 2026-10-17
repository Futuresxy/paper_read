# CAPA: Convertibility-Aware Mixed-Precision Accelerator for Asymmetric GEMM in Weight-Only Quantized LLMs 原文翻译

# CAPA：面向仅权重量化 LLM 中非对称 GEMM 的可转换性感知混合精度加速器

Jihyun Moon<sup>1,3</sup>、Insu Choi<sup>2</sup>、Seungchan Lee<sup>2</sup> 和 Joon-Sung Yang<sup>1,2,3</sup> <sup>1</sup>系统半导体工程系，<sup>2</sup>电气与电子工程系，<sup>3</sup>智能半导体技术 BK21 研究生项目，延世大学，韩国首尔 {jh.moon,insuoficial,snchanlee,js.yang}@yonsei.ac.kr

## 摘要

仅权重量化大语言模型（LLM）已成为高内存效率部署的标准范式，推动了整型（INT）量化权重（Qweights）与浮点（FP）激活之间的混合精度（FP-INT）运算。尽管 FP–INT GEMM 加速器有助于执行此类运算，但现有设计存在累加开销高、低精度量化权重的并行性利用不足以及通用性欠缺等问题。为解决这些问题，我们提出 CAPA，这是一种针对非对称 GEMM 优化的可转换性感知加速器，其中操作数的位精度存在显著差异。我们引入混合差分块浮点（Hybrid Delta Block Floating Point，HDBFP），这是一种新型格式，可在保持准确率的同时将 FP 激活转换为 INT，从而最大限度地提高基于 INT 的处理单元（PE）的效率。此外，我们的数据分解方案在统一的硬件数据通路内，显著提高了低精度量化权重的计算并行度。这些设计共同使 CAPA 架构能够全面支持常见的 FP16/BF16–INT8/INT4 组合。广泛的评估表明，CAPA 在各项语言评测基准上均保持了与 GPU 基线相当的高准确率，相较于 FP–FP 基线，面积效率提高了 3.98×，功耗效率提高了 4.38×。

## CCS 概念

• 计算机系统组织 → 脉动阵列。

## ACM 参考文献格式：

Jihyun Moon<sup>1,3</sup>, Insu Choi<sup>2</sup>, Seungchan Lee<sup>2</sup> and Joon-Sung Yang<sup>1,2,3</sup>. 2026. CAPA: Convertibility-Aware Mixed-Precision Accelerator for Asymmetric GEMM in Weight-Only Quantized LLMs. In 63rd ACM/IEEE Design Automation Conference (DAC ’26), July 26–29, 2026, Long Beach, CA, USA. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3770743.3804335

## 1 引言

基于 Transformer 的大语言模型（LLM）[1–4] 在广泛的自然语言处理（NLP）任务上取得了最先进的性能，推动了人工智能的实质性进步。然而，其庞大的参数量带来了严苛的内存和计算需求。量化 [5] 是最有前景的方法之一，它将高精度浮点（FP）值转换为整型（INT）等低精度数据格式。在各种量化技术中，仅权重量化 [6–10] 在压缩权重的同时保留较高精度的激活，在准确率与内存效率之间实现了有利的平衡。

![](images/6f0bb0ba99828168cf4bd1bbd1b118f2e3a852e2354a3af557b8a5e36c1d1fd2.jpg)

尽管具有这些优势，GPU 等传统硬件通常会将低精度权重反量化回 FP，以使用 FP 运算执行通用矩阵乘法（GEMM）内核 [11]。这种做法会产生大量计算开销，并且无法利用 INT 运算固有的效率优势。这一局限在多批次服务器场景中更加突出：内存压力得到缓解，而计算需求仍然存在，使工作负载趋向于计算受限的情形 [12, 13]。近期研究 [14–19] 通过 FP–INT GEMM 解决这一问题，直接对异构精度操作数进行运算，无需类型转换。尽管利用了低成本的 INT 运算，这些设计仍因保证精度所需的大位宽累加而未能充分发挥计算效率。与此同时，文献 [17] 提出了非对称 GEMM 的概念，用于描述操作数之间存在显著精度差异的运算，例如 16 位激活与 4 位权重之间的运算。尽管提出了并行计算方案，但其对 FP 运算的依赖以及受限的精度组合限制了整体通用性。

本文提出 CAPA，一种面向非对称 GEMM 的可转换性感知混合精度加速器。CAPA 在混合精度（FP-INT）场景下充分发挥计算效率，同时最大限度地提高非对称 GEMM 的并行性，并确保精度可转换性。

## 2 背景与动机

## 2.1 面向仅权重量化 LLM 推理的 FP-INT GEMM

在自回归大语言模型（LLM）推理中，输入向量或矩阵会在 Transformer 解码器块 [20] 中传播，经过众多投影层或线性层，如图 1(a) 所示。由于获取大型权重矩阵以及与激活执行通用矩阵乘法（GEMM）的成本较高，线性层占据了 LLM 计算工作负载的主要部分 [14, 21]。此外，仅权重量化 LLM 推理还会产生用于重建 FP 权重的反量化开销 [6–10]。

反量化包含两个部分：(i) 解包，即分离打包的 INT 量化权重，并通过应用零点偏移量恢复有符号值；(ii) 缩放，即应用缩放因子将其转换为浮点数。为缓解相关的计算低效问题，已有研究提出 FP–INT GEMM [14–19]，将 INT 量化权重直接与 FP 激活相乘。

![](images/5ec3ff3e54ec3a43b16072cf1099e8dd2f4747e620dba8ea787de0783f96fde8.jpg)  
图 1：LLM 架构与 MAC PE 设计

## 2.2 面向 FP-INT GEMM 的硬件设计

在乘累加（MAC）运算中发挥 FP-INT GEMM 的效率优势，推动了处理单元（PE）设计的发展。如图 1(b) 所示，FP–FP PE（❶）分别处理激活、权重和部分和（psum）的符号、指数和尾数。在 FP-INT PE（❷）中，尾数（含符号）被转换为二进制补码 INT。随后，将其直接与 INT 量化权重相乘，从而减小乘法器规模并简化符号控制。然而，指数比较和尾数对齐，以及随后的规格化和舍入，仍会带来显著的硬件成本。

相比之下，INT PE（❸）仅实现乘法和加法，具有更高的硬件效率。因此，近期设计 [14, 15] 聚焦于 FP 到 INT 的转换，使 MAC 运算能够完全在 INT 域中执行。FIGNA [14] 采用块浮点（BFP）[22–24] 将每个 FP 激活转换为 INT，从而在 INT PE 上执行 MAC 运算。然而，为缓解 FP 到 INT 转换过程中尾数截断造成的精度损失，需要扩大输入和部分和的位宽（至少 30 位），从而增加硬件成本。Precon [15] 则将 FP 尾数划分为更小的块以进行 INT 运算，但对多个部分和寄存器的需求仍会带来可观的硬件成本。

这些观察表明，部分和位宽是 MAC 硬件开销的主要来源。因此，我们致力于设计一种基于 INT 的 PE（❹），在 INT 域中执行对齐，以减小部分和位宽并降低总体成本。

## 2.3 FP 到 INT 的转换

图 2(a) 总结了 FP 和定点（FXP）格式，并展示了从 FP 到 INT 的逐元素转换。FP 值由三个字段组成：符号、指数和尾数。FXP 表示将符号、隐含的前导 1 和尾数合并为一个有符号 INT，而指数则单独存储。图 2(b) 展示了从 FP 到 INT 尾数的逐组转换。对于每个 FP 组，首先确定组内最大指数（最大指数），并结合每个元素的指数差，即 delta-exponent，执行对齐。对齐是指根据 FP 操作数之间的指数差对尾数进行移位。预对齐和后对齐分别指在乘累加（MAC）运算中的乘法之前和之后执行的对齐。

![](images/cc8686bafa498d2fd983bf9b847ad64806013db0db971a00d6101a233c4f34f8.jpg)  
图 2：数据格式及 FP 到 INT 的转换

采用预对齐（①）（例如 BFP）[22–24] 具有较高的 PE 硬件效率，但会因截断最低有效位而造成严重的精度损失。图 2(b) 给出了一个示例：当组内最大指数为 6 时，指数为 1 的尾数会丢失 5 个最低有效位（LSB），而指数为 5 的尾数仍会丢失 1 位，从而降低精度。相比之下，采用后对齐（②）可通过避免截断来提高精度。然而，其较大的移位量会增加部分和（psum）的位宽，并降低硬件效率。

因此，预对齐和后对齐体现了硬件效率与数值精度之间的根本权衡，这促使我们设计一种兼具两者优势的转换方案。

## 2.4 非对称 GEMM 中的并行性

混合精度（FP-INT）算术涉及数据类型不同、尤其是位宽不同的操作数。[17] 将具有显著精度差异的 GEMM 运算定义为非对称 GEMM。这种差异提供了内在的并行性。在等同于单个激活值位宽的空间内，多个低精度量化权重被打包到一个标准字（例如 INT16、INT32）中。例如，在相同位宽内，一个 FP16 激活值对应八个 INT2 量化权重。为利用这一特性，PacQ [17] 采用打包策略和 SIMT 架构来支持一对多计算。然而，PacQ 的通用性和可扩展性有限。它仅支持极低精度格式，并依赖 FP 算术执行 MAC 运算。

这促使我们设计一种非对称 GEMM，在保留基于 INT 的 PE 的效率的同时，充分利用已打包量化权重上的并行性。

![](images/76c30685697651a085ff65b8f5831f5f1d546c63db1bff2e43cb627fdf79c718.jpg)  
图 3：混合差分块浮点（HDBFP）格式、指数分布及精度与效率分析

![](images/c52f34ca28ba33ea269f1315b470c997431ef0743b735db0e12f1d41bc7e13f3.jpg)  
图 4：INT 量化权重分解方案

## 3 数据表示与架构

## 3.1 FP 激活值的数据格式

我们定义了混合差分块浮点（HDBFP），以结合预对齐的效率与后对齐的精度。HDBFP 指定了一个预先定义的最大指数差（max delta-exponent）。仅对指数差超过最大指数差的离群值选择性地进行预对齐，随后在有界指数差下对所有尾数进行后对齐。例如，如图 3(a) 所示，当最大指数差为 3 时，指数差为 0 和 1 的尾数跳过预对齐，而一个离群值（指数差为 5）则按超出的差值 2 进行预对齐。

HDBFP 数据格式具有三个关键优势。（1）大多数尾数得到保留，即使是重要性较低的离群值，也仅被少量截断，从而保持精度。（2）通过限制指数差，后对齐的移位量和部分和位宽受到约束，从而大幅降低 PE 的硬件成本。（3）使用指数差代替绝对指数，使 BF16 运算能够与 FP16 运算采用统一的实现。关于优势（3），BF16 和 FP16 具有相同的相对指数分布。因此，使用指数差可以消除其绝对指数位宽差异（8 位与 5 位）的影响，从而实现单一的统一实现。

HDBFP 数据格式对 LLM 工作负载尤其有效。在 LLM 中，输入激活值组的指数分布集中在最大指数附近，如图 3(b) 所示，这直接使优势（1）和（3）的收益最大化。对多种模型和层的广泛分析表明，绝大多数重要激活值都位于 7 的指数差范围内。因此，我们将最大指数差设为 7，该值足以用最小的 3 位位宽覆盖主要的指数分布。与 BFP 进行比较时，该配置实现了更低的误差 $3 . 3 5 \times { { 1 0 } ^ { - 3 } }$ (标准差：$6 . 0 6 \times 1 0 ^ { - 3 } )$，低于 FP16–FP16 基线 $( 4 . 8 9 \times 1 0 ^ { - 2 } )$，验证了优势（1）。此外，相较于 BFP INT PE，其面积开销降低了 1.4×（图 3(c)），证实了优势（2）。

## 3.2 INT 量化权重的数据分解

我们提出了一种由重新打包（Repack）和解包（Unpack）策略组成的数据分解方案，以实现并行计算。在标准布局中（图 4(a)），量化权重沿 k 维度打包成字。由 GEMM 的计算语义决定，这种打包方式使每个激活值与单个量化权重对齐。这种一对一依赖关系显著限制了并行执行效率。例如，在图 4(c)（红框）中，与量化权重 a0 配对的激活值无法复用于 a1、a2 或 a3。为解决这一问题，Repack 策略将打包维度从 k 转置为 n。因此，单个激活值映射到同一维度内的多个量化权重，从而有效实现一对多并行。例如，如图 4(c)（蓝框）所示，与量化权重 a0 配对的激活值在组内共享，使其能够与所有对齐元素进行计算，包括 b0、c0 和 d0。这种布局将非对称 GEMM 转化为沿 n 维度并行生成多个输出部分和，从而显著提高吞吐量。

然而，在这种重新打包的配置中，INT8 量化权重超出了 INT4 通道的宽度。Unpack 策略通过将两个 INT8 量化权重分解为四个 INT4 量化权重切片来解决这一问题（图 4(b)）。首先，将重新打包的字拆分为各个量化权重，并施加零点偏移，将无符号值转换为有符号值。接着，按照位切片方案 [25]，将每个 INT8 量化权重拆分为两个 4 位量化权重切片。传播 LSB 切片的符号，以确保两个切片均表示有符号值，并将 MSB 切片扩展至 5 位以防止溢出。例如，在图 4(d) 中，有符号值 0111\_1111（127）被拆分为切片 0111（7）和 1111（-1）。对 MSB 切片进行符号扩展得到 00111，再加上 LSB 切片的符号（1），结果为 01000（8）。由此得到有效的分解：$( 1 2 7 = 8 \cdot 2 ^ { 4 } + ( - 1 ) \cdot 2 ^ { 0 } )$，从而在同样的四个 PE 上同时支持四个 INT4 操作数和两个 INT8 操作数，并维持峰值核心利用率。总之，我们的数据分解方案充分利用已打包的量化权重来执行非对称 GEMM。Repack 最大限度地提高并行吞吐量，而 Unpack 则确保硬件通用性和精度支持。

![](images/8ff86bdf0de14721c383763bbaca7f53efaed7abd327a0723dfb329b961f0061.jpg)  
(a) CAPA 架构概览  
图 5：CAPA 架构及端到端数据流概览

## 3.3 CAPA 架构与数据流

3.3.1 架构概述。图 5(a) 展示了 CAPA 的整体架构。核心计算引擎采用二维输出驻留式脉动阵列拓扑。在此基础上，执行过程在用于乘累加（MAC）运算的计算阶段与用于排出结果的排空阶段之间交替进行。该阵列组织方式支持非对称 GEMM 并行，每个分块内的处理单元（PE）分为奇数 PE 和偶数 PE，每个 PE 均与量化权重切片紧密耦合。阵列周围的专用 SIMD 单元执行第 3.3.3 节和第 3.3.4 节详述的数据变换方案。这些单元以流式方式运行，仅引入可忽略的单周期延迟。同时，采用双缓冲 SRAM 存储激活值（64KB）、量化权重（64KB）和部分和（32KB），有效缓解 I/O 延迟。集中式控制器通过有限状态机（FSM）协调全局数据流与控制信号。CAPA 将外部精度与内部执行解耦。外围 SIMD 单元处理激活值（FP16、BF16）和量化权重（INT4、INT8）的精度变化，而脉动阵列执行单一的统一计算路径，确保硬件的简洁性与通用性。

3.3.2 PE 细节。处理单元（PE）是核心计算单元，对 INT 激活值与分解后的量化权重切片执行基于 INT 的 MAC 运算。输入的量化权重切片根据其位宽配置被路由至奇数和偶数通道，与基于 INT 的 PE 设计保持一致。在 MAC 运算期间，对齐器支持 0 到 7 的受限移位范围，与定义的最大指数差直接对应。对于累加，每个 PE 使用一个 25 位累加器，其中包含一个用于防止溢出的 MSB 保护位。由于部分和（psum）所需的位宽大于操作数，因此在 MAC 运算期间将其驻留在 PE 内至关重要。驻留式累加可尽量降低宽位宽数据移动通常带来的大量能耗。在排空阶段，累加得到的部分和切片在写回之前被舍入为 16 位（就近舍入、平局取偶模式）。这一舍入过程在优化位宽需求的同时减少精度损失，从而显著提高硬件的整体可扩展性。

3.3.3 激活值转换。图 5(b) 展示了数据流及相应的操作序列。

浮点激活值组串行进入阵列，以便进行即时转换。流水线首先由最大值搜索单元（❶）从片外激活值流中识别出各组的最大指数。随后，FP2FXP 单元（❷）将其转换为 HDBFP 格式。该单元将符号和尾数合并为有符号 INT 尾数，并计算相对于组内最大指数的指数差。然后，它对指数差超出限制的元素进行选择性预对齐。接着，该单元将指数差与 INT 尾数拼接，生成 INT 激活值，并将其以流式方式送入脉动阵列。（参见第 3.3.4 节中的 ❸。）在每个 PE（❹）内部，INT 激活值与 INT 量化权重切片执行乘累加运算，并应用受限后对齐。（参见第 3.3.4 节中的 ❺。）在排空阶段，聚合后的 INT 部分和从 PE 中排出，与组内最大指数结合，并在 FXP2FP 单元（❻）中转换回浮点格式。（参见第 3.3.4 节中的 ❼。）该流水线通过 HDBFP 将浮点激活值转换为仅保留必要信息的 INT 激活值，从而在基于 INT 的 PE 上实现高效的 MAC 运算。

3.3.4 量化权重分解。对于图 5(b) 所示的量化权重路径，其数据流与第 3.2 节及图 5(a) 中给出的实现细节相结合。

重打包与解包单元（❸）执行完整的量化权重分解。重打包单元获取分布在多存储体缓冲区中的多个字，并将其转置后存入突发缓冲区。随着转置后的字在每个周期发出，解包单元应用零点偏移，并将其拆分为量化权重切片。在 PE（❹）内部，分解后的 INT 量化权重切片与 INT 激活值执行乘累加运算。在排空阶段，聚合单元（❺）收集从每个 PE 排出的部分和切片，重构出与原始权重对应的单个 INT 部分和。（参见第 3.3.3 节中的 ❻。）最后，后缩放单元（❼）通过应用量化权重的组级缩放因子，完成对输出的反量化。该流水线通过从单个激活值到多个量化权重的一对多映射执行非对称 GEMM，从而最大限度地提高核心利用率并实现并行执行。

表 1：CAPA 与 GPU 基线在五项基准测试上的精度对比评估

<table><tr><td rowspan="3">格式</td><td rowspan="3">平台</td><td colspan="5">Phi2-2.7B</td><td colspan="5">Mistral-7B</td><td colspan="5">Llama2-13B</td></tr><tr><td colspan="3">语言理解†</td><td colspan="2">文本生成‡</td><td colspan="3">语言理解†</td><td colspan="2">文本生成‡</td><td colspan="3">语言理解†</td><td colspan="2">文本生成‡</td></tr><tr><td>BoolQ</td><td>PIQA</td><td>Wino.</td><td>Wiki-2</td><td>C4</td><td>BoolQ</td><td>PIQA</td><td>Wino.</td><td>Wiki-2</td><td>C4</td><td>BoolQ</td><td>PIQA</td><td>Wino.</td><td>Wiki-2</td><td>C4</td></tr><tr><td rowspan="2">FP16-INT8</td><td>GPU</td><td>0.8327</td><td>0.7862</td><td>0.7585</td><td>11.2441</td><td>30.5445</td><td>0.8517</td><td>0.7987</td><td>0.7411</td><td>10.1396</td><td>23.8734</td><td>0.8055</td><td>0.7905</td><td>0.7214</td><td>8.0521</td><td>15.0453</td></tr><tr><td>CAPA</td><td>0.8312</td><td>0.7856</td><td>0.7601</td><td>11.1658</td><td>30.5428</td><td>0.8508</td><td>0.7982</td><td>0.7411</td><td>10.1387</td><td>23.8745</td><td>0.8052</td><td>0.7900</td><td>0.7214</td><td>8.0517</td><td>15.0456</td></tr><tr><td rowspan="2">FP16-INT4</td><td>GPU</td><td>0.8159</td><td>0.7818</td><td>0.7482</td><td>11.8768</td><td>32.2588</td><td>0.8434</td><td>0.7982</td><td>0.7293</td><td>9.8964</td><td>23.5469</td><td>0.8092</td><td>0.7840</td><td>0.7222</td><td>7.8530</td><td>15.4998</td></tr><tr><td>CAPA</td><td>0.8144</td><td>0.7829</td><td>0.7459</td><td>11.8799</td><td>32.2572</td><td>0.8450</td><td>0.7992</td><td>0.7253</td><td>9.9028</td><td>23.5495</td><td>0.8083</td><td>0.7835</td><td>0.7206</td><td>7.8534</td><td>15.4997</td></tr><tr><td rowspan="2">BF16-INT8</td><td>GPU</td><td>0.8297</td><td>0.7911</td><td>0.7616</td><td>11.3179</td><td>30.7690</td><td>0.8517</td><td>0.7982</td><td>0.7435</td><td>10.1310</td><td>23.8643</td><td>0.8095</td><td>0.7927</td><td>0.7222</td><td>8.0568</td><td>15.0523</td></tr><tr><td>CAPA</td><td>0.8318</td><td>0.7829</td><td>0.7672</td><td>11.2543</td><td>30.7681</td><td>0.8508</td><td>0.7987</td><td>0.7411</td><td>10.1324</td><td>23.8740</td><td>0.8073</td><td>0.7894</td><td>0.7198</td><td>8.0553</td><td>15.0498</td></tr><tr><td rowspan="2">BF16-INT4</td><td>GPU</td><td>0.8147</td><td>0.7878</td><td>0.7553</td><td>11.8899</td><td>32.2821</td><td>0.8425</td><td>0.7987</td><td>0.7277</td><td>9.8911</td><td>23.5388</td><td>0.8095</td><td>0.7845</td><td>0.7174</td><td>7.8566</td><td>15.5034</td></tr><tr><td>CAPA</td><td>0.8159</td><td>0.7862</td><td>0.7522</td><td>11.8910</td><td>32.2817</td><td>0.8440</td><td>0.8003</td><td>0.7238</td><td>9.8859</td><td>23.5458</td><td>0.8104</td><td>0.7835</td><td>0.7159</td><td>7.8591</td><td>15.5079</td></tr></table>

指标 = 准确率（越高越好）  
<sup>‡</sup> 指标 = 词困惑度（越低越好）

表 2：CAPA 架构的面积与功耗明细

<table><tr><td rowspan="2">类别</td><td rowspan="2">组件</td><td rowspan="2">面积 (mm2)</td><td colspan="3">功耗 (W)</td></tr><tr><td>动态功耗</td><td>漏电功耗</td><td>总功耗</td></tr><tr><td rowspan="2">计算</td><td>脉动阵列 PE</td><td>0.7931</td><td>0.0121</td><td>0.0037</td><td>0.0157</td></tr><tr><td>脉动阵列寄存器</td><td>0.6465</td><td>0.1229</td><td>0.0037</td><td>0.1266</td></tr><tr><td rowspan="2">存储</td><td>激活值缓冲区 + 量化权重缓冲区</td><td>0.4480</td><td>0.0718</td><td>0.1762</td><td>0.2480</td></tr><tr><td>部分和缓冲区</td><td>0.1843</td><td>0.0320</td><td>0.0677</td><td>0.0996</td></tr><tr><td>其他</td><td>控制器、SIMD 单元</td><td>0.1450</td><td>0.0147</td><td>0.0018</td><td>0.0165</td></tr></table>

![](images/28a5b44ae541f82ca072ac9e10080289c448955790cc1ec1d91228c6a5ea47e3.jpg)

![](images/89d5d516c4f89e9ea3362137c003c099f166f3ae0516a829248952c1016a6051.jpg)  
图 6：PE 设计的面积和功耗比较

## 4 评估

## 4.1 方法

4.1.1 软件评估。我们使用来自 Huggingface [26] 的预训练仅权重量化 LLM 模型 [6]，不进行任何微调。我们选择了不同参数规模的模型：Phi2-2.7B [27]、Mistral-7B [28] 和 Llama2-13B [29]。量化权重的精度（INT4 或 INT8）由模型修订版本决定，而分组大小固定为 128。LLM 的准确性使用 Language Model Evaluation Harness [30] 进行评估。对于语言理解任务，采用 BoolQ、PIQA 和 WinoGrande（Wino.）基准。对于文本生成任务，选择 Wikitext-2（Wiki-2）和 C4 基准。我们使用配备 NVIDIA GeForce RTX 4090 GPU 的 GPU 服务器作为基线平台，并将其准确性与 CAPA 的准确性进行比较。

4.1.2 硬件实现。CAPA 使用 Verilog 在寄存器传输级（RTL）实现，并使用 Synopsys Design Compiler（DC）进行综合，目标为 28nm CMOS 工艺，工作频率为 250MHz。通过注入测试向量，在 0.72V、$1 2 5 ^ { \circ } \mathrm { C }$ 环境下使用 PrimeTime PX 进行功耗分析。片上暂存 SRAM 使用 CACTI [31] 建模。我们使用开源周期精确模拟器 SCALE-SIM [32] 对硬件配置进行建模，并提取总周期数、计算利用率和带宽利用率。在相同环境下，我们实现并综合了 FP-FP 基线、FP-INT、Precon [15]、FIGNA [14] 和 CAPA，以进行对比分析。

## 4.2 LLM 准确性分析

我们在三种参数规模和多种精度组合下评估了 CAPA，确认其能够复现 GPU 级别的准确性（表 1）。在分类任务（BoolQ、PIQA、Wino.）中，平均准确率差距被限制在 ≤ 0.2 个百分点。对于语言建模，困惑度（PPL）偏差仍然极小：在 Wiki-2 上平均 ≤ 0.13%，在 C4 上平均 ≤ 0.02%，且经常与基线一致到最后一位小数。关键的是，随着参数量增加，绝对差距逐渐减小，表明其具有稳定的可扩展性。这种稳健性在 FP16/BF16 激活值和 INT8/INT4 量化权重的各种组合下均得以保持。因此，在不同模型规模和格式下，CAPA 均可作为 GPU 推理的直接替代方案，同时保持准确性。

## 4.3 硬件分析

4.3.1 整体架构分析。表 2 汇总了整体架构的详细面积和功耗分析。脉动阵列（PE 和寄存器）占据了主要的面积，而功耗主要来自片上存储器。集中式控制器和重打包单元中的突发传输缓冲区构成了“其他”类别的大部分。然而，专门用于支持 FP-INT GEMM 的开销仍然极小。SIMD 单元仅占面积的 6.5% 和功耗的 3.3%，表明卸载精度处理所带来的开销可以忽略不计，同时显著降低了占主导地位的 PE 数据通路的复杂度。

![](images/f1e6604a2d2adbc4edd373e20e0d4e8fbdaed19e66f7047cf284bd5833e1fa5f.jpg)  
图 7：不同 LLM 工作负载的面积效率（上）和功耗效率（下）比较

4.3.2 处理单元分析。我们通过表征单个 MAC 单元级别的面积和功耗，验证了 PE 设计的有效性。我们针对常见的 FP16–INT4 配置综合了多种 PE 变体：FP-FP 基线、通用 FP-INT、Precon、FIGNA，以及我们的 CAPA PE（奇数（od）和偶数（ev））。图 6 展示了相对于基线归一化后的各项指标，并包含组件级的细分结果。结果表明，INT 量化权重较低的位宽降低了所有对比架构中的乘法器复杂度。Precon、FIGNA 和 CAPA 通过消除复杂的规格化和舍入逻辑，进一步降低了硬件成本。尽管 CAPA 保留了舍入阶段，但其仅依赖一个简单的加法器，因此带来的开销可以忽略不计。与 Precon 的冗余尾数拆分或 FIGNA 的固定分块宽度开销不同，我们的 PE 仅精确提取激活值中的有效尾数位。这一点与 HDBFP 所实现的较窄移位范围相结合，显著减小了乘法、对齐和加法的逻辑深度。因此，与 FP-FP 基线相比，CAPA PE 的面积平均减少了 77.5%，功耗平均降低了 82.6%。

4.3.3 LLM 工作负载的效率比较。为展示 CAPA 在 LLM 工作负载中的有效性，我们使用与准确性评估相同的模型，对注意力（attn）和 MLP（mlp）模块进行评估。所有架构均统一为配置达到峰值吞吐量的 32 × 32 脉动阵列结构，使我们能够比较由计算利用率带来的加速效果。值得注意的是，与针对特定精度优化的其他设计不同，我们评估了 CAPA\_ALL，即一种固定的 32 × 32 × 4 配置（每个计算块中有 4 个并行 PE），以展示其在并行性和通用性方面的有效性。为确保面积效率和功耗效率的公平比较，我们报告了相对于 FP-FP 基线归一化后的 TOPS/mm<sup>2</sup> 和 TOPS/W，如图 7 所示。

结果表明，CAPA\_ALL 显著优于基线，面积效率最高达到基线的 3.98×（相对于 FIGNA 为 1.30×），功耗效率最高达到基线的 4.38×（相对于 FIGNA 为 1.94×）。具体而言，对于 INT4，

CAPA\_ALL 的吞吐量达到基线的四倍。由于这种性能提升有效缓解了硬件开销的影响，整体效率指标显著改善。尽管通用性使 INT8 效率受到轻微影响，但 CAPA\_ALL 的面积效率和功耗效率仍分别至少达到最先进的 FIGNA 的 1.04× 和 1.02×。这项评估表明，CAPA 具有更高的效率，同时能够在不同精度需求下保持稳健的性能。

## 5 结论

我们提出了 CAPA，这是一种专为仅权重量化 LLM 设计的高效且通用的加速器架构。CAPA 直接解决了 FP-INT GEMM 的两个主要挑战：高硬件成本和有限的精度通用性。我们引入了混合差分块浮点（HDBFP）数据格式，显著降低了 PE 的硬件成本。此外，我们的数据分解方案能够对所有常见的 FP16/BF16-INT8/INT4 组合进行通用且并行的处理。因此，CAPA 在实现与 GPU 基线相当的准确性的同时，大幅提升了效率。我们的评估表明，CAPA 的面积效率和功耗效率最高分别达到基线的 3.98× 和 4.38×。

## 致谢

本研究部分得到了由韩国政府（MSIT）出资的信息通信技术规划与评估院（IITP）资助项目（No.RS-2025-09942968，AI 半导体创新实验室（延世大学））的支持，部分得到了由韩国政府（MSIT）出资的信息通信技术规划与评估院（IITP）资助项目（No. RS-2026-25507672，面向 AI 服务器的下一代 PIM 软件栈开发）的支持，并部分得到了由韩国政府（MSIT）出资的韩国国家研究基金会（NRF）资助项目（RS-2026-25540349）的支持。

## References

[1] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, et al. The Llama 3 herd of models. arXiv, 2024. URL https://arxiv.org/abs/2407.21783.

[2] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv, 2025. URL https://arxiv.org/abs/ 2503.19786.

[3] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chan Gao, Chen en Huan , Chenxu Lv, et al. Qwen3 technica report. arXiv, 2025. URL https://arxiv.org/abs/2505.09388.

[4] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-V3 technical report. arXiv, 2024. URL https://arxiv.org/abs/2412.19437.

[5] Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and Hufman coding. In Proceedings ofthe 4th International Conference on Learning Representations (ICLR), 2016. URL https://arxiv.org/abs/1510.00149.

[6] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. OPTQ: Accurate quantization for generative pre-trained transformers. In Proceedings of the 11th International Conference on Learning Representations (ICLR), 2023. URL https: //openreview.net/forum?id=tcbBPnfwxS.

[7] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation aware weight quantization for on-device LLM compression and acceleration. In Proceedings ofMachine Learning and Systems (MLSys), volume 6, a es 87–100, 2024.

[8] Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via block-wise quantization. In Proceedings ofthe 10th International Conference on Learning Representations (ICLR), 2022. URL https://openreview.net/forum?id= shpkpVXzo3h.

[9] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Eficient finetuning of quantized LLMs. In Advances in Neural Information Processing Systems (NeurIPS 2023), volume 36, pages 10088–10115, 2023.

[10] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. LLM.int8(): 8-bit matrix multiplication for transformers at scale. In Advances in Neural Information Processing Systems (NeurIPS 2022), volume 35, pages 30318–30332, 2022.

[11] Youn Jin Kim, Rawn Henr , Raf Fahim, and Han Hassan. Who sa s ele hants can’t run: Brin in lar e scale MoE models into cloud scale roduction. In Proceedings ofthe Third Workshop on Simple and Eficient Natural Language Processing (SustaiNLP 2022), pages 36–43. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.sustainlp-1.6. URL https://aclanthology.org/2022.sustainlp- 1.6/.

[12] Samuel Williams, Andrew Waterman, and David Patterson. Roofline: an insightful visual performance model for multicore architectures. Communications ofthe ACM, 52(4):65–76, 2009. doi: 10.1145/1498765.1498785.

[13] Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang, and Joel S Emer. Eficient processing of dee neural networks: A tutorial and survey. Proceedings ofthe IEEE, 105(12): 2295–2329, 2017. doi: 10.1109/JPROC.2017.2761740.

[14] Jaeyong Jang, Yulhwa Kim, Juheun Lee, and Jae-Joon Kim. FIGNA: Integer unit-based accelerator design for FP-INT GEMM preserving numerical accuracy. In Proceedings of the 2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pages 760–773, 2024. doi: 10.1109/HPCA57654. 2024.00064.

[15] Jongwoo Park, Hyeonseong Kim, Jiyun Han, and Seungkyu Choi. Precon: A precision-convertible architecture for accelerating quantized deep learning mod els across various domains including LLMs. In Proceedings of the 62nd Annual ACM/IEEE Design Automation Conference (DAC), pages 1–7, 2025. doi: 10.1109/DAC63849.2025.11133184.

[16] Gunho Park, Hyeokjun Kwon, Jiwoo Kim, Jeongin Bae, Baeseong Park, Dongsoo Lee, and Youngjoo Lee. FIGLUT: An energy-eficient accelerator design for FP INT GEMM using look-up tables. In Proceedings of the 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA), pages 1098–1111, 2025. doi: 10.1109/HPCA61900.2025.00085.

[17] Ruokai Yin, Yuhang Li, and Priyadarshini Panda. PacQ: A SIMT microarchitecture for eficient dataflow in hyper-asymmetric GEMMs. In Proceedings of the 62nd Annual ACM/IEEE Design Automation Conference (DAC), pages 1–7, 2025. doi: 10.1109/DAC63849.2025.11132101.

[18] Lei Wang, Lingxiao Ma, Shijie Cao, Quanlu Zhang,Jilong Xue, Yining Shi, Ningxin Zhen , Zimin Miao, Fan Yan , Tin Cao, et al. Ladder: Enablin eficient Low- Precision dee learning com uting through hardware-aware tensor transformation. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 307–323. USENIX Association, 2024.

[19] Jianyu Wei, Shijie Cao, Ting Cao, Lingxiao Ma, Lei Wang, Yanyong Zhang, and Mao Yang. T-MAC: CPU renaissance via table lookup for low-bit LLM deployment on edge. In Proceedings of the Twentieth European Conference on Computer Systems (EuroSys 2025), ages 278–292, 2025. doi: 10.1145/3689031.3696099.

[20] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS 2017), volume 30, pag<sup>es 5998–6008</sup>, <sup>2017.</sup>

[21] Seongho Kim, Jihyun Moon, Juntaek Oh, Insu Choi, and Joon-Sung Yang. Survey and evaluation of converging architecture in LLMs based on footsteps of operations. IEEE Open Journal of the Computer Society, 6:1214–1226, 2025. doi: 10.1109/OJCS.2025.3587005.

[22] Eric Chung, Jeremy Fowers, Kalin Ovtcharov, Michael Papamichael, Adrian Caulfield, Todd Massengill, Ming Liu, Daniel Lo, Shlomi Alkalay, Michael Haselman, et al. Serving DNNs in real time at datacenter scale with project brainwave. IEEE Micro 38(2):8–20 2018. doi: 10.1109/MM.2018.022071131.

[23] Urs Köster, Tristan Webb, Xin Wang, Marcel Nassar, Arjun K Bansal, William Constable, Oguz Elibol, Scott Gray, Stewart Hall, Luke Hornof, et al. Flexpoint: An adaptive numerical format for eficient training of deep neural networks. In Advances in Neural Information Processing Systems (NeurIPS 2017), volume 30, pag<sup>es 1742–1752</sup>, <sup>2017.</sup>

[24] Bita Darvish Rouhani, Ritchie Zhao, Ankit More, Mathew Hall, Alireza Khodamoradi, Summer Deng, Dhruv Choudhary, Marius Cornea, Eric Dellinger, Kristof Denolf, et al. Microscaling data formats for deep learning. arXiv, 2023. URL https://arxiv.org/abs/2310.10537.

[25] Insu Choi, Young-Seo Yoon, and Joon-Sung Yang. Bit-slice architecture for DNN acceleration with slice-level sparsity enhancement and exploitation. In Proceedings ofthe 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA), a es 821–835, 2025. doi: 10.1109/HPCA61900.2025.00067.

[26] Thomas Wolf, Lysandre Debut, Victor Sanh,Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, a es 38–45. Association for Com utational Linguistics, 2020. doi: 10.18653/v1/2020.emnlp-demos.6. URL https://aclanthology. org/2020.emnlp-demos.6/.

[27] Mojan Javaheripi and Sébastien Bubeck. Phi-2: The Surprising Power of Small Language Models. Microsoft Research Blog, December 2023. URL https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/.

[28] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7B, 2023. URL https://arxiv.org/abs/2310.06825.

[29] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv, 2023. URL https://arxiv.org/abs/2307.09288.

[30] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jefrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighof, Chris Ociepa,Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 2024. URL htt s://zenodo.or /records/12608602.

[31] HewlettPackard/cacti contributors. CACTI: A tool to model caches/memories. GitHub repository, 2014. URL htt s:// ithub.com/HewlettPackard/ cacti/commit/b288fc2eae7a9f35892060d5a589b816e0a35969. commit b288fc2eae7a9f35892060d5a589b816e0a35969.

[32] Ritik Raj, Sarbartha Banerjee, Nikhil Chandra, Zishen Wan, Jianming Tong, Ananda Samajdhar, and Tushar Krishna. SCALE-Sim v3: A modular cycle accurate systolic accelerator simulator for end-to-end system analysis. In Proceedings of the 2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), pages 186–200, 2025. doi: 10.1109/ISPASS64960.2025.00026.

> 中文翻译与解读由 Codex/GPT 根据所提供的论文 PDF 生成。PDF 使用 MinerU 解析，乱码数学符号已对照原文校正；参考文献保留原文。技术总结和通俗讲解属于解读，请与全文译文结合阅读。[原文 DOI](https://doi.org/10.1145/3770743.3804335)。
