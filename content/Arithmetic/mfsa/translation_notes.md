# MFSA: A Multi-Format Systolic Array with Native Micro Scaling Support 原文翻译

# MFSA：原生支持微缩放的多格式脉动阵列

Jiayi Wang<sup>∗</sup>

华盛顿大学

美国华盛顿州西雅图

jwang710@uw.edu

Kearnan Bishop<sup>∗</sup> 华盛顿大学 美国华盛顿州西雅图 kearnan@uw.edu

Ang Li 华盛顿大学 美国华盛顿州西雅图 angliz@uw.edu

摘要—AI 工作负载越来越多地在不同模型、层和执行阶段采用 BF16 / FP16、FP8 变体、FP4、微缩放（MX）和 INT 模式等低精度格式，以平衡精度、内存占用和吞吐量，这进而要求硬件以灵活且高效的方式支持这些格式。我们提出了 MFSA，这是一种权重驻留的多格式脉动阵列，通过两项架构设计实现面积效率。在处理单元（PE）层面，MFSA 使用基于点积累加器的处理单元，由现有的开源跨精度 FPU 扩展而来，通过共享数据通路支持降低精度的格式（FP32/FP16/BF16/CFP8/FP4/INT16/INT8/INT4），并实现吞吐量的成比例扩展。在阵列层面，MFSA 针对块缩放格式采用在线块缩放重定基（OBSR），在块边界处将每个块的缩放因子融入持续更新的 FP32 部分和，无需增加第二个跨块累加器，从而减少支持块缩放所需的额外开销。在 TSMC 28 nm 工艺、1 GHz 频率下，吞吐量相同的 BF16/FP8/FP4 级联 FMA-PE 基线的面积分别为单格式 DPA-PE 的 1.5×/2.5×/4.5×，且差距随子字并行度的提高而增大。完成布线的 16×16 MFSA 在 FP32/BF16/FP8/FP4 下的峰值吞吐量达到 0.5/1/2/4 TOPS；与朴素 MX 实现基线相比，OBSR 将 MX 带来的额外面积和能耗开销分别降低了 64.2% 和 67.5%，同时不损害真实 LLM 执行轨迹上的数值稳定性。在 BF16+CFP8 脉动阵列的基础上增加对所有其他格式的支持，需要额外增加 25.7% 的面积。OBSR 还可扩展到非 2 的幂次的块缩放因子：将 E4M3 缩放因子的小数部分融入 MFPE 点积，即可同时支持 NVFP4 和 OCP MX，相比 CFP8+FP4 基础设计，面积和功耗分别增加 11.8% 和 7.3%，开销不到朴素实现的一半。

索引词—脉动阵列、多格式算术、融合点积、跨精度、AI 加速器、深度学习硬件。

## I. 引言

现代 AI 工作负载已不再采用单一的数值精度。物理信息神经网络等科学机器学习模型需要高数值精度，因为降低精度会导致准确性下降并使收敛停滞 [1], [2]；而大型语言模型（LLM）则越来越多地采用低精度格式。BF16 / FP16 仍常用于训练状态更新和优化器路径 [3], [4]，FP8 变体（E4M3/E5M2）面向高吞吐率训练和推理服务 [5], [6]，FP4 正逐渐用于更激进的推理 [7]，而微缩放（MX）格式将低精度元素与共享的块缩放指数相结合，以较低的存储和带宽开销恢复动态范围 [8]。

![](images/30a00d29cf48cc7bbe423a018775dbaa2d53e010085dbc7dfbf43e3b556b25c0.jpg)  
图 1. 格式编码与正数幅值范围

首选格式在训练与推理之间可能有所不同，甚至同一模型的不同层也可能采用不同格式：例如，DeepSeek-V4 即使在注意力和混合专家（MoE）子系统内部也混用多种精度：KV 条目在 RoPE 维度上保留 BF16，在其他维度上使用 FP8；闪电索引器使用 FP4 计算注意力；MoE 专家权重被量化为 FP4，并在计算时反量化为 FP8 [9]。Nemotron 3 Ultra 已发布的检查点按算子分配精度 [10]：MoE 路由专家采用 NVFP4，共享专家和 Mamba [11] 混合器采用 FP8，注意力和嵌入采用 BF16，并使用 FP8 KV 缓存；Kimi K2 Thinking 则将其专家量化为 INT4 [12]。较低精度能够减少内存占用和数据搬移开销，并提高有效算术吞吐率，但也会缩小动态范围、降低尾数精度，从而可能导致性能下降，因此格式选择仍对准确性敏感 [3], [5]。与此同时，为了在 AI 矩阵乘法特有的长归约过程中保持数值稳定性，累加需要采用高精度，因此生产系统对浮点输入矩阵乘法保留 FP32 累加 [6], [7], [13]–[15]。图 1 汇总了本工作所面向的代表性浮点、整数和 MX 模式，并突出了它们显著不同的幅值范围。

在硬件方面，商用矩阵引擎已通过显式支持多种格式来适应这一工作负载约定，尽管其处理单元（PE）的详细微体系结构尚未公开。表 I 根据当前公开的厂商文档汇总了代表性实例。NVIDIA Tensor Core [6], [7] 涵盖 BF16 / FP16 / FP8 / FP32 级 / INT 模式，而 Google TPU7x [16] 和 AWS Trainium3 [15] 则提供多种低精度模式下的矩阵引擎吞吐率，并明确规定 FP32 级的累加或输出行为。这些产品共同确立了体系结构目标：支持多种格式、且吞吐率大致按比例扩展的矩阵乘法引擎。然而，高效实现这一目标所需的微体系结构机制，以及支持额外格式带来的面积和能耗开销，在公开文献中仍缺乏充分刻画。

表 I  
代表性商用矩阵引擎：支持的输入格式及以各产品 BF16 模式（1×）为基准归一化的峰值吞吐率。

<table><tr><td>格式</td><td>H100 TC [6]</td><td>B200 TC [7]</td><td>TPU7x [16]</td><td>Trn3 (NCv4) [15]</td><td>Dojo D1 [17]</td></tr><tr><td>FP32</td><td> $0.5 \times$ </td><td> $0.5 \times$ </td><td>不适用</td><td> $0.27 \times$ </td><td> $0.06 \times$ </td></tr><tr><td>BF16</td><td> $1 \times$ </td><td> $1 \times$ </td><td> $1 \times$ </td><td> $1 \times$ </td><td> $1 \times$ </td></tr><tr><td>FP16</td><td> $1 \times$ </td><td> $1 \times$ </td><td>不适用</td><td> $1 \times$ </td><td>不适用</td></tr><tr><td>FP8</td><td> $2 \times$ </td><td> $2 \times$ </td><td> $2 \times$ </td><td>不适用</td><td> $1 \times$ </td></tr><tr><td>FP4</td><td>不适用</td><td> $4 \times$ </td><td>不适用</td><td>不适用</td><td>不适用</td></tr><tr><td>MXFP8</td><td>不适用</td><td> $2 \times$ </td><td>不适用</td><td> $4 \times$ </td><td>不适用</td></tr><tr><td>MXFP4</td><td>不适用</td><td> $4 \times$ </td><td>不适用</td><td> $4 \times$ </td><td>不适用</td></tr><tr><td>INT8</td><td> $2 \times$ </td><td> $2 \times$ </td><td>不适用</td><td>不适用</td><td>不适用</td></tr><tr><td>累加</td><td>FP32</td><td>FP32</td><td>FP32</td><td>FP32/BF16</td><td>FP32</td></tr></table>

TC = NVIDIA 张量核心；TPU = Google 张量处理单元；NCv4 = 第 4 代 AWS NeuronCore（AWS 数值对应 NeuronCore-v4 内部的 Tensor Engine）。FP32 行 = 在支持的情况下采用原生 FP32 矩阵计算，否则采用 FP32 级计算（例如 NVIDIA 上的 TF32）。Tesla Dojo 使用可配置的 CFP8/CFP16；其 CFP8（运行时可选择 E4M3/E5M2）以 BF16 的矩阵运算速率运行，而 FP32 的优先级则被大幅降低。

本工作聚焦于此类引擎的微体系结构设计，并量化支持额外格式带来的开销。我们采用以 TPU [18]–[20] 为代表的权重驻留式脉动矩阵乘法引擎，因为它们兼具较高的数据复用率和规则的局部连接，并且能够自然扩展到大型阵列。支持多种格式的一种简单方法是为每种格式复制一条独立的数据通路，但这样会增加面积、布线和控制开销，同时往往导致硬件利用不足，因为工作负载很少同时使用所有格式。因此，要在共享数据通路上实现按比例扩展的吞吐率，就需要精心设计微体系结构。除了格式支持本身，MX 格式还会引入额外的阵列级开销。在开放计算项目（OCP）的 MXFP{8, 6, 4} [21] 中，每个含 32 个元素的块都携带一个共享的 E8M0 缩放因子，因此，对于 PE 跨越多个 MX 块的直接实现的脉动阵列，需要级联经过块缩放的部分和，并在每一列中为每个块增加第二份 FP32 状态和第二个加法器。因此，本工作旨在提出一种高效、灵活的设计，以减少这些开销，并刻画支持额外格式所带来的增量成本。

综上，本工作的贡献如下：

• 我们提出多格式脉动阵列（MFSA），这是一种围绕基于点积累加（DPA）的多格式处理单元（MFPE）构建的权重驻留式脉动阵列。DPA 单元在开源 TransDot [22] 的基础上扩展，支持 FP32/BF16/FP16/CFP8/FP4 和 INT16/INT8/INT4 模式（CFP8 表示运行时可选择 E4M3/E5M2 的可配置 FP8，与 Tesla Dojo [17] 中的定义相同）。随着精度降低，MFPE 通过子字并行扩展吞吐率，同时保留宽位宽的 FP32/INT32 累加通路。

• 我们在块边界采用在线块缩放重定基（OBSR），以支持块缩放格式，代替添加第二条跨块 FP32 累加通路。OBSR 利用 E8M0 MX 缩放因子与 FP32 指数字段之间的对称性，将跨块重定基简化为指数域内的小规模加减运算。我们进一步讨论了支持分数型 NVFP4 [23]（E4M3 块缩放因子）的方法，并评估其面积/功耗开销。

• 我们采用 Cadence Genus/Innovus/Joules 流程，在 TSMC 28 nm 工艺下实现并评估 MFSA。在 PE 层面，吞吐率相同的 BF16/FP8/FP4 级联 FMA-PE 基线的面积分别为单格式点积累加器 PE 的 $1 . 5 { \times } / 2 . 5 { \times } / 4 . 5 { \times }$，且这一差距随着子字并行度的增加而扩大。通过 OBSR 添加 MX 支持，相较于朴素 MX 基线，MX 带来的增量面积和能耗开销分别降低了 64.2% 和 67.5%，同时在真实 LLM 执行轨迹上，与该基线的数值结果差异控制在几个 ULP 以内。我们进一步量化了此前缺乏充分刻画的逐步增加格式支持的成本：支持完整 FP32/FP16/BF16/CFP8/FP4/INT16/INT8/INT4/MXFP8/MXFP4 格式的阵列，其布线后单元面积仅比 BF16+CFP8 基线阵列增加 25.7%。

## II. MFSA 微体系结构

## A. 顶层阵列组织结构

MFSA 保留了 TPU 经典的权重固定式脉动数据流：由相互连接的 PE 构成阵列，每个 PE 均执行相同的融合乘加（FMA）操作。权重预先加载并保存在各个 PE 内部，激活值按常规的行间错位方式从西向东流动，部分和则沿固定的 FP32（或 INT32）级联通路从北向南传播。

图 2 详细展示了这一组织结构。MFSA 进一步将四个 PE 组织成一个 PE 块（BPE）（图 2(b)）；BPE 内的每个 PE 都是一个多格式 PE（MFPE）（图 2(c)）。MFPE 的算术核心是点积累加器（DPA）（图 2(e)），而非单个 FMA：它接收以子字并行方式处理的低精度操作数，并生成高精度结果，因此吞吐率随精度降低而成比例提升。由此，即使输入精度发生变化，阵列仍能保持一条不随模式变化的部分和通路。较窄的格式以自然打包的子字形式通过相同的激活值链路和权重链路传输，因此，MFSA 在执行低精度运算时，无需使用特定于模式的 PE 间连线，也无需加宽部分和网络。

对于 MX 模式，每个 BPE 增加一个在线块缩放因子基准重设（OBSR）单元（图 2(d)），在累加到当前块的结果之前执行跨块重缩放。

边带格式选择信号按计算分块配置 MFPE 数据通路；由于南北向级联通路在所有模式下均保持为 FP32/INT32，格式切换不会清空正在传输的部分和。随后，每列的 FP32 输出被写入结果累加器，由其合并跨通用矩阵乘法（GEMM）计算分块的部分和。

![](images/d5230214a4c2462604528f6b0f06e04271b6e4e2a3b3597ceb94abcbcde32a86.jpg)  
图 2. MFSA 层次结构。(a) 顶层 MFSA 阵列。(b) 为支持 MX，四个垂直相邻的 MFPE 与一个位于北侧边界的 OBSR 单元构成一个 PE 块（BPE）；MXFP4 的每个块使用一个 BPE，而 MXFP8 跨越两个上下堆叠的 BPE。(c) 带有 DPA 的 MFPE 微架构。(d) OBSR 单元微架构 (e) 每个 MFPE 内部的 DPA。

本节其余部分将详细介绍各个子模块。

## B. MFPE 数据通路

目标 MFPE 应支持 FP32/FP16/BF16/CFP8/FP4 输入并采用 FP32 累加，以及 INT16/INT8/INT4 输入并采用 INT32 累加，同时在各精度下提供成比例的吞吐率和较高的面积效率。点积累加天然契合这一目标：低精度操作数被打包到原有的宽输入数据通路中，并在各个 PE 内通过内部点积运算进行归约。这使 DPA 能够在各精度模式之间共享宽 FMA 数据通路的大部分组件，而无需为每种格式分别复制一个独立单元。

a) 浮点数据通路：[22] 中的开源跨精度点积单元已支持 FP32/FP16/FP8/FP4，并采用 FP32 累加。我们的 DPA 在其基础上扩展了现代 AI 模型所需的 BF16、CFP8 和整数模式。该数据通路包括：(1) 指数处理通路；(2) 围绕可重构的 24×24 多模式乘法器构建的尾数乘法通路，该乘法器可划分为 2、4 或 8 路较窄的乘积运算及其加法树；(3) 对齐移位器，用于对齐乘积尾数与已累加 FP32 值的尾数；(4) 宽累加器，用于对原始乘积和已累加 FP32 值的尾数求和；以及 (5) 输出前的规格化/舍入级。

多模式乘法器以进位保留形式生成 k 个乘积，其中 k 为点积运算中的项数，共享加法树对这些乘积进行归约，中间不进行舍入。进位保留结果通过单个移位器，按照 FP32 累加器的当前指数进行对齐；一个 76 位累加器（24 位超尾数 + 2 位保护/舍入/粘滞位 + FP32 动态范围对齐余量）容纳对齐操作，在累积移位范围内不损失精度；单个最近偶数舍入（RNE）舍入器生成下一个 FP32 部分和。因此，MFPE 对 k 个乘积的求和表达式仅进行一次 IEEE-754 规格化和 RNE 舍入，而非分别进行 k 次规格化/舍入。

b) 整数扩展：为尽量降低支持整数运算的面积开销，INT-X 复用 FP-2X 的尾数乘法器槽位：INT16 使用 FP32 的 24×24 乘法器（k=1）；INT8 使用双通道 FP16-DP 通路（k=2）；INT4 使用四通道 FP8-DP 通路（k=4，由于 E4M3 通道承载 4 位有效数，即 3 个存储的尾数位加上隐含位，因此恰好匹配）。乘法器和累加器直接复用浮点通路中的组件，无需修改；整数专用逻辑仅包括对二进制补码操作数进行整数符号扩展、有符号/无符号模式选择，以及绕过浮点规格化和舍入的输出多路选择器。子字 INT 沿用与 FP 相同的逐级 2× 吞吐率级联关系，但其绝对吞吐率仅为同位宽 FP 的一半，因为 INT-X 复用的是 FP-2X 的槽位，而非 FP-X 的槽位。

## C. 用于 MX 的在线块尺度重定基（OBSR）

a) MX 块算术：对于 OCP MX 格式 [21]，每个操作数向量都被划分为包含 32 个元素的块，每个块除包含 32 个逐元素低精度值 m<sup>(b,i)</sup> 外，还携带一个 8 位 E8M0 共享指数 $2 ^ { s ^ { ( b ) } }$，其中每个低精度值本身都是具有各自指数和尾数的 FP8/FP6/FP4 数。两个 MX 向量 A 和 B 的点积为

$$
A \cdot B = \sum_ {b} 2 ^ {s _ {a} ^ {(b)} + s _ {b} ^ {(b)}} \sum_ {i = 0} ^ {3 1} m _ {a} ^ {(b, i)} m _ {b} ^ {(b, i)}.\tag{1}
$$

内部的逐块乘积 $\begin{array} { r } { \sum _ { i } m _ { a } ^ { ( b , i ) } m _ { b } ^ { ( b , i ) } } \end{array}$ 已由每个 MFPE 内的 DPA 计算完成；只剩外层的跨块尺度组合需要在阵列层面处理。

b) 朴素脉动实现基线：每个 MFPE 每周期消耗 k 个元素（即点积宽度），因此每隔 ⌈32/k⌉ 个垂直相邻的 MFPE 就会出现一个 MX 块边界：图 2(b) 中由 4 个 MFPE 组成的 BPE 是 MXFP4（k=8）的天然块单元，而 MXFP8（k=4）则跨越两个上下堆叠的 BPE。由于相邻块携带不同的共享尺度，其部分和不能直接相加。因此，一种直接的基线实现会为每列保留两个 FP32 值（一个块内累加器，以及一个单独缩放的跨块运行和），并在每个块边界配置一个专用的 FP32 跨块加法器。这条额外的 FP32 累加路径是支持 MX 的主要开销（第 IV-B 节）。

c) OBSR：对运行累加器重定基：OBSR 通过对传递的部分和重定基来消除第二条路径：在累加新块的乘积之前，将继承的累加器重新缩放至新块的组合尺度，使两者具有相同的尺度，并直接在现有的 FP32 级联路径上相加，无需第二个累加器，也无需跨块 FP32 加法器。

一般而言，这种重新缩放需要按照尺度比进行除法和乘法运算，这并非没有代价，其开销可能高于它所替代的第二条 FP32 累加路径。OBSR 的低开销恰恰源于 MX 块尺度采用 E8M0，即纯粹的二的幂：按二的幂进行重新缩放，只需对 FP32 指数字段做整数加法，而尾数保持不变。

具体而言，OBSR 维护单个运行 FP32 累加器 $\tilde { S }$ 和一个有符号的 10 位运行组合尺度寄存器 $s _ { \mathrm { c u r } } ,$，二者共同表示 $V = 2 ^ { s _ { \mathrm { c u r } } } \tilde { S } .$。在每个块边界，两个传入的块尺度首先组合成一个二的幂尺度，

$$
2 ^ {s _ {\text { new }}} = 2 ^ {s _ {a} ^ {(b + 1)}} \cdot 2 ^ {s _ {b} ^ {(b + 1)}} \implies s _ {\text { new }} = s _ {a} ^ {(b + 1)} + s _ {b} ^ {(b + 1)},\tag{2}
$$

随后仅通过调整累加器的指数字段，将其重定基至这一尺度，

$$
\tilde {S} \leftarrow \tilde {S} \cdot 2 ^ {s _ {\mathrm{cur}} - s _ {\mathrm{new}}} \equiv e (\tilde {S}) + = s _ {\mathrm{cur}} - s _ {\mathrm{new}},\tag{3}
$$

其中，$e ( \cdot )$ 为 FP32 指数字段，并为下一个块更新运行尺度，

$$
s _ {\mathrm{cur}} \leftarrow s _ {\mathrm{new}}.\tag{4}
$$

因此，边界逻辑仅涉及整数运算：一次用于组合尺度的加法（公式 (2)），一次用于计算 $s _ { \mathrm { c u r } } - s _ { \mathrm { n e w } }$ 的减法，以及一次对指数字段的加法（公式 (3)）：共两次加法和一次减法，尾数保持不变。随后，下一个块的乘积通过常规 MFPE 级联路径累加到重定基后的 S<sup>˜</sup> 中。

d) 特殊值与 MXFP6 转换：对于有限编码 0x00–0xFE，每个 E8M0 尺度均为二的幂 $2 ^ { e - 1 2 7 }$；0xFF 是预留的 NaN 编码（没有零或无穷大的编码），NaN 尺度会以静默 NaN 的形式通过 FP32 累加器传播。重定基操作跳过 ±0 和 Inf/NaN，并在重定基后的指数超出范围时钳位至 FP32 的边界值（溢出时为 ±∞，下溢时刷新为零）。由于 OBSR 保存的是 $\tilde { S } = V \cdot 2 ^ { - s _ { \mathrm { c u r } } }$，当某个块的尺度与累积历史相差很大时，$\tilde { S }$ 可能发生钳位，而保存 V 的独立 FP32 累加器则不会；不过，在实际轨迹中未发生此类事件（第 IV-C 节）。通过即时转换为 MXFP8，可以支持 MXFP6。

![](images/47e6fce44d8c452cdad9159dc545d3613102039dca53a19fbce62ae2aa85ac52.jpg)  
图 3. 单格式 DPA-PE 与吞吐量匹配的 FMA-PE；FMA 柱条由 k 个级联的单格式单元堆叠而成。

## III. 评估方法

## A. 硬件实现

我们使用 RTL 实现 MFSA，采用 Cadence Genus [24] 进行综合，并采用 Cadence Innovus [25] 完成整个布局布线流程，工艺为 TSMC 28 nm，工作角为 SS $0 . 7 2 \mathrm { { V } } / 1 2 5 ^ { \circ } \mathrm { { C } }$，目标频率为 1 GHz。布局布线采用签核级 RC 提取和自动时钟门控；阵列在核心利用率为 59% 时完成布线。功耗使用 Cadence Joules [26] 进行估算，输入包括布线后寄生参数和门级 GEMM 波形，其中的激活值与权重采样自 Qwen2.5-7B [27] 的预填充轨迹（采用 RMSNorm 之后的分布，而非均匀随机分布）。我们报告在激活数据流窗口内取平均的峰值稳态功耗，排除复位、权重预加载和排空周期。

## B. 基线

我们将 MFSA 与三种基线进行比较。(i) 为单独考察 DPA 在 PE 内部带来的收益，我们将每种单格式 MFPE 与吞吐量匹配的 FMA PE 进行比较，后者由 k 个级联的单级 FMA 单元构成（例如，将一个延迟为 4 的双项 BF16 MFPE 与两个级联的、各自延迟为 2 的 BF16 FMA 单元进行比较，使吞吐量与延迟均保持一致）。(ii) 为量化 OBSR 的效果，我们将采用 OBSR 的 16×16 MFSA 阵列与使用朴素 MX 基线的同一阵列进行比较，该基线维护第二个 FP32 级联字和每列一个跨块加法器。(iii) 为表征增量添加格式支持的成本，我们生成采用不同格式掩码的 MFSA RTL 变体，每个变体启用不同的格式子集，并对每个变体执行布局布线，以获得布线后面积。

## IV. 评估结果

## A. MFPE 面积效率

DPA-PE 与 FMA-PE：图 3 将每种单格式 DPA-PE 与吞吐量和延迟均匹配的级联 FMA-PE（第 III 节）进行比较：二者均能在 1 GHz 下满足时序要求，并在每个周期产生一个累加结果，FMA 柱条则堆叠表示其 k 个级联的单格式单元。DPA 的优势随子字并行度的提高而增大：当 k=2 时，级联 FMA-PE 的面积为 DPA-PE 的 1.5×（FP16 和 BF16 均如此），在 FP8（k=4）下扩大至 2.5×，在 FP4（k=8）下扩大至 4.5×。这正体现了延迟规格化与舍入的特点：将一次 IEEE-754 规格化与舍入的开销分摊到 k 个乘积上，随着 k 的增大可节省更多开销，因此，融合点积结构恰好在主导现代推理的低精度场景中获益最多。

![](images/5f2b186ca1a244e999e7fc9cb1244da2b5a82c33392f25a7fa80308838255727.jpg)

![](images/6652255338b993fc79388860a56cb991e7df92a17f0a5d21836d467e8fdc99e6.jpg)

<table><tr><td>FP32</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>√</td><td>√</td></tr><tr><td>FP16</td><td>✗</td><td>√</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>√</td><td>√</td><td>√</td></tr><tr><td>BF16</td><td>✗</td><td>✗</td><td>√</td><td>✗</td><td>✗</td><td>√</td><td>✗</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>CFP8</td><td>✗</td><td>✗</td><td>✗</td><td>√</td><td>✗</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>FP4</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>√</td><td>✗</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>INT16/8/4</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>√</td></tr></table>

图 4. 不同启用格式子集下的 PE 单元面积与布线后的 16×16 阵列单元面积。

![](images/b050c2008995f29de235534fb739411a4ff7d0bb9ab0a7ba766a7fcfccc48962.jpg)

![](images/b737aa3579a1179e6f730b1254e468eb4d3e40baade2758cb11a70b68fba79a4.jpg)  
图 5. 在同一个布线后的 16×16 CFP8/FP4 阵列上支持 MX 的代价。

多格式支持的代价：图 4 在 PE 级和阵列级遍历了各个启用格式子集；每个 MFPE 数据点均使用相应的格式掩码单独进行综合，每个阵列数据点均经过完整的布局布线。我们同时报告 PE 单元面积和布线后的阵列面积。对于布线后的阵列，扩展 FP 支持的代价很低：相对于 BF16+CFP8 阵列 $( 2 . 6 0 \mathrm { m m } ^ { 2 } )$，启用完整的六格式 FP 集合（增加 FP32、FP16 和 FP4）仅增加 +6.9% 的面积 $( 2 . 7 8 \mathrm { m m } ^ { 2 } )$，因为共享的跨精度数据通路得以复用，而无需为每种格式复制一份。在此基础上增加 INT16/8/4 支持，由于新增的整数符号扩展和 MUX 逻辑，面积进一步增加 $+ 1 5 . 4 \% \ ( 3 . 2 1 \mathrm { { m m ^ { 2 } } ) }$。

![](images/ac55cfebe2f94e93195ce1dc466d21bf16f64d9700ad8485feb3777df8dba237.jpg)

![](images/ddef9cdd65b658af85fe7b3da61fcdc09e544e9aa160bcce272027716d3c4d77.jpg)  
图 6. OBSR 在真实 LLM 激活值上的鲁棒性（预填充与解码、投影与注意力 GEMM）。（a）严格更接近 FP64 黄金参考值的输出占比。（b）两种具有非零累加误差的格式的 p95 误差随阵列规模的变化；未展示的格式逐位一致。

![](images/224aa734325d1f993f488a7a43f1420c2004c5b4f90ccd219e990020e3c4236d.jpg)  
(a)

![](images/67098d63ffb25e1272d7938e34514db60e2114c9dbcd0ecb0bd6817529f3994e.jpg)  
(b)  
图 7. 采用 TSMC 28 nm 工艺、运行频率为 1 GHz 的 16×16 MFSA 布线后版图（芯片尺寸为 2.40×2.40 mm<sup>2</sup>，签核单元面积为 $3 . 2 7 \mathrm { { \dot { m m } ^ { 2 } } }$）。（a）布线后的芯片。（b）将 256 个 MFPE 分别以阴影标示的同一芯片。

## B. OBSR 与朴素 MX 支持基线的比较

图 5 报告了在 16×16 CFP8+FP4 阵列上提供 MX 支持的代价，对比了 OBSR 与朴素 MX 基线（第 II-C 节）。相对于同一阵列不支持 MX 的版本，OBSR 的面积开销为 +2.7%，每次操作能耗开销为 +5.4%，而朴素基线的代价约为前者的 2.5–3×（面积开销为 +7.6%，每次操作能耗开销为 +16.8%）。因此，OBSR 将 MX 的增量面积开销降低了 64.2%，将每次操作的增量能耗开销降低了 67.5%。在启用所有格式的完整 MFSA 上，MX 支持仅额外增加 $+ 1 . 8 \%$ 的面积，使布线后的单元面积达到 $3 . 2 7 \mathrm { m m ^ { 2 } }$。这两种开销都会在每个 PE 行块中出现一次，因此，它们的比值由每个位置的开销决定，而非由阵列规模决定；扩展至 128×128 阵列时，由于固定开销（时钟树、全局信号）所占的比例增大，这两种开销均被略微摊薄。

## C. OBSR 的数值鲁棒性

在长序列浮点归约中，数值精度可能取决于各项的累加顺序。每次浮点加法都会根据操作数的指数进行对齐；因此，当一个较小的项与大得多的当前累加和相加时，较小操作数的低位可能会因舍入而丢失。尽管平衡的成对求和通常能通过减缓误差增长，相较于纯顺序累加提高精度，但最准确的顺序取决于数据，尤其是在存在相消或操作数幅值差异很大的情况下。

表 II  
完整 MFSA 各格式的吞吐量与能耗

<table><tr><td>输入格式</td><td>累加格式</td><td>吞吐量 (TOPS)</td><td>功耗 (W)</td><td>每次操作能耗 (pJ/op)</td></tr><tr><td>FP32</td><td>FP32</td><td>0.512</td><td>1.89</td><td>3.68</td></tr><tr><td>FP16</td><td>FP32</td><td>1.024</td><td>1.87</td><td>1.83</td></tr><tr><td>BF16</td><td>FP32</td><td>1.024</td><td>1.85</td><td>1.80</td></tr><tr><td>CFP8</td><td>FP32</td><td>2.048</td><td>1.85</td><td>0.90</td></tr><tr><td>FP4</td><td>FP32</td><td>4.096</td><td>1.85</td><td>0.45</td></tr><tr><td>INT16</td><td>INT32</td><td>0.512</td><td>1.74</td><td>3.41</td></tr><tr><td>INT8</td><td>INT32</td><td>1.024</td><td>1.72</td><td>1.68</td></tr><tr><td>INT4</td><td>INT32</td><td>2.048</td><td>1.74</td><td>0.85</td></tr><tr><td>MXFP8</td><td>FP32</td><td>2.048</td><td>1.85</td><td>0.90</td></tr><tr><td>MXFP4</td><td>FP32</td><td>4.096</td><td>1.85</td><td>0.45</td></tr></table>

这一效应与我们的 $N \times N$ 阵列有关，因为朴素第二累加器数据通路与 OBSR 会产生不同的 FP32 加法顺序。在朴素设计中，首先将每个内层 MX 块中的 32 个乘积归约为一个 FP32 部分和。随后，由第二个 FP32 累加器顺序累加所得到的 $N / 3 2$ 个块和。相比之下，OBSR 保留了原始权重驻留脉动阵列的累加顺序，这相当于直接顺序累加 N 个乘积。

我们使用预填充和解码轨迹中的 168 个 GEMM 来量化这一效应，覆盖投影层（down $\cdot / \mathsf { u p } / \mathsf { o \_ p r o j }$，三个解码器深度）以及注意力中的 $Q K ^ { \top }$ 和 P·V GEMM，其中 $P { \cdot } V \mathbf { \bar { s } }$ 归约规模随上下文增长；上下文长度为 74 至 ∼430 个词元，涵盖四个 LLM（Qwen2.5-0.5B/1.5B/3B-Instruct 和 SmolLM2-1.7B-Instruct）。我们将 FP32 输出与按幅值升序累加得到的 FP64 基准结果进行比较。对于块和可由 FP32 精确表示的 MXFP4 和 MXFP6 E2M3，OBSR 与朴素基线的结果逐比特完全一致；对于其余格式，两者的差异在几个 ULP 以内（在投影 GEMM 上，MXFP8 的 p95 在 6 ULP 以内，在注意力 GEMM 上则在 0.62 ULP 以内）。两种数据通路都没有始终更接近基准结果（图 6(a)）；朴素方法更接近基准结果的次数略多，因为其块内局部归约更接近成对求和。在从 16 行到 256 行的各种阵列尺寸下，以及预填充和解码两种阶段中，p95 曲线几乎重合（图 6(b)），而 ULP 值较大的离群情况仅在相消使 FP64 参考值接近零时出现：最大绝对误差低于 $3 \times 1 0 ^ { - 5 }$，比 MX 量化误差下限至少小 395×。

真实 GEMM 轨迹中没有发生钳位事件。定向构造的对抗性块（宽跨块动态范围、符号交替相消、离群值、接近非规格化数的缩放因子）在四个定向 GEMM 和所有阵列尺寸下共触发了 2242 次钳位事件。触发钳位时，MXFP4/MXFP6 的偏差被限制在 $\sim 1 0 ^ { - 2 }$ 以内，但 MXFP8 E4M3 的偏差可达到 $8 . 1 9 \times 1 0 ^ { 3 }$；因此，只有当某个块缩放因子与此前累加历史的偏离大到足以使 S<sup>˜</sup> 发生钳位时，OBSR 才会与基线产生差异，而这种情况在实测轨迹中从未发生。

对合成块缩放因子分布（高斯、随机游走和交替指数模式）的扫描表明，OBSR 的结果与朴素基线相符，且没有钳位事件，直至块缩放因子指数的离散程度达到 $\sigma \approx 3 2$；此时乘积覆盖 FP32 的整个指数范围，而实测 LLM 轨迹处于 $\sigma { \approx } 0 . 7 - 1 . 4$

表 III  
完整 MFSA 与 VMXDOTP 的 MX 格式 PPA

<table><tr><td rowspan="3">工作</td><td rowspan="3">面积 (mm2)</td><td rowspan="3">功耗 (W)</td><td colspan="6">格式</td></tr><tr><td colspan="3">MXFP8</td><td colspan="3">MXFP4</td></tr><tr><td>T/A</td><td>T/W</td><td>pJ</td><td>T/A</td><td>T/W</td><td>pJ</td></tr><tr><td>MFSA（本文）, 28 nm</td><td>3.27</td><td>1.85</td><td>0.63</td><td>1.11</td><td>0.90</td><td>1.25</td><td>2.21</td><td>0.45</td></tr><tr><td>Vmxdotp [28] ‡, 12 nm</td><td>0.18</td><td>0.17§</td><td>0.69</td><td>0.75</td><td>1.33</td><td>1.37</td><td>1.57</td><td>0.64</td></tr><tr><td>Vmxdotp [28] ‡, 28 nm†</td><td>0.99</td><td>0.40</td><td>0.13</td><td>0.32</td><td>3.10</td><td>0.25</td><td>0.67</td><td>1.49</td></tr></table>

$\mathrm { T } / \mathrm { A } = \mathrm { T O P S } / \mathrm { m m ^ { 2 } }$ $\mathrm { T } / \mathrm { W } = \mathrm { T O P S } / \mathrm { W } ,$ pJ = pJ/op；两种设计均运行于 1 GHz；MFSA 行对应已完成布线的 16×16 阵列（MX 模式功耗）。<sup>‡</sup> 仅包含 FPU；<sup>§</sup> 功耗由所报告的能效推算。<sup>†</sup> 一阶工艺节点缩放。

总之，在真实 LLM 工作负载下，OBSR 与朴素第二累加器之间的差异仅为低于量化误差的残差，同时通过数据通路复用节省面积和能耗；两种方案仅在远超实测统计范围的合成块缩放因子剧烈变化下才会产生偏差。然而，在采用大分块的商用引擎中（例如 128×128），分块尺寸远大于块大小，专用块归约路径也可以是一种有意优化精度的选择；朴素基线对这种组织方式进行了建模。

## D. 完整多格式 MFSA + OBSR 的最终 PPA

表 II 总结了所实现的 16×16 MFSA 阵列在各格式下的功耗、性能和面积（PPA），该阵列支持完整的格式集合：FP32、FP16、BF16、CFP8、FP4、INT16/8/4 和 MXFP8/MXFP4。图 7 展示了布线后版图，芯片尺寸为 $2 . 4 0 { \times } 2 . 4 0 \mathrm { m m } ^ { 2 }$，在 59% 的核心利用率下，签核单元面积为 $3 . 2 7 \mathrm { m m ^ { 2 } }$。每个 MFPE 的延迟为 4 个周期，阵列在布线后以 1 GHz 实现时序收敛。在真实 LLM 输入轨迹下，各格式的峰值功耗介于 1.72 至 1.89 W 之间。每次操作能耗随吞吐量变化，从 FP32 的 3.68 pJ/op 降至 FP4 的 0.45 pJ/op。

## V. 相关工作与比较

## A. 跨精度浮点单元

单元级跨精度 FPU 和融合点积单元，例如 FPnew [29]、跨精度浮点单元 [30]、Trans-Dot [22] 和 ExSdotp [31], [32]，确立了采用延迟舍入和 FP32 累加的多格式算术共享数据通路范式。MFSA 的 MFPE 将这一方向扩展至更完整的格式集合，通过编译时和运行时配置，在同一条数据通路上支持 BF16、CFP8 和 INT16/8/4。

## B. 块浮点与微缩放

在一组尾数之间共享一个指数的做法由来已久，从经典块浮点到 Flexpoint [33] 和 Microsoft Floating Point [34] 均采用了这一思路；OCP MX [21] 是其中一种细粒度实例，将由 32 个低精度元素组成的块与一个 E8M0 共享缩放因子配对。这类格式保留一个独立的块共享指数，并在执行与顺序无关的归约之前，将尾数与该指数对齐。近期的 MX 硬件在单元或集群范围内遵循这一模式：Vmxdotp [28] 和 MXDOTP [35] 为 RISC-V 核心增加了逐块 MX 点积运算，一种精度可扩展的数据通路面向边缘机器人 [36]，而 VEGETA [37] 将块缩放因子多播到稀疏/稠密 GEMM 阵列中；这些设计均在乘法器输入端应用缩放因子，并在各通道中复制缩放处理逻辑。相比之下，OBSR 在归约过程中，将每个块的缩放因子原地并入活跃的 FP32 运行累加器的指数中，使 MFSA 能够在大型权重驻留脉动阵列中原生支持 MX，无需独立的块指数、逐 PE 缩放逻辑或第二个跨块累加器。表 III 将 MFSA 的 MX 模式与 Vmxdotp [28] 进行了比较。

## C. 脉动阵列与精度可扩展阵列

包括 TPU/Trainium、Tesla Dojo [17]、RaPiD [38]、Bit Fusion [39] 和 BitBlade [40] 在内的脉动阵列与精度可扩展阵列，探索了相关的空间组织方式。RaPiD 为 FP 和 INT 使用独立的数据通路，而非共享硬件；商业芯片可能结合了多种格式，但其微架构尚未公开，多格式支持所带来的开销也尚未得到量化。此外，它们报告的结果包括 SRAM、控制逻辑、片上网络和存储控制器等芯片级组件，这些组件超出了本工作的范围，因此不适合直接比较。我们因此不与它们进行比较。MFSA 在 FP、INT 和 MX 模式之间保持统一的算术运算基础架构，并量化每种模式的增量成本。因此，其贡献在于架构层面：以先前的多格式 FDP PE 为基础，MFSA 展示了如何将该 PE 集成到完成布线的权重驻留脉动阵列中，同时实现不随模式变化的部分和传输以及低开销的 MX 支持。

## VI. 讨论

## A. 含小数部分的块缩放因子（NVFP4）

OBSR 利用了 E8M0 块缩放因子的特性：2 的幂次缩放因子可以通过一次精确的整数加法并入累加器的指数。相比之下，NVFP4 [23] 将包含 16 个元素的块与含小数部分的 E4M3 缩放因子配对，以在 4 位精度下恢复模型准确率；若要整体应用此类缩放因子，则每个边界都需要一个 FP 乘法器，而 OBSR 无法直接支持这种方式。

使用 OBSR 支持含小数部分的缩放因子的一种方法，是将缩放因子的指数与尾数分离，即 $s =$ $2 ^ { e } \left( 1 . m \right)$。其中，2 的幂次部分 $2 ^ { e _ { a } + e _ { b } }$ 沿用 OBSR 路径，无需改动。小数部分 $\mu = ( 1 . m _ { a } ) ( 1 . m _ { b } )$ 是一个 8 位整数，被并入 MFPE 内部的 FP4 点积运算：先对四个通道的幅值求和，再在与 FP32 加数合并之前，通过一次 11×8 乘法以 µ 进行缩放。根据分配律，这种并入是精确的，无需舍入或除法器，且缩放因子级联保持其 E8M0 位宽。

布线后，采用 OBSR+µ 支持 OCP MX 和 NVFP4 会额外增加 +11.8% 的单元面积和 +7.3% 的功耗。朴素方法使用额外的乘法器和一个跨块 FP32 加法器进行缩放还原，带来 +25.1% 的面积和 +24.0% 的功耗开销。我们还测试了第二种替代方案：在每个 PE 中对乘积进行缩放还原，从而完全避免跨块加法器和 OBSR，但需要在每个 PE 中增加一个乘法器和一次动态指数加法；该方案带来 +15.6% 的面积和 +15.7% 的功耗开销。

在这些方法中，OBSR+µ 仍然最为高效：与次优方法（逐 PE 缩放还原）相比，额外面积开销降低了 24%，额外功耗开销降低了 53%；与双累加器相比，额外面积/功耗开销则降低了 53%/70%。

## B. 以整数运算为主的边缘推理

整数格式共用 FPU 的乘法器，因此其处理速率为相同位宽 FP 格式的一半（INT8 的处理速率与 BF16/FP16 一致，而非 CFP8）。对于以整数运算为主的模型，可以增加一条专用整数路径，将整数吞吐量翻倍。在相同配置下，增加一条额外的 INT 点积路径以匹配 FP 的吞吐量，会使每个 PE 额外增加 1.39 k µm<sup>2</sup>（+10.3%）的面积。

## VII. 结论

MFSA 将共享数据通路的点积累加器（DPA）集成为完成布线的权重驻留脉动阵列的叶级单元，在 FP32、FP16、BF16、CFP8、FP4、INT16/8/4 以及 MX 块缩放模式之间保持单一的共享数据通路。两项设计选择实现了这种高效性：基于共享数据通路 DPA 的 MFPE，以及在线块缩放重定基机制，后者将 MX 块缩放因子并入运行中的 FP32 累加器，而非第二个跨块累加器。在采用 TSMC 28 nm 工艺、工作频率为 1 GHz、完成布线的 16×16 阵列中，从 BF16 到 FP4，吞吐量匹配的级联 FMA-PE 的面积为单格式 DPA-PE 的 1.5–4.5×；OBSR 将 MX 的增量面积开销降低了 64.2%，增量能耗开销降低了 67.5%，且在真实 LLM GEMM 中，其数值结果与朴素基线的差异在几个 ULP 以内。在 BF16+CFP8 脉动阵列上增加对所有其他格式的支持，需要额外增加 25.7% 的面积。OBSR 并不局限于 2 的幂次缩放因子：将 E4M3 缩放因子分解为沿 OBSR 路径处理的指数，以及并入 MFPE 点积运算的 8 位尾数，即可在支持 OCP MX 的同时支持 NVFP4，面积和功耗开销分别为 11.8% 和 7.3%，约为朴素跨块累加器方案的一半。这些设计共同使 MFSA 成为一种紧凑的多格式运算基础架构。

## REFERENCES

[1] C. Xu, D. Liu, A. Nassereldine, and J. Xiong, “FP64 is all you need: Rethinking failure modes in physics-informed neural networks,” in Advances in Neural Information Processing Systems (NeurIPS), 2025.

[2] J. Hayford, J. Goldman-Wetzler, E. Wang, and L. Lu, “Speeding up and reducing memory usage for scientific machine learning via mixed precision,” 2024. [Online]. Available: https://arxiv.org/abs/2401.16645

[3] S. Wang and P. Kanwar, “BFloat16: The Secret to High Performance on Cloud TPUs,” Google Cloud Blog, Aug. 2019, accessed: 2026-06-09. [Online]. Available: https://cloud.google.com/blog/products/ai-machine -learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus

[4] D. Kalamkar, D. Mudigere, N. Mellempudi, D. Das, K. Banerjee, S. Avancha, D. T. Vooturi, N. Jammalamadaka, J. Huang, H. Yuen, J. Yang, J. Park, A. Heinecke, E. Georganas, S. Srinivasan, A. Kundu, M. Smelyanskiy, B. Kaul, and P. Dubey, “A study of BFLOAT16 for deep learning training,” 2019. [Online]. Available: https://arxiv.org/abs/1905.12322

[5] P. Micikevicius, D. Stosic, N. Burgess, M. Cornea, P. Dubey, R. Grisenthwaite, S. Ha, A. Heinecke, P. Judd, J. Kamalu, N. Mellempudi, S. Oberman, M. Shoeybi, M. Siu, and H. Wu, “FP8 formats for deep learning,” 2022. [Online]. Available: https: //arxiv.org/abs/2209.05433

[6] NVIDIA Corporation, “NVIDIA H100 Tensor Core GPU Architecture,” NVIDIA Whitepaper, 2022, GTC 2022. Accessed: 2026-06-09. [Online]. Available: https://resources.nvidia.com/en-us-hopper-architect ure/nvidia-h100-tensor-c

[7] ——, “NVIDIA Blackwell Architecture Technical Overview,” NVIDIA Technical Brief, 2024, accessed: 2026-06-09. [Online]. Available: https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-arc hitecture-technical-brief

[8] B. Darvish Rouhani, R. Zhao, V. Elango, R. Shafipour, M. Hall, M. Mesmakhosroshahi, A. More, L. Melnick, M. Golub, G. Varatkar, L. Shao, G. Kolhe, D. Melts, J. Klar, R. L’Heureux, M. Perry, D. Burger, E. Chung, Z. S. Deng, S. Naghshineh, J. Park, and M. Naumov, “With shared microexponents, a little shifting goes a long way,” in Proceedings of the 50th Annual International Symposium on Computer Architecture, ser. ISCA ’23. New York, NY, USA: Association for Computing Machinery, 2023.

[9] DeepSeek-AI, “DeepSeek-V4: Towards highly efficient million-token context intelligence,” 2026. [Online]. Available: https://arxiv.org/abs/26 06.19348

[10] NVIDIA, “Nemotron 3 Ultra: Open, efficient mixture-of-experts hybrid Mamba-Transformer model for agentic reasoning,” 2026. [Online]. Available: https://arxiv.org/abs/2606.15007

[11] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in First Conference on Language Modeling (COLM), 2024.

[12] Moonshot AI, “Kimi K2 Thinking Model Card,” Hugging Face model card, 2025. [Online]. Available: https://huggingface.co/moonshotai/Ki mi-K2-Thinking

[13] Google Cloud, “TPU v5p,” Google Cloud Documentation, 2026, accessed: 2026-06-09. [Online]. Available: https://docs.cloud.google.co m/tpu/docs/v5p

[14] Amazon Web Services, “Trainium2 Architecture,” AWS Neuron Documentation, 2024, accessed: 2026-06-09. [Online]. Available: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/ arch/neuron-hardware/trainium2.html

[15] ——, “Trainium3 Architecture,” AWS Neuron Documentation, 2026, accessed: 2026-06-09. [Online]. Available: https://awsdocs-neuron.read thedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/traini um3.html

[16] Google Cloud, “TPU7x (Ironwood),” Google Cloud Documentation, 2026, accessed: 2026-06-09. [Online]. Available: https://docs.cloud.goo gle.com/tpu/docs/tpu7x

[17] Tesla, Inc., “Tesla Dojo Technology: A Guide to Tesla’s Configurable Floating Point Formats and Arithmetic,” Tesla white paper, 2021, accessed: 2026-06-09. [Online]. Available: https://tesla.relayto.com/e/te sla-dojo-technology-nwq57gsbyb33x

[18] N. P. Jouppi et al., “In-datacenter performance analysis of a tensor processing unit,” SIGARCH Comput. Archit. News, vol. 45, no. 2, pp. 1–12, Jun. 2017.

[19] N. P. Jouppi, D. Hyun Yoon, M. Ashcraft, M. Gottscho, T. B. Jablin, G. Kurian, J. Laudon, S. Li, P. Ma, X. Ma, T. Norrie, N. Patil, S. Prasad, C. Young, Z. Zhou, and D. Patterson, “Ten lessons from three generations shaped Google’s TPUv4i: Industrial product,” in 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), 2021, pp. 1–14.

[20] N. Jouppi, G. Kurian, S. Li, P. Ma, R. Nagarajan, L. Nai, N. Patil, S. Subramanian, A. Swing, B. Towles, C. Young, X. Zhou, Z. Zhou, and D. A. Patterson, “TPU v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings,” in Proceedings of the 50th Annual International Symposium on Computer Architecture, ser. ISCA ’23. New York, NY, USA: Association for Computing Machinery, 2023.

[21] Open Compute Project, “OCP Microscaling Formats (MX) Specification, Version 1.0,” Open Compute Project, 2023, accessed: 2026-06-09. [Online]. Available: https://www.opencompute.org/documents/ocp-mic roscaling-formats-mx-v1-0-spec-final-pdf

[22] J. Wang, M. Nie, S.-C. Lin, C.-J. R. Shi, and A. Li, “TransDot: An area-efficient reconfigurable floating-point unit for trans-precision

dot-product accumulation for FPGA AI engines,” 2026. [Online]. Available: https://arxiv.org/abs/2605.07245

[23] NVIDIA, “Pretraining large language models with NVFP4,” 2025. [Online]. Available: https://arxiv.org/abs/2509.25149

[24] Cadence Design Systems, “Genus Synthesis Solution,” Cadence product documentation. [Online]. Available: https://www.cadence.com/en US/ home/tools/digital-design-and-signoff/synthesis/genus-synthesis-solutio n.html

[25] ——, “Innovus Implementation System,” Cadence product documentation. [Online]. Available: https://www.cadence.com/en US/home/tools/digital-design-and-signoff/soc-implementation-and-flo orplanning/innovus-implementation-system.html

[26] ——, “Joules RTL Power Solution,” Cadence product documentation. [Online]. Available: https://www.cadence.com/en US/home/tools/digit al-design-and-signoff/power-analysis/joules-rtl-power-solution.html

[27] Qwen Team, “Qwen2.5 Technical Report,” 2024. [Online]. Available: https://arxiv.org/abs/2412.15115

[28] M. Wipfli, G. <sup>˙</sup>Islamoglu, N. K. Purayil, A. Garofalo, and L. Benini,˘ “Vmxdotp: A RISC-V vector ISA extension for efficient microscaling (MX) format acceleration,” in 2026 Design, Automation & Test in Europe Conference (DATE). IEEE, 2026, pp. 1–7.

[29] S. Mach, F. Schuiki, F. Zaruba, and L. Benini, “FPnew: An open-source multiformat floating-point unit architecture for energy-proportional transprecision computing,” IEEE Transactions on Very Large Scale Integration (VLSI) Systems, vol. 29, no. 4, pp. 774–787, Apr. 2021.

[30] G. Tagliavini, S. Mach, D. Rossi, A. Marongiu, and L. Benini, “A transprecision floating-point platform for ultra-low power computing,” in 2018 Design, Automation & Test in Europe Conference & Exhibition (DATE). IEEE, 2018, pp. 1051–1056.

[31] L. Bertaccini, G. Paulin, T. Fischer, S. Mach, and L. Benini, “MiniFloat-NN and ExSdotp: An ISA extension and a modular open hardware unit for low-precision training on RISC-V cores,” in 2022 IEEE 29th Symposium on Computer Arithmetic (ARITH). IEEE, 2022, pp. 1–8.

[32] L. Bertaccini, G. Paulin, M. Cavalcante, T. Fischer, S. Mach, and L. Benini, “MiniFloats on RISC-V cores: ISA extensions with mixedprecision short dot products,” IEEE Transactions on Emerging Topics in Computing, vol. 12, no. 4, pp. 1040–1055, Oct. 2024.

[33] U. Koster, T. J. Webb, X. Wang, M. Nassar, A. K. Bansal, W. H.¨ Constable, O. H. Elibol, S. Gray, S. Hall, L. Hornof, A. Khosrowshahi, C. Kloss, R. J. Pai, and N. Rao, “Flexpoint: An adaptive numerical format for efficient training of deep neural networks,” in Advances in Neural Information Processing Systems (NeurIPS), 2017.

[34] B. Darvish Rouhani, D. Lo, R. Zhao, M. Liu, J. Fowers, K. Ovtcharov, A. Vinogradsky, S. Massengill, L. Yang, R. Bittner, A. Forin, H. Zhu, T. Na, P. Patel, S. Che, L. C. Koppaka, X. Song, S. Som, K. Das, S. Tiwary, S. Reinhardt, S. Lanka, E. Chung, and D. Burger, “Pushing the limits of narrow precision inferencing at cloud scale with Microsoft floating point,” in Advances in Neural Information Processing Systems (NeurIPS), 2020.

[35] G. <sup>˙</sup>Islamoglu, L. Bertaccini, A. S. Prasad, F. Conti, A. Garofalo, and˘ L. Benini, “Mxdotp: A RISC-V ISA extension for enabling microscaling (MX) floating-point dot products,” in 2025 IEEE 36th International Conference on Application-specific Systems, Architectures and Processors (ASAP). IEEE, 2025, pp. 81–84.

[36] S. Cuyckens, X. Yi, N. S. Murthy, C. Fang, and M. Verhelst, “Efficient precision-scalable hardware for microscaling (MX) processing in robotics learning,” in 2025 IEEE/ACM International Symposium on Low Power Electronics and Design (ISLPED). IEEE, 2025, pp. 1–7.

[37] K. B. Nine, C. Talley, A. S. Mandadi, T. Krishna, and A. Raychowdhury, “Optimizing sparse/dense VEGETA accelerator performance with microscaling quantization,” in 2025 IEEE International Symposium on Circuits and Systems (ISCAS), 2025, pp. 1–5.

[38] S. Venkataramani et al., “RaPiD: AI accelerator for ultra-low precision training and inference,” in 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), 2021, pp. 153–166.

[39] H. Sharma, J. Park, N. Suda, L. Lai, B. Chau, J. K. Kim, V. Chandra, and H. Esmaeilzadeh, “Bit fusion: Bit-level dynamically composable architecture for accelerating deep neural networks,” in 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), 2018, pp. 764–775.

[40] S. Ryu, H. Kim, W. Yi, and J.-J. Kim, “BitBlade: Area and energyefficient precision-scalable neural network accelerator with bitwise summation,” in 2019 56th ACM/IEEE Design Automation Conference (DAC), 2019, pp. 1–6.

> 来源与版本：用户提供的 arXiv:2610.19645v1（2026-10-14 预印本），不标作已录用会议论文。经 MinerU 解析，由 Codex/GPT 翻译与解读，未调用 GLM 翻译 API。全文翻译保留参考文献；其余报告为解读。

[原文入口](https://arxiv.org/abs/2610.19645) · [多精度算术目录](../index.md)
