# An Algorithm-Hardware Co-design Based on Revised Microscaling Format Quantization for Accelerating Large Language Models 原文翻译

# 基于修订版微缩放格式量化的大语言模型加速算法与硬件协同设计

Yingbo Hao $^{\dagger*}$ , Huangxu Chen $^{\ddagger*}$ , Yi Zou $^{\dagger\S}$ , Yanfeng Yang $^{\dagger}$ $^{\dagger}$ 华南理工大学，$^{\ddagger}$ 香港科技大学（广州），中国广州 $^{\dagger}\{202410193423@mail., zouyi@, mi_yyf@mail.\}scut.edu.cn, ^{\ddagger}\{hchen499@connect.\}hkust-gz.edu.cn$

摘要——窄位宽数据格式对于降低现代深度学习应用的计算和存储成本至关重要，尤其是对于基于大语言模型（LLM）的应用。微缩放（MX）格式已被证明可以在现有推理框架中直接替代基准 FP32，且给用户带来的额外负担很小。然而，将这种新格式部署到现有硬件系统中仍然具有挑战性，低精度 LLM 推理的主流解决方案仍然是低比特量化。这尤其限制了此类 LLM 在大规模实际部署中的战略性应用。在本工作中，我们提出了一种算法与硬件协同设计，采用两级修订版 MX 格式量化（RMFQ）和修订版 MX 格式加速器（RMFA）架构设计。RMFQ 提出了修订版 MX（RMX）格式，并提供了一种采用创新分组方向的新型量化框架。此外，RMFA 提供了一种适配 RMX 的硬件架构和一种 RMX 编码方案。最终，RMFQ 将 4 比特和 6 比特量化的性能极限推向了新的最先进水平，而 RMFA 超越了 OliVe 等现有离群值感知加速器，实现了 1.28× 的加速和 1.31× 的能耗降低。

索引词——大语言模型（LLM）、修订版微缩放（RMX）格式、加速器、两级量化。

## I. 引言

人工智能（AI）能力的最新进展，例如对话式问答和智能代码补全 $[1]$，主要通过扩大底层深度学习模型的规模来实现 $[2]$。然而，这种规模扩展显著增加了对计算能力和内存容量的需求，以满足训练和部署此类大规模模型的要求。

与此同时，模型量化已被证明是一种有效技术，它通过使用更窄的位宽表示数值来降低内存使用量和计算需求 [3]。面向 AI 的硬件如今普遍原生支持低位宽格式。例如，NVIDIA 自 Turing 架构起就在张量核心中引入了 INT8 整数类型 [4]。具体而言，8 比特量化需要为每个张量设置一个缩放因子，并且在模型参数量相对较小时已展现出显著效果 [5], [6]。然而，在处理基于 Transformer 的大语言模型（LLM）时，模型性能很容易受到极少部分 $(< 0.1\%)$ 离群值的影响 [7], [8]。例如，对于 OPT 模型，常规量化方法在参数规模达到 6.7B 时就会失效 [8]。

表 I
量化架构比较

<table><tr><td rowspan="2">方法</td><td rowspan="2">编码方案</td><td colspan="2">片外 / 片上存储器</td><td rowspan="2">计算位宽</td><td rowspan="2">额外面积占比</td></tr><tr><td>是否对齐</td><td>位宽</td></tr><tr><td>OLAccel [12]</td><td>坐标列表</td><td>否</td><td>5.81</td><td>4.36</td><td>71%</td></tr><tr><td>GOBO [11]</td><td>坐标列表</td><td>否</td><td>4.04/6.81</td><td>16</td><td>55%</td></tr><tr><td>OliVe [10]</td><td>离群值—牺牲值对</td><td>是</td><td>4.23/8</td><td>4.23/8</td><td>4%</td></tr><tr><td>RMFA（本文）</td><td>RMX 检测</td><td>是</td><td>4.06/6.06</td><td>4.06/6.06</td><td>2%</td></tr></table>

此外，研究表明，由于动态范围有限，张量级缩放不足以满足基于 Transformer 的 LLM 的需求。为处理离群值，研究者提出了多种方法，其中一种常见做法是对正常值和离群值采用不同的处理方式 $[9]$–$[11]$。例如，OliVe $[10]$ 采用离群值—牺牲值对（OVP）量化，通过牺牲正常值来在局部容纳离群值。GOBO $[11]$ 和 OLAccel $[12]$ 使用坐标列表来区分离群值与正常值。然而，这些进展不仅导致了不对齐的内存存储和访问，还因额外的离群值控制器而造成了较大的面积开销。

另一种方法是使用新的数据格式来同时容纳离群值和常规值。研究表明，使用微缩放数据格式，即将缩放因子关联到张量的细粒度子块，是一种很有前景的基于 Transformer 的 LLM 量化方法 $[13]$–$[16]$，而且在低于 8 比特的情况下更加有效 $[14]$–$[16]$。然而，这些工作都针对现有 GPU 进行了优化，而现有 GPU 在架构上与这些新数据格式并不兼容，往往需要数据识别和重排等复杂的动态量化算法。微缩放（MX）数据格式 $[17]$ 已被提出，成为面向深度学习推理的一系列微缩放数据格式的首个开放标准。然而，MX 的部署仍有很长的路要走，需要产业界和学术界共同努力。

本文提出了一种算法与硬件协同设计，采用两级修订版 MX 格式量化（RMFQ）和修订版 MX 格式加速器（RMFA）架构设计，以解决 LLM 的低精度量化问题。在软件算法方面，我们设计了一种基于修订版 MX（RMX）格式的新型量化框架，以更有效地处理离群值。同时，在硬件架构方面，我们设计了 RMFA 以支持 RMX 格式的计算，主要包括适配 RMX 格式的脉动阵列和一种 RMX 编码方案。表 I 将 RMFQ 与以往的离群值感知量化架构进行了比较。RMFA 以极小的硬件开销实现了高性能，并且能够高效地集成到现有硬件加速器中。主要贡献如下：

\- 我们提出了 RMFQ，这是一种使用 RMX 格式的两级量化方法，引入了自适应整数/浮点和一种新颖的分组方向。与原始 MX 格式相比，该方法显著降低了等效位宽，并能更有效地处理 LLM 中的离群值。

\- 我们设计了 RMFA 以利用 RMFQ 的优势，主要包括适配 RMX 的脉动阵列架构和一种 RMX 编码方案。该架构优化了数据传输和存储，从而有效降低了数据传输的带宽需求。

\- 我们在具有代表性的 LLM 上实现并验证了 RMFQ 的有效性。一方面，RMFQ 在 LLM 的 4 比特和 6 比特量化中实现了最先进（SOTA）的量化精度。另一方面，对 RMFA 的评估表明，相比 ANT、OLAccel 和 OliVe，其分别实现了 $3.13 \times$、$1.88 \times$ 和 $1.28 \times$ 的加速。

## II. 背景与所提方法

在本节中，我们总结了 LLM 量化面临的挑战，并提出了 RMX 格式。在此基础上，我们提出了新的量化方法。

## A. LLM 量化的挑战

过去几年中，人们提出了多种量化方法。两类主要方法是训练后量化（PTQ）[3]、[7]、[10]、[13]、[15] 和量化感知训练（QAT）[3]、[9]、[18]。QAT 需要微调，并能够实现更低位宽的量化 [19]。然而，重新训练 LLM 极其消耗资源，因此 PTQ 成为更适合 LLM 的方法。近期最先进的 PTQ 方法 [7]、[15] 表明，通过基于分组的量化方法（例如，为张量中 g 个连续权重设置一个缩放因子），可以将 LLM 中的权重量化至四位甚至三位，同时保持具有竞争力的模型性能。

然而，由于存在离群值，使用相同方法对激活进行量化的效果并不好。一种主流方法是对涉及离群值的计算使用更高精度 [8]、[14]、[16]，但这同时会引入复杂的动态量化感知过程，从而降低速度并增加系统实现的难度。图 1 展示了 OPT-6.7B [20] 中一个典型线性层的激活张量和权重张量的绝对值。可以看出，离群值仅存在于激活的特定通道中。直观的方法是对激活使用逐通道量化（例如，为激活张量中的每个通道设置一个缩放因子）。然而，逐通道激活量化无法很好地映射到硬件加速的 GEMM 内核。假设采用一种使用 INT4 的典型逐通道量化方案。量化过程可表示为：

$$
X ^ {I N T 4} = R o u n d (\frac {X ^ {F P 1 6}}{\Delta}), \Delta = \frac {\max (| X ^ {F P 1 6} |)}{2 ^ {4 - 1} - 1},\tag{1}
$$

其中，$x^{FP16}$ 是待量化的张量，$x^{INT4}$ 是其量化后的对应张量，$\Delta$ 是量化步长。矩阵乘法过程可表示为：

$$
Y = \operatorname{diag} (\Delta_ {x} ^ {F P 1 6}) \cdot (X ^ {I N T 4} \cdot W ^ {I N T 4}) \cdot \operatorname{diag} (\Delta_ {w} ^ {F P 1 6}).\tag{2}
$$

在这些内核中，缩放只能沿矩阵乘法的外侧维度（即激活的词元维度）进行。因此，以往的工作都使用逐词元或逐张量的激活量化，即使有工作使用逐通道平滑因子来保持线性层在数学上的等价性，其过程仍然很复杂 $[13]$。

![](images/e56261cd9062f879b5f57f960ec67019c38e396d0d2b1791be5d81117ce41737.jpg)  
(a)

![](images/11ba59bb3e884916025a20e117a9cb862a1749b1c44736dc45e2eaa411234c3e.jpg)  
(b)  
图 1. OPT-6.7B 中 (a) 激活张量和 (b) 权重张量的绝对值。
B. RMX 格式

MX 格式已在包括 LLM 在内的多种深度学习基准测试的推理中展现出有效性 $[21]$。然而，该方法依赖于伪仿真，而当前 GPU 架构终究并不兼容 MX 格式。因此，将 MX 格式融入现有量化方法，并提出实用的硬件适配架构，对于充分释放 MX 格式的潜力至关重要。基于这一分析，我们提出了 RMX 格式。图 2 对比了 MX 格式与我们提出的 RMX 格式。

![](images/61ff4373d740f7a9b01eedfabfc56df9adc4dd26bc574da476b544e0c9367134.jpg)  
图 2. MX 格式与 RMX 格式的比较

1) 混合类型与精度：首先，我们将固定的整数（例如 MXINT4）或浮点数（例如 MXFP4、MXFP6）替换为采用混合整数/浮点类型与 4 位/6 位精度的格式。这基于以下观察：在神经网络的不同层中，各张量的分布特征不同，可能呈现类均匀分布、类高斯分布或类拉普拉斯分布 [9]。对不同层使用相同的数据类型并不合适，例如对激活张量使用 MXINT4。此外，这一修改对硬件友好。这 6 种数据类型都可以在一个支持 E3M3（3 位隐式指数、3 位尾数）的单元中实现乘加（MAC）运算。请注意，补零用于说明 RMX 格式的计算机制；4/6 位 RMX 格式在内存存储和访问时仍保持其原始位宽。

2) 降低等效位宽 $bit_{e}$：其次，我们将单级的分组缩放因子替换为逐张量加分组的两级缩放因子。由于 MX 和 RMX 格式都是粒度较小的微缩放格式，分组缩放因子会带来相当大的内存开销。为此，我们提出了一个记为 $bit_{e}$ 的新指标，用于计算微缩放格式的等效位宽，并衡量内存节省效果。$bit_{e}$ 定义为：

$$
b i t _ {e} = \frac {b i t \cdot g + b i t _ {s}}{g},\tag{3}
$$

其中，bit 是量化位宽，g 是分组大小，$bit_{s}$ 是分组缩放因子的位宽。假设 g = 64，我们将在第 IV-B 节中对此进行说明。

例如，在 4 位量化中，假设 OPT-6.7B 在 WikiText [22] 上的一个激活张量形状为 [4096, 2048]。逐张量缩放因子分摊到每组的平均位宽开销为 $16 \div \frac{4096 \cdot 2048}{64} = 10^{-4}$，这一数值太小，因此在 $bit_e$ 中不予考虑。所以，$bit_s$ 对 $bit_e$ 至关重要。如图 3 所示，我们计算了 OPT-6.7B 和 GPT2-XL [1] 的最大共享指数和共享指数溢出（$>2^3 - 1 = 7$）情况。显然，权重的共享指数很少发生溢出。然而，激活中存在大量溢出。因此，我们选择 4 作为分组共享指数的位宽，使最大共享指数为 15。由此，原始 MX 格式的 $bit_e$ 为 $\frac{4 \cdot 32 + 8}{32} = 4.25$，而所提出的 RMX 格式的 $bit_e$ 为 $\frac{4 \cdot 64 + 4}{64} = 4.06$。

![](images/88b135f76178b14cf6022a39a399cd8a77f5c014b71b411a1660282ca7ee1453.jpg)

![](images/1288caf4f9804c32979ace4f4ecc3679e30eae5a768ee8ddad1245517dfb1797.jpg)  
(a) OPT-6.7B 在 WikiText 上  
(b) GPT2-XL 在 WikiText 上  
图 3. 代表性 LLM 的最大共享指数和共享指数溢出百分比。权重的溢出百分比太小，因此在图中难以看清。

## C. 量化方法

我们的直觉是利用 RMX 格式中的隐式缩放因子，使分组量化能够沿矩阵乘法的内部维度进行。图 4 展示了所提出的两级量化方法。我们采用典型的逐张量方法进行 $1^{\mathrm{st}}$ 量化。为处理激活张量的通道级离群值，我们调整了以往对激活张量采用的分组方向，改为沿通道方向分组。对于权重张量，我们发现其分布模式与激活张量相同，如图 1 所示。因此，我们也沿输出通道方向分组，以处理权重张量中不太显著的离群值。尽管如此，我们在第 III 节提出了兼容这种分组方向的硬件系统。因此，$1^{st}$ 4 位量化过程可表示为：

$$
\overline {{X}} ^ {F P 1 6} = \frac {X ^ {F P 1 6}}{\Delta^ {1 s t}}, \Delta^ {1 s t} = \frac {\max (| x ^ {F P 1 6} |)}{\max (| Q ^ {g r i d} |)},\tag{4}
$$

其中，$Q^{grid}$ 为量化网格（例如，RMXINT4 对应 [-8,..., 6, 7]），$\overline{X}^{FP16}$ 为 $1^{st}$ 量化结果。一个组的 $2^{nd}$ 量化过程可表示为：

$$
\begin{array}{c} \frac {1}{\Delta^ {2 n d}} = \max (\lfloor \log_ {2} \frac {\max (| Q ^ {g r i d} |)}{\max (| \overline {{X}} _ {g r o u p} ^ {F P 1 6} |)} \rfloor - 1, 1 5), \\ X ^ {R M X 4} = R o u n d (\frac {\overline {{X}} _ {g r o u p} ^ {F P 1 6}}{\Delta^ {2 n d}}), \end{array}\tag{5}
$$

(6)

其中，$\overline{X}_{group}^{FP16}$ 为一个组的 $1^{st}$ 量化结果，$\lfloor\cdot\rfloor$ 为向下取整函数，$X^{RMX4}$ 为 $2^{nd}$ 量化结果。通过执行上述量化过程，我们可以获得最优的组级共享指数。

![](images/48bbea3d281a70ef577e1352af18f67c59bd9bb5335ef2f74f917380c9e1b769.jpg)

![](images/b07cc6cb6415cb02c88b27f18b5e7d2aba709df4f907e39dd25c44ea4bb8e798.jpg)  
图 4. 所提出的两级修订 MX 格式量化方法（即 RMFQ）示意图。

我们还采用均方误差（MSE）作为量化优化目标，许多先前研究 [3]、[9]、[23] 已证明其有效性。MSE 指标可由下式定义：

$$
M S E = \int (X ^ {F P 1 6} - X ^ {R M X 4}) ^ {2} p (x) d x,
$$

其中，$p(x)$ 为概率密度函数。通过搜索最小 MSE，我们可以为每一层选择最优的 RMX 数据类型，并以更高精度对 MSE 最大的层进行量化，即进行混合数据类型/精度量化。

## III. 硬件架构

本节详细介绍本工作在 RMFA 方面的贡献，主要包括改进的脉动阵列和一种硬件友好的 RMX 编码方案。

![](images/f454ba2b0adf9b7b04c86583ed288c3c29f6a8b4eeba461cee0d2a6ae58ac3b4.jpg)  
图 5. RMFA 脉动阵列架构的简要概览。

## A. 用于 RMFA 的脉动阵列架构

图 5(a) 展示了所提出的脉动阵列架构。鉴于组大小可能与阵列大小不同，有必要将脉动阵列划分为与最小可行组大小相对应的分块。每个分块在每个周期独立计算一个组，共享指数则以脉动方式传播。显然，对于给定的阵列大小，减小组大小会增加分块数量，从而带来更多硬件开销。例如，当阵列大小为 $a = 16$、组大小为 $g = 4$ 时，额外的硬件需求包括 $(\frac{a}{g})^2 \times (2g - 1) = 112$ 个 5 位寄存器、$2(\frac{a}{g})^2 = 32$ 个 4 位寄存器，以及 $(\frac{a}{g})^2 = 16$ 个 4 位加法器。尽管如此，减小组大小虽然可以提高精度，但也会显著增加硬件和带宽开销。

图 5(b) 展示了针对 RMX 格式设计的一个分块，以组大小 g = 4 为例。由于脉动阵列的固有特性，后续列的数据必须延迟一个周期传输到下一列 MAC 单元。因此，属于同一组的数据首先通过网格进行划分，对应的共享指数用相同颜色表示。MX 数据类型的矩阵运算公式表示为：

$$
\sum_ {1} ^ {n} \sum_ {1} ^ {g} 2 ^ {e _ {s} ^ {a}} a _ {m x} \times 2 ^ {e _ {s} ^ {w}} w _ {m x} = \sum_ {1} ^ {n} \sum_ {1} ^ {g} (a _ {m x} \times w _ {m x}) \ll e _ {s} ^ {a} + e _ {s} ^ {w},
$$

其中，$\ll$ 表示位移操作，$e_{s}^{a}$ 和 $e_{s}^{w}$ 分别为激活与权重的共享指数，$a_{mx}$ 和 $w_{mx}$ 为采用 RMX4 或 RMX6 表示的激活与权重，n 和 g 分别为组数和组大小。因此，我们首先将激活与权重的共享指数相加，然后对所得乘积进行移位。

修改后的 MAC 单元如图 5(c) 所示。与传统 MAC 单元相比，所提出的模型需要将双输入移位器替换为三输入移位器，并增加累加器的位宽，以应对共享指数带来的移位范围压力。

## B. 硬件友好的 RMX 编码方案

共享指数的计算在 RMFA 中至关重要。一旦确定共享指数，就可以根据 RMX 格式规范直接应用编码和解码算法。量化电路如图 6 所示。非线性运算产生的数据通常采用 FP16 或 BF16 格式，首先使用软件校准阶段预先确定的因子进行缩放，然后截断为 16 位整数。这种截断方式为数值幅度较小的分组保留了精度。对于每个数据组，电路识别高 16 位中第一个“1”的位置，并将该位置编码为一个 4 位整数，以确定该组的共享指数。随后，将数据右移、截断为 8 位，并根据指定的 RMX 类型量化为 RMX 数据。

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
算法 1 RMX 编码算法
输入：Group_Data $_{G,N}$ , Scale, RMX_Type
输出：Group_Data $_{G,N}$ , Shared_Exponent $_{N}$ 
1: Group_Data $_{G,N} \leftarrow$  Group_Data $_{G,N}$  * Scale
2: Group_Max_Data $_{N} \leftarrow$  Max(Group_Data $_{G,N}$ , dim = 0)
3: Max_Data  $\leftarrow$  Max(Group_Max_Data $_{N}$ )
4: Shared_Exponent $_{N} \leftarrow$  Frexp( $\frac{\text{Max\_Data}}{\text{Group\_Max\_Data}_N}$ )
5: Group_Data $_{G,N} \leftarrow$  Group_Data $_{G,N}$  * (2 $^{\text{Shared\_Exponent}_N}$ )
6: Group_Data $_{G,N} \leftarrow$  Quantize(Group_Data $_{G,N}$ , RMX_Type)
7: 返回 Group_Data $_{G,N}$ , Shared_Exponent $_{N}$
</div>

在软件层面，主要目标是模拟量化算法引入的误差。算法 1 概述了 RMX 编码过程。首先，对分组数据进行缩放。随后，通过比较组内最大值与全局最大值来确定共享索引，并利用 Frexp 函数提取浮点数的指数。最后，将所有数据量化为整数，并根据共享索引执行移位，以近似模拟张量量化引入的误差。

在硬件推理期间，量化电路如图 6 所示。非线性运算产生的数据通常采用 FP16 或 BF16 格式，首先使用软件校准阶段预先确定的因子进行缩放，然后截断为 16 位整数。这种截断方式为数值幅度较小的分组保留了精度。对于每个数据组，电路识别高 16 位中第一个“1”的位置，并将该位置编码为一个 4 位整数，以确定该组的共享指数。随后，将数据右移、截断为 8 位，并根据指定的 $RMX\_type$ 量化为 RMX 数据。

![](images/6b66a100fb43e1b24d6226e03c144e7d19ecfc73b5edd841e0ecd01e2abc8061.jpg)  
图 6. RMX 编码器的硬件架构

## IV. 实验

## A. 方法

量化框架与基准 为了评估我们提出的 RMFQ 框架，我们在 PyTorch 上开展实验。为了展示所提方法在 LLM 上的普遍适用性，我们在模型规模从 6.7B 到 13B 不等的 OPT 系列 $[20]$、LLaMA $[24]$、Llama-2 $[25]$ 以及 GPT2-XL $[1]$ 上评估了该方法。当模型规模扩大到约 6.7B 时，离群值会成为一个显著问题。我们使用 WikiText（Wiki）$[22]$ 和 C4 $[26]$ 数据集评估语言建模任务。我们采用困惑度作为评估指标，该指标广泛用于自回归模型；困惑度越低，模型性能越好。此外，为了证明我们方法的通用性，我们使用 ImageNet 数据集 $[28]$ 对 ResNet-18 $[27]$ 进行了评估；准确率越高，模型性能越好。

量化基线 我们将所提框架与现有量化框架进行了比较，包括 ANT [9]、OliVe [10] 和 Tender [29]。ANT 是一种量化框架，引入了一种新颖的数据类型，以更好地表示数据，并适应不同层之间的不同数据格式。OliVe 是一种硬件友好的量化框架，使用 OVP 编码方法保护离群值，同时确保内存对齐。Tender 是一种分解式量化技术，使用 2 的幂来表示分解后矩阵的缩放因子，以避免显式重新量化。我们通过 PTQ 实验进行评估。每次 PTQ 量化均使用 128 个随机采样的校准数据集。

加速器基线 我们将 RMFQ 加速器的性能和能耗与三种硬件友好的量化加速器进行了比较，包括 OliVe、ANT 和 OLAccel。我们使用 4 位和 6 位混合精度量化，以确保与其他三种量化框架具有相同的困惑度。

RMFQ 实现 我们使用 Synopsys Design Compiler（DC）工具 $[30]$ 对所提出的脉动阵列和编码器进行评估。我们选择 TSMC N28 工艺节点，并将时序约束设为较宽松的 2ns，以便在面积和功耗指标方面进行公平比较。此外，为了保证分析的一致性，我们将脉动阵列规模和分组大小均统一设为 64。另外，我们使用 CACTI $[31]$ 评估缓冲区的面积和功耗。我们使用周期级模拟器 DnnWeaver $[32]$ 进行性能和能耗仿真。我们使用 DeepScaleTool $[33]$ 将其他设计缩放到相同工艺节点，以进行公平比较。

## B. 量化性能结果

分组大小探索 我们采用 CNN 模型和基于 Transformer 的 LLM，分别用于计算机视觉和自然语言处理任务。4 位 RMX 在 WikiText 上的实验结果如图 7 所示。可以看出，分组大小在 8 到 64 之间时，对模型准确率的影响可以忽略不计，而对 6 位 RMX 的影响只会更小。然而，随着分组大小继续增大，对模型准确率的影响变得显著。

![](images/4640c266b886a6f2225c478ab12f38a66ab7c58cd963ca51302f83e9f61f706c.jpg)  
图 7. 不同分组大小下的准确率比较

为了进一步展示 RMFQ 的优越性，我们使用 OPT-6.7B 模型，在 WikiText 上针对数据类型和分组方向开展实验。实验结果如图 8 所示。我们仅展示激活的量化方法，因为权重的量化方法与之配对。可以看出，随着分组大小增大，我们采用的逐词元分组量化表现出更强的鲁棒性，并且 4 位混合 RMX 格式优于 RMXINT4。此外，逐通道分组量化在较小的分组大小下也取得了有竞争力的结果，表明 RMFQ 具有集成到现有 GEMM 内核（例如张量核心）中的潜力。

![](images/9b5bdf76e197dcc198f67d8619afdffddd04cec9d1835f2474cca7a23d336681.jpg)  
图 8. 不同情况下的准确率比较

基准测试与比较 对于 4 位 RMFQ，我们以 INT8 为基准进行比较，而对于 6 位 RMFQ，我们以 FP16 为基准进行比较。结果见表 II。总的来说，所提出的 RMFQ 框架为 LLM 的 4 位和 6 位量化确立了新的 SOTA 水平。在所有 LLM 任务中，4 位 RMFQ 的性能均超过 INT8，而 6 位 RMFQ 几乎无损（与 FP16 基线相比，困惑度最大增加 3.4%）。与 6 位量化相比，离群值在 4 位量化中对模型性能的影响显著更大。这是因为将离群值与其他值一起量化会导致更大的缩放因子。在 4 位 PTQ 本身较少的量化级别下，这种影响更加明显。我们提出的 RMFQ 允许将离群值和正常值分别置于独立的组中进行量化，并提供更小的量化粒度。在所有场景中，我们的方法均优于现有工作，包括 ANT、OliVe 和 Tender。4 位 RMFQ 表现出好得多的 \* RMX 格式的解码电路包含在脉动阵列中。

表 II  
各种模型和配置的性能指标

<table><tr><td rowspan="2">方法</td><td colspan="2">OPT-6.7B</td><td colspan="2">OPT-13B</td><td colspan="2">GPT2-XL</td><td colspan="2">Llama-2-7B</td><td colspan="2">Llama-2-13B</td><td colspan="2">LLaMA-7B</td><td colspan="2">LLaMA-13B</td><td colspan="2">Resnet-18</td></tr><tr><td>Wiki ↓</td><td>C4 ↓</td><td>Wiki</td><td>C4</td><td>Wiki</td><td>C4</td><td>Wiki</td><td>C4</td><td>Wiki</td><td>C4</td><td>Wiki</td><td>C4</td><td>Wiki</td><td>C4</td><td>PTQ ↑</td><td>QAT ↑</td></tr><tr><td>INT8</td><td>16.91</td><td>17.28</td><td>4E+3</td><td>4E+3</td><td>15.51</td><td>17.35</td><td>8.79</td><td>9.96</td><td>20.52</td><td>22.23</td><td>8.52</td><td>9.73</td><td>7.49</td><td>8.61</td><td>69.38</td><td>69.64</td></tr><tr><td>4 位 ANT</td><td>4E+4</td><td>4E+4</td><td>4E+3</td><td>5E+3</td><td>25.39</td><td>29.57</td><td>142.73</td><td>161.34</td><td>113.27</td><td>119.32</td><td>60.56</td><td>66.71</td><td>73.75</td><td>80.32</td><td>59.52</td><td>65.67</td></tr><tr><td>4 位 OliVe</td><td>43.44</td><td>32.41</td><td>45.61</td><td>37.56</td><td>16.36</td><td>18.14</td><td>34.24</td><td>38.85</td><td>36.73</td><td>40.12</td><td>28.64</td><td>30.83</td><td>24.12</td><td>25.77</td><td>59.49</td><td>65.61</td></tr><tr><td>4 位 Tender</td><td>13.56</td><td>13.71</td><td>16.43</td><td>17.33</td><td>16.12</td><td>17.93</td><td>36.47</td><td>41.72</td><td>55.08</td><td>28.14</td><td>23.85</td><td>25.26</td><td>13.68</td><td>14.11</td><td>64.23</td><td>66.41</td></tr><tr><td>4 位 RMFQ*</td><td>12.69</td><td>12.30</td><td>15.11</td><td>15.96</td><td>15.83</td><td>17.48</td><td>7.39</td><td>8.25</td><td>7.83</td><td>8.97</td><td>6.95</td><td>7.91</td><td>6.56</td><td>7.62</td><td>65.74</td><td>67.79</td></tr><tr><td>FP16</td><td>10.40</td><td>10.63</td><td>10.13</td><td>10.32</td><td>14.69</td><td>16.30</td><td>5.83</td><td>6.36</td><td>5.25</td><td>6.43</td><td>5.86</td><td>6.97</td><td>5.19</td><td>5.49</td><td>69.68</td><td>69.87</td></tr><tr><td>6 位 ANT</td><td>11.76</td><td>11.90</td><td>11.42</td><td>11.31</td><td>14.84</td><td>16.46</td><td>6.23</td><td>6.91</td><td>6.03</td><td>7.29</td><td>6.82</td><td>7.93</td><td>6.71</td><td>6.93</td><td>68.99</td><td>69.54</td></tr><tr><td>8 位 OliVe</td><td>10.93</td><td>10.73</td><td>10.28</td><td>10.47</td><td>14.70</td><td>16.37</td><td>6.16</td><td>6.62</td><td>5.98</td><td>7.01</td><td>6.79</td><td>7.52</td><td>6.23</td><td>7.03</td><td>68.82</td><td>69.45</td></tr><tr><td>8 位 Tender</td><td>10.93</td><td>10.76</td><td>10.17</td><td>10.39</td><td>14.72</td><td>16.35</td><td>6.09</td><td>6.53</td><td>5.43</td><td>6.61</td><td>6.05</td><td>7.24</td><td>5.41</td><td>5.58</td><td>69.21</td><td>69.89</td></tr><tr><td>6 位 RMFQ</td><td>10.48</td><td>10.67</td><td>10.21</td><td>10.38</td><td>14.72</td><td>16.33</td><td>6.03</td><td>6.47</td><td>5.39</td><td>6.57</td><td>6.03</td><td>7.12</td><td>5.34</td><td>5.52</td><td>69.30</td><td>70.02</td></tr></table>

\* 为探索最佳的 4 位量化方法，这里展示分组大小为 8 时的最佳结果，其与分组大小为 64 时的结果接近。请注意，在分组大小为 64 时，RMFQ 仍优于所有其他方法。

困惑度，优于其他方法，尤其是在 Llama-2 和 LLaMA 上。6 位 RMFQ 的结果与 8 位 OliVe 和 Tender 的结果相当，并且在大多数情况下优于它们。

## C. 加速器实现结果

面积与功耗 表 III 给出了 MXFA 的架构配置。在 500 MHz 的工作频率下，MXFA 占用 $3.09 \, mm^{2}$ 的面积，功耗为 472.5 mW。脉动阵列集成了第 II-B 节所述的 RMX 格式解码电路。为确保充足的计算吞吐量，我们设计了一个 64 KB 的输入缓冲区、一个 32 KB 的权重缓冲区和一个 32 KB 的输出缓冲区，其中输入缓冲区的容量加倍，以便进行乒乓操作。此外，我们统一了所有基线设计中的 MAC 单元数量和缓冲区大小。为进行公平比较，在后续的性能和能耗评估中，我们将所有 MAC 单元设计的累加器位宽设为 24 位。

性能 图 9(a) 展示了不同加速器设计在各个模型上的归一化周期数。ANT 由于缺乏对离群值的保护，性能最差。尽管 OLAccel 采用混合精度来保护离群值，但其性能受到非对齐内存访问和带宽限制的制约。OliVe 使用 OVP 编码来保护离群值，带宽开销可以忽略不计，但这牺牲了精度，导致其在试图恢复到相同精度时性能不佳。RMFQ 通过共享指数，以可忽略不计的带宽开销实现了更高的精度，从而在相同精度水平下取得了更优的性能。总体而言，在 LLM 中，与 ANT、OLAccel 和 OliVe 相比，RMFA 分别实现了 $3.13\times$、$1.88\times$ 和 $1.28\times$ 的性能提升。

能耗 图 9(b) 展示了不同加速器设计在各个模型上的静态和动态能耗，包括 DRAM、片上缓冲区和核心的能耗。得益于面向硬件友好的脉动阵列设计，RMFA 的能耗最低。相比之下，ANT 和 OLAccel 由于采用混合精度，产生了更多的内存访问和计算周期，因此能耗最高。与 OliVe 相比，精度的牺牲导致需要更多高精度层，能耗也更高。总体而言，在 LLM 中，与 ANT、OLAccel 和 OliVe 相比，RMFA 的能耗分别降低至其 $1/2.38$、$1/1.86$ 和 $1/1.31$。

表 III  
RMFA 的面积与功耗

<table><tr><td>组件</td><td>配置</td><td>面积/ $mm^2$ </td><td>功耗/mW</td></tr><tr><td>脉动阵列*</td><td>64×64 MACs</td><td>1.41</td><td>249.9</td></tr><tr><td>编码器</td><td>64</td><td>0.03</td><td>3.0</td></tr><tr><td>输入缓冲区</td><td>64KB</td><td>0.49</td><td>49.5</td></tr><tr><td>权重缓冲区</td><td>32KB</td><td>0.65</td><td>94.6</td></tr><tr><td>输出缓冲区</td><td>32KB</td><td>0.51</td><td>75.5</td></tr><tr><td>总计</td><td></td><td>3.09</td><td>472.5</td></tr></table>

![](images/3f6dbcc724ab849349084bc8057df830ce4e76db6942bf535e95a6aefc5633b4.jpg)  
(a)

![](images/b5c6d52597bc5b7c4ba0367a5fc3e938d62141d402d19c048f5db09d5a727b18.jpg)  
(b)  
图 9. 不同加速器设计的归一化 (a) 周期数和 (b) 能耗比较

## V. 结论

本文提出了一种用于加速 LLM 的算法与硬件协同设计。为降低量化等效位宽并支持混合数据类型/精度量化，我们提出了 RMX 格式。为更有效地处理 LLM 中的离群值，我们提出了采用新型分组方向的两级 RMFQ。此外，我们设计了 RMFA，以高效支持 RMX 格式。综上，通过集成用于 4/6 位等低比特量化的 RMFQ，与 OliVe 等先前设计相比，RMFA 在精度损失可忽略不计的情况下，在 LLM 中实现了显著的 $1.28\times$ 性能提升和 $1.31\times$ 能效提升。

## VI. 致谢

本研究得到了 SBSTR 基金（编号 KJZD2024090310270801）以及 SCUT 基金（编号 K3200890）的部分资助。

[1] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., "Language models are unsupervised multitask learners," OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[2] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[3] M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. Van Baalen, and T. Blankevoort, “A white paper on neural network quantization,” arXiv preprint arXiv:2106.08295, 2021.

[4] J. Burgess, “Rtx on—the nvidia turing gpu,” IEEE Micro, vol. 40, no. 2, pp. 36–44, 2020.

[5] S. Kim, A. Gholami, Z. Yao, M. W. Mahoney, and K. Keutzer, “I-bert: Integer-only bert quantization,” in International conference on machine learning. PMLR, 2021, pp. 5506–5518.

[6] S. Shen, Z. Dong, J. Ye, L. Ma, Z. Yao, A. Gholami, M. W. Mahoney, and K. Keutzer, “Q-bert: Hessian based ultra low precision quantization of bert,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 05, 2020, pp. 8815–8821.

[7] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, “Awq: Activation-aware weight quantization for on-device llm compression and acceleration,” Proceedings of Machine Learning and Systems, vol. 6, pp. 87–100, 2024.

[8] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “Gpt3. int8(): 8-bit matrix multiplication for transformers at scale,” Advances in Neural Information Processing Systems, vol. 35, pp. 30318–30332, 2022.

[9] C. Guo, C. Zhang, J. Leng, Z. Liu, F. Yang, Y. Liu, M. Guo, and Y. Zhu, “Ant: Exploiting adaptive numerical data type for low-bit deep neural network quantization,” in 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO). IEEE, 2022, pp. 1414–1433.

[10] C. Guo, J. Tang, W. Hu, J. Leng, C. Zhang, F. Yang, Y. Liu, M. Guo, and Y. Zhu, “Olive: Accelerating large language models via hardware-friendly outlier-victim pair quantization,” in Proceedings of the 50th Annual International Symposium on Computer Architecture, 2023, pp. 1–15.

[11] A. H. Zadeh, I. Edo, O. M. Awad, and A. Moshovos, “Gobo: Quantizing attention-based nlp models for low latency and energy efficient inference,” in 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO). IEEE, 2020, pp. 811–824.

[12] E. Park, D. Kim, and S. Yoo, “Energy-efficient neural network accelerator based on outlier-aware low-precision computation,” in 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA). IEEE, 2018, pp. 688–698.

[13] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, "Smoothquant: Accurate and efficient post-training quantization for large language models," in International Conference on Machine Learning. PMLR, 2023, pp. 38087-38099.

[14] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “Qlora: Efficient finetuning of quantized llms,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[15] Y. Lin, H. Tang, S. Yang, Z. Zhang, G. Xiao, C. Gan, and S. Han, "Qserve: W4a8kv4 quantization and system co-design for efficient llm serving," arXiv preprint arXiv:2405.04532, 2024.

[16] Y. Zhao, C.-Y. Lin, K. Zhu, Z. Ye, L. Chen, S. Zheng, L. Ceze, A. Krishnamurthy, T. Chen, and B. Kasikci, “Atom: Low-bit quantization

for efficient and accurate llm serving," Proceedings of Machine Learning and Systems, vol. 6, pp. 196–209, 2024.

[17] B. Darvish Rouhani, R. Zhao, V. Elango, R. Shafipour, M. Hall, M. Mesmakhosroshahi, A. More, L. Melnick, M. Golub, G. Varatkar et al., "With shared microexponents, a little shifting goes a long way," in Proceedings of the 50th Annual International Symposium on Computer Architecture, 2023, pp. 1–13.

[18] O. Zafrir, G. Boudoukh, P. Izsak, and M. Wasserblat, “Q8bert: Quantized 8bit bert,” in 2019 Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing-NeurIPS Edition (EMC2-NIPS). IEEE, 2019, pp. 36–39.

[19] K. Chowdhary and K. Chowdhary, “Natural language processing,” Fundamentals of artificial intelligence, pp. 603–649, 2020.

[20] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin et al., “Opt: Open pre-trained transformer language models,” arXiv preprint arXiv:2205.01068, 2022.

[21] B. D. Rouhani, R. Zhao, A. More, M. Hall, A. Khodamoradi, S. Deng, D. Choudhary, M. Cornea, E. Dellinger, K. Denolf et al., “Microscaling data formats for deep learning,” arXiv preprint arXiv:2310.10537, 2023.

[22] W. contributors, “68-95-99.7 rule - wikipedia,” in The Free Encyclopedia. [Online], 2022.

[23] R. Banner, Y. Nahshan, and D. Soudry, “Post training 4-bit quantization of convolutional networks for rapid-deployment,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[24] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al., “Llama 2: Open foundation and fine-tuned chat models,” arXiv preprint arXiv:2307.09288, 2023.

[25] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar et al., "Llama: Open and efficient foundation language models," arXiv preprint arXiv:2302.13971, 2023.

[26] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” Journal of machine learning research, vol. 21, no. 140, pp. 1–67, 2020.

[27] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[28] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009, pp. 248–255.

[29] J. Lee, W. Lee, and J. Sim, “Tender: Accelerating large language models via tensor decomposition and runtime requantization,” arXiv preprint arXiv:2406.12930, 2024.

[30] P. Kurup and T. Abbasi, Logic Synthesis Using Synopsys, 2nd ed. Springer Publishing Company, Incorporated, 2011.

[31] N. Muralimanohar, R. Balasubramonian, and N. Jouppi, “Cacti 6.0: A tool to model large caches. hp laboratories,” Tech. Rep. HPL-2009-85, 2009.

[32] H. Sharma, J. Park, D. Mahajan, E. Amaro, J. K. Kim, C. Shao, A. Mishra, and H. Esmaeilzadeh, “From high-level deep neural models to fpgas.” in 2016 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO). Ieee, 2016, pp. 1–12.

[33] S. Sarangi and B. Baas, “Deepscaletool: A tool for the accurate estimation of technology scaling in the deep-submicron era.” in 2021 IEEE International Symposium on Circuits and Systems (ISCAS). Ieee, 2021, pp. 1–5.

> 本文由 Codex/GPT 根据论文 PDF 生成中文译文与解读；PDF 由 MinerU 解析。全文译文保留原始参考文献，解读与原文结论请结合阅读。
