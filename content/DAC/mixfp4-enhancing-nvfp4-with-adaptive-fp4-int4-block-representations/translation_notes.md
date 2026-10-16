# MixFP4: Enhancing NVFP4 with Adaptive FP4/INT4 Block Representations 原文翻译

# MixFP4：利用自适应 FP4/INT4 块表示增强 NVFP4

Jiaxiang Zou、Yonghao Chen、Ruilong Wu、Xinyu Chen

作者单位：香港科技大学（广州），中国广州。通讯作者：Xinyu Chen（xinyuchen@hkust-gz.edu.cn）。论文首页标注：第 43 届国际机器学习大会（ICML），韩国首尔，PMLR 306，2026；版权归作者所有。预印本：arXiv:2605.31035v1 [cs.AR]，2026 年 5 月 29 日。

## 摘要

随着大语言模型不断扩大，NVFP4 等细粒度块缩放低精度格式因其显著的吞吐量和存储收益而得到越来越广泛的应用。然而，单一的 FP4 微格式往往无法匹配各数据块之间存在差异的张量统计特征。为在不改变标准块缩放 MMA/GEMM 执行路径的前提下解决这一问题，我们提出 MixFP4：一种面向 NVFP4 的混合微格式扩展，允许每个块在两种存储 FP4 微格式（E2M1 和 E1M2）之间进行选择。

MixFP4 复用 NVFP4 的缩放层次，并将 FP8 E4M3 块缩放因子的符号位改作格式选择位，因此不增加任何元数据。通过将两种微格式都解码为统一的内部 E2M2 计算表示，MixFP4 避免了重复的数据通路。在有代表性的大语言模型家族上，相比 NVFP4 基线，MixFP4 提高了 FP4 量化的鲁棒性和准确率，同时仅给张量核带来适度的开销：面积增加 3.1%，功耗增加 1.5%。

## 1. 引言

大语言模型（LLM）越来越多地部署于对延迟和成本敏感的场景（Achiam et al., 2023；Grattafiori et al., 2024；Liu et al., 2024a；Yang et al., 2025）。这些场景中，计算能力和存储带宽都会限制端到端吞吐量（Agrawal et al., 2024；Kamath et al., 2025；Zhu et al., 2025；Lin et al., 2026；Zhong et al., 2024；Du et al., 2025）。因此，量化是高效 LLM 服务和训练不可或缺的技术。但将量化进一步压到 4 比特仍然很困难：由于可表示值只有少数几个，误差会对局部张量统计特征高度敏感，简单的粗粒度量化很容易发生饱和，或引入过多噪声（Cook et al., 2025；Chen et al., 2025a；Guo et al., 2022；Hu et al., 2025a）。

为了在动态范围压力很大的情况下减少量化误差，近期硬件和软件栈逐渐采用 FP8、FP4 等低比特浮点格式，尤其是 NVFP4 这样的块缩放 FP4 格式（Abecassis et al., 2025）。这些格式为每个小块附加一个共享缩放因子，在维持 4 比特有效载荷存储的同时扩展有效动态范围。不过，块缩放本身并不能消除 LLM 张量分布的根本异质性：在同一组激活或权重中，少量罕见离群值可能与大面积相对“平坦”的小幅值区域并存；这些不同状态的占比也可能随模块、层和块而大幅变化（Zhao et al., 2024；Xiao et al., 2023）。

这些混合状态对单一码本提出了相互冲突的要求：离群值较多的块受益于指数位较多的间隔安排，而平坦的块则更适合均匀、类似整数的间隔。因此，即使在同一层内，强制所有块使用全局固定的 FP4 码本也可能不是最优方案。

推理时进行特征混合后，这个问题更加突出。结构化正交变换，例如 Hadamard 变换，可以将较大的坐标值分散到多个维度，从而减少离群值（Ashkboos et al., 2024；Liu et al., 2024b；Hu et al., 2025b；Sun et al., 2024）；但它们也会重塑局部分布，使相同选择规则下更合适的低比特格式发生变化。这些现象表明：在 4 比特精度下，没有任何单一 FP4 微格式能够在所有块和模块上普遍最优。格式选择应以细粒度适应局部统计特征，同时保持与通用矩阵乘法（GEMM）执行兼容。

本文从逐块统计特征的角度重新审视 FP4 设计空间，提出建立在 NVFP4 之上的混合格式扩展 MixFP4。MixFP4 保留块缩放结构和适合张量核的执行路径，同时允许每个块在指数位较多的 FP4 码本与间隔更均匀的类整数 FP4 码本之间选择：前者适合动态范围压力较大的块，后者往往适合经过混合、峰值因子降低的块。关键在于，MixFP4 复用了 NVFP4 缩放层次中已有的冗余位，保存逐块选择结果时不增加存储开销。


![图 1：NVFP4 与 MixFP4 数据格式结构](images/figure-01.png)

**图 1.** NVFP4 和 MixFP4 的数据格式结构示意图。每个块包含 16 个值，共享 E4M3 块缩放因子，并使用 FP32（E8M23）张量缩放因子。MixFP4 将 E4M3 缩放因子中的冗余符号位 S 改为格式类型位 T，不增加额外存储。

**本文贡献。** 本文主要研究将权重和激活量化为每个参数 4 比特，贡献如下：

- 刻画现代 LLM 中逐块张量统计特征的异质性，并说明正交混合能够重塑这些统计特征，改变更适合的低比特码本。
- 提出 NVFP4 的逐块混合微格式扩展 MixFP4。它为每个块从两种 FP4 码本中选择一种，并将选择位打包进第一级 E4M3 块缩放因子的冗余符号位，不增加存储开销。
- 建模在现代 GPU 张量核（例如 Blackwell FP4 张量核）上支持 MixFP4 所需的硬件开销。结果显示，只需轻量解码和小格式转换逻辑，就可以复用现有计算数据通路，所需开销很小。

## 2. 背景与动机

### 2.1. NVFP4 格式简介

根据 IEEE-754 标准（IEEE, 2019），浮点值由符号位 S、宽度为 $W_E$ 的指数域 E，以及宽度为 $W_M$ 的尾数域 M 表示。指数偏置为：

$$
B=2^{W_E-1}-1. \tag{1}
$$

浮点值 x 可以写作：

$$
x=(-1)^S
\begin{cases}
2^{E-B}\left(1+\dfrac{M}{2^{W_M}}\right),&1\le E\le 2^{W_E}-1,\\
2^{1-B}\dfrac{M}{2^{W_M}},&E=0.
\end{cases} \tag{2}
$$

式（2）表明，改变指数位数和尾数位数会改变码本的量化级间隔及动态范围，进而影响给定局部张量分布下的量化误差。

NVFP4 是一种块缩放量化格式，使用 FP4 E2M1（$W_E=2,W_M=1$）作为块内量化值的微格式。其幅值码本为 $\{0,0.5,1.0,1.5,2.0,3.0,4.0,6.0\}$，同时配有正的 FP8 E4M3 逐块缩放因子和 FP32 逐张量缩放因子。如图 1a 所示，两级缩放的过程为：（1）逐张量 FP32 缩放，将整个张量映射到块缩放格式可以表示的范围，即带 E4M3 块缩放的 FP4 范围；（2）逐块 E4M3 缩放，进一步将各块映射到 FP4 E2M1 的具体表示范围。

为方便讨论，我们定义 NVINT4：它类似 NVFP4，但将 FP4 E2M1 微格式替换为对称 INT4 值，幅值使用整数码本 $\{0,1,2,3,4,5,6,7\}$。

![图 2：跨层和模块的平均块内峰值因子](images/figure-02.png)

**图 2.** 块大小 $g=16$ 时，权重（上）与激活输入（下）在不同层和模块上的平均块内峰值因子。较亮的网格表示权重或激活的平均峰值因子较大，对应块内数值分布比暗色网格更尖锐。激活呈现较强的空间变化。图中比较 Llama-3.1-8B 与 Qwen3-8B，模块包括 Q/K/V、O、GATE/UP 和 DOWN。

### 2.2. 张量间与张量内的统计变化

不同张量类型（权重、激活、梯度）以及不同模型位置（层、模块，乃至块）之间的数据分布高度异质（Egiazarian et al., 2025；Liu et al., 2024b；Ashkboos et al., 2024；Hu et al., 2025b；Sun et al., 2024；Guo et al., 2022；Lee et al., 2025；Lin et al., 2025）。为量化这种异质性，我们使用峰值因子，即最大绝对值与均方根（RMS）的比值。它反映固定步长均匀量化器发生饱和或产生较大误差的可能性。

对于包含 16 个元素、使用 FP8 E4M3 块缩放的块，当峰值因子低于 2.224 时，NVINT4 的预测量化信噪比（QSNR）高于 NVFP4，也就是量化误差更小；证明见附录 A。


图 2 展示了张量间的数据多样性，并揭示出鲜明对比：权重张量的峰值因子相对稳定，而激活张量在不同层和模块之间表现出显著的空间变化。这种明显的空间异质性说明，单一的全局固定低比特格式不太可能在整个模型中始终保持最优。

![图 3：单个张量内部的逐块峰值因子](images/figure-03.png)

**图 3.** 即使在单个张量内（第 0 层 Q 投影，$g=16$），Llama-3.1-8B 和 Qwen3-8B 的权重（上）及激活输入（下）在不同块之间也存在很大的峰值因子差异。这表明张量内部动态范围高度异质，因此有必要采用逐块格式自适应。颜色表示块内峰值因子：较暗表示更平坦的块，较亮表示更容易出现离群值的块。

关键在于，这种变化不仅存在于层与层之间，也延伸到张量内部。如图 3 所示，即使是同一个张量，例如 Q 投影，不同块之间也存在明显异质性。热图中，各块的峰值因子差异很大，既有“平坦”区域，也有“离群值易发”热点；统一量化格式不足以刻画这些局部动态范围波动。

**设计启示。** 低比特格式应以细粒度适应局部张量统计特征，而不应只依赖全局固定格式。

### 2.3. 一种 FP4 微格式并不够

在 4 比特精度下，这种敏感性尤为突出：少量离散量化级需要表示差异很大的局部统计特征。因此，FP4 自然形成一个码本家族，不同的指数位 E 与尾数位 M 划分适合不同的局部张量分布。NVFP4 默认采用 E2M1，但其他 FP4 微格式可能更适合某些局部统计特征，例如 E1M2 提供类似整数的均匀间隔，而 E3M0 强调二的整数次幂量化级。

这些选择导出一个简单而重要的结论：没有任何单一 FP4 微格式能在所有张量和模块上普遍最优。当张量统计特征发生变化时，根据局部统计进行格式选择，是减少量化误差的一种有依据的策略。

相较 FP4 变体，在分布更均匀的情况下，整数格式也可能很有竞争力。此外，基于旋转的 Hadamard 变换可以降低峰值因子（Ashkboos et al., 2024；Egiazarian et al., 2025；Chen et al., 2025a），此时均匀 INT 量化的间隔更贴近近似均匀的局部统计特征，因而可能更合适。反之，对于仍然面临动态范围压力的块，指数位较多的浮点变体可以更有效地在不同幅值之间分配可表示值，保留准确率优势。综合来看，目标应当是跟踪局部统计特征的细粒度混合微格式选择。

**设计启示。** 逐块混合微格式选择应在类浮点码本与类整数码本之间选择，以匹配局部统计特征。

![图 4：扩充 FP4 候选格式的困惑度收益](images/figure-04.png)

**图 4.** Llama-3.1-8B 与 Qwen3-8B 使用 NVFP4 量化时的 WikiText 困惑度。`+FP4-E1` 表示增加 E1M2，`+FP4-E3` 表示再增加 E3M0。Llama 的困惑度依次为 8.26、8.06、8.01，BF16 为 7.33；Qwen 依次为 12.74、12.39、12.39，BF16 为 12.21。困惑度越低越好。

### 2.4. 格式种类增多的收益递减

尽管细粒度混合微格式可以减少量化误差，但扩大支持的格式集合，很快就会遇到准确率收益递减，同时增加系统成本。在准确率方面，一旦一小组代表性格式已覆盖主要动态范围状态，再增加相近变体通常只能带来有限收益。

如图 4 所示，我们按照已有工作采用的逐块 MSE 选择准则（Cook et al., 2025），对三个 FP4 候选集合进行消融。结果显示，在 NVFP4 中加入 E1M2 可显著提高准确率，但继续加入 E3M0 的额外收益很小。

为了进一步评估组合不同格式的收益，并确定哪些格式更合适，我们选择两个候选集合：$\{\mathrm{FP4\ E2M1}(6),\mathrm{FP4\ E2M1}(4)\}$（Cook et al., 2025），以及 $\{\mathrm{FP4\ E2M1}(6),\mathrm{FP4\ E1M2},\mathrm{FP4\ E3M0}\}$。E2M1(4) 表示将 E2M1 的最大量化值从 6 改为 4。如图 5 所示，扩大候选集合后，逐块选择呈现明显偏斜：少量格式占主导，尤其是 E2M1(6) 与 E1M2；其他格式很少被选中。在随机 Hadamard 变换下，这种偏斜更加明显。


![图 5：有无随机 Hadamard 变换时的格式选择比例](images/figure-05.png)

**图 5.** Llama-3.1-8B 和 Qwen3-8B 的格式选择，对比不使用与使用随机 Hadamard 变换的结果。（a）Llama 不使用变换；（b）Llama 使用变换；（c）Qwen 不使用变换；（d）Qwen 使用变换。每组分别展示权重和激活在 E2M1(6)/E2M1(4) 双格式集合，以及 E2M1(6)/E1M2/E3M0 三格式集合中的选择比例。

这一现象支持一种实用的设计选择：只保留小型、具有代表性的格式集合，即 E2M1(6) 与 E1M2，即可获得大部分准确率收益，同时避免为很少被选中的变体付出软件和硬件成本。附录 E 进一步展示了各张量中的逐块选择，并将它们与图 3 所示的峰值因子异质性联系起来。

**设计启示。** 保留少量有代表性的候选格式，可以在避免格式种类膨胀的同时获得大部分准确率收益。

### 2.5. 设计目标：兼顾细粒度自适应与适合 GEMM 的执行

要将数值改善转化为端到端加速，所选格式必须能够顺利映射到现代加速器以 GEMM 为中心的执行方式。近期加速器设计同样表明，数值格式与算术单元需要协同设计，才能高效执行低精度 GEMM（Hu et al., 2025a；Chen et al., 2025b；Zou et al., 2025；Chen et al., 2025c）。我们的目标是直接改进算术单元使用的低比特数值类型，只增加轻量前端解码和块缩放逻辑，保留标准 MMA/GEMM 计算模式。

综合这些现象，可以提出三项设计要求：（i）以细粒度格式自适应匹配局部张量统计特征；（ii）限制候选格式数量，避免格式种类膨胀；（iii）复用现有 GEMM 流水线，保证硬件执行友好。块缩放 FP4 已处于这一设计空间，建立在 NVFP4 之上的 MixFP4 则在保持适合 GEMM 执行的同时，进一步加入逐块自适应。

## 3. 设计

### 3.1. FP4 编码与 INT4 兼容性

根据第 2.4 节中逐块选择高度集中于少数主导类型的观察，MixFP4 专门支持两种 4 比特有效载荷微格式：E2M1 与 E1M2。表 1 总结它们的符号、指数、尾数（S.E.M）位布局及数值范围，包括零和次正规数。之所以选择这两种格式，是因为 E2M1 是 NVFP4 的有效载荷格式，而 E1M2 具有类似整数的均匀量化级间隔，在相同的块缩放层次下可以提供 INT4 兼容行为。

![表 1：FP4 二进制格式详情](images/table-01.png)

**表 1. FP4 二进制格式详情。** S 表示符号位。

| 项目 | E2M1 | E1M2 |
| --- | --- | --- |
| 指数偏置 | 1 | 0 |
| 零 | S.00.0 | S.0.00 |
| 最大正规数幅值 | S.11.1：$1.5\times2^2=6$ | S.1.11：$1.75\times2=3.5$ |
| 最小正规数幅值 | S.01.0：1 | S.1.00：2 |
| 最大次正规数幅值 | S.00.1：0.5 | S.0.11：$0.75\times2=1.5$ |
| 最小次正规数幅值 | S.00.1：0.5 | S.0.01：$0.25\times2=0.5$ |

![图 6：E1M2 到对称 INT4 码本的缩放映射](images/figure-06.png)

**图 6.** 均匀步长的 E1M2 码本可以通过缩放表示对称 INT4 值。图中 I、E、M 分别表示整数位、指数位、尾数位；E1M2 幅值 $[0,0.5,1,1.5,2,2.5,3,3.5]$ 乘以 2 后得到 INT4 幅值 $[0,1,2,3,4,5,6,7]$。

关键观察是：存储的 E1M2 幅值构成均匀码本 $\{0,0.5,1.0,\ldots,3.5\}$，而对称 INT4 的无符号幅值采用整数级 $\{0,1,\ldots,7\}$。MixFP4 通过对 E1M2 幅值进行固定的乘 2 重映射，将两个格点集合连接起来，得到精确的类 INT4 格点 $0,1,\ldots,7$（图 6）。这一重映射是常数因子，在解码时吸收到数值解释中；存储的 FP4 有效载荷和逐块 E4M3 缩放约定保持不变，而在块统计特征适合类整数间隔时，E1M2 实际上表现为真正的均匀步长 INT4 量化器。


![算法 1：MixFP4 数据类型量化算法](images/algorithm-01.png)

**算法 1. MixFP4 数据类型量化算法。** 输入张量 X；输出量化值 $\bar X$、FP32 张量缩放 $s_{32}$ 与 FP8 块缩放 $s_8$。

1. 定义 `MixFP4(X)`，计算 $s_{32}=\max(|X|)/2688$。这里 $6\times448=7\times384=2688$，448 和 384 均在 E4M3 范围内。
2. 计算 $X_{\mathrm{FP8}}=X/s_{32}$。
3. 对 E2M1 候选计算 $s_{8,\mathrm{E2M1}}=\mathrm{FP8\_E4M3}(\operatorname{block\!\!\!-max}(|X_{\mathrm{FP8}}|)/6)$；6 是 MixFP4 E2M1 的最大值。
4. 计算 $X^q_{\mathrm{E2M1}}=\mathrm{FP4\_E2M1}(X_{\mathrm{FP8}}/s_{8,\mathrm{E2M1}})$。
5. 反量化得到 $D_{\mathrm{E2M1}}=X^q_{\mathrm{E2M1}}s_{8,\mathrm{E2M1}}$，计算 $\mathrm{Err}_{\mathrm{E2M1}}=\operatorname{MSE}(D_{\mathrm{E2M1}},X_{\mathrm{FP8}})$。
6. 对 E1M2 候选计算 $s_{8,\mathrm{E1M2}}=\mathrm{FP8\_E4M3}(\operatorname{block\!\!\!-max}(|X_{\mathrm{FP8}}|)/7)$；7 是 MixFP4 E1M2 的最大值。
7. 计算 $X^q_{\mathrm{E1M2}}=\mathrm{FP4\_E1M2}(X_{\mathrm{FP8}}/s_{8,\mathrm{E1M2}})$。
8. 反量化得到 $D_{\mathrm{E1M2}}=X^q_{\mathrm{E1M2}}s_{8,\mathrm{E1M2}}$，计算 $\mathrm{Err}_{\mathrm{E1M2}}=\operatorname{MSE}(D_{\mathrm{E1M2}},X_{\mathrm{FP8}})$。
9. 若 $\mathrm{Err}_{\mathrm{E2M1}}<\mathrm{Err}_{\mathrm{E1M2}}$，选择 E2M1 候选的缩放因子和量化值；否则选择 E1M2 候选，同时设置 T=1。

> 译注：算法中 E1M2 的最大值 7，采用第 3.1 节介绍的乘 2 后的类 INT4 数值解释；表 1 中未重映射的 E1M2 最大幅值仍为 3.5。

### 3.2. MixFP4：一种块缩放格式中的双微格式

MixFP4 为 NVFP4 增加细粒度逐块混合格式能力：每个块从 E2M1 与 E1M2 两种 FP4 微格式中选择一种，以更好匹配局部张量统计特征，同时保留标准块缩放 MMA/GEMM 执行路径。核心思想是复用现有 NVFP4 缩放层次，以零额外元数据开销编码逐块格式选择。

图 1 展示了底层块缩放结构。NVFP4 使用 E2M1 编码 4 比特有效载荷、E4M3 编码逐块第一级缩放、FP32 编码逐张量第二级缩放。MixFP4 保留相同的缩放层次和粒度，但允许有效载荷采用 E2M1 或 E1M2。为了在不增加存储的情况下支持逐块混合格式选择，我们将格式类型 T 保存为一个块共享位，复用利用不足的 E4M3 第一级缩放符号位。因为只有一个这样的位可用，MixFP4 恰好支持两种格式。

如算法 1 所示，MixFP4 在相同的块缩放流程下，评估每一种候选微格式产生的均方误差，并选择 MSE 更低的一种。具体来说，对每个块分别计算 E2M1 和 E1M2 的候选缩放因子，在两种选择下分别量化、反量化，再比较 MSE。采用 MSE 作为选择准则，是因为已有实证表明，逐块格式选择时它比其他指标更可靠（Cook et al., 2025）。

![图 7：MixFP4 量化线性层的训练计算流](images/figure-07.png)

**图 7.** MixFP4 量化线性层的训练计算流程。前向传播 FPROP、输入梯度 DGRAD 和权重梯度 WGRAD 三个核心矩阵乘法均采用模拟 MixFP4（绿色路径）；主权重保持 FP32（蓝色），激活和梯度保持 BF16（橙色）。Q(MixFP4) 表示将块映射为 E2M1 或 E1M2 的量化操作。此外，使用随机舍入 SR 和随机 Hadamard 变换 RHT 稳定反向传播。图中还包括转置、二维块量化以及优化器更新。

图 7 展示了训练时将 MixFP4 集成到量化线性层的方法。遵循近期 FP4 训练配方（Abecassis et al., 2025），我们对权重梯度计算的输入应用随机 Hadamard 变换，并对权重矩阵实施二维块量化。在 GEMM 边界量化权重、激活和梯度，以模拟 MixFP4；主权重和优化器状态等高精度状态保持不变。预训练结果见第 4.2 节。

![表 2：多精度张量核的乘法器组成](images/table-02.png)

**表 2.** 目标 BF16/FP8/FP4 吞吐量比例为 4:8:16 时，多精度张量核的乘法器配置比较。

| 设计 | 乘法器组成 |
| --- | --- |
| 基线单元 | $4\times\mathrm{E8M10}+4\times\mathrm{E5M3}+8\times\mathrm{E2M1}$ |
| 本文单元 | $4\times\mathrm{E8M10}+4\times\mathrm{E5M3}+8\times\mathrm{E2M2}$ |

表注：为了反映张量核在不同指数/尾数配置间的硬件复用，我们用 E5M3 乘法器建模 FP8 通道，覆盖 E4M3/E5M2；用 E8M10 乘法器建模 BF16/FP16 通道，覆盖 FP16 E5M10 和 BF16 E8M7。

### 3.3. 支持 MixFP4 的张量核设计

我们的目标是原本就通过 NVFP4 支持块缩放 FP4 MMA 的张量核切片：数值编码为 E2M1，每个块由 FP8 E4M3 缩放。


![图 8：支持 MixFP4 的张量核](images/figure-08.png)

**图 8.** 支持 MixFP4 的张量核。左侧是原设计，右侧是本文设计：增加解码器，将 FP4 E2M1 乘法单元换为 E2M2 单元，保留 BF16/FP8 乘法、块缩放、加法和部分和路径。

我们扩展该切片，使其也支持另一种 FP4 微格式 E1M2，同时保持三个不变量：FP16/FP32 累加、块缩放粒度、缩放因子的存储占用。因而这一扩展保留兼容性，并非重新设计整个数据通路。动机有两点：首先，即使局部统计相同，不同 FP4 码本也可能产生明显不同的误差；其次，E1M2 在块缩放下提供类整数的均匀间隔，当统计特征更均匀，例如正交混合之后，可以与 E2M1 互补。

为了支持混合 E2M1/E1M2 操作数，同时避免格式相关的额外计算开销，例如分离算术数据通路，我们在解码时将两种微格式都规范化为统一的 E2M2 内部表示。块缩放符号位被改作块共享类型信号：E2M1 通过补零填入缺少的尾数位，E1M2 则通过小型重映射查找表解码，使两条路径都产生 E2M2 表示（图 9、附录 B.5.1）。这一统一方式为下游计算提供单一数据格式，避免为不同格式建立独立计算通路。

![图 9：根据 T 将 E2M1 或 E1M2 解码到 E2M2](images/figure-09.png)

**图 9.** 由 T 控制的 E2M1/E1M2 到 E2M2 解码。T=0 时 E2M1 幅值编码补尾零，例如 100、101、110、111 分别得到表示 2、3、4、6 的 E2M2 编码；T=1 时 E1M2 经重映射得到表示 4、5、6、7 的 E2M2 编码。

我们将张量核建模为三维阵列，规范化吞吐量为 BF16:FP8(FP6):FP4=4:8:16，分别对应每周期 $4\times4\times4$、$4\times4\times8$ 和 $4\times4\times16$ 的 MMA（表 2）。为了聚焦归约维度，我们遵循已有多精度复用模型（Chen et al., 2025a），令 M=N=1，只分析一个输出通道，即 K 元素向量 MAC。

如图 8 所示，基线切片包括 4 个 BF16/FP16 MAC 单元、4 个 FP8 MAC 单元和 8 个 FP4 E2M1 MAC 单元。支持 MixFP4 的切片将后 8 个单元替换为 E2M2 MAC，并在每个通道之前放置轻量解码逻辑，支持 E2M1/E1M2 双模式；E2M2 MAC 单元之间的 4 个累加器由 E4M3 扩展为 E4M5，其余数据通路保持复用。

支持 E1M2 的增量硬件成本局限于 FP4 解码、块缩放控制与元数据路径。具体来说，解码只增加少量组合逻辑，例如几个多路选择器和简单逻辑门，在块共享类型信号控制下重映射 FP4 位域，无需修改 FP16/FP32 累加通路。我们在门级量化该开销，发现相对于基线张量核切片其占比很小；详细结果见附录 B.4.2。

## 4. 评估

本节评估在训练后量化和预训练量化中，MixFP4 对细粒度 4 比特模型性能的影响。遵循已有量化研究，我们量化以 GEMM 为主的线性投影，例如 Q/K/V/O 和 MLP 的 up/gate/down，但不量化嵌入、LM head、Attention 和非线性模块。

在 Qwen3 风格的 114M 参数预训练设置中，我们观察到稳定性改善；在已经训练好的 Llama（Grattafiori et al., 2024）、Qwen（Yang et al., 2025）和 Mamba（Gu & Dao, 2024）模型上，多种下游任务性能也得到提升。我们评估 WikiText（Merity et al., 2016）词级困惑度，以及 lm_eval_harness（Gao et al., 2024）上的下游准确率。下游任务包括 MMLU（Hendrycks et al., 2020）、ARC-Easy 和 ARC-Challenge（Clark et al., 2018）、HellaSwag（Zellers et al., 2019）以及 PIQA（Bisk et al., 2020）。所有软件实验均在单张 80GB H100 GPU 上运行。

在硬件方面，我们用 Verilog 实现支持 MixFP4 的张量核，并在 Synopsys Design Compiler 中评估面积及能耗开销。目标为台积电 28nm、1GHz，TT 工艺角、0.9V、25°C，采用 CCS 时序模型。

### 4.1. 训练后量化

我们评估 MixFP4 改善 NVFP4 训练后量化（PTQ）准确率的能力，并与同样旨在提高 NVFP4 准确率的先进基线 Four-over-Six（4/6，Cook et al., 2025）比较。MixFP4 既可以单独使用，也是一种修改 NVFP4 底层量化算法的通用方法，可以方便地与 SmoothQuant（Xiao et al., 2023）、GPTQ（Frantar et al., 2022）和 SpinQuant（Liu et al., 2024b）等已有 PTQ 方法结合。

表 3 报告采用最近邻舍入（RTN）量化时，在 WikiText 上的 PTQ 困惑度结果，数值越低越好。


![表 3：RTN 量化下的 WikiText 困惑度](images/table-03.png)

**表 3.** MixFP4 与各基线采用 RTN 量化时的 WikiText 词级困惑度。RHT 表示加入随机 Hadamard 变换；表中完整数值如下。

| 模型与设置 | BF16 | NVFP4 | NVINT4 | 4/6 | MixFP4 |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B | 11.57 | 13.90 | 14.79 | 13.60 | 13.48 |
| Llama-3.2-1B +RHT | 11.56 | 15.09 | 13.99 | 14.19 | 13.53 |
| Llama-3.1-8B | 7.33 | 8.26 | 8.37 | 8.15 | 8.06 |
| Llama-3.1-8B +RHT | 7.33 | 8.55 | 8.23 | 8.29 | 8.10 |
| Qwen3-4B | 16.45 | 17.34 | 18.88 | 17.61 | 17.42 |
| Qwen3-4B +RHT | 16.46 | 19.22 | 18.11 | 18.32 | 18.60 |
| Qwen3-8B | 12.21 | 12.74 | 12.73 | 12.56 | 12.39 |
| Qwen3-8B +RHT | 12.22 | 13.57 | 12.94 | 13.48 | 12.77 |
| Qwen3-14B | 10.78 | 11.31 | 11.31 | 11.21 | 11.03 |
| Qwen3-14B +RHT | 10.79 | 11.62 | 11.21 | 11.36 | 11.44 |
| Mamba-1.4B | 13.58 | 14.56 | 14.88 | 14.45 | 14.38 |
| Mamba-2.8B | 11.79 | 12.44 | 12.55 | 12.34 | 12.27 |

我们将 MixFP4 与 BF16 及 NVFP4、NVINT4、4/6 等代表性 4 比特基线比较。对 Transformer 模型，还报告加入随机 Hadamard 变换的消融结果。总体而言，MixFP4 在多数模型上获得最低或接近最低的困惑度，在相同设置下整体优于原始 NVFP4、NVINT4 和 4/6。

值得注意的是，RHT 可以明显改变各量化器的相对排序。Hadamard 混合往往有利于对离群值敏感的码本，而 MixFP4 仍保持竞争力，并经常保有最佳准确率，这表明其混合码本设计能够适应正交变换引起的分布变化。在少数设置中，例如 Qwen3-4B，NVFP4 或 NVINT4 获得最佳困惑度；即便如此，本文张量核仍兼容这些原始格式，可以在同一执行流水线中支持 NVFP4 与 NVINT4，无需增加格式专用硬件成本。

表 4 展示 Llama-3.2-1B、Llama-3.1-8B、Qwen3-4B 和 Qwen3-8B 与 SmoothQuant、GPTQ、SpinQuant 结合时的 WikiText 困惑度与下游任务准确率。详细设置见附录 C。结果显示，MixFP4 在多数配置中得到最低困惑度和最高平均分。

这些结果也说明，MixFP4 与已有校准及旋转式 PTQ 流程互补，并不局限于某一种前端算法。SmoothQuant、GPTQ、SpinQuant 以不同方式改变张量统计特征，但当 MixFP4 作为底层 4 比特块格式插入后，通常仍能获得最佳或接近最佳的困惑度和平均下游分数。少量例外也不需要单独执行路径，因为本文张量核保留对原始 NVFP4/NVINT4 的兼容性。这对于部署很重要：系统可以根据模型或层选择数值表现更合适的格式，同时维持相同的混合格式执行数据通路。

脚注 1：未展示 Qwen3-4B 与 SpinQuant 结合的结果，因为其中间维度为 $9728=38\times256$，不受 SpinQuant 支持。SpinQuant 仅支持维度为 $2^k$ 或 $K\times2^k$ 的情形，其源代码中没有 K=38。

![表 4：MixFP4 与多种 PTQ 方法的结合结果](images/table-04.png)

**表 4.** MixFP4 在 Llama3 和 Qwen3 模型上与不同量化方法结合时的 PTQ 性能。原表各列中文含义依次为：方法、量化格式、WikiText 困惑度、MMLU、ARC-Challenge、ARC-Easy、HellaSwag、PIQA、平均准确率。WikiText 越低越好，准确率越高越好。以下保留全部数值。

| 模型 | 方法 | 格式 | Wiki | MMLU | Arc-C | Arc-E | Hella. | PIQA | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B | 基准 | BF16 | 11.57 | 36.71 | 36.43 | 60.31 | 63.70 | 74.65 | 54.36 |
| Llama-3.2-1B | SmoothQuant | NVFP4 | 13.85 | 29.59 | 34.22 | 55.93 | 59.78 | 72.03 | 50.31 |
| Llama-3.2-1B | SmoothQuant | 4/6 | 13.50 | 28.41 | 34.04 | 55.81 | 59.72 | 70.89 | 49.78 |
| Llama-3.2-1B | SmoothQuant | MixFP4 | 13.38 | 31.70 | 33.45 | 55.26 | 60.37 | 72.42 | 50.64 |
| Llama-3.2-1B | GPTQ | NVFP4 | 13.24 | 30.69 | 33.96 | 56.36 | 59.85 | 72.96 | 50.76 |
| Llama-3.2-1B | GPTQ | 4/6 | 13.04 | 31.84 | 34.81 | 57.70 | 60.11 | 71.33 | 51.16 |
| Llama-3.2-1B | GPTQ | MixFP4 | 12.91 | 30.96 | 34.13 | 57.01 | 59.75 | 72.87 | 50.97 |
| Llama-3.2-1B | SpinQuant | NVFP4 | 14.44 | 29.35 | 33.11 | 53.96 | 58.93 | 68.44 | 48.76 |
| Llama-3.2-1B | SpinQuant | 4/6 | 13.69 | 27.05 | 33.96 | 55.18 | 59.18 | 69.53 | 48.98 |
| Llama-3.2-1B | SpinQuant | MixFP4 | 13.38 | 27.92 | 32.59 | 57.07 | 60.24 | 69.04 | 49.37 |
| Llama-3.1-8B | 基准 | BF16 | 7.33 | 63.38 | 53.24 | 81.10 | 78.92 | 81.07 | 71.54 |
| Llama-3.1-8B | SmoothQuant | NVFP4 | 8.22 | 60.53 | 50.17 | 76.68 | 77.36 | 79.71 | 68.89 |
| Llama-3.1-8B | SmoothQuant | 4/6 | 8.10 | 59.67 | 51.54 | 78.16 | 77.64 | 80.30 | 69.46 |
| Llama-3.1-8B | SmoothQuant | MixFP4 | 8.03 | 60.79 | 50.26 | 78.11 | 77.96 | 79.65 | 69.35 |
| Llama-3.1-8B | GPTQ | NVFP4 | 8.24 | 59.93 | 49.83 | 75.00 | 77.46 | 79.82 | 68.41 |
| Llama-3.1-8B | GPTQ | 4/6 | 8.16 | 60.60 | 51.19 | 79.04 | 77.86 | 80.09 | 69.76 |
| Llama-3.1-8B | GPTQ | MixFP4 | 8.05 | 60.67 | 50.60 | 76.73 | 77.96 | 79.92 | 69.18 |
| Llama-3.1-8B | SpinQuant | NVFP4 | 8.66 | 55.85 | 50.34 | 77.61 | 76.33 | 78.35 | 67.70 |
| Llama-3.1-8B | SpinQuant | 4/6 | 8.43 | 58.59 | 50.34 | 76.73 | 76.76 | 78.02 | 68.09 |
| Llama-3.1-8B | SpinQuant | MixFP4 | 8.26 | 59.36 | 52.73 | 77.74 | 77.38 | 78.84 | 69.21 |
| Qwen3-4B | 基准 | BF16 | 16.45 | 68.35 | 53.92 | 78.41 | 68.44 | 75.19 | 68.86 |
| Qwen3-4B | SmoothQuant | NVFP4 | 17.90 | 65.65 | 49.40 | 72.31 | 65.92 | 73.78 | 65.41 |
| Qwen3-4B | SmoothQuant | 4/6 | 17.66 | 65.99 | 50.00 | 73.36 | 66.58 | 74.10 | 66.01 |
| Qwen3-4B | SmoothQuant | MixFP4 | 17.52 | 66.98 | 51.11 | 76.05 | 66.90 | 73.56 | 66.92 |
| Qwen3-4B | GPTQ | NVFP4 | 17.20 | 65.99 | 49.32 | 73.40 | 66.06 | 73.07 | 65.57 |
| Qwen3-4B | GPTQ | 4/6 | 17.08 | 66.19 | 50.85 | 74.45 | 66.35 | 73.45 | 66.26 |
| Qwen3-4B | GPTQ | MixFP4 | 17.13 | 66.27 | 50.85 | 75.08 | 66.97 | 74.43 | 66.72 |
| Qwen3-8B | 基准 | BF16 | 12.21 | 73.00 | 56.14 | 80.68 | 74.87 | 77.53 | 72.44 |
| Qwen3-8B | SmoothQuant | NVFP4 | 12.72 | 71.13 | 54.86 | 80.18 | 73.09 | 75.95 | 71.04 |
| Qwen3-8B | SmoothQuant | 4/6 | 12.72 | 70.80 | 54.86 | 78.70 | 73.58 | 76.39 | 70.87 |
| Qwen3-8B | SmoothQuant | MixFP4 | 12.48 | 71.42 | 55.55 | 79.55 | 73.75 | 76.28 | 71.31 |
| Qwen3-8B | GPTQ | NVFP4 | 12.68 | 71.31 | 54.27 | 80.47 | 72.92 | 76.50 | 71.09 |
| Qwen3-8B | GPTQ | 4/6 | 12.60 | 71.56 | 56.91 | 79.17 | 73.33 | 76.17 | 71.43 |
| Qwen3-8B | GPTQ | MixFP4 | 12.44 | 71.04 | 55.63 | 78.75 | 73.46 | 76.12 | 71.00 |
| Qwen3-8B | SpinQuant | NVFP4 | 13.12 | 67.50 | 50.26 | 73.44 | 70.47 | 74.32 | 67.20 |
| Qwen3-8B | SpinQuant | 4/6 | 12.96 | 69.14 | 51.19 | 72.90 | 71.97 | 72.80 | 67.60 |
| Qwen3-8B | SpinQuant | MixFP4 | 12.98 | 70.45 | 53.33 | 77.99 | 72.66 | 74.48 | 69.78 |

### 4.2. 预训练量化

我们在 FineWeb-Edu 数据集（Penedo et al., 2024）上，使用本文方法及各基线，从头训练两个 Qwen3 风格的仅解码器模型。两者使用 Qwen3 的分词器及 Hugging Face 的 QwenForCausalLM，采用 RoPE（Su et al., 2024）、SwiGLU（Shazeer, 2020）以及 Query-Key 归一化（Henry et al., 2020）。图 10 中的模型含 114M 参数，训练 2.5B token，隐藏维度 512、8 个查询头、4 个键值头、中间维度 2048，共 9 层。


![图 10：114M 模型的预训练损失](images/figure-10.png)

**图 10.** 114M 参数 Qwen3 风格模型采用 BF16 与不同 FP4 方案时的预训练损失。（a）完整训练过程，横轴为已处理 token 数（十亿），纵轴为损失；（b）最后阶段 2.35B–2.50B token 的局部放大。

![图 11：476M 模型的预训练损失](images/figure-11.png)

**图 11.** 476M 参数 Qwen3 风格模型采用 BF16 与不同 FP4 方案时的预训练损失。（a）完整的 4B token 训练过程；（b）最后阶段 3.20B–4.00B token 的局部放大。

为了进一步检验规模影响，图 11 采用训练代码中较大的 476M 参数设置：隐藏维度 1024、16 个查询头、4 个键值头、中间维度 4096、18 层，共训练 4B token。

两个设置均使用 AdamW（Loshchilov & Hutter, 2017），$\beta_1=0.9$、$\beta_2=0.95$，权重衰减 0.1，梯度裁剪阈值 1.0，序列长度 2048，全局批大小 256。114M 实验的最大学习率为 $10^{-3}$、最小学习率为 $10^{-4}$；476M 实验最大学习率为 $6\times10^{-4}$，最小学习率与最大值的比例同样为 0.1。

遵循当前先进的 NVFP4 预训练配方（Abecassis et al., 2025），我们对梯度使用随机舍入，对权重梯度计算的两个输入应用随机 Hadamard 变换，并对权重矩阵进行二维块量化，如图 7 所示。我们发现随机舍入对 MixFP4 预训练同样有益；其具体影响见附录 D。

图 10 和图 11 展示在整个训练过程中应用 MixFP4 时的预训练损失轨迹。在 114M 设置中，原始 NVFP4 基线优化不稳定，而 MixFP4 的损失轨迹更平滑，最终损失也低于 4/6。更大的 476M 实验呈现相同的定性现象：在训练后期，MixFP4 始终低于基线，说明收益并不局限于小规模试验模型。这些结果表明，在所评估的设置下，逐块微格式选择可以改善 FP4 预训练的数值鲁棒性。

### 4.3. 块大小敏感性

全部主要软件实验均使用块大小 $g=16$，与 MixFP4 面向的细粒度 NVFP4 设置一致。表 5 对 Llama-3.1-8B 和 Qwen3-8B 的 WikiText 困惑度进行块大小消融。

![表 5：块大小对困惑度的影响](images/table-05.png)

**表 5.** WikiText 困惑度的块大小敏感性，越低越好。E2 表示 E2M1，E1 表示 E1M2，E3 表示 E3M0；带加号的列表示在基础 E2 上加入相应候选。

| 模型 | 块大小 | FP4-E2 | +FP4-E1 | +FP4-E3 | +E1+E3 |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | 8 | 8.04 | 7.83 | 8.00 | 7.79 |
| Llama-3.1-8B | 16 | 8.26 | 8.06 | 8.21 | 8.01 |
| Llama-3.1-8B | 32 | 8.49 | 8.34 | 8.40 | 8.27 |
| Llama-3.1-8B | 64 | 8.73 | 8.69 | 8.63 | 8.54 |
| Qwen3-8B | 8 | 12.48 | 12.39 | 12.46 | 12.40 |
| Qwen3-8B | 16 | 12.74 | 12.39 | 12.63 | 12.39 |
| Qwen3-8B | 32 | 12.82 | 12.78 | 12.77 | 12.73 |
| Qwen3-8B | 64 | 13.00 | 12.96 | 12.85 | 12.79 |

当 $g=16$ 时，FP4-E2+FP4-E1 是最佳双格式 NVFP4 扩展，几乎达到完整混合集合的效果。这支持复用缩放位的设计：附加类型位让每个 NVFP4 块在类浮点 E2M1 与类整数 E1M2 之间选择，无需额外存储。但块更大时，一个共享缩放因子覆盖的数值更异质，FP4-E1 的优势会减弱；在这种更粗的缩放粒度下，FP4-E2+FP4-E3 更有利，因为指数位更多的 E3M0 更适合动态范围较宽的块。

![图 12：相对面积和功耗开销](images/figure-12.png)

**图 12.** MixFP4 相对基线张量核的面积和功耗。基线归一化为 1，MixFP4 面积为 1.031，功耗为 1.015。堆叠部分包括寄存器、加法器、其他精度乘法器、E2M1/E2M2 乘法器和解码器。

### 4.4. 硬件评估

图 12 比较 MixFP4 与基线张量核切片的归一化面积、功耗开销，重点是计算数据通路。我们将数据通路划分为浮点加法器（Add）、浮点乘法器（Mult）、解码逻辑（Decoder）和寄存器（Reg）。乘法器进一步分成基线 FP4 E2M1 通道、MixFP4 E2M2 通道，以及 FP8、BF16 等其他乘法器（Other）。

总体上，MixFP4 相对基线仅增加 3.1% 面积和 1.5% 功耗，表明逐块混合格式执行具有较高的硬件效率。这些开销在张量核计算数据通路上测得；完整张量核中的非计算部分，以及 GPU 的其余部分，例如片上存储和控制逻辑，会进一步稀释芯片级影响。柱形图的组成也显示，新增成本集中于已有算术和解码部分，与局部扩展而非重复计算流水线的设计相符。

因此，虽然综合使用的是 28nm、1GHz，由于变化是增量的，且以局部算术逻辑为主，而不是大规模存储结构或全局互连，我们预计缩放到先进 GPU 工艺后仍会有相近的相对开销。

## 5. 讨论

MixFP4 旨在扩展标准 NVFP4 格式及其软硬件协同设计，使逐块采用类整数均匀步长格式成为可能。当前原型用 PyTorch 实现，目的是忠实模拟量化准确率。MixFP4 原生张量核执行的内核延迟和吞吐量取决于硬件支持与内核工程，这超出本文范围，将留待未来研究。

## 6. 结论

本文提出与张量核协同设计的 NVFP4 扩展 MixFP4。通过复用 E4M3 缩放因子的冗余位，MixFP4 不增加存储开销，并缓解单一固定 FP4 码本与异质逐块统计特征之间的不匹配。具体而言，它在统一的块缩放流水线中，为每个块选择 E2M1 或 E1M2，利用类浮点和类整数分布，更好匹配局部数据分布。实验表明，在训练后量化和预训练中，MixFP4 都能相对有代表性的先进 NVFP4 基线改善鲁棒性与准确率，而张量核计算数据通路的功耗只增加 1.5%、面积只增加 3.1%。

## 影响声明

本文旨在推动机器学习领域的发展。我们的工作可能带来多种社会影响，但我们认为没有需要在此特别强调的内容。


## References

Abecassis, F., Agrusa, A., Ahn, D., Alben, J., Alborghetti,
S., Andersch, M., Arayandi, S., Bjorlin, A., Blakeman,
A., Briones, E., et al. Pretraining large language models
with nvfp4. arXiv preprint arXiv:2509.25149, 2025.
Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I.,
Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S.,
Anadkat, S., et al. Gpt-4 technical report. arXiv preprint
arXiv:2303.08774, 2023.
Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra,
N., Gulavani, B., Tumanov, A., and Ramjee, R. Tam-
ing Throughput-Latency tradeoff in LLM inference with
Sarathi-Serve. In 18th USENIX Symposium on Operat-
ing Systems Design and Implementation (OSDI 24), pp.
117–134, 2024.
Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B.,
Cameron, P., Jaggi, M., Alistarh, D., Hoefler, T., and
Hensman, J. Quarot: Outlier-free 4-bit inference in ro-
tated llms. Advances in Neural Information Processing
Systems, 37:100213–100240, 2024.
Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reasoning
about physical commonsense in natural language. In Pro-
ceedings of the AAAI conference on artificial intelligence,
volume 34, pp. 7432–7439, 2020.

Chen, M., Wu, M., Jin, H., Yuan, Z., Liu, J., Zhang, C.,
Li, Y., Huang, J., Ma, J., Xue, Z., et al. Int vs fp: A
comprehensive study of fine-grained low-bit quantization
formats. arXiv preprint arXiv:2510.25602, 2025a.
Chen, Y., AbouElhamayed, A. F., Dai, X., Wang, Y., An-
dronic, M., Constantinides, G. A., and Abdelfattah, M. S.
BitMoD: Bit-serial mixture-of-datatype LLM accelera-
tion. In 2025 IEEE International Symposium on High
Performance Computer Architecture (HPCA), pp. 1082–
1097. IEEE, 2025b.
Chen, Y., Zou, J., and Chen, X. April: Accuracy-improved
floating-point approximation for neural network accelera-
tors. In 2025 62nd ACM/IEEE Design Automation Con-
ference (DAC), pp. 1–7, 2025c. doi: 10.1109/DAC63849.
2025.11133083.
Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A.,
Schoenick, C., and Tafjord, O. Think you have solved
question answering? try arc, the ai2 reasoning challenge.
arXiv preprint arXiv:1803.05457, 2018.
Cook, J., Guo, J., Xiao, G., Lin, Y., and Han, S. Four
over six: More accurate nvfp4 quantization with adaptive
block scaling. arXiv preprint arXiv:2512.02010, 2025.
Du, K., Wang, B., Zhang, C., Cheng, Y., Lan, Q., Sang, H.,
Cheng, Y., Yao, J., Liu, X., Qiao, Y., et al. Prefillonly:
An inference engine for prefill-only workloads in large
language model applications. In Proceedings of the ACM
SIGOPS 31st Symposium on Operating Systems Princi-
ples, pp. 399–414, 2025.
Egiazarian, V., Castro, R. L., Kuznedelev, D., Panferov, A.,
Kurtic, E., Pandit, S., Marques, A., Kurtz, M., Ashkboos,
S., Hoefler, T., et al. Bridging the gap between promise
and performance for microscaling fp4 quantization. arXiv
preprint arXiv:2509.23202, 2025.
Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. Gptq:
Accurate post-training quantization for generative pre-
trained transformers. arXiv preprint arXiv:2210.17323,
2022.
Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi,
A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li,
H., McDonell, K., Muennighoff, N., Ociepa, C., Phang,
J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika,
L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A.
The language model evaluation harness, 07 2024. URL
https://zenodo.org/records/12608602.
Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian,
A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A.,
Vaughan, A., et al. The llama 3 herd of models. arXiv
preprint arXiv:2407.21783, 2024.
Gu, A. and Dao, T. Mamba: Linear-time sequence mod-
eling with selective state spaces. In First conference on
language modeling, 2024.
Guo, C., Zhang, C., Leng, J., Liu, Z., Yang, F., Liu, Y.,
Guo, M., and Zhu, Y. Ant: Exploiting adaptive numerical
data type for low-bit deep neural network quantization.
In 2022 55th IEEE/ACM International Symposium on
Microarchitecture (MICRO), pp. 1414–1433. IEEE, 2022.
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika,
M., Song, D., and Steinhardt, J.
Measuring mas-
sive multitask language understanding. arXiv preprint
arXiv:2009.03300, 2020.
Henry, A., Dachapally, P. R., Pawar, S. S., and Chen, Y.
Query-key normalization for transformers. In Findings of
the Association for Computational Linguistics: EMNLP
2020, pp. 4246–4253, 2020.
Hu, W., Zhang, H., Guo, C., Feng, Y., Guan, R., Hua, Z.,
Liu, Z., Guan, Y., Guo, M., and Leng, J. M-ant: Efficient
low-bit group quantization for llms via mathematically
adaptive numerical type. In 2025 IEEE International
Symposium on High Performance Computer Architecture
(HPCA), pp. 1112–1126. IEEE, 2025a.
Hu, X., Cheng, Y., Yang, D., Xu, Z., Yuan, Z., Yu, J., Xu,
C., Jiang, Z., and Zhou, S. Ostquant: Refining large
language model quantization with orthogonal and scal-
ing transformations for better distribution fitting. arXiv
preprint arXiv:2501.13987, 2025b.
IEEE. Ieee standard for floating-point arithmetic. IEEE Std
754-2019 (Revision of IEEE 754-2008), pp. 1–84, 2019.
doi: 10.1109/IEEESTD.2019.8766229.
Kamath, A. K., Prabhu, R., Mohan, J., Peter, S., Ramjee,
R., and Panwar, A. Pod-attention: Unlocking full prefill-
decode overlap for faster llm inference. In Proceedings of
the 30th ACM International Conference on Architectural
Support for Programming Languages and Operating Sys-
tems, Volume 2, pp. 897–912, 2025.
Lee, J., Park, J., Cha, S., Cho, J., and Sim, J. Mx+: Push-
ing the limits of microscaling formats for efficient large
language model serving.
In Proceedings of the 58th
IEEE/ACM International Symposium on Microarchitec-
ture, pp. 869–883, 2025.
Lin, Y., Tang, H., Yang, S., Zhang, Z., Xiao, G., Gan, C., and
Han, S. Qserve: W4a8kv4 quantization and system co-
design for efficient llm serving. Proceedings of Machine
Learning and Systems, 7, 2025.

Lin, Z., Xu, H., Chen, G., Chen, Z., Lu, Y., and Zhang,
X.
Bullet: Boosting gpu utilization for llm serving
via dynamic spatial-temporal orchestration.
In Pro-
ceedings of the 31st ACM International Conference
on Architectural Support for Programming Languages
and Operating Systems, Volume 2, ASPLOS ’26, pp.
290–306, New York, NY, USA, 2026. Association for
Computing Machinery. ISBN 9798400723599. doi: 10.
1145/3779212.3790135. URL https://doi.org/
10.1145/3779212.3790135.
Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao,
C., Deng, C., Zhang, C., Ruan, C., et al. Deepseek-
v3 technical report. arXiv preprint arXiv:2412.19437,
2024a.
Liu, Z., Zhao, C., Fedorov, I., Soran, B., Choudhary, D., Kr-
ishnamoorthi, R., Chandra, V., Tian, Y., and Blankevoort,
T. Spinquant: Llm quantization with learned rotations.
arXiv preprint arXiv:2405.16406, 2024b.
Loshchilov, I. and Hutter, F. Decoupled weight decay regu-
larization. arXiv preprint arXiv:1711.05101, 2017.
Merity, S., Xiong, C., Bradbury, J., and Socher, R.
Pointer sentinel mixture models.
arXiv preprint
arXiv:1609.07843, 2016.
Penedo, G., Kydl´ıˇcek, H., Lozhkov, A., Mitchell, M., Raffel,
C. A., Von Werra, L., Wolf, T., et al. The fineweb datasets:
Decanting the web for the finest text data at scale. Ad-
vances in Neural Information Processing Systems, 37:
30811–30849, 2024.
Shazeer, N.
Glu variants improve transformer.
arXiv
preprint arXiv:2002.05202, 2020.
Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y.
Roformer: Enhanced transformer with rotary position
embedding. Neurocomputing, 568:127063, 2024.
Sun, Y., Liu, R., Bai, H., Bao, H., Zhao, K., Li, Y., Hu, J., Yu,
X., Hou, L., Yuan, C., et al. Flatquant: Flatness matters
for llm quantization. arXiv preprint arXiv:2410.09426,
2024.
Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han,
S. Smoothquant: Accurate and efficient post-training
quantization for large language models.
In Interna-
tional conference on machine learning, pp. 38087–38099.
PMLR, 2023.
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,
Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical
report. arXiv preprint arXiv:2505.09388, 2025.
Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi,
Y. Hellaswag: Can a machine really finish your sentence?
arXiv preprint arXiv:1905.07830, 2019.
Zhao, Y., Lin, C.-Y., Zhu, K., Ye, Z., Chen, L., Zheng, S.,
Ceze, L., Krishnamurthy, A., Chen, T., and Kasikci, B.
Atom: Low-bit quantization for efficient and accurate llm
serving. Proceedings of Machine Learning and Systems,
6:196–209, 2024.
Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin,
X., and Zhang, H. DistServe: Disaggregating prefill and
decoding for goodput-optimized large language model
serving. In 18th USENIX Symposium on Operating Sys-
tems Design and Implementation (OSDI 24), pp. 193–210,
2024.
Zhu, K., Gao, Y., Zhao, Y., Zhao, L., Zuo, G., Gu, Y., Xie,
D., Ye, Z., Kamahori, K., Lin, C.-Y., et al. NanoFlow: To-
wards optimal large language model serving throughput.
In 19th USENIX Symposium on Operating Systems De-
sign and Implementation (OSDI 25), pp. 749–765, 2025.
Zou, J., Chen, Y., Chen, X., Xu, C., and Chen, X. Ax-
core: A quantization-aware approximate gemm unit for
llm inference. In Proceedings of the 58th IEEE/ACM
International Symposium on Microarchitecture, MICRO
’25, pp. 839–853, New York, NY, USA, 2025. Associa-
tion for Computing Machinery. ISBN 9798400715730.
doi: 10.1145/3725843.3756094. URL https://doi.
org/10.1145/3725843.3756094.





## 附录 A. 使用精确 INT4 范围推导 NVINT4 与 NVFP4 的 QSNR 交点

本附录推导 NVINT4 与 NVFP4 具有相同量化信噪比（QSNR）时的峰值因子交点 $\kappa$。我们基于 Chen et al.（2025a）细化证明。这里 $\kappa$（非正式表述中有时记为 k）是块的峰值因子：

$$
\kappa=\frac{\max_i|X_i|}{\sigma},\qquad \sigma=\operatorname{RMS}(X). \tag{3}
$$

我们沿用该工作的高斯块假设 $X_i\overset{\mathrm{i.i.d.}}\sim\mathcal N(0,\sigma^2)$ 和 AbsMax 缩放约定，但不采用对 INT4 的 $Q\approx2^{b-1}$ 简化，而保留精确的对称 INT4 最大码值 Q=7。

### A.1. QSNR 与相对 MSE

QSNR 定义为：

$$
\operatorname{QSNR}=-10\log_{10}\left(\frac{\|X-X_q\|_2^2}{\|X\|_2^2}\right). \tag{4}
$$

在块内独立同分布模型下，可以用标量相对均方误差很好地近似 QSNR：

$$
R=\frac{\mathbb E[e^2]}{\mathbb E[X^2]}=\frac{\mathbb E[e^2]}{\sigma^2},\quad e=X-X_q,\quad \operatorname{QSNR}=-10\log_{10}R. \tag{5}
$$

因此，两种格式 QSNR 相等，等价于：

$$
R_{\mathrm{NVINT4}}(\kappa)=R_{\mathrm{NVFP4}}(\kappa). \tag{6}
$$

### A.2. 最大码值精确取 Q=7 时 NVINT4 的相对 MSE

NVINT4 使用对称有符号 INT4 码本 $\{-7,-6,\ldots,6,7\}$，即：

$$
Q=2^{b-1}-1=2^3-1=7. \tag{7}
$$

AbsMax 缩放中的理想缩放因子，将块最大值映射到 $\pm Q$：

$$
s=\frac{\max_i|X_i|}{Q}=\frac{\kappa\sigma}{Q}. \tag{8}
$$

NV 格式使用高精度缩放，即 E4M3 加第二级 FP32，因此取 $s'\approx s$，量化步长为 $\Delta=s'$。

**近似 A1：整数的高分辨率均匀误差。** 对不发生饱和的最近邻舍入均匀量化器，将逐元素误差建模为 $e\sim\operatorname{Uniform}[-\Delta/2,\Delta/2]$，于是：

$$
\mathbb E[e^2]\approx\frac{\Delta^2}{12}. \tag{9}
$$

代入 $\Delta\approx\kappa\sigma/Q$，得到基础整数相对 MSE：

$$
R_{\mathrm{INT}}(\kappa)\approx\frac1{12}\left(\frac\kappa Q\right)^2. \tag{10}
$$

**近似 A2：NV 格式中一个几乎无误差元素的修正。** 由于 NV 格式的块缩放精度较高，决定最大绝对值的块内最大元素几乎精确映射到 $\pm Q$，贡献的误差可以忽略。对于大小为 g 的块，这会将平均误差能量乘以 $(g-1)/g$：

$$
R_{\mathrm{NVINT4}}(\kappa)\approx\frac1{12}\left(\frac\kappa Q\right)^2\frac{g-1}{g}. \tag{11}
$$

NV 格式取 g=16，INT4 严格取 Q=7，因而：

$$
R_{\mathrm{NVINT4}}(\kappa)\approx\frac{\kappa^2}{12\cdot7^2}\frac{15}{16}. \tag{12}
$$

### A.3. NVFP4 的相对 MSE：包含次正规数的 FP4 E2M1

NVFP4 使用启用次正规数的 FP4 E2M1，AbsMax 缩放映射到最大有限正规数幅值。本文 E2M1 的常数为：

$$
M=1,\ B=1,\ Q_{\max}=6,\ N_{\min}=2^{1-B}=1,\ S_{\min}=2^{1-B-M}=2^{-1}=\tfrac12. \tag{13}
$$

最大绝对值映射到 $Q_{\max}$，在高精度 NV 缩放下：

$$
s'\approx s=\frac{\max_i|X_i|}{Q_{\max}}=\frac{\kappa\sigma}{6}. \tag{14}
$$

信号域中正规数区域与次正规数区域之间的阈值为：

$$
T_N:=s'N_{\min}=s'\quad\Rightarrow\quad t=\frac{T_N}{\sigma}=\frac\kappa6. \tag{15}
$$

#### A.3.1. 正规数区域误差项

定义正规数区域的能量占比：

$$
w_{\mathrm{norm}}(\kappa)=\frac{\mathbb E[X^2\mathbf1\{|X|\ge T_N\}]}{\sigma^2}. \tag{16}
$$

再次采用近似 A1，即每个浮点量化区间内的均匀误差：在正规浮点区间内，误差方差建模为 $\Delta(X)^2/12$。

**近似 A3：对数相位均匀。** 令 $r:=2^{\{\log_2(|X|/s')\}}\in[1,2)$ 表示小数部分对应的“对数相位”。原工作假设 r 在 [1,2) 上近似均匀，因此 $\mathbb E[1/r^2]=1/2$。在这一假设下，正规数区域的贡献为：

$$
\frac{\mathbb E[e^2\mathbf1\{|X|\ge T_N\}]}{\sigma^2}\approx\alpha_M w_{\mathrm{norm}}(\kappa),\qquad\alpha_M:=\frac1{24\cdot2^{2M}}. \tag{17}
$$

NVFP4 E2M1 的 M=1，因此：

$$
\alpha:=\alpha_{M=1}=\frac1{96}. \tag{18}
$$

#### A.3.2. 次正规数区域误差项

令落入次正规数区域的概率为：

$$
p_{\mathrm{sub}}(\kappa):=\mathbb P(|X|<T_N). \tag{19}
$$

在次正规数区域内，绝对步长是常数：

$$
\Delta_{\mathrm{sub}}:=s'S_{\min}=s'2^{1-B-M}. \tag{20}
$$

应用近似 A1，条件误差 $\mathbb E[e^2\mid|X|<T_N]\approx\Delta_{\mathrm{sub}}^2/12$。又因 $s'=\kappa\sigma/Q_{\max}$，次正规数对相对 MSE 的贡献为：

$$
\frac{\mathbb E[e^2\mathbf1\{|X|<T_N\}]}{\sigma^2}\approx\beta\kappa^2p_{\mathrm{sub}}(\kappa),\qquad\beta:=\frac{2^{2(1-B-M)}}{12Q_{\max}^2}. \tag{21}
$$

对于 NVFP4 E2M1，B=1、M=1、最大码值为 6，故：

$$
\beta=\frac{2^{-2}}{12\cdot6^2}=\frac1{1728}. \tag{22}
$$

#### A.3.3. NV 修正：从正规数误差预算中扣除块最大值

再次采用近似 A2：最大元素的误差可以忽略。高精度缩放使块最大值几乎精确映射到最大码值，因此从正规数区域预算中减去该元素的能量占比：

$$
\eta:=\frac{\max_i|X_i|^2}{g\sigma^2}=\frac{\kappa^2}{g}. \tag{23}
$$

修正后的 NVFP4 相对 MSE 模型为：

$$
R_{\mathrm{NVFP4}}(\kappa)\approx\alpha\left(w_{\mathrm{norm}}(\kappa)-\frac{\kappa^2}{g}\right)+\beta\kappa^2p_{\mathrm{sub}}(\kappa). \tag{24}
$$

### A.4. 高斯块下正规区能量占比与次正规区概率的闭式表达

令 $Z:=X/\sigma\sim\mathcal N(0,1)$。定义标准正态概率密度和累积分布函数：

$$
\varphi(z):=\frac1{\sqrt{2\pi}}e^{-z^2/2},\qquad\Phi(z):=\int_{-\infty}^z\varphi(u)\,du. \tag{25}
$$

根据式（15），$t=\kappa/6$，于是：

$$
p_{\mathrm{sub}}(\kappa)=\mathbb P(|Z|<t)=2\Phi(t)-1. \tag{26}
$$

此外：

$$
w_{\mathrm{norm}}(\kappa)=\mathbb E[Z^2\mathbf1\{|Z|\ge t\}]=2\int_t^\infty z^2\varphi(z)\,dz. \tag{27}
$$

利用 $\varphi'(z)=-z\varphi(z)$ 分部积分：

$$
\int_t^\infty z^2\varphi(z)\,dz=[-z\varphi(z)]_t^\infty+\int_t^\infty\varphi(z)\,dz=t\varphi(t)+1-\Phi(t). \tag{28}
$$

所以：

$$
w_{\mathrm{norm}}(\kappa)=2\bigl(t\varphi(t)+1-\Phi(t)\bigr),\qquad t=\kappa/6. \tag{29}
$$

### A.5. 交点方程及数值解：严格采用 Q=7

令式（12）与式（24）相等，并代入 g=16、$\alpha=1/96$、$\beta=1/1728$ 和 $t=\kappa/6$，得到关于峰值因子的标量方程：

$$
\frac1{96}\left(2\bigl(t\varphi(t)+1-\Phi(t)\bigr)-\frac{\kappa^2}{16}\right)+\frac{\kappa^2}{1728}\bigl(2\Phi(t)-1\bigr)=\frac{\kappa^2}{12\cdot7^2}\frac{15}{16},\quad t=\kappa/6. \tag{30}
$$

由于涉及正态累积分布函数，式（30）没有初等闭式解，需要使用二分法或牛顿法等数值方法求解。解得：

$$
\kappa^*=2.224277301764024. \tag{31}
$$

这意味着，当 $\kappa<\kappa^*$ 时，NVINT4 的 QSNR 更好；当 $\kappa>\kappa^*$ 时，NVFP4 的 QSNR 更好。在交点处，两者的相对 MSE 和 QSNR 分别为：

$$
R^*=R_{\mathrm{NVINT4}}(\kappa^*)=R_{\mathrm{NVFP4}}(\kappa^*)=0.007888089150418761. \tag{32}
$$

$$
\operatorname{QSNR}^*=-10\log_{10}R^*=21.03028189684982\ \mathrm{dB}. \tag{33}
$$

## 附录 B. 硬件开销分析

### B.1. 背景：将块缩放 FP4 表达为因式分解计算

块缩放 FP4 操作数 A 由打包的 FP4 张量与缩放张量表示：

$$
A\approx A_q\odot s_A. \tag{34}
$$

其中 $s_A$ 沿归约维度在每个 V 元素块内保持不变，本文 V=16。对一个块内长度为 V 的点积：

$$
\sum_{i=0}^{V-1}(a_i s_A)(b_i s_B)=(s_A s_B)\sum_{i=0}^{V-1}a_i b_i. \tag{35}
$$

式（35）启发了这样的微架构：先计算未缩放的局部点积，再为每个块执行一次缩放因子乘积的乘法。相比逐元素反量化，这将缩放相关算术操作减少到原来的 1/V。


### B.2. 统一的双模式 FP4：NVFP4-E2 与 NVFP4-E1

我们定义两种共享块大小和缩放类型的块缩放 FP4 模式：

- **NVFP4-E2**：块大小 16，缩放因子为 E4M3，FP4 类型为 E2M1。
- **NVFP4-E1**：块大小 16，缩放因子为 E4M3，FP4 类型为 E1M2。

**FP4 位布局。** 每个 FP4 数据包含一个符号位和三位有效载荷：$x_4=[s\mid p_2p_1p_0]$。不同模式对有效载荷的解释不同：

$$
\mathrm{E2M1}:\quad e=[p_2p_1],\quad m=[p_0]. \tag{36}
$$

$$
\mathrm{E1M2}:\quad e=[p_2],\quad m=[p_1p_0]. \tag{37}
$$

我们采用类似 IEEE 的共同约定：E2M1 偏置为 1，E1M2 偏置为 0；指数域为零时使用次正规数。不包含 Inf/NaN 等特殊值，这与许多 FP4 部署一致。

**类整数码本映射。** 在上述约定下，忽略符号时 E1M2 给出近似均匀的幅值集合 $\{0,0.5,1.0,\ldots,3.5\}$。以固定缩放 $\alpha=2$ 映射为：

$$
\alpha\cdot\{0,0.5,1.0,\ldots,3.5\}=\{0,1,2,\ldots,7\}. \tag{38}
$$

加入符号后得到 $\{-7,\ldots,7\}$，与常用对称 INT4 量化范围一致。这在不改变计算数据通路的情况下，建立了类浮点与类整数 FP4 语义之间的架构连接。

### B.3. 零额外开销的元数据打包：将类型放入缩放因子

一个实际挑战是，如何在不扩大元数据的情况下传递 FP4 子格式 E2M1/E1M2。我们利用如下观察：NVFP4 缩放因子保存为无符号的 FP8 E4M3 数值，因此其符号位固定为零，或未被数值编码使用。

我们将 8 比特缩放因子的最高位改作类型位：

```text
scale_packed[7]   = type
scale_packed[6:0] = ue4m3_mag
```

硬件通过强制符号位置零来恢复无符号缩放因子：

$$
\mathrm{scale\_ue4m3}=\{1'b0,\ \mathrm{scale\_packed}[6:0]\}. \tag{39}
$$

类型元数据因此不增加存储或带宽成本。实现中，类型在块内共享，以尽可能摊薄成本并简化控制。

### B.4. 以 NAND 等效门分析开销

我们比较两项硬件增量：（i）在 E2M1 之外启用 E1M2；（ii）实现块缩放机制，即缩放因子乘积和每块一次乘法。为了便于跨综合流程比较，成本以 NAND 等效门表示，并给出参数化公式。

依据已有工作（Chen et al., 2025a）的估计，我们建模总硬件开销。对齐器位宽为：

$$
n=\min\left(2^{x+1}+2y,\ \mathrm{psum\_bit\_width}\right). \tag{40}
$$

![表 6：k 通道 MAC 的门复杂度模型](images/table-06.png)

**表 6.** k 通道 MAC 单元的门复杂度模型。x、y 分别表示指数和尾数宽度；整数格式取 x=0；对齐器宽度 n 由式（40）给出。最后一列列出占主导的标准单元。FA 为全加器，HA 为半加器，MUX 为多路选择器。

| 子模块 | 整数乘法 | 浮点乘法 | 整数加法 | 浮点加法 | 主要单元 |
| --- | --- | --- | --- | --- | --- |
| 乘法器 | $k(x+y+1)^2$ | $k(y+1)^2$ | — | — | AND、FA、HA |
| 尾数/整数加法器 | — | — | $2k(x+y+1)$ | $kn$ | FA、HA |
| 指数加法器 | — | $kx$ | — | — | FA、HA |
| 指数减法器 | — | — | — | $kx$ | XOR、FA、HA |
| 比较器 | — | — | — | $kx$ | XOR、AND、OR |
| 桶形对齐器 | — | — | — | $kn\log_2n$ | MUX |
| 共享规格化器 | — | — | — | $n\log_2n$ | MUX、OR |

#### B.4.1. 成本模型

采用标准架构级近似：

$$
G_{\mathrm{NOT}}=1\ \mathrm{NAND}. \tag{41}
$$

$$
G_{\mathrm{AND2}}=2\ \mathrm{NAND}. \tag{42}
$$

$$
G_{\mathrm{OR2}}=2\ \mathrm{NAND}. \tag{43}
$$

$$
G_{\mathrm{HA}}=5\ \mathrm{NAND}. \tag{44}
$$

$$
G_{\mathrm{FA}}=12\ \mathrm{NAND}. \tag{45}
$$

$$
G_{\mathrm{MUX2}}=2\mathrm{AND}+\mathrm{OR}+\mathrm{NOT}. \tag{46}
$$

$$
G_{\mathrm{MUX2}}=7\ \mathrm{NAND}. \tag{47}
$$

#### B.4.2. 支持 E1M2 的增量成本

相对仅支持 E2M1 的解码（指数位和尾数位直接连线），双模式解码为每个 FP4 元素增加：

$$
\Delta G_{\mathrm{E1M2,elem}}\approx2G_{\mathrm{MUX2}}+2G_{\mathrm{AND2}}\approx18\ \mathrm{NAND}. \tag{48}
$$

块共享的类型取反 $\neg t$ 每块只需一个反相器，由 8 条通道摊销。对于一次块点积，A、B 两个操作数都需要解码，共 16 个 FP4 元素，增量成本为：

$$
\Delta G_{\mathrm{E1M2,block},A+B}\approx16\times18=288\ \mathrm{NAND}. \tag{49}
$$

此外还需常数个反相器，其成本可忽略。

将 E2M1 乘法器改为 E2M2，也会增加乘法器之间加法器的开销：乘法器从 $8\times4$ 个全加器增加到 $8\times9$，加法器从 $8\times10$ 个全加器增加到 $8\times12$，对齐器从约 $8\times30$ 个 MUX 增加到约 $8\times40$ 个 MUX。这里假设 MUX 约等于 MUX2。

最终，一个累加向量的估计总增量约为：

$$
\Delta G\approx288+40\mathrm{FA}+16\mathrm{FA}+80\mathrm{MUX}=288+480+192+560=1520\ \mathrm{NAND}. \tag{50}
$$

#### B.4.3. 局限性

硬件模型是粗粒度估计，仅用于把握预测结果的数量级。我们没有建模寄存器开销；虽然加入额外寄存器会降低所报告的相对开销百分比，但也会降低模型的保真度。

### B.5. 硬件设计细节

#### B.5.1. MixFP4 解码器

图 13 展示 MixFP4 使用的解码器。块缩放因子的符号位被改作块共享类型信号，用来选择 E2M1 移位路径或 E1M2 查表路径，两条路径都产生统一的 E2M2 内部表示。

![图 13：MixFP4 解码器](images/figure-13.png)

**图 13.** 块缩放符号位控制选择 E2M1 移位路径与 E1M2 查表路径，将 4 比特 MixFP4 输入转换为 5 比特 E2M2 内部表示。

对 E2M1，解码仅需简单位扩展，即在末尾补零；对 E1M2，则使用固定的小型查找表，将 4 比特有效载荷映射到同一个内部格式。这个简单解码器将两种输入统一为 E2M2，为下游计算数据通路提供一致的数据格式。

## 附录 C. 与 SmoothQuant、GPTQ、SpinQuant 集成的实验设置

### C.1. SmoothQuant

使用 WikiText-raw-v1 的训练划分作为 SmoothQuant 校准集。每个模型使用 512 个样本，每个样本含 512 token。所有相关实验的迁移强度超参数 $\alpha$ 都设为 0.5。

### C.2. GPTQ

采用 FP-Quant（Egiazarian et al., 2025）中的 GPTQ 实现。校准数据来自 WikiText-raw-v1 训练划分，共 1024 个样本，序列长度 2048。遵循 FP-Quant 仓库实现，在 GPTQ 过程中启用激活量化。

格式选择使用静态策略：GPTQ 算法开始前，基于校准数据确定每个权重块的最优格式。因此，GPTQ 误差补偿期间所选格式固定，不会被误差传播过程更新。

### C.3. SpinQuant

SpinQuant 同样使用 WikiText-raw-v1 的训练划分校准。遵循原 SpinQuant 仓库（Liu et al., 2024b）设置，旋转矩阵优化 100 个训练步骤，每步包含 2048 token。

旋转后的 GPTQ 步骤采用 SpinQuant 仓库原始配置，与附录 C.2 不同：这一过程中不量化激活；此外，格式选择使用动态策略，在误差更新过程中一边量化权重，一边选择使 MSE 最小的数据格式。

## 附录 D. 随机舍入消融

随机舍入以概率方式将一个值舍入到最近的两个可表示值之一，选择概率与距离成反比。对整数格式，例如范围为 [-7,7] 的格式，可以写为：

$$
\widetilde q_{\mathrm{scaled}}=q_{\mathrm{scaled}}+\epsilon-0.5,\qquad\epsilon\sim\mathcal U(0,1). \tag{51}
$$

$$
q=\operatorname{clamp}(\operatorname{round}(\widetilde q_{\mathrm{scaled}}),-7,7). \tag{52}
$$

表 7 展示在训练最后 2.45B–2.5B 已见 token 区间测得的验证损失消融结果。


![表 7：随机舍入有效性的消融实验](images/table-07.png)

**表 7.** 随机舍入有效性的消融实验。列标题为已处理的 token 数，单位十亿；各行为验证损失，越低越好。`+SR` 表示为上一行量化方案增加随机舍入。

| 方案 | 2.454 | 2.459 | 2.464 | 2.469 | 2.475 | 2.480 | 2.485 | 2.490 | 2.495 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | 3.7376 | 3.6300 | 3.6222 | 3.6696 | 3.6539 | 3.6427 | 3.5881 | 3.6514 | 3.6795 |
| NVFP4 | 3.8051 | 3.6964 | 3.6873 | 3.7319 | 3.7183 | 3.7068 | 3.6518 | 3.7163 | 3.7458 |
| NVFP4 +SR | 3.8055 | 3.6947 | 3.6868 | 3.7313 | 3.7190 | 3.7077 | 3.6523 | 3.7182 | 3.7446 |
| 4/6 | 3.7941 | 3.6850 | 3.6770 | 3.7222 | 3.7069 | 3.6986 | 3.6436 | 3.7072 | 3.7357 |
| 4/6 +SR | 3.7933 | 3.6849 | 3.6747 | 3.7224 | 3.7076 | 3.6986 | 3.6427 | 3.7063 | 3.7349 |
| MixFP4 | 3.7942 | 3.6866 | 3.6787 | 3.7215 | 3.7071 | 3.6962 | 3.6409 | 3.7070 | 3.7339 |
| MixFP4 +SR | 3.7919 | 3.6839 | 3.6748 | 3.7193 | 3.7047 | 3.6940 | 3.6381 | 3.7047 | 3.7320 |

## 附录 E. 逐块格式选择可视化

本节展示候选集合 $\{\mathrm{FP4\ E2M1}(6),\mathrm{FP4\ E1M2},\mathrm{FP4\ E3M0}\}$ 下的逐块格式选择可视化结果。图中蓝、绿、红分别表示 E2M1(6)、E1M2、E3M0；左列是权重，右列是激活。

![图 14：Llama Q 投影的逐块选择](images/figure-14.png)

**图 14.** Llama-3.1-8B 第 0、10、20、30 层 Q 投影模块的逐块格式选择。

![图 15：Qwen Q 投影的逐块选择](images/figure-15.png)

**图 15.** Qwen3-8B 第 0、10、20、30 层 Q 投影模块的逐块格式选择。

![图 16：Llama UP 投影的逐块选择](images/figure-16.png)

**图 16.** Llama-3.1-8B 第 0、10、20、30 层 UP 投影模块的逐块格式选择。

![图 17：RHT 后 Llama Q 投影的逐块选择](images/figure-17.png)

**图 17.** 使用随机 Hadamard 变换（RHT）后，Llama-3.1-8B 第 0、10、20、30 层 Q 投影模块的逐块格式选择。

![图 18：RHT 后 Llama UP 投影的逐块选择](images/figure-18.png)

**图 18.** 使用 RHT 后，Llama-3.1-8B 第 0、10、20、30 层 UP 投影模块的逐块格式选择。
