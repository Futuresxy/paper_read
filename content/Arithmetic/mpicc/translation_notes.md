# MPICC: Multiple-Precision Inter-Combined MAC Unit with Stochastic Rounding for Ultra-Low-Precision Training 原文翻译

# MPICC：面向超低精度训练、采用随机舍入的多精度交叉组合 MAC 单元

Leran Huang<sup>1</sup>, Yongpan Liu<sup>2</sup>, Xinyuan Lin<sup>2</sup>, Chenhan Wei<sup>2</sup>, Wenyu Sun<sup>2</sup>, Zengwei Wang<sup>2</sup>, Boran Cao<sup>1</sup>, Chi Zhang<sup>1</sup>, Xiaoxia Fu<sup>2</sup>, Wentao Zhao<sup>2</sup>, and Sheng Zhang<sup>1∗</sup>

<sup>1</sup>清华大学深圳国际研究生院先进传感器与集成系统重点实验室

<sup>2</sup>清华大学电子工程系

<sup>1</sup>中国深圳 <sup>2</sup>中国北京

zhang\_sh@mail.tsinghua.edu.cn\*

## 摘要

近期研究已证明超低精度（≤ 8 位）训练的可行性。然而，现有运算电路大多仅支持少数较高精度（FP16、FP32 等）的乘法运算，且累加位宽无法降低，导致面积效率和功耗效率较低。本文提出 MPICC，一种面向超低精度训练的多精度乘加（MAC）单元。它支持 18 种不同精度组合的交叉组合计算，包括 LOG4（以 4 为基数的 FP4）、FP4、FP6、FP8 和 INT4。它还通过优化的随机舍入（SR）策略，将累加精度从 FP16 降至 FP12，以进一步节省逻辑资源。此外，还设计了一个低成本模拟控制器，通过时分复用低精度 MAC 单元，为关键 DNN 层完成高精度计算。与现有多精度计算单元相比，该设计在 FP8 下的面积效率/能效分别提升 1.17×/1.19×，在 FP4 下则分别提升 4.69×/3.64×。SR 策略进一步使浮点累加器的面积/功耗分别降低 15.6%/14.9%。

## CCS 概念

• 硬件 → 集成电路；逻辑电路；算术与数据通路电路；

## 关键词

超低精度训练，乘加（MAC）单元，多精度计算，随机舍入（SR）

## ACM 参考文献格式：

Leran Huang, Yongpan Liu, Xinyuan Lin, Chenhan Wei, Wenyu Sun, Zeng wei Wang, Boran Cao, Chi Zhang, Xiaoxia Fu, Wentao Zhao, and Sheng Zhang. 2025. MPICC: Multiple-Precision Inter-Combined MAC Unit with Stochastic Rounding for Ultra-Low-Precision Training. In 30th Asia and South Pacific Design Automation Conference (ASPDAC ’25), January 20–23, 2025, Tokyo, Japan. ACM, New York, NY, USA, 6 pages. https://doi.org/10. 1145/3658617.3697641

表 1：用于深度学习的 PE 的特性比较

<table><tr><td rowspan="2">设计</td><td colspan="5">支持的低精度</td><td rowspan="2">随机舍入</td><td rowspan="2">MPICC</td><td rowspan="2">高精度模拟</td></tr><tr><td>FP16</td><td>FP8</td><td>FP4</td><td>LOG4</td><td>INT4</td></tr><tr><td>本文方案</td><td>○</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>[16]</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>[7]</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>√</td><td>×</td><td>×</td></tr><tr><td>[10]</td><td>×</td><td>×</td><td>√</td><td>×</td><td>√</td><td>×</td><td>√</td><td>×</td></tr></table>

◦ 支持 FP12，且累加准确度与 FP16 相同

## 1 引言

DNN 在各个领域均取得了显著的性能表现。然而，训练期间对计算资源、时间和功耗的高需求带来了重大挑战。因此，研究人员采用量化后的低位宽数据进行训练，显著减少了内存使用量以及数据搬移和处理的相关开销。早期量化技术使用 16 位浮点数进行训练[11]，相较于 32 位系统，性能提升了 4-8×。后续研究 [9] 将精度降至 8 位，相较于 16 位系统，性能进一步提升了 2-4×。低于 8 位的训练需要根据张量的分布特征选择最优数据格式，因此产生了对多精度交叉组合计算（MPICC）的需求。例如，[2] 使用 FP4 权重和 FP6 激活实现了语言模型训练。此外，[1, 13] 探索了采用 FP4、INT4 和 LOG4 等多种 4 位格式的训练，相较于 16 位系统，实现了 4-7× 的性能提升。他们还发现，对关键运算采用高精度可以提高网络的准确度和兼容性。

设计专用硬件架构可以最大限度地发挥量化算法带来的收益。然而，现有计算硬件通常不支持低于 8 位的精度和 MPICC[7, 14]，因而无法兑现超低精度训练的优势。此外，随机舍入（SR）对超低精度训练至关重要 [1]，它能够解决训练停滞现象 [8]，并在保证准确度的前提下降低累加器位宽 [10]，从而节省资源。然而，当前硬件尚未广泛支持 SR。

为应对这些挑战，本文提出了一种面向超低精度训练的 MAC 单元，主要贡献如下：

• 提出了一种多精度交叉组合计算（MPICC）架构，将基本类型解码为统一格式[3]，并使用多项点积单元执行 LOG4（E4M0）、FP4（E3M0、E2M1）、FP6（E3M2、E2M3）、FP8（E5M2、E4M3）和 INT4 的交叉组合计算。该设计根据尾数位宽灵活执行 1/2/4 项点积，最大限度地提高资源利用率。

• 在累加过程中引入了 Variable-SR 结构，在保持准确度的同时，将累加器位宽从 10 位降至 6 位。这种显著节省资源的方法可应用于任何架构的 MAC 单元。

• 设计了一种低成本高精度模拟控制器，将一次 FP12 计算分解为多次 FP8 计算。通过复用低精度 MAC 单元实现高精度计算，仅以 5.7% 的面积开销扩展了设计所支持的精度范围。

本文组织如下。第 2 节回顾算法和硬件方面的相关工作。第 3 节介绍 MPICC 架构、Variable-SR 结构和高精度模拟控制器的实现。第 4 节给出实验与比较。第 5 节对全文进行总结。

## 2 相关工作

超低精度训练对计算硬件提出了两项要求[1]。首先，硬件必须支持多种8位及以下精度的格式。但当前大多数硬件支持的最低精度仅为8位。例如，[6]提出了一种多精度乘法单元，支持四种精度类型：INT8、FP8、FP16和FP32。[14]提出了一种基于POSIT的存内计算宏，用于执行8位训练，但采用低于8位的POSIT进行训练的可行性尚未得到证明。其次，硬件必须能够直接对不同位宽和数据格式的数据执行运算（MPICC）。尽管[16]在输入精度不同时，采用精度转换电路在计算前统一精度，但其计算单元本身并不支持MPICC。在[3]中，不同格式首先被解码为统一格式，然后再进行计算，从而在4位精度下实现了MPICC。然而，这种方法适用的网络范围有限。

在超低精度训练中，累加器位宽的减小会加剧停滞现象，即一系列较小的反向传播增量因舍入而丢失。能够提供无偏估计的随机舍入已成为解决这一问题的一种方案[8]。SR最早被用于实现8位浮点（FP）训练[9]，并且已有研究证明，对数量化和SR对于4位训练是必要的[1]。在硬件实现方面，市场上已经出现了多个支持SR的平台，如Graphcore IPU、Intel Loihi和Tesla Dojo。在学术界，[7]设计了首个支持SR的开源浮点MAC单元，[10]则提出了用于浮点加法器的Eager-SR结构，通过提前执行SR来缩短关键路径。然而，这种结构只能用于位宽相等的加法，因此无法用于乘积与加数位宽不一致的MAC单元。

对关键计算采用高精度，可以显著提高超低精度训练的准确率，并扩大其支持的网络范围[9]。例如，[13]将ResNet-50中Conv 1x1层的精度从FP4提高到FP8，使准确率损失从1.28%降至0.34%。[1]在4位训练后进行了三轮FP16微调，将ResNet-50的准确率损失从1.1%降至0.32%，吞吐量降低了24%。由于支持高精度会带来大量硬件开销，[12, 15]使用低精度硬件模拟高精度计算。他们在Tensor Core上使用四次FP16计算来模拟FP32计算。然而，这种方法仅适用于配备复杂编译器的高端芯片。对于低端芯片，定制硬件以模拟高精度更为实际。

表2：所提出的MAC单元支持的精度模式

<table><tr><td>K</td><td>精度</td><td>K</td><td>精度</td><td>K</td><td>精度</td></tr><tr><td>1/3</td><td>FP12×FP12</td><td>2</td><td>FP8×FP4</td><td>1</td><td>FP8×FP8</td></tr><tr><td>1/2</td><td>FP12×FP8</td><td>2</td><td>FP8×LOG4</td><td>1</td><td>FP8×FP6</td></tr><tr><td>1/2</td><td>FP12×FP6</td><td>2</td><td>FP6×FP4</td><td>1</td><td>FP6×FP6</td></tr><tr><td>4</td><td>FP4×FP4</td><td>2</td><td>FP6×LOG4</td><td>1</td><td>FP8×INT4</td></tr><tr><td>4</td><td>FP4×LOG4</td><td>2</td><td>INT4×FP4</td><td>1</td><td>FP6×INT4</td></tr><tr><td>4</td><td>LOG4×LOG4</td><td>2</td><td>INT4×LOG4</td><td>1</td><td>INT4×INT4</td></tr></table>

综上，本设计通过开发一种支持MPICC、改进的SR策略以及用低精度模拟高精度的MAC单元，满足了超低精度训练的三项主要需求。表1比较了本设计与其他处理单元（PE）的功能特性。

## 3 所提出的MPICC架构

所提出的MPICC MAC单元根据输入数据的尾数长度，在每次操作中执行1/2/4个点积，并以FP12/INT8格式累加结果，如公式1所示。表2列出了本设计支持的所有精度模式，其中K表示每次操作执行的点积数量。本设计由四个组件组成，如图1所示。

$$
\mathrm{F} = \left\{ \begin{array}{l l} A _ {1} \times B _ {1} + C & K \leq 1 \\ A _ {1} \times B _ {1} + A _ {2} \times B _ {2} + C & K = 2 \\ A _ {1} \times B _ {1} + A _ {2} \times B _ {2} + A _ {3} \times B _ {3} + A _ {4} \times B _ {4} + C & K = 4 \end{array} \right.\tag{1}
$$

1) 解码器：将浮点、定点和对数格式解码为由符号、指数和尾数组成的统一格式。

2) 多点积单元：以统一格式执行多个点积；当输入全部为INT4时，输出格式为INT8，否则输出格式为FP12。

3) 高精度模拟控制器：将输入的FP12分解为两个FP8，并通过累加多次低精度计算来模拟高精度计算。

4) 随机数生成器：利用线性反馈移位寄存器（LSFR）生成一个12位随机数，用于SR。

![](images/aa4042306d4fb8c5ea57eff3357d8eb5c6b3f703912d6d4bb991299ede02d2ad.jpg)  
图1：所提出的MPICC MAC单元架构

## 3.1 MPICC实现原理

[4]提出，任意2<sup>n</sup>位乘法都可以分解为多个2位乘法结果移位后的和。例如，为实现4位乘法，先将输入分解为高2位和低2位两部分，再执行四次2位乘法。随后将乘积分别左移4/2/2/0位，并使用8位加法器树进行归约，如图2所示。通过将移位量分别调整为2/2/0/0位和0/0/0/0位，该电路还可以执行两次4×2位乘法或四次2×2位乘法。

我们将这种方法扩展到浮点乘法。由于FP8包含最多4位的尾数，而FP4/LOG4包含最多2位的尾数，因此图2中电路的一种变体可以执行一次FP8×FP8、两次FP8×FP4或四次FP8×FP4乘法。所需的修改包括根据符号进行补码转换、根据指数差调整移位量，以及添加解码、规格化和舍入等浮点特有的处理过程。

![](images/5eee34f4a92a39d65d7c94a9669521a3928a097e387a98723185892edfc4df17.jpg)  
图2：通过四次2位乘法实现4位乘法

## 3.2 解码器单元

解码器将各种基本类型解码为符号、指数和尾数格式。为实现兼容，FP4被解码为E4M1，以与LOG4（E4M0）和E2M1对齐，而FP8和FP6被解码为E5M3，以与E5M2和E4M3对齐。图3以浮点数为例说明了解码过程，其中定点数被视为指数为0的浮点数。

解码器的16位输入由四个FP4或两个FP8组成，在某些情况下，仅部分数据有效，如图3(a)所示。符号解码的结果为四个乘积的符号，如图3(b)所示。解码指数时，需要调整指数的偏置，例如将E3M0（偏置=3）解码为E4M1（偏置=7）时需要加4。FP8指数解码的部分结果无效，会被置零，如图3(c)所示。尾数解码的8位结果包含前导位。为执行FP8所需的4位乘法，我们根据图2所示的逻辑，将4位尾数分解为四个2位尾数，并在左、右解码器中采用不同的排列方式，如图3(d)所示。

## 3.3 多点积单元

图 4 展示了所提出的五级流水线多点积单元的架构，其中 LZC 表示前导零计数。前三个流水级以中间格式生成 1/2/4 个点积结果，后两个流水级将其与部分和相加，得到最终结果。定点运算和浮点运算复用全部计算逻辑，图 4 中未展示这一点。

## 流水级 1：乘积解包与抵消

首先，对输入符号进行异或运算，得到四个乘积的符号，并将输入被乘数的指数相加，得到四个乘积的指数。然后，将指数和尾数相同但符号相反的乘积置零，从而避免因两个较大且符号相反的乘积而对小乘积进行右移所造成的精度损失。

## 流水级 2：乘法、指数比较与对齐

我们采用 [5] 中描述的并行指数比较方案，找出四个乘积中的最大指数。接下来，根据指数差和乘法形式确定移位量。随后调整该最大指数的偏置，使其与结果指数的偏置一致。最后，根据移位量和符号对乘积进行对齐，并在移位后取其二进制补码。

## 流水级 3：归约与乘积规格化

经加法器树归约后，需要对点积结果进行规格化，以便后续执行标准浮点加法。规格化尾数的高 9 位用于浮点加法，低 5 位用于提前进行 SR。

## 流水级 4：点积结果与部分和相加

在解码、指数比较和对齐之后，将部分和与点积结果相加。提前 SR 与加法并行执行，生成提前舍入进位。加法结果的高 10 位用于规格化左移，低 6 位用于 SR 校正。

## 流水级 5：加法结果规格化与舍入

对加法结果进行第二次规格化，生成舍入前的符号、指数和尾数。SR 校正利用规格化结果、一个 2 位随机数以及提前舍入进位来确定最终舍入进位。舍入输出支持非规格化数。

![](images/33310a29a58f9a4e34787f062193e5cbc6e2c09a023c01085e3b376b24774b10.jpg)  
图 4：多点积单元架构

![](images/6192a63364a0e36d63e6a9c0be5594cdaf2067582e52f3a75c2031ec49611ca4.jpg)  
(a) 输入排列

(b) 符号解码  
![](images/d61b05628050e0b059edcc82c120bbc7224f22ae65b1ff121cda266bf0476da2.jpg)  
(c) 指数解码

![](images/744f7db935785e08d11b734025d36c2f2283cabf3ad218eea3d729b0f6745467.jpg)  
(d) 尾数解码

图 3：解码器单元逻辑  
![](images/e22043cd95361930df82405bb70881d831699f1fbcbfe3e37fc074708f01d57f.jpg)

(a) 尾数相加期间的提前 SR  
![](images/b6749d24119f70ff7476c8d69bf0172b24233e8de83d5781c445884b4d37f921.jpg)  
(b) 规格化期间的 SR 校正  
图 5：可变 SR 策略

## 3.4 可变 SR 策略

传统 SR 在规格化之后执行，会导致较大的延迟。为优化这一点，[10] 提出了 Eager-SR 策略，在规格化之前执行提前 SR，并在规格化期间进行 SR 校正。然而，该策略仅适用于操作数位宽相等的浮点加法。考虑到 MAC 单元中点积结果（PROD）与部分和（PSUM）的位宽往往不同，本文提出了一种适用于不等位宽加法的可变 SR 策略，如图 5 所示。通过采用提前 SR 来优化延迟并减小规格化位宽，实现了 PPA 的全面改善。

## 尾数相加期间：提前 SR

操作数对齐后，将 10 位随机数插入对齐结果中。其高 8 位用于尾数相加，低 10 位用于两次并行执行的 SR 加法，如图 5(a) 所示。当 PROD 被对齐时，将随机数插入 PSUM 的 LSB 之后，并且仅执行第一次 SR 加法。当 PSUM 被对齐时，将随机数的低 5 位插入 PROD 的 LSB 右侧，执行第一次 SR 加法，然后将该加法结果与随机数的高 5 位相加，执行第二次 SR 加法，得到提前 SR 舍入进位 $( S _ { 1 } S _ { 2 } )$

## 规格化期间：SR 校正

根据加法结果的前导零数（CNT），分四种情况进行 SR 校正，如图 5(b) 所示。

1) CNT=0：尾数相加产生进位；将加法结果的低 2 位、一个 2 位随机数和 $\mathrm { S } _ { 1 } \mathrm { S } _ { 2 }$ 相加，产生最终舍入进位 $\left( \mathrm { C } _ { 1 } \mathrm { C } _ { 2 } \right)$

2) CNT=1：尾数相加未产生进位；将其 LSB、一个 1 位随机数和 $\mathrm { S } _ { 1 } \mathrm { S } _ { 2 }$ 相加，产生 $\mathrm { C } _ { 1 } \mathrm { C } _ { 2 }$

3) CNT=2：尾数相加产生 1 位退位；$\mathrm { C } _ { 1 } \mathrm { C } _ { 2 }$ 等于 S<sub>1</sub>S<sub>2</sub>。

4) CNT>2：尾数相加发生抵消，且加法结果是精确的，因此无需 SR。为保持精度，根据 PROD 与 PSUM 的指数差，将 PROD 的低 5/6 位放入规格化移位器的低位。

## 3.5 高精度模拟控制器

一个 FP12 数可以分解为两个 FP8 数之和，即 FP12 $= \mathrm { F P } 8 _ { H } + \mathrm { F P } 8 _ { L }$，如图 6 所示。因此，一次 FP12 × FP12 运算可以分解为四次 FP8 × FP8 运算，并以 FP12 精度累加，即 FP12<sub>1</sub> × $\mathrm { F P 1 } 2 _ { 2 } = \mathrm { F P } 8 _ { H 1 } \times \mathrm { F P } 8 _ { H 2 } + \mathrm { F P } 8 _ { H 1 } \times \mathrm { F P } 8 _ { L 2 } + \mathrm { F P } 8 _ { L 1 } \times \mathrm { F P } 8 _ { H 2 } + \mathrm { F P } 8 _ { L 1 }$ $\mathrm { ~ \times ~ F P 8 } _ { L 2 }$。其中，$\mathrm { F P } 8 _ { L 1 } \times \mathrm { F P } 8 _ { L 2 }$ 对结果的影响极小 [15]，因此被省略。FP12 × FP8 计算采用类似的方法，但只需要两次乘累加计算。高精度模拟控制器执行 FP12 分解，并将分解结果依次输入 MPICC MAC 单元，以实现高精度计算。

具体的分解过程如图 6 所示。首先，FP8<sub>H</sub> 的符号、指数及尾数的前 2 位与 FP12 保持一致。然后，根据 FP12 尾数的第 3 位（R）对 FP8<sub>H</sub> 进行就近舍入。通过调整 FP8<sub>L</sub> 的指数，我们用它来表示 FP12 尾数的低位。当 R=0 时，符号与 FP12 相同；对 FP12 的最后 4 位进行 LZC，其中第一个前导 1 作为 FP8<sub>L</sub> 的隐含位，随后的 2 位用作尾数。当 R=1 时，符号与 FP12 相反，并且在执行 LZC 之前对 FP12 的最后 4 位取补。

当 FP12 的指数大于或等于 7 时，上述分解完全准确。否则，当 $\mathrm { F P } 8 _ { L }$ 的理论指数小于 1 时，分解可能失败。通过支持 FP8<sub>L</sub> 的非规格化数表示，我们将能够准确分解的 FP12 的最小指数从 7 降低到 5。

![](images/7db7df769a1e50488a54914cb820bf02b7637077e5c58176ccc09970721b416c.jpg)  
符号 ʍ R 指数 - LZC - 3 前导 1 之后  
图 6：一个 FP12 数分解为两个 FP8 数之和

![](images/15a746f798ac8c2d7d2c9be3d7941beb4b0b2e7c2f86a704a086c51a78685ac4.jpg)  
图 7：不同累加器的累加结果

## 4 实验与比较

我们使用 Synopsys Design Compiler 2021.09 和 PrimeTime PX 2021.06，在 TSMC 28nm 工艺的典型工艺角（TT、0.9 V、25°C）下探索了 MPICC MAC 单元的设计空间。在不使用流水线级的情况下，实现了较宽松的 200 MHz 时钟频率，MPICC MAC 单元的总面积为 1619.16 µm2。其中，多路点积单元、解码器和高精度模拟控制器分别占总面积的 61.6%、32.7% 和 5.7%，总功耗为 0.234 mW。

## 4.1 不同累加器模式的分析

为了分析舍入模式和尾数位宽对 MPICC MAC 累加器精度的影响，我们将采用 6 位尾数（FP12）的最近偶数舍入（RNE）、Eager-SR 和 Variable-SR，与采用 10 位尾数（FP16）的 RNE 进行了比较。运算模式为 FP8×FP8，其中一个输入设为 1，另一个输入服从均匀分布（均值=1，标准差=1），共累加 15,000 次。使用 FP64 作为基准。

不同精度下的累加结果如图 7 所示。FP12 RNE 的精度最差，在累加值达到 512 时饱和，这是因为增量相对于累加值过小，因而始终被丢弃。FP16 RNE 在超过 4500 后逐渐偏离理论值，并在 8192 处饱和。两种 SR 方法均通过在小增量情况下以一定概率进行舍入，避免了饱和与停滞。Variable-SR 的结果更接近基准，因为它只执行一次 SR，而 Eager-SR 需要执行两次。

我们进一步比较了 MPICC MAC 累加器不同模式的性能，如表 3 所示。为了能够应用 Eager-SR，我们对 PROD 的低 6 位执行了一次随机舍入，以确保 PROD 和 PSUM 在相加前具有相同的位宽。

实验结果表明，与常规 SR 累加器相比，Variable-SR 累加器的面积和功耗分别降低了 6.6% 和 11.3%。这一改进主要得益于关键路径的分解，以及规格化位宽从（尾数 + 随机数 + 1）位缩减为（尾数 + 2）位。与 Eager-SR 相比，Variable-SR 无需对 PROD 执行额外的 SR，因此在获得更高精度的同时，节省了 LSFR 和舍入运算的开销。通过使用 SR，我们将规格化和舍入数据通路的位宽减少了 4 位，使延迟降低了 16.7%，面积和功耗节省了约 15%。

表 3：不同累加器的性能比较

<table><tr><td>配置</td><td>面积（$\mu m^{2}$）</td><td>功耗（$\mu W/MHz$）</td><td>延迟（ns）</td><td>精度</td></tr><tr><td>E5M6 RNE</td><td>320.81</td><td>0.282</td><td>5</td><td>低</td></tr><tr><td>E5M6 Variable-SR</td><td>390.94</td><td>0.337</td><td>5</td><td>高</td></tr><tr><td>E5M6 Eager-SR[10]</td><td>401.16</td><td>0.342</td><td>5</td><td>中</td></tr><tr><td>E5M6 常规 SR</td><td>418.52</td><td>0.375</td><td>5</td><td>高</td></tr><tr><td>E5M10 RNE</td><td>463.64</td><td>0.396</td><td>6</td><td>高</td></tr></table>

表 4：ResNet-20 在 CIFAR-100 上的训练准确率

<table><tr><td>精度</td><td>舍入模式</td><td>SR 位宽</td><td>准确率（%）</td></tr><tr><td>E8M23</td><td>RNE</td><td>-</td><td>75.05</td></tr><tr><td>E8M10</td><td>RNE</td><td>-</td><td>74.03</td></tr><tr><td>E8M6</td><td>RNE</td><td>-</td><td>72.88</td></tr><tr><td>E8M6</td><td>SR</td><td>16</td><td>74.39</td></tr><tr><td>E8M6</td><td>SR</td><td>12</td><td>74.41</td></tr><tr><td>E8M6</td><td>SR</td><td>8</td><td>74.23</td></tr><tr><td>E8M6</td><td>SR</td><td>4</td><td>73.94</td></tr></table>

## 4.2 网络训练的实际结果

由于第 4.1 节仅展示了 SR 在理想实验条件下的优势，本节将通过实际网络训练的结果进一步证明这些优势。

我们将指数统一为 8 位，并选择 23/10/6 位尾数，以展示使用 SR 降低累加精度的可行性。我们使用基于 Pytorch 和 CUDA 的低精度累加模拟器，在 ResNet-20 模型和 CIFAR-100 数据集上完成实验。所有实验均使用随机梯度下降优化器，初始学习率设为 0.1，采用梯度缩放和梯度裁剪等技术，动量参数为 0.9。模型以 128 的批量大小训练了 200 个轮次。

训练结果如表 4 所示。在 RNE 模式下，随着尾数位数逐渐减少，网络准确率显著下降。准确率最低的 E8M6 RNE 相比基准 FP32 损失了 2.17% 的准确率。相比之下，使用 8/12/16 位尾数的 SR 在准确率上优于 RNE 模式。准确率最高的 E8M6 SR12 比 M10 RNE 高 0.38%，仅比基准低 0.64%，这表明与 RNE 相比，SR 能够在更低的尾数位宽下保持训练准确率，使 M6 SR 累加器优于 M10 RNE 累加器。文献 [10] 表明，指数更小时，SR 的优势会更加明显。然而，这需要更先进的训练策略，并且不易在芯片上实现，超出了本文的研究范围。

## 4.3 高精度仿真的准确性

我们在 MPICC MAC 单元上使用不同大小的随机矩阵执行通用矩阵乘法（GEMM），以比较高精度仿真对均方误差（MSE）的影响，如图 8 所示。矩阵大小的增加使计算结果变大，从而增大了 MSE。使用 FP8 进行计算的 MSE 是使用 FP12 × FP8 的 3 倍，是使用 FP12 进行计算的 22 至 135 倍，这表明在低精度硬件中进行高精度仿真能够显著减少计算误差。与 [12] 中的直接精度分解方案（不使用舍入或次正规数）相比，我们的方案能够仿真范围更广的尾数和指数，从而大幅降低 MSE。

![](images/657adbec54fae02db609052b8706860ad54e45e602ca59f0341e8b88ad019de0.jpg)  
图 8：不同大小的 GEMM 计算的 MSE

表 5：与已有工作的性能比较

<table><tr><td>设计</td><td colspan="2">TVLSI&#x27;21 [16]</td><td colspan="2">GLSVLSI&#x27;23 [6]</td><td colspan="2">ITETC&#x27;24 [7]</td><td colspan="2">本工作</td></tr><tr><td>结构</td><td colspan="2">向量 FMA</td><td colspan="2">n 项 MULT</td><td colspan="2">向量 MAC</td><td colspan="2">n 项 MAC</td></tr><tr><td>工艺</td><td colspan="2">22nm</td><td colspan="2">28nm</td><td colspan="2">12nm</td><td colspan="2">28nm</td></tr><tr><td>电压</td><td colspan="2">0.8V</td><td colspan="2">-</td><td colspan="2">0.8V</td><td colspan="2">0.9V</td></tr><tr><td>频率</td><td colspan="2">0.92GHz</td><td colspan="2">2.5GHz</td><td colspan="2">1.26GHz</td><td colspan="2">1.43GHz</td></tr><tr><td>面积</td><td colspan="2"> $49000\mu m^2$ </td><td colspan="2"> $9875\mu m^2$ </td><td colspan="2"> $19000\mu m^2$ </td><td colspan="2"> $2295\mu m^2$ </td></tr><tr><td></td><td>FP8</td><td>1245</td><td>FP8</td><td>206.6</td><td>FP8</td><td>1631</td><td>FP8</td><td>1948</td></tr><tr><td rowspan="3"> $GFLOPS^\alpha /W$ </td><td>FP16</td><td>497.6</td><td>FP16</td><td>206.6</td><td>FP16</td><td>815.5</td><td>FP12</td><td>865.1</td></tr><tr><td>BF16</td><td>581.9</td><td>INT8</td><td>413.2</td><td>-</td><td>-</td><td>FP4</td><td>5931</td></tr><tr><td>FP8</td><td>302.7</td><td>FP8</td><td>1013</td><td>FP8</td><td>1063</td><td>FP8</td><td>1246</td></tr><tr><td rowspan="2"> $GFLOPS/mm^2$ </td><td>FP16</td><td>151.4</td><td>FP16</td><td>1013</td><td>FP16</td><td>531.5</td><td>FP12</td><td>415.4</td></tr><tr><td>BF16</td><td>151.4</td><td>INT8</td><td>2025</td><td>-</td><td>-</td><td>FP4</td><td>4985</td></tr></table>

<sup>α</sup> 1 MULT = 1 FLOP，1 FMA/MAC = 2 FLOP，1 n 项 MAC = 2n FLOP。

## 4.4 性能比较

我们从功能和性能方面，将 MPICC MAC 单元与最新的多精度 PE 进行了比较，如图 9 和表 5 所示。在精度支持方面，MPICC MAC 单元支持 18 种不同的精度组合，远超其他设计，并允许采用更多量化策略。在性能方面，与 SOTA 设计[7]相比，MPICC MAC 单元的面积效率和能效在 FP8 下分别提升了 1.17×/1.19×，在 FP4 下的最佳条件下分别提升了 4.69×/3.64×。这一性能提升主要有三个原因：

1) 针对性设计：我们的设计面向超低精度训练，去除了对 [6, 7, 16] 中 FP16 和 FP32 等冗余较高精度的支持，显著缩减了数据通路宽度。

2) 集成式多精度支持：与 [7, 16] 通过组合多种单精度 PE 来支持多精度不同，我们的设计在单个 MAC 单元上支持多精度，最大限度地实现了不同精度之间的资源复用。

3) MPICC：我们的设计支持不同精度格式之间的直接计算，而 [7, 16] 在计算前使用精度转换电路来统一精度，引入了额外的时序和逻辑资源开销。

## 5 结论

为最大限度地发挥超低精度训练算法带来的硬件性能提升，本文提出了一种面向超低精度训练的 MPICC MAC 单元，支持多种浮点、定点和对数格式之间的相互运算。我们引入了 Variable-SR 策略，在保持准确性的同时减少累加器位宽，并设计了高精度仿真控制器，以低成本支持高精度计算。实验表明，在 ResNet-20 训练中，采用 6 位尾数精度的 SR 比采用 10 位尾数的 RNE 能够获得更高的准确率。基于这一发现，我们使用 Variable-SR 策略来减少累加器位宽，节省了累加器 15.6% 的面积和 14.9% 的功耗。与 SOTA 设计相比，MPICC MAC 单元支持的精度种类最为丰富，并且在 FP8 下面积效率/能效提升了 1.17×/1.19×，在 FP4 下提升了 4.69×/3.64×。

![](images/2c7293b3b084f95a075cd1c5f95718c19eb19227e47b86b7619186a762c30ee1.jpg)  
图 9：各 PE 的面积效率/能效比较

## REFERENCES

[1] Brian Chmiel et al. 2023. Accurate Neural Training with 4-bit Matrix Multiplications at Standard Formats. In The Eleventh International Conference on Learning Representations.

[2] Bita Darvish Rouhani et al. 2023. Microscaling Data Formats for Deep Learning. arXiv:2310.10537

[3] Cong Guo et al. 2022. ANT: Exploiting Adaptive Numerical Data Type for Lowbit Deep Neural Network Quantization. In 2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO). 1414–1433.

[4] Hardik Sharma et al. 2018. Bit Fusion: Bit-Level Dynamically Composable Architecture for Accelerating Deep Neural Network. In 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA). 764–775.

[5] Jongwook Sohn et al. 2016. A Fused Floating-Point Four-Term Dot Product Unit. IEEE Transactions on Circuits and Systems I: Regular Papers 63, 3 (2016), 370–378.

[6] Jing Zhang et al. 2023. Low-Cost Multiple-Precision Multiplication Unit Design For Deep Learning. In Proceedings of the Great Lakes Symposium on VLSI 2023. 9–14.

[7] Luca Bertaccini et al. 2024. MiniFloats on RISC-V Cores: ISA Extensions with Mixed-Precision Short Dot Products. IEEE Transactions on Emerging Topics in Computing (2024), 1–16.

[8] Matteo Croci et al. 2022. Stochastic rounding: implementation, error analysis and applications. Royal Society Open Science 9 (2022).

[9] Naigang Wang et al. 2018. Training Deep Neural Networks with 8-bit Floating Point Numbers. In Advances in Neural Information Processing Systems, Vol. 31.

[10] Sami Ben Ali et al. 2024. A Stochastic Rounding-Enabled Low-Precision Floating-Point MAC for DNN Training. In 2024 Design, Automation Test in Europe Conference Exhibition (DATE). 1–6.

[11] Suyog Gupta et al. 2015. Deep learning with limited numerical precision. In Proceedings ofthe 32nd International Conference on International Conference on Machine Learning, Vol. 37. 1737–1746.

[12] Stefano Markidis et al. 2018. NVIDIA Tensor Core Programmability, Performance Precision. In 2018 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW). 522–531.

[13] Xiao Sun et al. 2020. Ultra-Low Precision 4-bit Training of Deep Neural Networks. In Advances in Neural Information Processing Systems, Vol. 33. Curran Associates, Inc., 1796–1807.

[14] Yang Wang et al. 2024. 34.1 A 28nm 83.23TFLOPS/W POSIT-Based Compute-in-Memory Macro for High-Accuracy AI Applications. In 2024 IEEE International Solid-State Circuits Conference (ISSCC), Vol. 67. 566–568.

[15] Zixuan Ma et al. 2022. Eficiently emulating high-bitwidth computation with low-bitwidth hardware. In In Proceedings ofthe 36th ACM International Conference on Supercomputing (ICS’22). 12 pages.

[16] Zürich Switzerland et al. Stefan Mach Integrated Systems Laboratory, ETH Zürich. 2021. FPnew: An Open-Source Multiformat Floating-Point Unit Architecture for Energy-Proportional Transprecision Computing. IEEE Transactions on Very Large Scale Integration (VLSI) Systems 29, 4 (2021), 774–787.

> 来源与版本：本页仅翻译 MPICC_paper.pdf。按用户要求，MPICC_ppt.pdf 不解析、不翻译、不发布。FP8_H/FP8_L 等字体乱码按论文 PDF 修复。

> 本文根据用户提供的 PDF 经 MinerU 解析，由 Codex/GPT 翻译和解读，未调用 GLM 翻译 API。参考文献保留原文；技术总结、通俗讲解和图表说明属于解读。[原文入口](https://doi.org/10.1145/3658617.3697641)。
