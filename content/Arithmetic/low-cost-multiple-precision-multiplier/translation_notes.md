# Low-Cost Multiple-Precision Multiplication Unit Design For Deep Learning 原文翻译

# 面向深度学习的低成本多精度乘法单元设计

Jing Zhang 国防科技大学 中国长沙 zhangjing@nudt.edu.cn

Ling Yang 国防科技大学 中国长沙 yanglingnudt@nudt.edu.cn

Libo Huang 国防科技大学 中国长沙 libohuang@nudt.edu.cn

Zhong Zheng<sup>∗</sup> 国防科技大学 中国长沙 zheng\_zhong@nudt.edu.cn

Hongbing Tan 国防科技大学 中国长沙 tanhongbing@nudt.edu.cn

Qianming Yang 国防科技大学 中国长沙 yqm21249@nudt.edu.cn

## 摘要

低精度格式已被提出并应用于深度学习算法，以加速训练和推理。本文提出了一种面向深度学习的新型多精度乘法单元（MU）。所提出的 MU 支持四种精度的浮点（FP）数——FP8-E4M3、FP8-E5M2、FP16、FP32——以及 8 位定点（FIX）数。该 MU 可以在一个周期内同时执行四路并行 FP8 乘法和八路并行 FIX8 乘法，或以全流水线方式执行四路并行 FP16 乘法，延迟为 1，或执行一次 FP32 乘法，延迟为一个周期。同时执行 FIX8 和 FP8 运算可以满足特定深度学习算法的需求。得益于低精度组合（LPC）和向量化设计方法，任意精度的乘法均可实现 100% 的乘法器资源利用率，并且该 MU 可以采用更低的时钟延迟，从而在所有数据类型下实现更好的性能。与现有面向深度学习设计的多精度单元相比，该 MU 能够以更低的面积开销支持更多种低精度格式；并且在 FIX8 下表现出更高的吞吐量，提升至少达到 8×。

## CCS 概念

• 硬件 → 硬件加速器；可编程逻辑元件。

## 关键词

低精度，多精度，乘法单元，深度学习，LPC

ACM 参考格式：Jing Zhang, Libo Huang, Hongbing Tan, Ling Yang, Zhong Zheng, and Qianming Yang. 2023. Low-Cost Multiple-Precision Multiplication Unit Design For Deep Learning. In Proceedings ofthe Great Lakes Symposium on VLSI 2023 (GLSVLSI ’23), June 5–7, 2023, Knoxville, TN, USA. Knoxville, TN, USA, June 5–7, 6 pages. https://doi.org/10.1145/3583781.3590269

## 1 引言

深度学习已在许多领域取得巨大成功[1–3]，是当前最热门的研究领域之一。深度学习的训练和推理过程涉及海量计算，而基于 FP32 数据类型的传统计算存在延迟较长等问题。近期研究发现，使用低精度数据进行计算可以保持深度学习算法的准确率，同时显著提升性能[4–6]。

此外，各种采用低精度数据类型的深度学习加速器也已被提出。Venkataramani 等人设计了一种名为 RapiD 的 AI 加速器，支持 FP16、FP8 数以及 FIX4、FIX8 数，并通过低精度方式提高 AI 负载的能效比[7]。Wang 等人设计了一种灵活的混合低精度 CNN 推理硬件架构，在 Xilinx Virtex UltraScale+13P FPGA 上可实现 58.9 TOPS 的峰值性能[8]。因此，面向通用应用设计一种支持低精度数据的有效硬件架构，对于推动深度学习的发展具有重要意义。

近年来，许多支持低精度格式的硬件设计已被提出。Li Kai 等人提出了一种面向高性能计算的多精度 FP 处理单元，其乘法硬件利用率达到 100%[9]。Zhang Hao 等人提出了一种支持 FP16 和 FP32 的高效多精度 FP 融合乘加（FMA）单元，与常规设计相比，为支持混合精度运算增加了 6.5% 的面积开销[10]。然而，这两种设计由于支持深度学习中很少使用的 FP64，面积成本较高。文献 [11] 为深度学习处理器设计了一种高效的 FIX/FP 合并 FMA 单元。该单元支持 FP16、FP32 和 FIX8 运算。在文献 [12] 中，作者提出了一种支持多种模式的 FMA，支持 FP16 和 BF16。在文献 [13] 中，研究者设计了一种面向深度学习的 FP32 与 FP16 组合乘法器，可以方便地配置为在 FP16 模式下运行以节省功耗，或在 FP32 模式下运行以保证准确性。然而，这些设计不支持 FP8 和 FIX8 等更低精度格式。

本文提出了一种面向深度学习的新型多精度乘法单元，支持包括 FP8 和 FIX8 在内的低精度数据。本文的主要贡献如下：

• 通过 LPC 方法设计了一种硬件开销较低的多精度乘法架构。该 MU 在所有精度模式下均可实现 100% 的乘法器利用率。

• 我们提出了一种混合乘法器设计，可以通过相同的计算路径执行两次 4x8 乘法或一次 8x8 乘法。

本文其余部分组织如下。第 II 节介绍我们设计的背景。第 III 节阐述整体架构。第 IV 节给出评估和比较。第 V 节总结全文。

## 2 背景

IEEE-754 标准中定义的 FP 数由三部分组成：符号位（S）、偏置指数（E）和尾数的尾随有效数字字段（M）[14]。该标准为 FP 数定义了四种精度：FP16、FP32、FP64 和 FP128。一个 FP 数 F 可以表示如下：

$$
F = (- 1) ^ {S} \times (1 + 0. M) ^ {(E - b i a s)}
$$

其中，bias 是为使偏置指数非负而选取的常数。

文献 [15] 中定义的两种 FP8 数据格式 E3M4 和 E5M2 符合 IEEE-754 标准。所有受支持数据类型的更多细节见图 1。

![](images/17429f571d2f340948279a13f072362b636fb8dcb73455ea36d83ab4f65bd7af.jpg)  
图 1：所提出的 MU 支持的数据格式。

IEEE-754 标准还定义了四种舍入方向——向最近值舍入（RNT）、向零舍入、向正无穷舍入和向负无穷舍入。其中，RNT 模式下的表示与真实值之间的误差最小，因此被应用广泛采用。该 MU 中设计的所有 FP 乘法均采用 RNT 模式。

与 FIX 乘法不同，两个 FP 数的乘法涉及指数求和、尾数相乘、归一化、舍入等步骤，因此 FP 乘法的硬件实现比 FIX 乘法更复杂，也更消耗面积。两个 FP 数相乘的详细描述见算法 1，其中 sticky 用于保留尾数乘积被截断部分的信息，并用于正确的 RTN 舍入。

```python
算法 1 两个浮点数相乘
要求：
    两个 n 位 FP 数：
    {S_0,E_0,M_0}, {S_1,E_1,M_1}
保证：
    1: // 计算结果的符号位。
    2: o_sig = S_0 ⊕ S_1
    3: // N 表示一个 FP 数的位宽。
    4: // M 表示尾数的位宽。
    5: // 计算 E 的和与 M 的乘积。
    6: exp_sum[n-m-1:0] = E_0 + E-1 - bias
    7: man_mul[2m-1:0] = {1'b1,M_0} × {1'b1,M_1}
    8: // 归一化和舍入。
    9: norm_man[m-2:0] = man_mul[2m-1] ? man_mul[2m-2:m] : man_mul[2m-3:m-1]
    10: rd = man_mul[2m-1] ? man_mul[m-1] : man_mul[m-2]
    11: sticky = |man_mul[m-3:0] | (man_mul[m-2] & man_mul[2m-1])
    12: rd_man[m-1:0] = rd(norm_man[m-2:0], rd, sticky)
    13: o_man[m-2:0] = rd_man[m-2:0]
    14: o_exp[n-m-1:0] = exp_sum[n-m-1:0] + man_mul[47] + rd_man[23]
    15: o_mul_rslt[n-1:0] = {o_sig, o_exp, o_man}
    16: // 输出结果。
    return o_mul_rslt
```

## 3 总体架构

图 2 展示了本文提出的 MU 的整体数据通路。该架构由左右两部分组成。所提出的 MU 具有两级流水线。首先，输入处理模块根据不同的模式控制信号对输入信号进行处理。

左侧部分的输入包括：两组 8 位 FP8 源操作数 fp\_src0 和 fp\_src1，均为 32 位；1 位 FP8 精度选择信号 fp\_mode，用于控制 E3M4 乘法或 E5M2 乘法；1 位模式选择信号 mode。Mul\_fp 模块由四个 4x4 无符号乘法器组成，负责 FP8 的尾数乘法；Exp\_add 模块由一个 9 位加法器和三个 6 位加法器组成，前者负责所有精度的指数求和。Nm\_rd\_Fp8 模块负责 FP8 数的规格化和舍入过程。

右侧部分的输入包括：两组 64 位操作数 src0 和 src1；用于模式选择的信号 mode；用于控制 FIX8 数执行有符号运算还是无符号运算的信号 sign；浮点精度选择信号 $f p _ { - } p r c ,$，在 mode = 1 时有效。Mul\_int 模块由四个 8x8 乘法器和四个 4x8/8x8 混合乘法器组成。Csa\_fp 模块用于计算 FP16 或 FP32 的尾数乘积。Nm\_rd 模块用于对 FP16 和 FP32 数进行规格化和舍入。选择器选择 FP16 和 FP32 的计算结果，以得到最终结果。

![](images/106677d82703a511849e98029ef9f3e922688ca23f26f4264e3189ea452aed58.jpg)  
图 2：所提出的乘法器单元的数据通路。

## 3.1 FP8 模式

当 mode = 1 时，MU 执行 FP8 乘法。根据 fp\_mode 信号，输入处理阶段生成两组 20 位指数向量和两组 16 位尾数向量，向量长度均为 4。两种 FP8 数据格式在指数向量和尾数向量中的分布如图 3 所示。在 E4M3 模式下，4x4 乘法器的 8 位输出用作尾数乘积，加法器输出的 6 位阶码的高 5 位用作指数之和；而在 E5M2 模式下，乘法器输出的低 6 位用作乘积，加法器的全部 6 位输出用作指数之和。

![](images/be2060937e8b878964f4337017caeb17c5189bbbdc73f776dedde107be072025.jpg)  
图 3：E4M3 和 E5M2 的指数在乘法器中的排列及尾数在加法器中的排列。

E4M3 的尾数宽度为 4 位，因此使用一个 4x4 无符号乘法器来完成尾数部分的乘法。4x4 无符号乘法器采用改进的 Booth-4 编码[16]实现。E5M2 的指数宽度为 5，因此使用一个 6 位加法器来计算阶码之和。额外的一位用于存储进位。

由于浮点数的指数带有偏置，因此在将两个带偏置的指数相加以得到其和时，必须减去偏置值。E5M2 的指数偏置值为 15，其 5 位宽二进制补码为 1\_0001；FP-E4M2 的偏置值为 7，其 5 位宽二进制补码为 1\_1001。因此，进行求和时，首先根据信号 fp\_mpde 选择正确的偏置值；然后通过一个 5 位 CSA3:2，将两个带偏置的指数和负偏置值压缩为两个加数；最后通过一个 6 位加法器计算其和。该过程如图 4 所示。得到乘积与和之后，将按照算法 1 所述执行规格化和舍入过程。

在得到尾数乘积 mul\_rslt 和指数之和 sum\_exp 后，在同一周期内进行规格化。以 E4M3 模式为例，根据尾数乘积的最高位（mul\_rslt[7]）生成尾数乘法的进位，并在相同条件下更新指数之和。当 mul\_rslt[7] = 1 时，规格化后的尾数为 mul\_rslt[6 : 4]。同时，mul\_rslt[4] 和 mul\_rslt[3] 分别为保护位和舍入位，对 mul\_rslt[2 : 0] 进行或运算的结果为粘滞位。当 $mul\_rslt[7] = 0$ 时，这些位分别取上述情况中对应位右侧一位的值。舍入过程在规格化之后立即执行，是浮点乘法的最后一步。根据保护位、舍入位和粘滞位生成舍入进位，具体可见算法 1。当进位等于 1，且规格化后尾数的所有位均等于 1 时，将产生一个非规格化数。在这种情况下，需要再进行一步规格化才能得到最终结果。

![](images/8ed261a7df97c5d49d3705a7fd6b27372b7bc54e6252786a01ddda5bd88c723f.jpg)  
图 4：FP8 指数求和的数据通路。

## 3.2 FIX8 模式

FIX8 乘法可以在与 FP8 相同的条件下执行，即 mode = 1，并在信号 sign 的控制下选择有符号（sign = 1）或无符号（sign = 0）模式。负责 FIX8 计算的 Mul\_int 模块需要同时支持有符号乘法和无符号乘法，其中包含 4 个 mul\_8x8 普通乘法器和 4 个 mul\_2x4x8 混合乘法器。

全部八个 8x8 乘法均由这 4 个 mul\_8x8 和 4 个 mul\_2x4x8 乘法器计算，其中，每个乘法器的部分积都采用改进的 Booth-4 编码，然后通过压缩树和全加器求和。符号位采用文献[17]中提到的方法进行扩展。图 5 展示了 mul\_8x8 乘法器符号位扩展方法的细节。图中的黑点表示部分积的各个位，S 表示乘数编码的符号位，即 $b _ { 2 k + 1 }$。E 是有符号编码中的一个特殊位。当乘数编码的符号位等于被乘数的符号位，或者乘数编码全为 0 时，E 等于 0，否则 E = 0。混合乘法器 mul\_2x4x8 将在第 3.4 节中介绍。

当 mode = 0 时，FP8 和 FIX8 乘法可以同时且独立地执行，此时乘法器利用率为 100%。在 FP8 模式下，如果没有 FIX8 计算需求，可以对 FIX8 数据通路进行门控以降低功耗；同样，在 FIX8 模式下，也可以对 FP8 数据通路进行门控。

![](images/5587c8934a334bd67bbb49224ed26dc8da539dc6160a3ec6299c2d99bf58ca3d.jpg)  
图 5：mul\_8x8 中的符号位扩展方式。

## 3.3 FP16 和 FP32 模式

当 mode = 1 且 $f p _ { - } p r c = 0 ,$ 时，在两个周期内执行四路并行 FP16 乘法，src0 和 src1 用作八个 FP16 数的输入。该模式需要同时使用 FP8 数据通路和 FIX8 数据通路，以在两个周期内完成四路并行 FP16 乘法。

每两个 FP16 数的 11 位尾数乘法需要一个 mul\_4x4、一个 mul\_2x4x8 和一个 mul\_8x8 来计算各个部分，如图 6 所示。图的顶部展示了 11 位尾数在 12 位操作数中的排列。Mul\_4x4 计算 P3 部分，mul\_2x4x8 计算 P1 和 P2 部分，mul\_8x8 计算 P0 部分。然后将所有短乘积组合并相加，得到 11 位无符号乘法的乘积，如图右侧所示。由于 FP16 的指数和偏置值的宽度与 E5M2 相同，因此 FP16 的指数求和可以通过现有的 FP8 6 位加法器完成。规格化和舍入过程与 FP8 的方式相同。

与 FP16 运算过程类似，当 mode = 1 且 fp\_prc = 1 时，执行一次 FP32 乘法。24x24 尾数乘法被拆分为四个部分，然后由所有乘法器计算各部分，再组合成最终结果，如图 6 所示。

如上所述，FP16 和 FP32 模式都使用了所有乘法器，因此在这些精度模式下，乘法器利用率为 100%。

## 3.4 4x8/8x8 混合乘法器

4x8/8x8 混合乘法器是整个架构的关键。它有两种工作模式：用于 FP16 和 FP32 精度的两次 4x8 乘法运算，以及用于 FIX8 精度的一次 8x8 乘法运算。该乘法器有六个输入：两个 8 位乘数 a 和 b、两个 4 位乘数 c 和 d、一个用于控制工作模式的模式选择信号 mode，以及一个用于控制在 FIX8 模式下执行有符号还是无符号乘法的符号位信号。该乘法器有两个编码器和两组压缩树。

![](images/8bb8fea70bf4dc9985b12555a659403205256a492a0f3a3605cc3cb8340e3322.jpg)  
图 6：FP16 和 FP32 乘法器的组织结构。

表 1：最小时延设计下各模块的面积和功耗（delay = 0.4ns，total\_area = 9874.87μm<sup>2</sup>，total\_power = 48.4mW）

<table><tr><td>模块</td><td>面积( $um^2$ )</td><td>功耗(mW)</td></tr><tr><td>Mul_int</td><td>3660.93</td><td>14.9</td></tr><tr><td>Csa_fp</td><td>1831.41</td><td>12.2</td></tr><tr><td>Mul_fp</td><td>445.03</td><td>1.6</td></tr><tr><td>Deal_int</td><td>417.82</td><td>1.0</td></tr><tr><td>Deal_fp</td><td>45.49</td><td>0.1</td></tr><tr><td>Nm_rd_fp8</td><td>227.30</td><td>1.3</td></tr><tr><td>Nm_rd_fp16</td><td>186.84</td><td>1.1</td></tr><tr><td>Nm_rd_fp32</td><td>292.32</td><td>1.6</td></tr><tr><td>Exp_add</td><td>121.97</td><td>0.4</td></tr><tr><td>其他</td><td>2645.76</td><td>14.2</td></tr></table>

执行两次 4x8 乘法时，两个编码器各自生成独立的部分积，然后分别使用 CSA3:2 和 CPA 得到乘法结果。执行 8x8 乘法时，将乘数 b 分成两个等长部分，再由两个编码器生成部分积。编码及符号位扩展方式如图 7 所示。注意，在 FIX8 模式下，高位编码器的最低位需要使用低位编码器输入中的最高位来生成第一个部分积，即图中的绿色部分；低位编码器则使用乘数的有符号或无符号扩展位来生成最后一个部分积，即图 7 中的橙色部分。

图 8 展示了混合乘法器的详细数据通路。可以看出，所有部分积的生成过程和部分压缩过程共享相同的硬件资源。

![](images/91127eed10eaf563dba19ad9424216ed568cce9deddb75cc5e97cacba546bcbf.jpg)  
图 7：mul\_2x4x8 乘法器的符号位扩展。

![](images/7b95230fc4e006ae671f6e01ccada539709791e41fb0a9af61b5019ffc641182.jpg)  
图 8：mul\_2x4x8 混合乘法器的数据通路。

## 4 评估与比较

## 4.1 方法

所提出的 MU 使用 Verilog HDL 实现，并在 28-nm CMOS 工艺的典型工艺角下进行评估。

## 4.2 评估

我们在不同的时延要求下对所提出的 MU 进行了综合，面积和功耗如图 9 所示。随着时钟周期缩短，面积和功耗随之增加。为实现最佳吞吐量，本文选择最小时延 0.4ns 作为设计点，此时面积为 9874.87μm<sup>2</sup>，功耗为 48.4mW。

表 1 给出了 MU 在最小时延下各模块的详细面积和功耗。由表可见，在所有模块中，Mul\_int 模块的面积和功耗占比最大，分别为 37.1% 和 30.8%。该模块包含四个 8x8 普通乘法器和四个 4x8/8x8 混合乘法器，所有这些乘法器都可以执行有符号或无符号乘法。

## 4.3 比较

我们选择了几种支持低精度格式的现有设计进行比较。这些设计与我们所提出的 MU 的制造工艺、频率、面积、功耗和吞吐量如表 2 所示。

表 2：与以往工作的性能比较。

<table><tr><td rowspan="2">设计</td><td rowspan="2">工艺节点(nm)</td><td rowspan="2">频率(GHz)</td><td rowspan="2">面积( $um^2$ )</td><td rowspan="2">功耗(mW)</td><td colspan="4">吞吐量(GOPS)</td></tr><tr><td>FP8</td><td>FP16</td><td>FP32</td><td>FIX8</td></tr><tr><td>[9]-TCAS-II&#x27;22</td><td>28</td><td>1.35</td><td>15000</td><td>38.0</td><td>-</td><td>43.2</td><td>10.8</td><td>-</td></tr><tr><td>[10]-TC&#x27;19</td><td>90</td><td>0.50</td><td>794790</td><td>177.0</td><td>-</td><td>4.00</td><td>2.0</td><td>-</td></tr><tr><td>[11]-ISCAS&#x27;18</td><td>90</td><td>1.13</td><td>42711</td><td>14.1</td><td>-</td><td>1.25</td><td>-</td><td>2.5</td></tr><tr><td>[12]-TVLSI&#x27;23</td><td>28</td><td>2.22</td><td>13032</td><td>59.3</td><td>-</td><td>4.4</td><td>2.2</td><td>-</td></tr><tr><td>本文设计</td><td>28</td><td>2.50</td><td>9875</td><td>48.4</td><td>10</td><td>10</td><td>2.5</td><td>20</td></tr></table>

![](images/19bc39272abb955f9676e0ee65e66ed97cf32a14dd14466b4f2b8cd432b72a7d.jpg)  
图 9：所提出的 MU 在不同时延下的面积–时延和功耗–时延折线图。

可以看出，所提出的 MU 的硬件开销最小，至少降低了 24.2%；频率最高，至少提高了 12.6%。此外，除 [9] 外，所提出的 MU 在所有支持的数据类型上的吞吐量均超过了所列出的其他工作。文献 [9] 提出了一种用于多精度 FP 计算的向量加速器。因此，以更大的面积和更低的频率为代价，其 FP16 和 FP32 吞吐量高于本文提出的设计。

## 5 结论

本文提出了一种新型乘法单元。该 MU 支持 FP8、FIX8 和 FP16 等低精度数据，以满足深度学习不断涌现的计算需求。该 MU 还支持 FP32，以进行常规精度计算。通过低精度组合方法的设计，可以采用高频率来实现更好的性能，且所提出的 MU 在所有精度模式下均可实现 100% 的乘法器利用率。本文详细讨论了整体数据通路和关键设计。

所提出的 MU 使用 Verilog HDL 实现，并进行了正确性测试。我们展示了不同的时延要求下的综合结果，还给出了最小时延下各模块的面积和功耗。与以往工作相比，所提出的 MU 在用于深度学习的 FIX8 乘法中，以更低的面积开销实现了更好的性能。所提出的设计可为嵌入式设备或大规模深度学习加速器的算术单元提供参考。

## 致谢

本工作得到了 NSFC（资助编号：62272475 和 62090023）、HNNSFC（资助编号：2022JJ10064）以及 NDKL（资助编号：WDZC20215250111）的支持。

## REFERENCES

[1] Alex Krizhevsky, Ilya Sutskever, and Geofrey E. Hinton. Imagenet classification with deep convolutional neural networks. Communications ofthe ACM, 60:84 – 90, 2012.

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2015.

[3] Yann LeCun, Yoshua Bengio, and Geofrey Hinton. Deep learning. Nature, 521:436–444, 2015.

[4] Dhiraj D. Kalamkar, Dheevatsa Mudigere, and Naveen Mellempudi et al. A study of bfloat16 for deep learning training. ArXiv, abs/1905.12322, 2019.

[5] Feng Zhu, Ruihao Gong, and Fengwei Yu et al. Towards unified int8 training for convolutional neural network. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1966–1976, 2019.

[6] Sae Kyu Lee, Ankur Agrawal, and Joel Silberman et al. A 7-nm four-core mixedprecision ai chip with 26.2-tflops hybrid-fp8 training, 104.9-tops int4 inference, and workload-aware throttling. IEEE Journal ofSolid-State Circuits, 57(1):182–197, 2022.

[7] Swagath Venkataramani, Vijayalakshmi Srinivasan, and W. Wang et al. Rapid: Ai accelerator for ultra-low precision training and inference. 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pages 153–166, 2021.

[8] Junbin Wang, Shaoxia Fang, and Xi Wang et al. High-performance mixed-low precision cnn inference accelerator on fpga. IEEE Micro, 41:31–38, 2021.

[9] Kai Li, Wei Mao, and Junzhuo Zhou et al. A vector systolic accelerator for multi-precision floating-point high-performance computing. IEEE Transactions on Circuits and Systems II: Express Briefs, 69:4123–4127, 2022.

[10] Hao Zhang, Dongdong Chen, and Seok-Bum Ko. Eficient multiple-precision floating-point fused multiply-add with mixed-precision support. IEEE Transactions on Computers, 68:1035–1048, 2019.

[11] H. Zhang, Hyuk-Jae Lee, and Seok-Bum Ko. Eficient fixed/floating-point merged mixed-precision multiply-accumulate unit for deep learning processors. 2018 IEEE International Symposium on Circuits and Systems (ISCAS), pages 1–5, 2018.

[12] Hongbing Tan, Gan Tong, and Libo Huang et al. Multiple-mode-supporting floating-point fma unit for deep learning processors. IEEE Transactions on Very Large Scale Integration (VLSI) Systems, 31(2):253–266, 2023.

[13] Tuan D. Nguyen and James E. Stine. A combined ieee half and single precision floating point multipliers for deep learning. 2017 51st Asilomar Conference on Signals, Systems, and Computers, pages 1038–1042, 2017.

[14] Dan Zuras, Michael F. Cowlishaw, and Alex Aiken et al. Ieee standard for floatingpoint arithmetic. 2008.

[15] Paulius Micikevicius, Dusan Stosic, and Neil Burgess et al. Fp8 formats for deep learning. ArXiv, abs/2209.05433, 2022.

[16] Andrew Donald Booth. A signed binary multiplication technique. Quarterly Journal ofMechanics and Applied Mathematics, 4:236–240, 1951

[17] Gary Bewick. Fast multiplication: Algorithms and implementations. 1994.

> 来源与版本：字体乱码变量、模式信号及面积/功耗单位已按原 PDF 恢复。原文中 FP8 格式命名、mode 控制值及延迟措辞有不一致，译文保留原文，解读中另行指出，不自行改写硬件合同。

> 本文根据用户提供的 PDF 经 MinerU 解析，由 Codex/GPT 翻译和解读，未调用 GLM 翻译 API。参考文献保留原文；技术总结、通俗讲解和图表说明属于解读。[原文入口](https://doi.org/10.1145/3583781.3590269)。
