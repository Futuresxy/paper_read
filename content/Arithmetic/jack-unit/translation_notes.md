# Jack Unit: An Area- and Energy-Efficient Multiply-Accumulate (MAC) Unit Supporting Diverse Data Formats 原文翻译

![](images/4ec4144dcd4a157ddc40b1b7895db3a7909df81624227e8f293ba49b146c0806.jpg)

# Jack Unit：一种支持多种数据格式、兼具面积效率与能量效率的乘加（MAC）单元

Seock-Hwan Noh<sup>∗</sup>, Sungju Kim<sup>†</sup>, Seohyun Kim<sup>∗</sup>, Daehoon Kim<sup>†</sup>, Jaeha Kung<sup>‡</sup> <sup>§</sup> and Yeseong Kim<sup>∗§</sup>

<sup>∗</sup>DGIST，韩国大邱

<sup>†</sup>延世大学，韩国首尔

<sup>‡</sup>高丽大学，韩国首尔

nosh3332@dgist.ac.kr; jhkung@korea.ac.kr; yeseongkim@dgist.ac.kr

摘要——在本工作中，我们提出了一种兼具面积效率与能量效率的乘加（MAC）单元，名为 Jack Unit。它是一个多面手，支持整数（INT）、浮点（FP）和微缩放数据格式（MX）等多种数据格式。它通过以下方式提供位级灵活性并提高硬件效率：i）将 FP 乘法器中的进位保存乘法器（CSM）替换为精度可伸缩的 CSM；ii）在 CSM 内部根据指数差调整有效数；iii）利用二维子字并行。为评估其有效性，我们实现了 Jack 单元和三种基线 MAC 单元的版图。此外，我们设计了一款配备 Jack 单元的 AI 加速器，以便与支持多种数据格式的先进 AI 加速器进行比较。与基线 MAC 单元相比，所提出的 MAC 单元实现了 1.17∼2.01× 的面积缩减和 1.05∼1.84× 的功耗降低。在五个 AI 基准测试中，采用 Jack 单元设计的加速器在多种数据格式下的能效达到基线的 1.32∼5.41×。

## I. 引言

随着 AI 准确率的显著提升，许多应用开始利用 AI 算法执行语言翻译、自动驾驶和语音识别等各种任务。AI 能够实现高准确率，其成功得益于对海量数据的处理。例如，OpenAI 的 GPT-4 和 Meta 的 LLaMA 使用参数量从数百亿到数万亿不等的模型，以高准确率处理多模态查询 [3]、[40]。然而，使用计算机系统中常用的 IEEE 双精度（FP64）或单精度（FP32）格式来处理如此庞大的数据量，会带来过高的硬件成本，尤其是在内存占用和能耗方面。为减轻这一负担，研究人员一直在积极探索针对多种数据格式降低位宽的算法方法，包括整数（INT）[8]、[42]、[43]、浮点（FP）[15]、[16]、[35] 和微缩放（MX）格式 [6]、[25]、[30]。

基于 AI 的应用会根据目标设备和任务目标采用不同的精度与数据类型。具体而言，需要高准确率的应用通常采用高精度，以尽量减少准确率损失；而在智能手表和无人机等小型设备上运行的应用，则可能因功耗和面积限制而采用低精度。此外，由于数据表示的动态范围不同，AI 模型的训练和推理过程对数据格式也有不同的要求。为满足这些多样化需求，许多研究工作和商用 AI 处理器将支持各种数据格式的多个计算单元集成到单个芯片（或芯片组）中 [14]、[25]、[26]、[41]。然而，这种方法会导致较低的面积效率。图 1-(a) 展示了量化 AI 算法中广泛使用的几种数据格式所对应的硬件成本，即面积和功耗延迟积（PDP）。为支持多种数据格式，像图 1-(b) 所示那样配置与所支持精度数量相同的计算单元，会增加乘加（MAC）单元的面积成本。此外，AI 算法通常倾向于使用单一数据格式，这使得 MAC 单元的部分电路处于空闲状态。因此，集成多个计算单元的加速器面积效率较低。为解决这一问题，一些研究工作采用多个加速器来适应 AI 模型的异构性 [10]、[11]，但这会增加硬件和软件栈两方面的开发成本。这是因为必须为异构系统中部署的每个加速器开发设备驱动程序，同时还需要更新系统编译器，以管理多个加速器底层平台 [39]。

<table><tr><td>运算</td><td>面积 [um2]</td><td>PDP [pJ]</td></tr><tr><td>bfloat16 加法</td><td>775</td><td>0.59</td></tr><tr><td>INT16 加法</td><td>141</td><td>0.04</td></tr><tr><td>bfloat16 乘法</td><td>1388</td><td>1.08</td></tr><tr><td>FP8 乘法</td><td>500</td><td>0.31</td></tr><tr><td>INT8 乘法</td><td>471</td><td>0.24</td></tr><tr><td>INT4 乘法</td><td>103</td><td>0.03</td></tr></table>

![](images/1f39b094c9600cee97196dabd4728a1e5b60855327583d3baa477f014d05e87f.jpg)  
(b)  
图 1：（a）各种数据格式下计算单元的硬件成本，即面积和功耗延迟积（PDP）。这些硬件成本基于采用 65 nm CMOS 工艺的 Synopsys DesignWare IP 得出。（b）为每种支持的数据格式部署专用乘法器的加速器中的 MAC 单元结构。

考虑到这些局限性，本文提出了一种兼具面积效率与能量效率的 MAC 单元，名为 Jack 单元，为各种数据格式提供通用解决方案。具体而言，所提出的 MAC 单元通过采用三项关键设计策略，以较低的硬件成本为 INT、FP 和 MX 数据格式（例如 MXINT 和 MXFP）提供位级灵活性（本工作的贡献）：i）将 FP 乘法器中的进位保存乘法器（CSM）替换为精度可伸缩的 CSM；ii）在 CSM 内部调整有效数；iii）利用二维子字并行。用精度可伸缩的 CSM 替换乘法器后，可以在单个集成乘法器内执行各种数据格式的乘法，从而无需为每种支持的格式设置独立乘法器（第 III-A1 节）。在 CSM 内部根据指数差调整有效数，使得乘法结果可以使用 INT 加法器树而非 FP 加法器树进行累加（第 III-A2 节）。此外，二维子字并行通过在 INT 加法器之间共享移位器，并降低加法器树的输入位宽，减小了累加所用加法器树的面积和功耗（第 III-A3 节）。采用这些策略后，与商用 MAC 单元相比，Jack 单元在面积和功耗上分别实现了 2.01× 和 1.84× 的缩减。

为进一步展示 Jack 单元在实际应用中的有效性，我们设计了一款集成 Jack 单元的 AI 加速器，并将其与一款先进的商用 AI 加速器进行比较。我们针对执行不同任务的五种代表性 AI 算法开展了架构评估。与基线加速器相比，采用 Jack 单元设计的加速器实现了 2.02× 的面积缩减，并在多种数据格式下实现了基线 1.32∼5.4× 的能效。

## II. 预备知识

## A. 数据表示

计算机基于双态器件运行，这意味着它们使用二进制数，即二进制数字（比特），作为其主要语言。计算机系统通过对二进制数进行分组，为程序员提供各种数据类型。在这些数据类型中，整数（INT）数据类型是最常用的数值形式之一。在 INT 计数系统中，数据的所有数位表示的都是整数，不含分数或小数部分。一个 n 位的 INT 数可以表示 $2 ^ { n }$ 个数；例如，无符号 INT 数可以编码从 0 到 $2 ^ { n } - 1$ 的非负值，而有符号 INT 数的取值范围为 $[ - 2 ^ { n - 1 } , + 2 ^ { n - 1 } - 1 ]$ ]。与 INT 数据类型不同，另一种常用的数据类型——浮点（FP）数据类型通过浮动的基数点来表示小数部分。FP 数表示为

$$
x _ {i} = (- 1) ^ {s _ {i}} \times (1. m _ {i}) \times 2 ^ {e _ {i}},\tag{1}
$$

其中，$s _ { i }$ 是符号，m<sub>i</sub> 是尾数，$e _ { i }$ 是数 $x _ { i }$ 的指数。由于在相同位宽下，这种格式能够表示比整数更宽的数值动态范围，因此 FP 数通常用于对精度敏感且要求较宽数值表示范围的应用。FP 数据类型能够编码的数值范围为 $| 2 ^ { B - 1 } , ( ( 2 - 2 ^ { - M } ) ) \stackrel { . . . } { \times } 2 ^ { B } |$，其中 B 为偏置值（计算为 $2 ^ { E - 1 } ; E$ 为指数的位数），M 为尾数的位宽。

最近，多项研究已在 AI 算法中应用微缩放（MX）数据格式 [6], [25]。MX 格式是一种替代表示方式，能够有效平衡逻辑密度与表示宽范围数值的能力。在这种格式中，多个数被分为一个块，这些数共享一个公共指数值。当一个块内的 MX 格式数表示为

$$
\vec {x} = [ \hat {x _ {1}}, \hat {x _ {2}}, \dots , \hat {x _ {N}} ] \cdot 2 ^ {e _ {x}} = \vec {\hat {x}} \cdot 2 ^ {e _ {x}},\tag{2}
$$

时，两个块之间的点积计算如下

$$
\vec {x} \cdot \vec {y} = (\vec {\hat {x}} \cdot \vec {\hat {y}}) \cdot 2 ^ {(e _ {x} + e _ {y})}.\tag{3}
$$

在式 (3) 中，$\vec { x }$ 和 $\vec { y }$ 的元素可以是 INT 数，也可以是 FP 数，分别称为 MXINT 格式和 MXFP 格式。使用 MX 格式时，即使 $\vec { x }$ 和 $\vec { y }$ 的元素被量化为低位宽，共享指数（即 $2 ^ { e _ { x } }$ 和 $2 ^ { e _ { y } } )$ 也能实现宽动态范围的表示，同时使点积 $( \vec { \hat { x } } \cdot \vec { \hat { y } } )$ 能够以较低的硬件成本计算。此外，通过在元素之间共享公共指数，还能减少存储指数所占用的内存空间。鉴于这些显著优势，Intel、Meta、Microsoft 和 NVIDIA 等多家领先科技公司正在推动 MX 格式的标准化，将其作为面向 AI 的下一代窄精度数据格式 [28]。

## B. 相关工作

旨在降低位宽的研究推动了支持位级灵活性的 AI 加速器设计。例如，多项研究 [17], [33] 使用位串行 MAC 单元，支持 INT 数据类型的可变精度配置。相比之下，文献 [23], [34] 和 [31] 中的工作采用空间方法，将构成 MAC 单元的各个子乘法器所生成的结果进行合并。针对 AI 算法的训练，研究人员还研究了能够在 MX 格式下进行精度缩放的 MAC 单元 [19], [25]。由于简化了 MAC 运算，并且能够覆盖所需精度的广泛范围，这些 MAC 单元具有较低的硬件成本，并能提供容错的训练结果。在超大规模云基础设施中，AI 加速器通常支持 INT 和 FP 混合精度，使云系统能够在同一个加速器节点上处理推理和训练任务 [14], [26], [41]。最近，面向移动和便携设备设计的商用 AI 硬件平台通过嵌入定制开发的 ASIC，已具备 INT 位宽灵活性或混合精度支持 [2], [27]。然而，值得注意的是，目前尚无工作同时集成对 INT、FP 和 MX 格式的位级灵活性支持（尚未解决的挑战）。云服务提供商提供的大多数 AI 加速器（例如 Google TPU [14]、IBM RaPiD [41] 和 NVIDIA Tensor Core [26]）通过为每种受支持的数据格式部署多个计算单元来实现位宽缩放。

## III. JACK 单元

近期在降低位宽方面的前沿进展已在多种数据格式（例如 bfloat16 {s:1, e:8, m:7}、FP8 {s:1, e:4, m:3}、INT8、INT4、MXINT8、MXINT4 和 MXFP8 {s:1, e:4, m:3}<sup>1</sup>）下实现了良好的训练和推理准确率，同时提升了性能和能效 [6], [8], [15], [16], [25], [35], [42], [43]。受这些近期研究工作的启发，我们提出了一种名为 Jack 单元的新型 MAC 单元，能够支持广泛的数据格式。

## A. 进位保存乘法器（CSM）的重构

1) 采用数据通路流水线的位宽可伸缩 CSM：为降低 MAC 单元的硬件开销，我们首先重构 FP 乘法器的 CSM。两个 FP 数的乘法结果为 $x \times y = ( - \mathrm { \hat { 1 } } ) ^ { S _ { x } \oplus S _ { y } } \times ( \boldsymbol { M } _ { x } \mathrm { \hat { \times } } \boldsymbol { M } _ { y } ) \times 2 ^ { E _ { x } + E _ { y } - B i a s }$，其中，$M _ { x }$ 和 $M _ { y }$ 分别为 x 和 y 的有效数（即 $1 . m _ { x }$ 和 $1 . m _ { y } )$。CSM 是 FP 乘法器的一个子模块，用于执行有效数的乘法。在综合先前工作 [1] 中介绍的 bfloat16 和 FP8 乘法器时，CSM 在乘法器的各个子模块中占据最大的面积和功耗比例。具体而言，在

![](images/cdd50f1f7426581c3c85d1148bd6a0bd6c0543d4523c0c870449fa9d88271ce5.jpg)

![](images/5a848a0eca5dc9b4fd7afe8c77421a8166954943ccc03dd72dcba927fa8f3a72.jpg)  
(a)  
(b)

![](images/74239dcafa2c130e2f68cadadb73567e1f1008eabed680029c07d1bd3a4d6074.jpg)  
(c)

图 2：精度可伸缩 CSM 执行 (a) 8 位×8 位和 (b) 4 位×4 位乘法的示意图。(c) 本设计中重构后的 CSM 的输入数据通路。  
![](images/bebe8e17f47d9aa7fd833e292860967a6402e4758d08626b137639f3d77fae0f.jpg)  
图 3：(a) FP 加法的计算流程，(b) 通过 FP 加法器将两个乘法结果相加的 MAC 单元，(c) 在 CSM 内根据指数差调整有效数的 MAC 单元，(d) 我们提出的 Jack 单元中使用的重构后的 CSM。

最高工作频率下，CSM 在 bfloat16 乘法器中占据 73.3% 的面积和 71.1% 的功耗，在 FP8 乘法器中占据 53.8% 的面积和 47.3% 的功耗。因此，如图 1-(b) 所示，在 MAC 单元中为每种支持的精度级别分配专用乘法器，会因需要为每种支持的数据格式集成如此昂贵的 CSM 而显著增加硬件开销。为缓解这一低效问题，我们采用与 [34] 类似的方法，将 bfloat16 乘法器的 CSM 替换为硬件开销更低的精度可伸缩乘法器，它能够支持 4 位和 8 位组合的有符号/无符号乘法。图 2-(a) 和 (b) 展示了替换后的 CSM 的结构，并通过两个示例说明了精度可伸缩 CSM 如何在 8 位×8 位和 4 位×4 位模式下运行。如图所示，替换后的 CSM 由四个 4 位×4 位子乘法器组成，并通过动态融合子乘法器的结果来支持 4 位和 8 位配置。然而，这一结构面临一个设计挑战，即带宽（BW）利用率会随所支持的精度级别而变化<sup>2</sup>。例如，在 4 位精度模式下，输入带宽得到充分利用（图 2-(b)），而在 8 位精度模式下，带宽利用率仅为 50%（图 2-(a)）。为缓解这一问题，我们将输入数据通路修改为可旁路的有线链路，如

<sup>2</sup>高带宽（BW）可能增加布线面积，从而导致总体面积增加。我们以 90% 的核心利用率对两个 8×8 MAC 阵列进行了布局布线（P&R）。这两个阵列分别采用非流水线式 JACK 单元（图 2-(a-b)）和流水线式 JACK 单元（图 2-(c)）设计。我们观察到，非流水线式 MAC 阵列产生了约 42% 的布局开销。

![](images/beef5da30623da45a4820ecc5d8551744f1f8addd0822c2eeef83bc4ed74f91f.jpg)  
图 4：(a) 所提出的 Jack 单元的整体结构，(b) 指数提取器的结构图。(c-f) Jack 单元在不同运算模式（即 FP8、INT8、MXINT8 和 MXFP8 模式）下各子模块的激活情况。

图 2-(c) 所示。该可旁路链路为每个输入（即 X 和 W）配备 8 条信号线，并通过流水线路径在多个周期内传输更宽的输入（> 8 位），从而确保无论所支持的精度级别如何，带宽利用率均为 100%。

2) 在 CSM 中调整有效数：图 3-(a) 展示了将两个乘法结果相加的 FP 加法计算流程。当对指数不同的 FP 数进行相加时，需要将较小数的指数与较大数的指数对齐。为实现这一点，FP 加法包含两个主要步骤：i) 比较指数，并将较小值的有效数按指数差进行移位；ii) 将对齐后的有效数相加。这些步骤是 INT 加法中不存在的复杂过程。因此，在相同位宽下，FP 加法器的硬件开销高于 INT 加法器（图 1-(a)）。尽管如上一小节所述，CSM 已被替换为精度可伸缩 CSM，但由于包含 FP 加法器，MAC 单元的开销仍然很高（图 3-(b)）。为以低得多的硬件开销实现 MAC 运算，我们探索了一种允许使用 INT 加法器执行加法的方法<sup>3</sup>。该方法的核心思想是提前计算指数差，并在 CSM 内对较小值的有效数进行移位。图 3-(c) 展示了支持这一方法的 MAC 单元框图。子乘法器的输出在送入 CSM 内部的加法器树之前，立即由桶形移位器（图中标记为 B.S.）进行移位。这种方法使加法和乘法均可在仅支持 INT 的计算单元上执行。

3) 二维子字并行：为进一步降低硬件开销，我们重新排列子乘法器，以实现操作数的二维子字并行。在精度可伸缩 CSM 中，子乘法器完成乘法后，依次执行左移操作。值得注意的是，不同 CSM 中处于相同位置的子乘法器，其输出具有相同的左移参数。例如，如图 3-(c) 所示，红框内子乘法器的输出均按相同的数值进行移位。（具体而言，红框内的子乘法器执行子字 A 和 B 的乘法（即 $M _ { P _ { 3 } } \ \times \ M _ { Q _ { 3 } }$ 和 $M _ { Y _ { 3 } } \times M _ { Z _ { 3 } } )$，并通过桶形移位器以相同的移位参数，即

$2 ^ { 8 }$，进行左移。）因此，我们将具有相同移位参数的子乘法器分组，使这些组能够共享左移运算器。图 3-(d) 展示了 Jack 单元所使用的重构后的 CSM 的最终结构。该重构后的 CSM 将来自四个精度可伸缩 CSM 的子乘法器进行聚合。通过这种方法，与未采用二维子字并行的四个 CSM 相比，重构后的 CSM 将移位器数量减少了 75%。

## B. Jack 单元的总体结构

图 4-(a) 展示了我们提出的 Jack 单元的总体结构。Jack 单元由重构后的 CSM、异或门组、指数提取器、规格化器和舍入器组成。重构后的 CSM 将四个位宽可扩展 CSM 组合在一起，作为一个基本构建模块。如第 III-A1 节所述，当操作数的有效数长度分别为 8 位（例如 bfloat16）和 4 位（例如 FP8）时，精度可扩展 CSM 分别执行一次 8 位×8 位乘法和四次 4 位×4 位乘法。因此，当有效数长度为 8 位时，Jack 单元产生四个乘法结果；当有效数长度为 4 位时，则产生十六个乘法结果。为了匹配重构后的 CSM 产生的运算结果数量，异或门组和指数提取器的内部结构中分别包含十六个异或门和十六个指数计算器。异或门组计算具有浮动基数点的数据格式（例如 FP 和 MXFP）的符号，而指数提取器使用指数计算器和比较器，确定这些格式的运算结果中的最大指数值（图 4-(b)）。异或门组和指数提取器的结果被传送至重构后的 CSM，并用于有效数的计算过程。在重构后的 CSM 中完成 MAC 运算后，最大指数值和重构后的 CSM 的输出被传送至规格化器。规格化器检测前导 1，并在发生溢出时通过递增指数来对尾数进行规格化。最后，舍入器将结果数据的位宽截断至 16 位。请注意，为了避免累加误差，Jack 单元产生单个 16 位输出，即 FP16 或 INT16，这与 [41] 中提出的设计一致。

## C. 工作模式

所提出的 Jack 单元支持针对各种数据类型的灵活位宽配置，例如 FP、INT 和 MX 数据格式。这种灵活性通过选择性电源门控方案实现，该方案在 Jack 单元内部的子模块不被使用时将其关闭。图 4-(c-f) 展示了在几种受支持的数据格式模式下，即 FP8、INT8、MXINT8 和 MXFP8 模式下，子模块的激活方式。在 FP8 模式下，Jack 单元的所有子模块均被激活，如图 4-(c) 所示。相比之下，在 INT8 模式下，仅重配置后的 CSM 被激活（图 4-(d)）。在这种情况下，重配置后的 CSM 的结果通过专用数据通路直接传送至 Jack 单元的输出，如图 4-(a) 所示。在 MXINT8 模式下，由于块张量中的元素共享同一个指数值，因此只激活一个指数计算器。在 MXFP8 模式下，与 FP8 模式类似，所有子模块均被激活；不过，指数计算器接收的偏置包含了块的共享指数，即式 3 中的 $2 ^ { e _ { x } }$ 和 $2 ^ { e _ { y } }$。（具体而言，在 MXFP 模式下，指数计算器将偏置 + $\boldsymbol { e } _ { x } + \boldsymbol { e } _ { y }$ 作为其偏置值。）这使得 Jack 单元能够在单元内部完成对块共享指数的计算。请注意，在 MX 格式模式下，可以通过激活多个 Jack 单元中的指数计算器来调整块大小的粒度。例如，在 MXINT16 模式下，通过在四个 Jack 单元中仅激活一个指数计算器，可以使十六个元素共享同一个指数值。

(b) 功耗  
![](images/2e120ee44e5ebdda2d50144c2c56f8a4f453014a357559047edb616bd35213ed.jpg)

![](images/b54a94b3c9c5ca7c88d3a25e471b434b85931ce670710a94c073c7ebcbf9a786.jpg)  
图 5：MAC 单元的 (a) 面积和 (b) 功耗构成。

## IV. 评估

## A. Jack 单元的延迟、面积和功耗分析

基准方案与方法：为了定量评估我们所提出方法的影响，我们在寄存器传输级（RTL）实现了 Jack 单元和三种基准方案。其中一种基准方案是 MAC 单元（MAC-1），它与商用 AI 加速器 [14]、[26]、[41] 类似，通过为每种支持的数据格式配备专用乘法器来支持多种数据格式。另外两种基准方案，即 $\mathrm { \underline { { M A C } } } - ( 2 )$ 和 $\mathrm { { M A C } } - \textcircled { 3 } .$，是图 3-(b) 和 (c) 所示的 MAC 单元。$\mathbf { M A C - } ( 2 )$ 是一种将 CSM 替换为精度可扩展 CSM 的 MAC 单元，而 MAC-3 不仅将 CSM 替换为精度可扩展 CSM，还在 CSM 内部执行有效数调整。在量化算法近期进展 [8]、[15]、[16]、[35]、[42]、[43] 的推动下，这些基准方案支持 bfloat16、FP8、INT8 和 INT4 等数据格式，并具有与所提出的 Jack 单元相同的吞吐量。所有设计均使用 Synopsys Design Compiler [37]，基于 65 nm CMOS 库的典型 PVT 角（1.1 V，25 C）进行综合。对于延迟，我们报告不会引起建立时间违例的最小可实现时钟周期。我们给出了从 Synopsys IC Compiler [36] 获得的面积数据，以及使用 Synopsys PrimeTime PX [38] 估算的功耗数据；为进行公平比较，这些数据均基于在相同的 286 MHz 时序约束下得到的布局布线（P&R）结果。

延迟分析：$\mathbf { M A C - } \textcircled{1 }$ 可在最高 286 MHz 的时钟频率下运行而不发生时序违例（$T _ { c l k } ~ = ~ 3 . 5 ~ \mathrm { n s }$）。由于将 CSM 替换为位宽可扩展 CSM 导致关键路径变长，MAC-2 的延迟为 3.6 ns，比 $\mathbf { M A C - } \textcircled { 1 } ,$ 更长。然而，$\mathbf { M A C - } ( 3 )$ 通过直接在 CSM 内部调整有效数，无需通过 FP 加法器树进行累加，而是使用成本更低的 INT 加法器树，从而将延迟改善至 3.4 ns。Jack 单元采用 2D 子字并行，减小了 INT 加法器树的输入位宽，进一步将延迟降低至 3.3 ns。位宽的减小缩短了累加加法器内部的进位传播延迟，从而缩短了关键路径。

表 I：采用我们的 Jack 单元设计的加速器与基准加速器（即类 RaPiD 加速器 [41]）的设计配置。

<table><tr><td>AI 加速器</td><td>采用 Jack 单元的加速器</td><td>类 RaPiD 加速器</td></tr><tr><td>工艺节点</td><td>商用 65 nm</td><td>商用 65 nm</td></tr><tr><td>工作频率</td><td>400 MHz</td><td>400 MHz</td></tr><tr><td>阵列尺寸</td><td>32×32</td><td>128×128</td></tr><tr><td>乘法器数量</td><td>128×128(bfloat16 / INT8 / MXINT8)512×512(FP8 / INT4 / MXFP8 / MXINT4)</td><td>128×128(bfloat16 / INT8)512×512(FP8 / INT4)</td></tr><tr><td>片上缓冲区(I / W / O) [KB]</td><td>512 / 512 / 256</td><td>512 / 512 / 256</td></tr></table>

表 II：用于评估的 AI 基准测试和数据集。

<table><tr><td>模型</td><td>应用</td><td>数据集</td></tr><tr><td>ConvNeXt-T (CNN) [18]</td><td>图像分类</td><td>ImageNet [7]</td></tr><tr><td>BERT (NLP) [9]</td><td>语言建模</td><td>WMT14 [5]</td></tr><tr><td>GPT2-Small (LLM) [29]</td><td>语言建模</td><td>Wikitext-2 [20]</td></tr><tr><td>NeRF [21]</td><td>面向 AR/VR 的视图合成</td><td>合成 NeRF [22]</td></tr><tr><td>QuickSRNet [4]</td><td>图像超分辨率</td><td>DIV2K [12]</td></tr></table>

![](images/52f1c8c5d7140b7f7f28e25d745744d6a19daea41c70770e1769daa2a7864690.jpg)  
图 6：采用 Jack 单元设计的加速器与基准加速器（即类 RaPiD 加速器 [41]）的面积构成。

面积与功耗分析：图 5 展示了基准 MAC 单元和 Jack 单元的面积与功耗。$\mathbf { M A C - } \textcircled{1 } .$ 为每种支持的数据格式采用专用乘法器，占用约 $1 1 , 0 8 4 \mathrm { u m ^ { 2 } }$ 的面积，功耗为 1.67mW。$\mathbf { M A C - } ( 2 )$ 通过采用支持位宽灵活性的 CSM 来替代专用乘法器，与 MAC-1 相比，实现了 1.37× 的面积缩减和 1.06× 的功耗缩减。通过在精度可扩展 CSM 内部执行 FP 加法所需的有效数调整，硬件开销得到进一步降低。$\mathrm { \mathbf { M A C } } - \textcircled { 3 }$ 引入了这一改进，与 $\mathbf { M A C - } ( 2 )$ 相比，面积和功耗分别降低了 20.15% 和 39.23%。我们提出的 Jack 单元具有最小的面积和最低的功耗，与基准单元相比，实现了 1.17∼2.01× 的面积缩减和 1.05∼1.84× 的功耗缩减。

## B. 架构评估

基线与方法：我们在 RTL 层面实现了一个 32×32 的二维脉动阵列，以评估 Jack 单元的实际有效性。作为比较，我们实现了一款近期的 AI 加速器，即 IBM RaPiD [41]，它具有 128×128 个 MAC 单元，并在其支持的所有数据格式模式下提供与采用 Jack 单元的加速器相同的吞吐量。两款加速器中采用的 Jack 单元和 MAC 单元均通过 2 级寄存器实现流水线化，以达到 400 MHz 的目标时钟频率。为进行架构评估，我们使用与第 IV-A 节所述相同的 EDA 工具和 CMOS 库进行了布局布线（P&R）。

![](images/aa43cde583a69834f50e7e3c76d771e61ff2004a62adf11d01d9ba9c33f54c53.jpg)

![](images/18484dc6e297a5988a69e572d2bec316c387b3d8da5e3d17434af35004a1407b.jpg)  
图 7：(a) 基于 JACK 单元的加速器在 bfloat16 和 INT4 模式下的推理延迟。(b) 在 bfloat16 和 INT4 两种精度模式下，基于 Jack 单元的 AI 加速器（J）与基线加速器（B）的计算密度（即面积效率）比较。

![](images/30294ea2b454471a45295a12492090610beceb0bf03c8be2fe4e5844e280205e.jpg)  
图 8：对于每种支持的精度模式，基于 JACK 单元的加速器（J）和基线加速器（B）在目标 AI 基准测试中的平均能效。

此外，我们通过在功耗分析工具 Synopsys PrimeTime PX 中调整控制信号的翻转活动，获得了每种支持的数据格式下的功耗。为进行系统级分析，我们在估算能耗时同时考虑了片上和片外存储器访问。片上存储器模块使用 CACTI-6.0 [24] 建模，而片外功耗和时序参数则采用双通道 HBM 的规格 [13]。为估算这两种架构的时钟周期数，我们使用了开源的周期级模拟器 SCALE-sim [32]。我们修改了该模拟器，以反映两种架构的片上和片外存储器配置及数据流。表 I 汇总了两种架构的关键设计规格。表 II 列出了所评估的 AI 基准测试及其对应的数据集。面积比较：图 6 展示了采用 Jack 单元设计的加速器和类 RaPiD 加速器的面积构成。与基线加速器的相应面积相比，基于 Jack 单元的加速器在 MAC 阵列面积和布线面积上分别实现了 1.93× 和 1.42× 的缩减。这些缩减归功于 Jack 单元的高面积效率设计和流水线化的数据通路。因此，与基线加速器相比，基于 Jack 单元的加速器实现了 1.60× 的总体面积缩减。

延迟与计算密度分析：图 7-(a) 展示了目标基准测试在 bfloat16 和 INT4 两种精度模式下的推理运行时间。在 INT4 模式下，Jack 单元的乘法器数量为 bfloat16 模式下的 16×（表 I），推理速度达到其 9.06∼13.08×。然而，由于采用流水线化的数据通路，基于 Jack 单元的加速器的片上缓冲区访问延迟增加了 69%，使得总体推理延迟比基线加速器长 6.65%。尽管延迟更长，所提出的基于 JACK 单元的加速器仍表现出比基线加速器更高的单位面积性能。图 7-(b) 展示了两款加速器的单位面积计算效率，表明由 JACK 单元构成的 MAC 单元平均实现了 1.80× 的计算强度。这一提升源于 JACK 单元的高面积效率设计策略（详见

第 III-A 节），以及数据通路的流水线化。

能效分析：图 8 展示了两款加速器在所支持的各种精度模式下的能效。得益于所提出的 Jack 单元的高能效设计，在 INT 和 FP 模式下采用相同位宽配置时，基于 Jack 单元的加速器的能效达到基线加速器的 1.32∼5.41×。此外，与基线加速器不同，基于 Jack 单元的加速器支持 MX 格式<sup>4</sup>。如图中红色星号标记的数据所示，与在 bfloat16 {s:1, e:8, m:7} 模式下运行的基线加速器相比，基于 Jack 单元的加速器在 MXINT8 {s:1, e shared:8, m:7} 模式下实现了 7.13× 的能效。这一提升源于 MX 格式所支持的低精度乘法，相关讨论见第 II 节。图中蓝色星号标记的数据展示了在 MXFP8 {s:1, e shared:8, e local:4, m:7} 模式下运行的基于 Jack 单元的加速器，以及在 FP8 {s:1, e:4, m:3} 模式下运行的基线加速器的能效。尽管 Jack 单元以更大的位宽运行，但得益于其高能效设计，它仍实现了 4.98× 的能效。

## V. 结论

本研究提出了 Jack 单元，这是一种兼具高面积效率和高能效的 MAC 单元，通过三种设计策略支持位宽灵活的数据格式：i) 精度可伸缩的乘法器，ii) CSM 内部的有效数调整，以及 iii) 二维子字并行。通过这些方法，与商用 AI 加速器中部署的 MAC 单元相比，所提出的 MAC 单元实现了 2.01× 的面积缩减和 1.84× 的功耗降低。我们预计，Jack 单元将因其能够以经济高效的方式满足多样化设计要求和应用需求，而在 AI 加速器中得到广泛采用。

## VI. 致谢

本研究获得了韩国政府（MSIT）资助的信息通信技术规划与评估院（IITP）项目支持，项目编号为 2022-0-00991；同时获得了科学与信息通信技术部资助的韩国国家研究基金会（NRF）项目支持，项目编号为 NRF-2023R1A2C2006290。本研究还获得了韩国产业通商资源部（MOTIE, Korea）资助的技术创新计划支持（RS-2024-00445759，“开发利用基于视觉语言模型的视觉信息、用于理解未学习空间中动态环境的导航技术”）。本研究使用的 EDA 工具由韩国集成电路设计教育中心（IDEC）提供支持。

## REFERENCES

[1] M. Al-Ashrafy et al. An efficient implementation of floating point multiplier. In Proc. of SIECPC, 2011.

[2] ARM. Ethos-N78. https://developer.arm.com/Processors/Ethos-N78, 2023.

[3] J.-A. Baktash et al. Gpt-4: A review on advancements and opportunities in natural language processing. In arXiv:2305.03195, 2023.

[4] G. Berger et al. Quicksrnet: Plain single-image super-resolution architecture for faster inference on mobile platforms. In Proc. of CVPR, 2023.

[5] Ondˇrej Bojar et al. Findings of the 2014 workshop on statistical machine translation. In Proc. of WMT, 2014.

[6] Bita Darvish Rouhani et al. Microscaling data formats for deep learning. In Proc. of NeurIPS, 2023.

[7] Jia Deng et al. Imagenet: A large-scale hierarchical image database. In Proc. of CVPR, 2009.

[8] Z. Dettmers et al. Gpt3.int8(): 8-bit matrix multiplication for transformers at scale. In Proc. of NeurIPS, 2022.

[9] J. Devlin et al. BERT: pre-training of deep bidirectional transformers for language understanding. In Proc. of NAACL-HTL, 2019.

[10] J. Fowers et al. A configurable cloud-scale dnn processor for real-time ai. In Proc. of ISCA, 2018.

[11] B. Hickmann et al. Intel nervana neural network processor-t (nnp-t) fused floating point many-term dot product. In Proc. of ARITH, 2020.

[12] Andrey Ignatov et al. Pirm challenge on perceptual image enhancement on smartphones: report. In Proc. of ECCV Workshop, 2019.

[13] JEDEC. Standard High Bandwidth Memory (HBM) DRAM, 2015.

[14] N. Jouppi et al. Tpu v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings. In Proc. of ISCA, 2023.

[15] D. Kalamkar et al. A study of bfloat16 for deep learning training. In arXiv:1905.12322, 2019.

[16] A. Kuzmin et al. Fp8 quantization: The power of the exponent. In Proc. of NeurIPS, 2022.

[17] J. Lee et al. Unpu: A 50.6tops/w unified deep neural network accelerator with 1b-to-16b fully-variable weight bit-precision. In Proc. of ISSCC, 2018.

[18] Z. Liu et al. A convnet for the 2020s. In Proc. of CVPR, 2022.

[19] Y.-C. Lo et al. Bucket getter: A bucket-based processing engine for low-bit block floating point (bfp) dnns. In Proc. of MICRO, 2023.

[20] Stephen Merity et al. Pointer sentinel mixture models. 2016.

[21] B. Mildenhall et al. Nerf: representing scenes as neural radiance fields for view synthesis. In Proc. of ECCV, 2020.

[22] Ben Mildenhall et al. Nerf. https://www.matthewtancik.com/nerf, 2020.

[23] Vishesh Mishra, Divy Pandey, Saurabh Singh, Sagar Satapathy, Kaustav Goswami, Babita Jajodia, and Dip Sankar Banerjee. Art-mac: Approximate rounding and truncation based mac unit for fault-tolerant applications. In Proc. of ISCAS, 2022.

[24] N. Muralimanohar. CACTI 6.0: A Tool to Model Large Caches. www. hpl.hp.com/techreports/2009/HPL-2009-85.pdf, 2009.

[25] S.-H. Noh et al. Flexblock: A flexible dnn training accelerator with multi-mode block floating point support. In journal of TC, 2023.

[26] NVIDIA. NVIDIA H100 Tensor Core GPU Architecture. https://www.advancedclustering.com/wp-content/uploads/2022/03/ gtc22-whitepaper-hopper.pdf, 2022.

[27] J.-S. Park et al. A multi-mode 8k-mac hw-utilization-aware neural processing unit with a unified multi-precision datapath in 4nm flagship mobile soc. In Proc. of ISSCC, 2022.

[28] Open Computer Project. Ocp microscaling formats (mx) specification version 1.0. 2023.

[29] Alec Radford et al. Language models are unsupervised multitask learners. 2019.

[30] Darvish Rouhani et al. With shared microexponents, a little shifting goes a long way. In Proc. of ISCA, 2023.

[31] S. Ryu et al. Bitblade: Area and energy-efficient precision-scalable neural network accelerator with bitwise summation. In Proc. of DAC, 2019.

[32] A. Samajdar et al. A systematic methodology for characterizing scalability of dnn accelerators using scale-sim. In Proc. of ISPASS, 2020.

[33] S. Sharify et al. Loom: Exploiting weight and activation precisions to accelerate convolutional neural networks. In Proc. of DAC, 2018.

[34] H. Sharma et al. Bit fusion: Bit-level dynamically composable architecture for accelerating deep neural network. In Proc. of ISCA, 2018.

[35] X. Sun et al. Hybrid 8-bit floating point (hfp8) training and inference for deep neural networks. In Proc. of NeurIPS, 2019.

[36] Synopsys. IC Compiler II Implementation User Guide: Version L-2016.03, 2016.

[37] Synopsys. Design Compiler User Guide: Version N-2017.09, 2017.

[38] Synopsys. Synopsys primetime px power analysis solution achieves broad market adoption. https://news.synopsys.com/home?item=123041, 2024.

[39] Maroun Tork et al. Lynx: A smartnic-driven accelerator-centric architecture for network servers. In Proc. of ASPLOS, 2020.

[40] H. Touvron et al. Llama: Open and efficient foundation language models. In arXiv:2302.13971, 2023.

[41] S. Venkataramani et al. Rapid: Ai accelerator for ultra-low precision training and inference. In Proc. of ISCA, 2021.

[42] M. Wortsman et al. Stable and low-precision training for large-scale vision-language models. In Proc. of NeurIPS, 2024.

[43] X. Wu et al. Understanding int4 quantization for language models: latency speedup, composability, and failure cases. In Proc. of ICML, 2023.

> 来源与版本：ISLPED 2025 出版状态由作者实验室确认，技术内容和页码依据用户提供的作者 PDF；原文入口另列 arXiv:2507.04772。

> 本文根据用户提供的 PDF 经 MinerU 解析，由 Codex/GPT 翻译和解读，未调用 GLM 翻译 API。参考文献保留原文；技术总结、通俗讲解和图表说明属于解读。[原文入口](https://arxiv.org/abs/2507.04772)。
