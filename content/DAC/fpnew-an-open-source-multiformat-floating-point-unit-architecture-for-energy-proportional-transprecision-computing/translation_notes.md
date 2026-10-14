# FPnew: An Open-Source Multiformat Floating-Point Unit Architecture for Energy-Proportional Transprecision Computing 原文翻译

# FPnew：一种面向能耗按比例调节的跨精度计算的开源多格式浮点单元架构

Stefan Mach , Fabian Schuiki , Florian Zaruba , Graduate Student Member, IEEE, and Luca Benini , Fellow, IEEE

摘要— 摩尔定律的放缓和功耗墙的限制，使得我们必须转向精细可调精度（又称跨精度）计算，以降低能耗足迹。因此，我们需要能够在广泛精度范围内执行浮点运算、并具有高能耗按比例调节能力的电路。我们提出了 FPnew，这是一种高度可配置的开源跨精度浮点单元，能够支持多种标准和自定义浮点格式。为了展示 FPnew 在通用处理器架构中的灵活性与高效性，我们对 RISC-V ISA 进行了扩展，增加了半精度、bfloat16 以及一种 8 位浮点格式上的运算，同时支持 SIMD 向量和多格式运算。将该跨精度浮点单元集成到 32 位 RISC-V 核心中，相较于 FP32 基线，可将混合精度应用的执行速度提升 1.67×，同时保持端到端精度，并将系统能耗降低 37%。我们还将 FPnew 集成到 64 位 RISC-V 核心中，支持五种浮点格式的标量运算或 2、4、8 路 SIMD 向量运算。对于该核心，我们在 Globalfoundries 22FDX 工艺下对流片制造的硅芯片进行了测量，电压范围覆盖 0.45 至 1.2 V。该单元在实测能效上达到业界领先水平，介于 178 Gflop/sW（FP64）与 2.95 Tflop/sW（8 位 mini-float）之间，性能介于 3.2 至 25.3 Gflop/s 之间。

索引术语— 能耗高效、浮点单元、多格式、RISC-V、跨精度计算。

## I. 引言

在过去的十年中，人们对能效架构与系统的追求出现了爆发式增长。随着摩尔定律逐渐失效，一个由 CMOS 技术缩放驱动的时代正在走向终结。所谓的“功耗墙”（power-wall）障碍正在推动计算范式向前发展，这些范式将能效作为任何硬件设计的终极衡量指标。

与此同时，快速演化的工作负载（例如机器学习）成为计算行业的焦点，并且始终要求在恒定或不断降低的功耗预算下获得更高的计算性能，其应用范围涵盖数据中心、高性能计算（HPC）直至物联网（IoT）领域。在这种环境下，要在数值计算中实现高能效，就需要在精度和性能上可微调的架构与电路。这类电路可以通过以敏捷的方式使性能和精度适配应用需求，从而最小化每次操作的能耗。“跨精度计算”（transprecision computing）范式 [1] 旨在创建一个覆盖算法、软件到硬件与电路的整体性框架，提供众多可微调工作负载的旋钮。

在现代系统上执行数值计算最灵活、最动态的方式是浮点（FP）运算。它由 IEEE 754 标准化，已在绝大多数计算领域真正无处不在：从通用处理器、图形计算加速器（GPU）到超级计算机，并且日益普及于高性能嵌入式系统和超低功耗微控制器。虽然通常使用整数数据通路的定点计算有时能成为 FP 的高效替代方案，但它的灵活性和通用性远不如 FP。将 FP 工作负载转换为定点表示通常需要人类专家的领域专用知识，因为数值范围与精度的权衡必须被人工管理和跟踪。IEEE 754 内建的舍入模式、优雅的下溢处理以及无穷大的表示，都是为了让 FP 运算对数值误差更加鲁棒和容忍 [2]。此外，许多应用（例如涉及物理和化学模拟的科学计算）在定点表示下是不可行的，需要 FP 所提供的高动态范围。

高效的跨精度计算所要求的 FP 精度调制，过去在 CPU 和 GPU 中仅限于常见的 “double” 和 “float” 格式。然而，FP 格式出现了名副其实的“寒武纪大爆发”，例如 Intel Nervana 的 Flexpoint [3]、Microsoft Brainwave 的 9-bit 浮点数 [4]、Google TPU 的 16-bit “bfloats” [5]，以及 NVIDIA 的 19-bit TF32，它们被实现在诸如 Tensor Cores [6] 等专用加速器中。这表明，新的架构需要具备极端的跨精度灵活性来支持 FP 计算，而这在很大程度上是由机器学习算法和应用驱动的。我们的目标是创建一种灵活、可定制的跨精度浮点单元（transprecision floating-point unit，TP-FPU）架构，使其能够应用于种类繁多的计算系统与应用之中。

为了利用这种跨精度使能的硬件，当然也必须在整个软件栈中获得支持与感知。指令集架构（ISA）构成了硬件与软件之间的接口。RISC-V [7] 是一种开源 ISA，原生支持常见的 “double” 和 “float” 格式的运算。此外，该 ISA 明确允许非标准扩展，架构师可以自由添加自己的指令。近来，得益于其开放与可扩展的特性，以及来自软硬件项目日益增长的支持，RISC-V 在工业界和学术界都获得了广泛关注。在本工作中，我们利用 RISC-V ISA 的开放性与可扩展性，添加了针对额外 FP 格式运算的扩展，而这些格式在当前的 RISC-V 处理器实现中并不存在 [8]。

在本工作中，我们还展示了完整的开源 TP-FPU 在 22 nm 工艺下、集成于一个 RISC-V 应用级内核中的全功能流片实现 [9]。该流片架构支持广泛的数据格式，包括 IEEE 754 double（FP64）、single（FP32）和半精度浮点数（FP16），以及 16-bit bfloat（FP16alt）和首次在 [10] 中引入的自定义 8-bit 格式（FP8）。此外，还全面支持单指令多数据（SIMD）向量化，以及向量化的转换与数据打包。

总而言之，我们的贡献如下。

1) 以 SystemVerilog 编写的、高度可配置的跨精度浮点单元架构设计。所有标准 RISC-V 运算均获支持，并辅以诸多新增功能，例如 SIMD 向量、多格式融合乘加（FMA）运算，以及动态创建打包向量的转换与打包（convert-and-pack）功能。该单元完全开源，因此可进一步扩展以支持更多功能。与现有最优（SOA）设计不同，我们通过增加电路面积来实现高能耗比例性与高能效。

2) RISC-V ISA 的扩展，以支持 FP64、FP32、FP16、FP16alt 和 FP8 上的跨精度 FP 运算 [10]。程序员可以通过高级编程语言中的标准运算利用跨精度，使用编译器使能的自动向量化，或通过编译器内建函数（compiler-intrinsic）调用跨精度指令进行进一步优化 [8]。

3) 将 TP-FPU 集成到 RI5CY [11]（一个 32 位嵌入式 RISC-V 处理器内核）中。一个应用案例研究表明，使用跨精度 ISA 扩展相对于 FP32 基线可实现 1<sup>.</sup>67× 的加速，且不牺牲结果的任何精度。此外，完成该工作负载所需的处理器能耗降低了 37%。

4) 将 TP-FPU 集成到 Ariane [12]（一个 64 位应用级 RISC-V 处理器内核）中，并随后在 GLOBALFOUNDRIES 22FDX [9] 中实现流片。对制造出的裸片进行的能耗与性能测量证实了我们架构显著的能耗比例性与领先的能效。我们对每条指令的能耗成本、向量化收益进行了详细分解，并评估了电压/频率缩放和体偏置（body biasing）对流片硅片的影响。我们的设计在灵活性和效率两方面均超越了已发表的浮点单元（FPU）设计的现有最优水平。

符号 指数（无符号） 尾数（无符号二进制小数）
<table><tr><td>s</td><td colspan="4">exp</td><td colspan="8"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>mant</td><td>•</td><td></td><td></td><td></td></tr><tr><td colspan="3">←1→←</td><td>Wexp</td><td></td><td>&gt;&lt;</td><td></td><td></td><td>-Wmant</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">格式宽度：</td><td> $w _ { f } = 1 + w e x p + w _ { m a n t }$ </td><td></td><td></td><td></td><td></td><td>! exp = 0 ⇒ v = (−1)s × 2(1−bias) × 0.mant</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">指数偏置：</td><td></td><td> $b i a s = 2 ^ { \left( w e x p - 1 \right) } - 1$ </td><td></td><td></td><td></td><td>! exp = 2ωexp − 1 ∧ mant = 0 ⇒ v = (−1)s × ∞</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td colspan="7">编码值： v = (−1)s × 2(exp−bias) × 1.mant ! exp = 2ωexp − 1 ∧ mant ≠ 0 ⇒ v = NaN</td><td></td><td></td><td></td></tr></table>

图 1. IEEE 754 规定的 FP 格式编码及其解释。

本文其余部分的组织如下。第II节深入描述所提出的TP-FPU的需求与架构。第III节概述我们在transprecision ISA扩展方面的工作、将硬件单元集成到两个处理器核中的实现，以及最终的流片实现。第IV节给出在RI5CY核系统上进行的transprecision案例研究，以及Ariane核系统的硅片测量结果。本文最后几节将我们的工作与相关研究进行对比，并给出总结。

## II. 架构

FPnew是一个灵活的开源硬件IP块，遵循IEEE 754标准原则，采用SystemVerilog编写。其目标是为广泛的可能系统提供浮点（FP）能力，例如通用处理器核以及领域专用加速器。

## A. 需求

为了满足众多可能的目标系统、应用和技术的需求，可配置性是FPnew开发过程中的核心驱动因素之一。与现有设计的易于集成性以及利用目标特定工具流程的可能性，也是该设计的指导原则。我们在设计该单元时考虑了以下几项关键需求：

1) FP格式编码：如第I节所述，在广泛的系统中引入定制FP格式（通常窄于32位）正变得越来越有吸引力。虽然前文提到的许多系统为了追求性能或电路复杂度方面的优化而放弃了标准合规性，但通用处理器通常必须遵循IEEE 754标准。因此，TP-FPU被设计为支持任意数量的任意FP格式（就位宽而言），这些格式均遵循IEEE 754-2008二进制格式的原则，如图1所示。

2) 运算：为了提供完整的FPU解决方案，我们旨在提供IEEE 754所规定的通用运算，支持算术运算、比较和转换。尤其值得注意的是，最早于1990年被纳入处理器的FMA运算 [13] 随后被加入IEEE 754-2008标准，如今在高效的AI和BLAS类型内核中无处不在。该运算仅通过一次最终舍入步骤即可计算 $( a \times b ) + c$。我们旨在原生支持至少RISC-V ISA中规定的所有FP运算。

此外，对于支持多于一种FP格式的实现，还需要在所有支持的FP格式与整数之间进行转换。非标准的多格式算术也正变得越来越普遍，例如在张量加速器 [5], [6] 中执行FMA运算时，乘法和累加使用两种不同的格式。

3) SIMD向量：如今，大多数通用计算平台都提供SIMD加速扩展，即将多个窄操作数打包进宽数据通路以提高吞吐量。虽然可以通过将整个窄位宽FPU复制为多个向量通道来构建这种向量化的宽数据通路，但运算将被限制为使用相同的窄位宽。FP类型之间的灵活转换对于transprecision应用中高效的即时精度调整至关重要 [14]，并且需要对向量化数据的支持。因此，TP-FPU的架构必须能够支持这种向量化，以支持对SIMD向量的多格式运算。

4) 可变流水线深度：为了实现高性能和高工作频率，FPU内部常用的运算需要进行流水线化。然而，FP运算的流水线延迟需求在很大程度上依赖于系统架构和实现技术的选择。例如，GPU通常倾向于最小面积实现，并能够通过其架构很好地隐藏较大的延迟；而在嵌入式通用处理器核中，运算延迟的影响则可能显著得多 [14]。

因此，TP-FPU不得依赖硬编码的特定流水线深度，以支持尽可能广泛的应用场景。由于电路复杂度因运算和FP格式的不同而差异显著，每项运算和格式的寄存器数量均应可独立配置。

5) 设计工具流程：为了确保与现有设计流程的可访问性和互操作性，TPnew采用行业标准SystemVerilog HDL（IEEE 1800）编写，所有常用设计流程均支持该语言。诸如Chisel [15] 之类的新型硬件构造语言未被纳入考虑，因为缺乏原生工具支持会在标准和验证流程方面显著增加复杂度。

面向特定目标的综合流程（例如针对专用集成电路（ASIC）或现场可编程门阵列（FPGA）技术的流程）在可用的优化模块上有所不同，其更倾向于可推导的运算符而非直接实例化。综合工具会为算术原语选择最优实现，例如FPGA中的DSP切片，或在高时序压力下为ASIC选择基于Wallace树的乘法器。由于不同目标之间可用的优化也有所不同，该单元的描述方式使其能够在可能的地方启用自动优化，包括时钟门控和流水线化。

## B. 基本构件

在下文中，我们对图 2 所示的 TP-FPU 进行总体架构描述。针对处理器核集成以及在硅片上实现的具体配置将在第 III 节和第 IV 节中讨论。

1) FPU 顶层：在 TP-FPU 的顶层（见图 2-1），每个时钟周期最多可有三个 FP 操作数进入该单元，同时还有确定操作类型以及所涉及格式的控制信号。一个 FP 结果连同当前操作根据 IEEE 754-2008 产生的状态标志位一起离开该单元。输入和输出操作数的宽度是参数化的，以下将其称为单元宽度（$\left( \mathrm { w } _ { \mathrm { f p u } } \right)$）。

输入操作数被路由到四个操作组块之一，每个操作组块专门负责一类指令。仲裁器将操作组的输出馈送到该单元的输出端。由于在任意给定时钟周期内只有一个操作组块可以接收新数据，因此可以采用时钟门控和数据通路门控来静默 FPU 中未使用的分支，从而消除虚假的开关活动。

2) 操作组块：构成 TP-FPU 的四个操作组块如下。

1) ADDMUL：加法、乘法和 FMA。

2) DIVSQRT：除法和平方根。

3) COMP：比较和位操作。

4) CONV：FP 格式之间以及与整数之间的转换。

这些块中的每一个都构成供操作流经的独立数据通路（见图 2-2）。当单元中存在多种 FP 格式时，这些块可以容纳多个格式切片，这些切片既可以实现为格式专属（并行）的，也可以实现为多格式（合并）的。在并行情况下，每个切片只承载一种 FP 格式，在路径延迟和延迟方面提供了最大的灵活性，因为每个切片可以包含其内部流水线。虽然这种复制会增加电路面积，但在先进工艺的 ASIC 技术中面积成本很低，它提供了高效实现每种格式的灵活性。非活动的格式切片可以被时钟门控而静默。相比之下，合并切片可以通过共享硬件、承载多种格式来降低总面dolgo成本，但灵活性降低。此外，合并可能会带来能耗和延迟开销，因为小格式会复用同一过度设计的数据通路，且所有格式共享相同的流水线深度。

3) 格式切片：格式切片承载功能单元，这些功能单元执行该块所专门负责的操作。

如果某种格式可以打包到单元宽度中（$( \mathsf { w } _ { \mathrm { f p u } } ~ \geq ~ 2 ~ \times ~ \mathsf { w } _ { f } )$），则可以创建 SIMD 向量数据通路。在这种情况下，切片将承载多个向量通道，记为 lane[1]...lane[k]。

在并行情况下 [见图 2-3(a)]，这些通道是同一功能单元的重复实例，其数量和宽度确定如下

$$
\begin{array} { r } { \mathrm { { k } } _ { \mathrm { p a r a l l e l } } = \left\lfloor \frac { \mathrm { W f p u } } { \mathrm { W } _ { f } } \right\rfloor } \\ { \mathrm { w } _ { \mathrm { l a n e , p a r a l l e l } } = \mathrm { w } _ { f } . } \end{array}
$$

在合并情况下 [见图 2-3(b)]，通道总数由所支持的最小格式决定，而每个通道的宽度取决于所包含的格式。合并切片内的各个通道具有位宽各不相同的特殊性质，并且每个通道需要支持不同的格式集合 [见图 2-4(b)]

$$
\begin{array} { r } { \displaystyle \operatorname { k } _ { \mathrm { m e r g e d } } = \left\lfloor \frac { \operatorname { W } _ { \mathrm { f p u } } } { \operatorname { M i n } } \right\rfloor } \\ { \displaystyle \operatorname { W i n e r g e d } = \frac { \operatorname* { m a x } } { \operatorname { w f o r m a t } \operatorname { s l i c e } } \mathbb { W } _ { f } \Big \rfloor _ { \mathbb { W } _ { \mathrm { f } } \le \frac { \operatorname { w } _ { \mathrm { f p u } } } { i } } . } \end{array}
$$

根据当前操作是标量操作还是向量操作，将使用一个或多个通道来计算切片的结果，而未使用的通道则被静默。CONV 块中的合并切片需要更复杂的数据分发和收集方案来处理 SIMD 向量，因为输入和输出格式的宽度可能不同。此外，可以将两个标量 FP 操作数进行转换并作为向量的元素插入，从而在运行时动态创建向量。如果禁用 SIMD，则每个切片只有一个通道。

![](images/6d90228bbe42e0d4ce69d481fe9d9b6362fc5581b0e7cfc0a4672b711decd75d.jpg)  
图 2. TP-FPU 的数据通路框图及其底层层次结构。它支持多种 FP 格式，且数据通路宽度可配置。

4) 功能单元：切片内的功能单元既可以采用全流水线实现，也可以采用阻塞式（例如迭代式）实现。

ADDMUL 块使用符合 IEEE 754-2008 的全流水线 FMA 单元，采用单路径架构 [13]、[16] 实现，可在 1/2 ulp 内提供结果。合并切片中使用 FMA 单元的多格式版本，支持使用多种格式的混合精度操作。即，乘法以源格式 src_fmt 完成，而加法以目标格式 dst_fmt 完成，与 C 风格的函数原型 dst_fmt fma(src_fmt, src_fmt, dst_fmt) 相匹配。

在 DIVSQRT 块中，除法和平方根使用迭代式不恢复余数除法器计算。该单元的迭代部分每时钟周期计算结果值的三个尾数位，并以合并切片的形式实现。迭代的次数可以被覆盖为少于正确舍入结果所需的次数，以在跨精度计算中用精度换取吞吐量。COMP 块中的操作单元由一个比较器以及附加的选择和位操作逻辑组成，用于执行比较、符号操作以及 FP 分类。最后，CONV 块具有多格式转换单元，可以在任意两种 FP 格式之间、从整数到 FP 格式以及从 FP 格式到整数格式之间进行转换。

5) 单元控制流：虽然每个周期只有一个操作可以进入 FPU，但来自不同延迟路径的多个值可能在同一时钟周期到达切片输出端。来自所有切片和块的结果数据通过公平的轮询仲裁进行合并。为了暂停内部流水线，在内部层次结构以及该单元的外部接口上均使用简单的同步 valid-ready 握手协议。

由于该单元大量使用握手机制，数据可以在不需要预先知道操作延迟的情况下流经 FPU。因此，基于握手信号的细粒度时钟门控可以在各个流水线级内部进行，静默未使用的部分，并通过允许数据追上流水线中被暂停的头部来“挤出”流水线气泡。如果流水线中不存在有效数据，则可以使用粗粒度时钟门控来禁用操作组或整个 TP-FPU。

## C. 配置、参数化与使用

TP-FPU 通过 SystemVerilog 包和实例参数提供了充足的可配置性。在包中，可以定义任意数量的自定义格式，包含任意数量的指数位和尾数位，不限于传统计算系统中的 2 的幂次格式宽度。<sup>1</sup> 格式按照 IEEE 754-2008（见图 1）处理，并支持所有标准舍入模式。通过参数，已定义格式上的运算可以实现为并行切片或合并切片，也可以完全禁用。此外，可以为所有满足 $\ w _ { f } ~ \leq ~ \mathrm { w f p u } / 2$ 的格式全局启用 SIMD 向量，并且 w<sub>fpu</sub> 可以选择得远宽于所支持的最大格式（例如用于向量加速器）。同样可参数化的是每种格式和运算组的流水线级数，其中合并切片的所有格式共享一条流水线。对于给定的目标技术，可能需要综合工具的重定时（retiming）功能来优化这些寄存器的布局。

所有 TP-FPU 实例参数在综合时固定到硬件中，可以从同一个 FPnew RTL 无需修改地创建多个不同的实例。事实上，本工作中的所有实现都是通过默认包进行参数化的。由于该单元是开源的，设计者可以自由地提取或替换单元的功能，或添加他们自己的更多运算组。

## III. 在 RISC-V 核中集成 FPNEW

TP-FPU 已被集成到多个设计中，本工作重点介绍其在 RISC-V 处理器核内的实现。为了在 RISC-V 平台上发挥转精度计算的优势，我们通过特殊指令对 ISA 进行了扩展。我们将该单元集成到了 RI5CY（一个 32 位低功耗核）和 Ariane（一个 64 位应用级处理器）中。

## A. ISA 扩展

RISC-V ISA 为自定义运算的扩展提供了充足的机会。因此，我们添加了非标准 FP 格式和指令，以实现转精度计算，并在通用处理器中充分发挥 TP-FPU 的能力。

1) FP 格式：除了 RISC-V "F" 和 "D" 标准扩展分别包含的 IEEE 754 binary32 和 binary64 格式之外，我们还提供了 [10] 中提出的位数小于 32 的格式。我们实现中可用的 FP 格式包括：

1) binary64 (FP64)：IEEE 754 双精度 (11, 52)；
2) binary32 (FP32)：IEEE 754 单精度 (8, 23)；
3) binary16 (FP16)：IEEE 754 半精度 (5, 10)；
4) binary16alt (FP16alt)：自定义半精度 (8, 7)<sup>2</sup>
5) binary8 (FP8)：自定义四分之一精度 minifloat (5, 2)。所有这些格式中的数据都按照标准 RISC-V FP 格式的方式处理，包括对 denormal（非规格化数）、NaN 以及宽 FP 寄存器内窄值的 NaN-boxing 的支持。

2) 运算：新运算大致可分为三个部分，即标量、向量与辅助扩展 [8]。

a) 标量指令：标量扩展将 "F" 标准扩展中的所有运算（如算术运算、比较、转换和数据移动）映射到新引入的格式上。我们添加了所有支持的 FP 类型之间的转换，以便在转精度应用中实现高效的运行时精度缩放。

b) 向量指令：我们在所有窄于 FP 寄存器堆大小（RISC-V 术语中的 FLEN）的 FP 格式上添加 SIMD 能力。因此，在一个最高支持到 FP32（FLEN 为 32）的核中，可以打包 2 × FP16 的向量。我们的 ISA 扩展包含所有标量指令的向量版本。此外，我们还添加了这些运算的向量-标量版本，其中第二个操作数为标量。该标量操作数会被复制到输入向量的所有元素，例如，这使得 SIMD 矩阵乘法计算无需在内存中对矩阵进行转置。

由于两种格式的向量长度可能不同，向量之间的转换需要特别处理。在 FLEN 为 32 的系统中，将 2 × FP16 的向量转换为 FP8 只会得到 4 元素（32 位）目的操作数中的两个元素（16 位）。反过来，将 4 × FP8 的向量转换为 FP16 会产生 64 位的结果，无法放入寄存器。因此，我们提供了单独的指令来使用向量的低半部分或高半部分进行向量转换，从而在转精度应用中实现灵活的精度缩放。

c) 辅助指令：一些面向转精度计算系统需求的非标准运算完善了我们的 ISA 扩展。例如，我们添加了一个扩展（expanding）FMA 运算，其在比乘法更大的 FP 格式上执行加法，映射到 TP-FPU 架构的合并 FMA 切片的多精度运算。我们的 cast-and-pack 指令使用向量转换硬件转换两个标量操作数，随后将它们打包为结果向量的元素。

3) 编码：这些新指令的编码是作为 RISC-V brown-field（棕地）非标准 ISA 扩展实现的。<sup>3</sup> 由于标量指令仅引入新格式，因此标准 RISC-V FP 指令的编码被复用并加以调整。我们使用一个保留的格式编码来表示 FP16 格式，并复用四倍精度标准扩展 "Q" 中的编码来表示 FP8，因为我们并不针对任何能够提供 128 位 FP 运算的 RISC-V 处理器。FP16alt 上的运算被编码为带有指令字中保留舍入模式的 FP16。向量扩展利用整数操作码空间中大量未使用的空间，类似于为 RI5CY 核实现的 DSP 扩展的编码方式 [11]。辅助指令根据其操作对象是标量还是向量，分别编码在未使用的 FP 或整数操作码空间中。

4) 编译器支持：程序员需要计算机体系结构中新特性的高级语言支持才能高效地使用它们。因此，上述格式和运算已被添加到 RISC-V GCC 编译器工具链中，以便在用户程序中为转精度运算提供原生支持 [8]。自定义格式可以像 C/C++ 编程语言中熟悉的原生 FP 类型一样使用，例如，新的 float8 C 类型表示一个 FP8 变量。

## B. 搭载 Transprecision FPU 的 RI5CY

RI5CY 是一款开源的 32 位、四级流水线、顺序执行的 RISC-V RV32IMFC 处理器。<sup>4</sup> 这个小型核心面向嵌入式与 DSP 应用，包含若干为提升性能、代码密度和能效而设计的自定义非标准 RISC-V 扩展 [11]。借助这个核心，我们希望在一个已被业界广泛采用的<sup>5</sup>低功耗 MCU 级开源 RISC-V 核心中展示非标准 transprecision 运算。

1) ISA 扩展支持：RI5CY 支持 RISC-V “F” 标准 ISA 扩展，该扩展要求包含 32 个 32 位 FP 寄存器。该核心提供了省略 FP 寄存器、将 FP 数据存放于通用寄存器堆中的选项，以节省面积并减少数据搬运。<sup>6</sup>

表 I
配置到 RI5CY 核心中的 TP-FPU 配置情况。FPU 位宽为 $\mathrm { \Delta \ w f { p u } } = 3 2 \mathrm { b i t }$
<table><tr><td>格式</td><td colspan="4">实现方式（周期数，通道数）</td></tr><tr><td></td><td>ADDMUL</td><td>DIVSQRT</td><td>COMP</td><td>CONV</td></tr><tr><td>FP32</td><td>合并式 (1,1)</td><td>禁用  $( - , - )$ </td><td>并行式 (1,1)</td><td>合并式 (1,2)</td></tr><tr><td>FP16</td><td>合并式 (1,2)</td><td>禁用 (-,-)</td><td>并行式 (1,2)</td><td>合并式 (1,2)</td></tr><tr><td>FP16alt</td><td>合并式 (1,2)</td><td>禁用 (-,-)</td><td>并行式 (1,2)</td><td>合并式 (1,2)</td></tr></table>

我们增加了对 FP16 与 FP16alt 运算的支持，包括 packed SIMD 向量。通过将 FP 值复用于通用寄存器堆，我们可以借助自定义 DSP 扩展中整数数据通路上已有的 SIMD 数据重排（shuffling）功能。该核心还加入了 cast-and-pack 以及扩展 FMA 的支持。

2) 核心修改：为处理这些新指令，我们对处理器的译码器进行了扩展，加入了相应的指令编码。RISC-V 要求对较窄的 FP 值进行所谓的 NaN-boxing，即 FP 寄存器中所有未使用的高位必须置为逻辑高电平。我们扩展了核心的 load/store 单元，通过修改原有的符号扩展电路，实现对标量窄 FP 数据的一扩展（one-extending）。不过，我们并未在运算单元中强制检查 NaN-boxing，以便在需要时将 SIMD 数据当作标量处理。除了将 RI5CY 的 FP32 FPU 替换为 TP-FPU 之外，相较于 “F” 扩展已有的基础设施，核心本身的改动并不算大。

3) FPU 配置：如表 I 所示，我们在不增加任何额外流水级的情况下启用对上述格式的支持。低功耗 MCU 通常瞄准相对宽松的时钟目标，因此 FP 运算可以在单个周期内完成。由于 XLEN 为 32 且 FP 运算使用通用寄存器堆，$\mathtt { w _ { f p u } }$ 被设为 32 位。

ADDMUL 模块采用合并式多格式（multiformat）切片实现，以支持 FP16[alt] 与 FP32 之间的多格式运算。DIVSQRT 模块被禁用，因为我们在案例研究中并未用到它，同时也是为了展示 TP-FPU 的细粒度可配置性。CONV 模块在合并式切片中使用两个 32 位通道，以实现对两个 FP32 操作数的 cast-and-pack 操作。

## C. 搭载 Transprecision FPU 的 Ariane

Ariane 是一款开源的 64 位、六级流水线、部分乱序（partially in-order）的 RISC-V RV64GC 处理器。<sup>7</sup> 它具备运行操作系统的完整硬件支持，并配有私有的指令与数据 cache。为加速顺序代码的执行，它还配备了返回地址栈（return address stack）、分支历史表（branch history table）和分支目标缓冲器（branch target buffer）[12]。我们的目标是借助该核心将完整的 transprecision 计算系统推向硅片实现，支持支持多种格式的能量比例（energy-proportional）计算。

1) ISA 扩展支持：Ariane 支持 RISC-V “F” 和 $\mathrm { ^ { 6 6 } D ^ { 3 } }$ 标准 ISA 扩展，这使得该核心的 FP 寄存器堆为 64 位宽。我们增加了对 FP16、FP16alt 和 FP8 运算的支持，以及包括 FP32 在内的所有这些格式的 SIMD 运算。我们支持灵活的 cast-and-pack 操作，但该版本的核心未配备扩展 FMA 操作。

表 II
配置到 Ariane 核心中的 TP-FPU 配置情况。FPU 位宽为 $\mathrm { \Delta w f p u } = 6 4 \mathrm { b i t }$
<table><tr><td rowspan="2">格式</td><td colspan="4">实现方式（周期数，通道数）</td></tr><tr><td>ADDMUL</td><td>DIVSQRT</td><td>COMP</td><td>CONV</td></tr><tr><td>FP64</td><td>并行式 (4,1)</td><td>合并式 (21,1*)</td><td>并行式 (1,1)</td><td>合并式 (2,2*)</td></tr><tr><td>FP32</td><td>并行式 (3,2)</td><td>合并式 (11,0)</td><td>并行式 (1,2)</td><td>合并式 (2,0)</td></tr><tr><td>FP16</td><td>并行式 (3,4)</td><td>合并式 (7,0)</td><td>并行式 (1,4)</td><td>合并式 (2,2*)</td></tr><tr><td>FP16alt</td><td>并行式 (3,4)</td><td>合并式 (6,0)</td><td>并行式 (1,4)</td><td>合并式 (2,0)</td></tr><tr><td>FP8</td><td>并行式 (2,8)</td><td>合并式 (4,0)</td><td>并行式 (1,8)</td><td>合并式 (2,4)</td></tr></table>

合并式通道，支持所有等宽及更窄的格式。

2) 核心修改：我们用我们的设计替换了该核心的 FPU，扩展了处理器译码器以支持新的操作，并扩展了核心的 load/store 电路，同样实现对较窄 FP 数据的一扩展，以完成正确的 NaN-boxing。这些额外的核心控制电路改动并非时序关键路径，其开销相对于核心其余部分的资源而言可以忽略不计。

3) FPU 配置：我们将 TP-FPU 配置为包含上述格式，并如表 II 所示加入与格式相关的流水线深度。流水寄存器的数量设置以使处理器核心能够达到约 1 GHz 的时钟频率为目标。$\mathtt { w _ { f p u } }$ 被设为 FP 寄存器堆的宽度 64 位，因此 FP64 格式不存在 SIMD 向量。

我们选择采用并行式实现的 ADDMUL 模块，以使不同格式运算的延迟各不相同，避免窄 FP 格式承受不必要的能量与延迟开销。由于所用除法器硬件的迭代特性，DIVSQRT 的延迟与格式相关；为节省面积，该操作不适用于 SIMD 数据。除法器每个时钟周期产生三个尾数位，外加用于前置和后处理的固定三个周期。除法运算需要 4（FP8）到 21（FP64）个周期，鉴于除法和平方根运算在性能优化的代码中相对罕见，这一延迟是可以接受的。转换模块再次采用合并式切片实现，其中两个 64 位宽的通道用于对两个 FP64 值执行 cast-and-pack 操作。此外，还分别设有两个 16 位通道和四个 8 位通道，以覆盖所有可能的转换。

## IV. 实现结果

## A. PULPissimo：搭载 Transprecision FPU 的 RI5CY

为了在支持 TP 的 RI5CY 核心上对应用进行基准测试，我们对一个包含该核心的完整平台进行了全面布局布线实现。本节展示实现结果，而第 IV-C 节则展示在已实现设计上的应用案例研究。

1) 实现：我们利用 PULPissimo<sup>8</sup> 来实现一个完整系统。PULPissimo 是一个基于 RI5CY 核心的单核 SoC 平台，包含 512 kB 的存储器以及众多标准外设，例如 UART、

![](images/4494004d09a481386945fccb9ada10b46eff5d767a66033e06ffa76bf642bb90.jpg)  
Fig. 3. 使用 RI5CY 核心与 TP-FPU 的 PULPissimo SoC（不含焊盘框架）的版图规划与框图。

![](images/5e240019b12d46adfeac90ab30ac1ece1fe18bd2525fdff28ebc445d168790ee.jpg)  
Fig. 4. PULPissimo SoC 与 RI5CY 核心的面积分布（单位 kGE，$1 \mathrm { G E } \approx 0 . 1 9 9 \mu \mathrm { m } ^ { 2 } )$

I<sup>2</sup>C 和 SPI。我们使用第 III-B 节所述的扩展 RI5CY 核心，包括表 I 所示的单周期 TP-FPU 配置。

该系统已在 GLOBALFOUNDRIES 22FDX 工艺（一种 22 nm FD-SOI 节点）中完成全面的综合、布局和布线，采用低压下的低阈值 8 轨单元库。整个 SoC（不含 I/O 焊盘）的最终版图如图 3 所示。综合与布局布线分别使用 Synopsys Design Compiler 和 Cadence Innovus 完成，采用最差情况低压约束（SSG，0.59 V，125 ◦C），最终设计目标频率为 150 MHz，综合时额外加入 20% 的时钟不确定度。在标称低压条件下（TT，0.65 V，25 ◦C），系统运行频率为 370 MHz。该设计的关键路径位于存储器与核心之间，涉及 SoC 互连。

2) TP-FPU 的影响：搭载 TP-FPU 的 RI5CY 核心总面积为 147 kGE，其中 FPU 占据 69 kGE（47%），而包含存储器在内的整个 PULPissimo 系统为 5.1 MGE，参见图 4。承载所有格式合并多格式 FMA 单元的 ADDMUL 模块占据了 FPU 面积的 76%，而 COMP 与 CONV 模块分别占 4% 和 18%。

与仅支持 FP32 的标准 RI5CY 核心相比，面积增加 29%，静态能耗增加 37%。能耗增幅高于面积增幅的原因在于，FPU 相比处理器其余部分使用了相对更多的高驱动、短栅极单元：虽然 TP-FPU 并非时序关键路径，但由于单周期单元中较长的路径，其相比核心其余部分多使用了 2.4 倍的 20 nm 晶体管（相对于 28 nm 而言）。在系统层面，所增加的面积与静态能耗仅分别占 0.7% 和 0.9%，这是由于存储器占了系统面积和漏电的 92% 和 96%。

![](images/543257c357148523a27985f40ac667204ed0600a781e36277eb591540134688c.jpg)  
Fig. 5. 整个 Kosmodrom 芯片的显微照片与框图，展示了两个搭载 TP-FPU 的 Ariane 核心宏的位置。

从每操作能耗的角度来看，将 FP32 FMA 指令与 RI5CY 中可用的 32 位整数乘累加（MAC）指令进行比较颇有意义。在 370 MHz 的标称低压条件下，这些浮点与整数指令在各自执行单元中的平均能耗分别为 3.9 pJ 和 1.0 pJ。考虑系统级能耗，在运行滤波内核（见第 IV-C 节）时，处理 FP32 数据平均每周期消耗 22.2 pJ，而整数变体则需要 21.2 pJ，二者性能相当。这些在系统层面面积以及静态与动态能耗上的微小差异表明，即使在 MCU 语境下，浮点计算也是可负担的。

## B. Kosmodrom：搭载 Transprecision FPU 的 Ariane

我们实现了一个搭载 TP Ariane 核心的完整测试系统并流片，并对 FP 指令的每操作能效进行了详细分析。

1) 硅实现：我们在 GLOBALFOUNDRIES 22FDX 工艺中实现了一个名为 Kosmodrom 的完整测试系统 [17]。图 5 包含硅片显微照片以及设计中主要模块的架构概览。两个功能相同、搭载 TP-FPU 的 Ariane 核心采用不同的单元工艺和目标频率制造，并共享 1.25 MB 的 L2 存储器以及中断和调试基础设施等公共外设。我们支持五种 FP 格式，每种格式均有专用数据通路，并利用了表 II 所示的各格式特定延迟。

综合使用 Synopsys Design Compiler，其中较快的、高性能核心采用低阈值 8 轨单元库，而较慢的低功耗核心则采用 7.5 轨单元库。由于两个核心可独立进行时钟控制与供电，后续的性能与能效分析将仅聚焦于高性能核心。综合时设置了 1-GHz 的最差情况约束（SSG，0.72 V，125 ◦C）以及 20% 的时钟不确定度。我们在综合过程中广泛使用了自动时钟门控插入（FPU 中超过 96% 的寄存器被门控）。未被门控的寄存器仅包括握手令牌以及控制除法与平方根的有限状态机。整个 FPU 中流水线寄存器的位置通过综合工具的寄存器重定时功能进行了优化。

布局布线使用 Cadence Innovus 完成，采用 1 GHz 约束的多模式多角流程，涵盖全部温度与掩模对准偏差角，共八个。作为角部修剪评估的一部分，我们在最差情况角和典型角中假设采用前向偏置电压，以缓解关键路径的压力并降低漏电。最终的后端设计在最差情况条件下（SSG，0.72 V，0.8 V 偏置，40/125 ◦C）达到 0.96 GHz，在标称条件下（TT，0.8 V，0.45 V 偏置，$2 5 ~ ^ { \circ } \mathrm { C } ) .$）达到 1.29 GHz，在最佳情况条件下（FFG，0.88 V，0 V 偏置，40/125 ◦C）达到 1.76 GHz。

![](images/b8de613683eb42b1c9317afc0db8f813fd63de2c8e39b10b4772271f14cc6114.jpg)  
Fig. 6. 整个 Ariane RISC-V 核心的面积分布（不含缓存存储器）（单位 kGE，1 GE 0.199 µm<sup>2</sup>）。

我们还对逻辑单元组合（阈值电压与晶体管沟道长度）进行了大量探索，以最大化能效。该设计包含 74% 的 LVT 和 26% 的 SLVT 单元；以及 86% 的 28 nm、8% 的 24 nm 和 6% 的 20 nm 晶体管。

2) TP-FPU 的静态影响：搭载 TP-FPU 的 Ariane 核心总面积为 742 kGE（包含缓存则为 1.4 MGE）。面积明细如图 6 所示。FPU 的总规模为 247 kGE，其中 160 kGE 为各类 FMA 单元，13 kGE 为比较与位操作电路，19 kGE 用于迭代式除法与平方根单元，47 kGE 用于转换单元。与仅支持标量 FP32 与 FP64（"F" 与 "D" 扩展）的完整 Ariane 核心（包含缓存）相比，面积与静态能耗分别增加 9.3% 和 11.1%。考虑到新增了三种格式的 FP 运算以及 SIMD 支持（使用 FP8 向量时可将 FP 运算吞吐量提升最多 8 倍），处理器中所增加的面积与能耗开销是适中的。

## 3) 芯片实测：

a) 评估方法：我们通过在 Ariane 上使用合成应用对各个运算分别进行压力测试，提取了 FPU 内部能耗的详细分解情况。每条指令都输入随机分布的正常 FP 值，并加以约束使运算不会遇到溢出，从而通过在 TP-FPU 的数据通路内部提供高翻转活性，构造出功耗方面的最坏情况。测量在流水线充满、FPU 以峰值性能运行的条件下进行，以提供公平的比较。

内核与存储器功耗的芯片实测是在处理器和 FPU 执行矩阵-矩阵乘法时完成的。利用经过校准的布局布线后仿真（具备完整的层次可见性），我们得以确定各个硬件模块的相对能耗贡献。布局后功耗仿真使用典型工艺角库在标称条件下（TT，VDD = 0.8 V，25 ◦C）进行。芯片实测在无偏置标称条件（0.8 V，0 V 偏置，25 ◦C）下进行，此时可达到 923 MHz，除非另有说明。电压缩放对性能与能效的影响通过实测流片后的芯片获得。

b) FPU 指令能效与性能：图 7 顶部展示了算术标量运算在 FPU 内的平均单指令<sup>9</sup>能耗。较小格式的能耗比例性（energy proportionality）在 ADDMUL 模块中尤为显著，这是由于乘法器的高影响（前三组柱状图）。例如，FP64 FMA 的 fmadd.d 消耗 26.7 pJ，而在 FP32 上执行相同运算可节省 65% 的能量。进一步缩减 FP 格式宽度时，与相邻更大格式相比，FP16、FP16alt 和 FP8 分别额外消耗降低 48%、54% 和 49% 的能量。使用 FP16alt 代替 FP16 在能耗上便宜 12%，这是因为 FP16 所需的尾数乘法器更小。类似地，缩减 FP 格式宽度后，与相邻更大格式相比的相对能耗收益分别为：FP 乘法在 FP32、FP16、FP16alt 和 FP8 下为 65%、47%、52%、47%；FP 加法为 53%、47%、57%、47%；FP 比较为 38%、34%、35%、22%。由此可见，较小格式上的标量运算在能耗上至少是直接成比例 advantageous 的。

![](images/abb01a4121dcc57ed75b45d837b40a5b5cbf247d589ab31e7213a7a78cf7898b.jpg)  
图 7. 全流水标量运算（左上）、向量运算（右上）的 FPU 单指令能耗，按 FMA、乘法、加法和比较分组；以及标量转换运算（左下）和向量转换运算（右下）。

直觉上，SIMD 指令在所有格式上的能耗应当非常接近，因为 64 位数据通路被完全利用。然而，我们发现向量运算也呈现出渐进的能耗比例性，进一步放大了节能效果。以 FP32 SIMD FMA vfmac.s 的 20.0 pJ 能耗为起点，FP16、FP16alt 和 FP8 相对相邻更大格式的单指令能耗收益分别为 20%、31% 和 20%。类似地，乘法为 21%、32% 和 21%，加法为 20%、31% 和 19%，比较为 14%、23% 和 8%。尽管数据通路被完全利用，使用更窄 FP 格式的打包（packed）运算仍提供了超比例的能耗收益，同时提高了单指令吞吐量。这种有利的缩放特性归功于各格式执行单元的分离式设计，其中空闲的切片（slice）被时钟门控关闭，而这在使用传统共享数据通路的方法中是较难实现的。若计入吞吐量的提升，SIMD FMA 相对相邻更大格式的单数据能耗收益达到 60%、66% 和 58%，优于直接比例关系。

在架构的 CONV 模块中，转换指令在所有格式间共享一个合并的切片，这是能耗缩放不那么显著的一个例子。图 7 底部展示了标量和向量转换的平均单指令能耗。指令的能耗同时受输入格式和目标格式的影响。

对于标量 FP-FP 转换，转换到更大的格式在能耗上更便宜，因为只有输入数据通路的一部分发生翻转，且输出尾数的大部分由常量零填充。转换到更小的格式则更昂贵，因为宽输入值会在转换单元内引起动态翻转以产生输出。为与上述缩放结果进行对比，我们比较输入和输出格式同时减半的转换，例如 fcvt.s.d、fcvt.h.s 和 fcvt.b.h。以 FP64/FP32 转换的 7.0 pJ 为起点，我们发现格式宽度减半时能耗仅分别降低 30% 和 35%。与上述能耗缩放结果相比，这种缩放效果较差，这是由于使用了单一合并单元，数据通路中未使用的部分更难被关闭。

对于 SIMD 向量，单指令能耗比例性的效果再次显现；从 FP32/FP16 转换到 FP16/FP8 转换便宜 9.5%。虽然不及向量 FMA 那样显著，这一收益来自用于转换小格式的额外向量通道更窄且支持的格式更少。

灵活的转换并打包（cast-and-pack）指令允许将两个 FP64 值转换并打包到目标向量的两个元素中，其能耗仅比执行一次从 FP64 到目标格式的标量转换高约 30%。值得注意的是，若没有这一功能，则需要两次标量转换外加一次额外的打包操作，而这些操作在 ISA 中并非直接可用。

对整数宽度固定的标量 FP-整数转换的测量同样显示出相对较小的收益，FP16alt/int32 相比 FP32/int32 转换最多仅便宜 25%，远差于直接比例关系。向量 FP-整数转换作用于与 FP 格式宽度相同的整数。此时，与其他格式共享向量通道的影响可能使得对多个窄值进行 SIMD 转换的能耗高于较大格式，例如 FP8/int8 转换，从而削弱了与并行切片相比的单数据能耗缩放。

在标称条件下，我们的 TP-FPU 在所支持的各格式上实现标量 FMA 的能耗为 2.5 至 26.7 pJ，SIMD FMA 的单数据项能耗为 1.6 至 10.0 pJ。FP-FP 转换对标量数据的能耗为 7.0 至 26.7 pJ，对向量数据为 2.2 至 4.9 pJ。我们将单元划分为并行切片的方法已被证明能够有效实现标量和 SIMD 数据的高能耗比例性。在芯片实测的 923 MHz 标称频率下，各格式的 FMA 对应的性能与能效分别为 1.85 Gflop/sW 至 14.83 Gflop/sW 和 75 Gflop/sW 至 1245 Gflop/sW。

c) 电压缩放的影响：图 8 展示了电压和频率缩放对流片后芯片的影响。我们测量了 0.425 至 1.2 V 供电电压下的最高可达频率及相应功耗。我们观测到的峰值算力与能效分别为：FP64 为 3.17 Gflop/s 和 178 Gflop/sW，FP32 为 6.33 Gflop/s 和 473 Gflop/sW，FP16 为 12.67 Gflop/s 和 1.18 Tflop/sW，FP16alt 为 12.67 Gflop/s 和 1.38 Tflop/sW，FP8 为 25.33 Gflop/s 和 2.95 Tflop/sW。

![](images/301c1670dd641475fb49613bb18dc1e69f93629aa971752c0f930b99ddae1d71.jpg)  
图 8. FP64 FMA 的算力与能效随供电电压的变化（左），通过调节供电电压和工作频率实现的算力与能效之间的权衡。在流片后的芯片上实测。

![](images/cc4906d526440f24318c651b9600898cd2931531d51a6042c60b5c6adef3509f.jpg)  
Fig. 9. Ariane 核心中各操作处理每项数据的能耗成本。

d) 核心级能效：由于 TP-FPU 仅仅是整个处理器系统的一部分，我们现在简要考察整个核心内执行操作所消耗的能量。图 9 展示了 Ariane 核心中执行各种操作的处理器模块的每数据项能耗。在 FP64 FMA——能耗最高的 FP 操作——期间，FPU 占 Ariane 核心总能耗的 39%，其能耗与访存操作的能耗相当。尽管得益于出色的能耗比例性（energy proportionality），FP8 FMA 相比同样操作在 FP64 上的 FPU 能耗低了 10.5 倍；但整体核心能耗仅降低了 38%。对于小型和嵌入式应用而言，标量 FPU 级别的节能可能已经足够，但对于诸如 Ariane 这类应用级核心，由于相对较大的核心侧开销，仅靠标量级节能无法通过跨精度计算带来能效上的最大收益。

采用 SIMD 向量化能够显著缓解这种核心开销对每数据项能耗成本的影响。例如，当 FP8 FMA 作为向量化 FMA 的一部分执行时，其所需的总核心能耗进一步降低 6.2 倍。

## C. 跨精度内核的性能与编程

为了直观展示跨精度应用所面临的挑战与收益，我们在启用跨精度功能的 RI5CY 核心上展示一个多格式应用内核。此外，我们还讨论在面向跨精度平台编程时需要考虑的因素。

1) 跨精度应用案例研究：我们考虑对两个输入流的元素逐项相乘后求和的累加操作，这种操作常见于信号处理或 SVM 等许多应用中。

float16 \*a, \*b; float16 \*a, \*b;   
float16 sum = 0; float sum = 0   
a) b)   
float16 \*a, \*b;   
float sum = 0;   
c)   
for (int i = 0; i < n; ++i)   
macex f16(sum, a[i], b[i]);  
Fig. 10. 对两个输入流 a 和 b 中的元素逐项乘积进行累加。输入为 FP16，结果在 采用 FP16 累加，在 中则采用 FP32。代码 c) 使用编译器内联函数来调用跨精度指令。

表 III
与图 11 中汇编代码对应的应用指标
<table><tr><td></td><td>正确位数</td><td>结果相对精确值的误差</td><td>相对精确 FP16</td><td>核心相对能耗</td><td>系统</td></tr><tr><td>精确结果</td><td>37</td><td>0.0</td><td></td><td></td><td></td></tr><tr><td>转换为 FP16</td><td>12</td><td> $1 . 9 \times 1 0 ^ { - 4 }$ </td><td>0.0</td><td></td><td></td></tr><tr><td>结果 11 a)</td><td>9</td><td> $2 . 7 \times 1 0 ^ { - 3 }$ </td><td> $2 . 9 \times 1 0 ^ { - 3 }$ </td><td>0.60</td><td>0.63</td></tr><tr><td>结果 11 b)</td><td>22</td><td> $2 . 0 \times 1 0 ^ { - 7 }$ </td><td>0.0</td><td>1.00</td><td>1.00</td></tr><tr><td>结果 11 c)</td><td>19</td><td> $1 . 6 \times 1 0 ^ { - 6 }$ </td><td>0.0</td><td>1.16</td><td>1.03</td></tr><tr><td>结果 11 d)</td><td>19</td><td> $1 . 6 \times 1 0 ^ { - 6 }$ </td><td>0.0</td><td>0.97</td><td>0.75</td></tr><tr><td>结果 11 e)</td><td>22</td><td> $2 . 0 \times 1 0 ^ { - 7 }$ </td><td>0.0</td><td>0.63</td><td>0.63</td></tr></table>

最终结果被转换为 FP16，并与转换为 FP16 后的精确结果进行比较

a) 方法：图 10 展示了与我们评估相关的负载的 C 语言表示。输入流以 FP16 值的形式存储在内存中，累加结果采用 FP16 或 FP32。我们利用我们的跨精度 ISA 扩展得到图 11 中的汇编代码，具体如下：图 11(a) 是图 10(a) 中仅使用 FP16 的负载，理想情况下每对输入需要三条指令。图 11(b) 全部在 FP32 上执行操作以获得最精确的结果，但由于需要类型转换，总共需要五条指令。图 11(c) 试图通过在 FP16 中执行乘法来节省能量，以用加法替代 FP32 FMA。图 11(d) 使用 SIMD 对前一段代码中的 FP16 部分进行加速，将指令数降至 3.5 条。图 11(e) 利用扩展的多格式 FMA 指令，将计算与转换合并，重新只需三条指令。

完整的应用会利用 RI5CY 中提供的零开销硬件循环（hardware loop）和后递增加载（post-incrementing load）指令，对全部输入数据重复上述操作。由于该负载数据强度较低，进一步的手动循环展开只能用于隐藏指令延迟开销。

b) 性能与能耗结果：我们在表 III 中汇总了这些程序的最终结果精度和能耗。能耗结果来自对第 IV-A 节中介绍的 RI5CY + TP-FPU 设计的布局后仿真。

图 11(a) 所得结果的精度相对较低，仅 9 位正确（约三个十进制数字），而该操作的精确结果需要 37 位精度。由于舍入误差的累积，该结果偏离了精确结果在 FP16 中可能的最精确表示（12 位正确）。图 11(b) 中的代码可提供 22 位精度，但由于执行时间增加以及 FP32 操作的每条指令能耗更高，核心级和系统级能耗分别增加了 66% 和 59%。图 11(c) 的精度有所下降（19 位），并且与 FP32 代码相比甚至需要 16% 更多的核心能耗（系统能耗多 3%），因为 FP16 乘法在能耗上远高于其替代的类型转换操作。与 FP32 情况相比，图 11(d) 中 SIMD 的使用使核心能耗降低 3%，系统总能耗降低 25%。在核心层面，性能提升带来的收益略微超过了 FPU 能耗的增加；而在系统层面，访存操作数量的减少带来了显著影响。图 11(e) 中使用扩展的乘加操作则兼得了两方面的优势：性能与朴素的仅 FP16 版本相同，而精度则与完全在 FP32 上执行时一样。转换为 FP16 后，其结果比仅使用 FP16 时精确 14<sup>.</sup>6 倍，与 FP32 情况相比，核心与系统功耗降低 37%。这些结果凸显了跨精度计算与灵活的硬件实现相结合时所具备的节能潜力。

2) 编译器支持：我们将低层跨精度指令以一组编译器内联函数（intrinsic function）的形式提供，使编译器和程序员能够充分利用跨精度硬件。标量类型和基本操作由编译器通过使用适当的类型（float16、float16alt、float8）和运算符（+、\* 等）透明地处理。自定义 FP 格式上的向量化操作则通过 GCC 向量扩展来推断。事实上，编译器可以从图 10(b) 的代码生成如图 11(d) 所示的程序。

然而，诸如 FMA 之类的操作，以及使用 SIMD 向量的优化访问与转换模式，往往无法干净地映射到高级编程语言的运算符和语义。即便在非跨精度代码中，性能优化的 FP 代码通常也需要低层的手工调优才能充分利用可用硬件，这一点十分普遍。我们可以且应当通过对内联函数的调用来利用诸如 cast-and-pack 或扩展 FMA 等无法推断的操作，如图 10(c) 所示，从而生成图 11(e) 的汇编代码。基于编译器的这一方法的优势与局限在 [8] 中有进一步的探讨。

## V. 相关工作

## A. 商用 ISA 中的 SIMD 与 Transprecision

Intel 的 x86-64 SSE/AVX 扩展在 FP32 和 FP64 上提供了非常宽的 SIMD 操作（AVX-512 中高达 512 位）。它们包含一条 FP 点积（dot-product）指令，可分别对 2× FP64 或 4× FP32 的向量进行操作，产生一个标量结果。目前尚不支持非标准 FP 格式，但未来支持 AVX-512 扩展的 CPU（Cooper Lake）将包含 BF16 格式（FP16alt），并支持 cast-and-pack（转换并打包）操作，以及对数值对的扩展 SIMD 点积操作。

ARM NEON 扩展可选地支持 FP16，并为 SIMD 操作提供了一个独立的寄存器堆，支持根据所使用的 FP 格式，通过不同的寻址视图进行寄存器融合。寻址方式隐含在指令内格式的使用中，从而能够对始终作用于整个寄存器的多格式（扩展、收缩）操作进行非常一致的处理。这种方式与我们的 ISA 扩展形成对比——后者需要多种编码来在向量转换过程中切分输入或输出向量。ARM ISA 的一个补充扩展是可伸缩向量扩展（SVE），它仅面向高性能 64 位架构，可将向量长度扩展到远超 128 位。SVE 包含对 BF16 作为存储格式的可选支持，在计算中使用时会隐式地将所有 BF16 输入数据转换为 FP32，并产生 FP32 结果。将 FP 数据转换为 BF16 用于存储也是可行的。

![](images/b4e908ea25231826328373ac67326b6c0786958ab153f0465fcd5c95ba259a42.jpg)  
Fig. 11. 图 10（灰色部分）中循环体两次迭代的 RISC-V 汇编实现。

面向 ARM M 系列处理器的一个新 ISA 扩展称为 MVE。它将 FP 寄存器堆重新配置为由八个 128 位向量寄存器组成的寄存器组，每个寄存器被划分为四个 32 位的 "beat"。虽然向量指令始终作用于整个向量寄存器（固定的 128 位向量长度），但实现可以自由选择每时钟周期计算一个、两个或全部四个 beat——这实质上允许在较低端硬件上串行化执行。该 ISA 扩展的浮点变体可以在 FP16 和 FP32 上操作，但不支持多格式操作。一种类似 MVE 精神的执行方案可以以极小的实现开销应用于使用我们 ISA 扩展的处理器。对于单条向量指令，发出四条或两条 SIMD FP 操作的序列，即可分别在 FP 寄存器宽度为 32 位和 64 位时复现单 beat 或双 beat 系统的行为。此外，MVE 还支持我们扩展中所没有的单个向量 lane 的谓词化（predication）、交错（interleaving）以及 scatter-gather 操作。

目前存在 RISC-V "V" 标准向量扩展的工作草案。"V" 扩展增加了一个独立的向量寄存器堆，具有 Cray 风格的向量操作语义和可变向量长度。多个寄存器也可以被融合，以有效地增加每条指令的向量长度。该标准向量扩展包含加宽（widening）和收窄（narrowing）操作，这些操作在操作的一端融合寄存器，从而无需对寄存器的各个部分进行寻址即可一致地处理数据。它支持 FP16、FP32、FP64 和 FP128，以及加宽 FMA 操作。这些操作的运行方式与我们对 fmacex 操作的实现相同，但限制是目标格式必须恰好是源格式的 2 倍宽。此外，还存在用于向量内积求和的归约（reduction）操作。

## B. 开源可配置 FPU 模块

大多数开源 FPU 设计都以固定格式实现特定功能，面向特定的系统或工艺<sup>10,11</sup>；不过也存在一些值得关注的可配置工作。

例如，FloPoCo [18] 是一个主要面向 FPGA 实现的 FP 函数生成器，以 VHDL 实体的形式生成各个函数。FP 格式可以在指数和尾数宽度方面自由配置；但所产生的硬件模块并不符合 IEEE 754 标准。具体而言，无穷大和 NaN 值不会编码在 FP 操作数本身中，并且作为换取 FloPoCo FP 格式中略高动态范围的折中，不支持次正规数（subnormal）。FMA 操作不可用。

另一方面，Hardfloat<sup>12</sup> 提供符合 IEEE 754 标准的参数化 FP 函数。它是一组用 Chisel 编写的硬件模块，具有参数化的 FP 格式，并包含 FMA 运算器。虽然 Chisel 在商用 EDA 工具流程中并未被广泛采用，但也提供了生成的标准 Verilog 版本。Hardfloat 在内部以一种非标准的重编码（recoded）表示来操作 FP 值。不过，其操作遵循 IEEE 754 标准，并且提供了到标准交换编码（即 TP-FPU 中所使用的编码）的转换模块。

这两个工作都只提供单独的功能模块，而非功能完整的 FPU。然而，得益于 TP-FPU 的层次化架构，用外部库中的实现替换其功能单元将是很容易的。

## C. 面向 RISC-V 的 FPU

IEEE 754 在所谓实现定义行为方面存在一些模糊之处，导致了 FP 代码在不同软件和硬件平台间的可移植性和可复现性问题。FP 行为可能因所使用的处理器型号和编译器版本而有很大差异。为了至少避免与硬件相关的问题，RISC-V 精确规定了 IEEE 754 中开放点应如何实现，包括 NaN 结果的精确位模式以及数值何时被舍入。因此，用于 RISC-V 处理器的 FPU（例如本工作）通常在其行为上是保持一致的。

源自 UCB 的 RISC-V 核心（Rocket 和 BOOM [19]、[20]）中所使用的 FPU 基于以针对 RISC-V 的特定配置的 Hardfloat 组件。

Kaiser 等人 [21] 发表了在 FP64 上 FMA 操作的 RISC-V 专用实现，采用了与本工作相同的 GLOBALFOUNDRIES 22FDX 工艺。我们在本节末尾将我们的 TP-FPU 与他们的实现及其他实现进行比较。

## D. 新型算术/超精度浮点加速器

近年来，非标准浮点系统日益流行，这主要是由主流机器学习算法的需求所驱动的。

例如，Google TPU [5] 和 NVIDIA 的 Tensor Cores [6] 都针对缩减精度（reduced-precision）浮点格式上的优化运算提供了高吞吐量，以实现快速的神经网络推理。二者均提供一种称为 bfloat16（BF16）的自定义格式，该格式与本工作中 FP16alt 使用的编码相同。后者此外还支持 FP16 以及一种名为 TensorFloat32 的全新 19 位浮点格式，该格式由 FP32 输入值的 19 个最高有效位构成，并产生 FP32 的结果。然而，这两种架构都省略了标准中规定的某些特性，例如非规格化数（denormal numbers）或忠实舍入（faithful rounding），以追求更高的吞吐量和更低的电路面积。在 BF16 上的性能与效率提升是直接可用的，因为该格式已在深度学习（DL）应用中得到普及，且精度损失可忽略不计 [22]。

面向神经网络训练的专用加速器，例如 NTX [23]，采用了非标准的乘加电路，使用快速的内部定点累加。虽然不符合 IEEE 754 标准，但它们可以为累加运算提供更高的精度和动态范围。

与浮点相关的数系也在被采用，例如通用数（universal numbers，UNUMs）[24]、Posits [25] 或对数数系（logarithmic number systems，LNSs）。基于 UNUM 的硬件实现已在 [26] 和 [27] 中提出。快速乘法和超越函数已在 [28] 中被实现到一个 RISC-V 核心中。尽管我们的 TP-FPU 的重点是提供类似 FPSTD 的浮点能力，但我们的工作可以通过若干正交的方式与这些更为新颖的数系相结合。例如，可以针对这些格式设置专用的功能单元，将其作为新的运算组与当前的浮点功能一同纳入 TP-FPU 中，以加速特定的工作负载。此外，我们的功能单元实现可以被用作实现其中一些新型算术函数的起点。Posit 算术所需的数据通路与一个具有大量可能输入格式的合并功能单元非常相似。最后，TP-FPU 可以作为一份架构蓝图，填充任意功能单元，从而利用我们架构的能耗比例特性。

## E. 多模式运算模块

据我们所知，目前尚无支持多种格式的全功能 TP-FPU 发表。然而，近年来已有多模式 FMA 架构被提出，通常支持两种或三种 FP 格式的计算 [29]–[32]，或整数与 FP 的组合支持 [34]。表 IV 在标称条件下将所提架构与我们的实现进行了比较。需要注意的是，我们工作中报告的结果测量的是执行 FMA 操作时整个 TP-FPU 的能耗，而非相关工作中所测量的孤立的 FMA 模块。

表 IV
标称条件下本工作（完整 FPU）与其他独立[多模式]架构的 FMA 比较
<table><tr><td>格式</td><td>L/T*</td><td>性能† [Gflop/s]</td><td>能耗 [pJ/flop]</td><td>能效 [Gflop/s W]</td><td>相对值</td></tr><tr><td colspan="6">本工作，22 nm，0.8 Va，0.049 mm²（完整 FPU），923 MHz</td></tr><tr><td>FP64</td><td>标量 4/1</td><td>1.85</td><td>13.36</td><td>74.83</td><td>1.0×</td></tr><tr><td>FP32</td><td>标量 3/1</td><td>1.85</td><td>4.72</td><td>211.66</td><td>2.8×</td></tr><tr><td>FP16 标量</td><td>3/1</td><td>1.85</td><td>2.48</td><td>403.08</td><td>5.4×</td></tr><tr><td>FP16alt 标量</td><td>3/1</td><td>1.85</td><td>2.18</td><td>458.56</td><td>6.1×</td></tr><tr><td>FP8</td><td>标量 3/1</td><td>1.85</td><td>1.27</td><td>786.30</td><td>10.5×</td></tr><tr><td>FP32 向量</td><td>3/2</td><td>3.71</td><td>5.01</td><td>199.70</td><td>2.7×</td></tr><tr><td>FP16</td><td>向量 3/4</td><td>7.42</td><td>2.01</td><td>497.67</td><td>6.7×</td></tr><tr><td>FP16alt 向量</td><td>3/4</td><td>7.42</td><td>1.72</td><td>581.96</td><td>7.8×</td></tr><tr><td>FP8</td><td>向量 2/8</td><td>14.83</td><td>0.80</td><td>1244.78</td><td>16.6×</td></tr><tr><td colspan="6">Kaiser et al. [21]，22 nm，0.8 Vc，0.019 mm2，1.8 GHz</td></tr><tr><td>FP64</td><td>标量 3/1</td><td>3.60</td><td>26.40</td><td>37.88</td><td></td></tr><tr><td colspan="6">Manolopoulos et al. [29]，130 nm，1.2 Vb，0.287 mm2，291 MHz</td></tr><tr><td>FP64 标量</td><td></td><td>0.58</td><td>60.53</td><td></td><td>1.0×</td></tr><tr><td>FP32 向量</td><td>3/1 3/2</td><td>1.16</td><td>30.26</td><td>16.52 33.05</td><td>2.0×</td></tr><tr><td colspan="6">Arunachalam et al. [30]，130 nm，1.2 Vb，0.149 mm2，308 MHz</td></tr><tr><td>FP64 标量</td><td>8/1</td><td>0.62</td><td>28.86</td><td>34.64</td><td>1.0×</td></tr><tr><td>FP32</td><td>向量 8/2</td><td>1.23</td><td>14.43</td><td>69.29</td><td>2.0×</td></tr><tr><td colspan="6">Zhang et al. [31]，90 nm，1 Vc，0.181 mm2，667 MHz</td></tr><tr><td>FP64 标量</td><td>3/1</td><td>1.33</td><td>32.85</td><td>30.44</td><td>1.0×</td></tr><tr><td>FP32 向量</td><td>3/2</td><td>2.67</td><td>16.43</td><td>60.88</td><td>2.0×</td></tr><tr><td>FP16 向量</td><td>3/4</td><td>5.33</td><td>8.21</td><td>121.76</td><td>4.0×</td></tr><tr><td colspan="6">Kaul et al. [32]，32 nm，1.05 Va，0.045 mm2，1.45 GHz</td></tr><tr><td>FP32</td><td>标量 3/1</td><td>2.90</td><td>19.4</td><td>52.00</td><td>1.0×</td></tr><tr><td>FP20 向量</td><td>3/2</td><td>5.80</td><td>10.34</td><td>96.67</td><td>1.9×</td></tr><tr><td>FP14 向量</td><td>3/4</td><td>11.60</td><td>6.2</td><td>161.11</td><td>3.1×</td></tr><tr><td colspan="6">Pu et al. [33]，28 nm，0.8 Va，0.024§ /0.018| mm2，910§ /1360| MHz</td></tr><tr><td>FP64</td><td>标量 6/1</td><td>1.82</td><td>45.05</td><td>43.70</td><td>1.0×</td></tr><tr><td>FP32</td><td>标量 6/1</td><td>2.72</td><td>18.38</td><td>110.00</td><td>2.5×</td></tr></table>

延迟 [周期] / 吞吐率 [操作/周期] † 1 FMA = 2 flops a 芯片实测结果 b 布局后结果 c 综合后结果 ‡ FP20 = FP32 仅使用 12 位精度，FP14 = FP32 仅使用 6 位精度 § FP64 FMA 设计 | FP32 CMA 设计

来自 [21] 的兼容 RISC-V 的 FMA 单元仅支持 FP64，尽管 RISC-V 要求同时支持 FP32，但其并不支持。在与我们的实现相同的 22 nm 工艺下综合后，该单元执行相同 FMA 操作时的能效比我们的 FPU 低 49%。

[29]–[31] 中的架构在 FMA 数据通路内部大量依赖硬件共享，这迫使所有格式使用相同的延迟，不支持较小格式的标量操作，同时也缺乏实质性的能耗比例性。这些架构只能实现成正比的能耗降低，而我们的能效增益随着格式变小而愈发显著——FP8 操作的能耗相对 FP64（位宽缩减 8 倍）最多降低至 1/16.6。通过利用电压调节这一手段，该效率增益还可再提升 2.3 倍，从而使我们的实现在使用窄 FP 格式时获得超越比例的收益，而非简单的 2:1 折中。

[32] 中的 FMA 实现采用了一种向量化方案，将 FP32 尾数数据通路按 2 或 4 分割，采用非常细粒度的门控技术，同时保持指数位宽恒定为 8 位。该架构的用途是在打包的窄数据通路上以 4 倍吞吐率尝试进行大批量 FP32 计算，能耗降低至 1/3.1。通过追踪不确定性，不精确的结果会先使用 2 倍缩减的数据通路重新计算，然后再以完整 FP32 恢复该操作。因此，该单元中使用的中间格式并不对应任何标准的 IEEE 754 格式。

FPMax [33] 针对 FP32 和 FP64 分别实现了独立的 FMA 操作，未采用任何数据通路共享，面向高速 ASIC。比较其两个最高效的实例（采用不同的内部架构）的能耗成本，可以发现其能耗比例性略低于我们的完整 FPU 实现。这一结果进一步印证了在整个 FPU 的规模上为不同格式提供独立数据通路的价值，也促使我们在未来探索适用于不同格式的特定 FMA 架构。

## F. 我们的 TP-FPU 的其他用途

FPnew 的开源特性，以及其采用可综合 SystemVerilog 编写的事实，降低了在新系统中实现 FP 功能的门槛，无需额外的 IP 授权或对标准设计流程进行更改。FPnew 或其子组件已在一些近期工作中得到实际应用。

Ara [35] 是一个可扩展的 RISC-V 向量处理器，实现了 “V” 扩展的草案版本，它利用 FPnew 实例在 16 个 64 位向量通道上高效执行矩阵运算。Snitch [36] 是一个小型伪双发射 RISC-V 处理器，配备强大的双精度 FPU。它采用一个 FPnew 实例来提供 FP64 和 SIMD FP32 计算能力。Montagna et al. [37] 利用 FPnew 的灵活性，在多核 CPU 集群中探索不同的流水线和共享配置。GAP9,<sup>13</sup> 是 GreenWaves Technologies 发布的一款商用 IoT 应用处理器，其中集成了 FPnew 以支持 sub-32 位 transprecision FP 计算。欧洲处理器计划（European Processor Initiative，EPI）项目 [38] 中的大部分 FP 硬件均基于开源的 FPnew 设计。

## VI. 结论

我们提出了 FPnew，这是一个可配置的开源 transprecision 浮点单元，能够支持任意 FP 格式。它提供 FP 算术运算以及高效的类型转换与打包操作，同时具有标量和 SIMD 向量化两种变体，并具备高能效与良好的比例缩放特性。我们将该 TP-FPU 实现到一个嵌入式 RISC-V 处理器核中，并利用我们的 transprecision RISC-V ISA 扩展展示了 transprecision 计算的潜力。在我们的案例研究中，我们实现了 FP32 精度，同时与最优的标量 FP16 基线相比未引入任何性能开销，相对于 FP32 实现将系统能耗降低了 34%。此外，我们将该单元作为 RISC-V 应用级处理器核的一部分，完成了首个支持五种 FP 格式的完整 TP-FPU 硅实现（基于 GLOBALFOUNDRIES 22FDX 工艺）。自适应电压与频率调节使能效高达 2.95 Tflop/sW，并且对于 8 路 FP8 SIMD 操作，计算性能高达 25.33 Gflop/s。处理器中额外的面积开销（9.3%）和静态能耗开销（11.1%），相对于 TP-FPU 所带来的显著性能与效率提升而言是可以接受的。

得益于我们架构中采用的并行数据通路方法，我们的设计比其他多模式 FMA 设计实现了更好的能效缩放。凭借其开放的本质，FPnew 可被用于多种不同的应用场景，既已在嵌入式 IoT 应用中得到了应用，也已被用于高性能向量处理加速器。

## REFERENCES

[1] A. C. I. Malossi et al., “The transprecision computing paradigm: Concept, design, and applications,” in Proc. Design, Autom. Test Eur. Conf. Exhib. (DATE), Mar. 2018, pp. 1105–1110.

[2] W. Kahan, J. D. Darcy, E. Eng, and H.-P. N. Computing, “How Java’s floating-point hurts everyone everywhere,” in Proc. ACM Workshop Java High-Performance Netw. Comput., Stanford, CA, USA: Stanford Univ., 1998, p. 81.

[3] U. Köster et al., “Flexpoint: An adaptive numerical format for efficient training of deep neural networks,” in Proc. Adv. Neural Inf. Process. Syst., 2017, pp. 1742–1752.

[4] E. Chung et al., “Serving DNNs in real time at datacenter scale with project brainwave,” IEEE Micro, vol. 38, no. 2, pp. 8–20, Mar. 2018.

[5] N. P. Jouppi et al., “In-datacenter performance analysis of a tensor processing unit,” in Proc. ACM/IEEE 44th Annu. Int. Symp. Comput. Archit. (ISCA), Jun. 2017, pp. 1–12.

[6] S. Xie et al., “Extreme datacenter specialization for planet-scale computing: ASIC clouds,” ACM SIGOPS Operating Syst. Rev., vol. 52, no. 1, pp. 96–108, Aug. 2018.

[7] A. Waterman and K. Asanovic, Eds. The RISC-V Instruction Set Manual: User-Level ISA, Document Version 20191213, vol. 1. Cham, Switzerland: RISC-V Foundation, Dec. 2019.

[8] G. Tagliavini, S. Mach, D. Rossi, A. Marongiu, and L. Benini, “Design and evaluation of SmallFloat SIMD extensions to the RISC-V ISA,” in Proc. Design, Autom. Test Eur. Conf. Exhib. (DATE), Mar. 2019, pp. 654–657.

[9] S. Mach, F. Schuiki, F. Zaruba, and L. Benini, “A 0.80pJ/flop, 1.24Tflop/sW 8-to-64 bit transprecision floating-point unit for a 64 bit RISC-V processor in 22nm FD-SOI,” in Proc. IFIP/IEEE 27th Int. Conf. Very Large Scale Integr. (VLSI-SoC), Oct. 2019, pp. 95–98.

[10] G. Tagliavini, S. Mach, D. Rossi, A. Marongiu, and L. Benin, “A transprecision floating-point platform for ultra-low power computing,” in Proc. Design, Autom. Test Eur. Conf. Exhib. (DATE), Mar. 2018, pp. 1051–1056.

[11] M. Gautschi et al., “Near-threshold RISC-V core with DSP extensions for scalable IoT endpoint devices,” IEEE Trans. Very Large Scale Integr. (VLSI) Syst., vol. 25, no. 10, pp. 2700–2713, Oct. 2017.

[12] F. Zaruba and L. Benini, “The cost of application-class processing: Energy and performance analysis of a linux-ready 1.7-GHz 64-bit RISC-V core in 22-nm FDSOI technology,” IEEE Trans. Very Large Scale Integr. (VLSI) Syst., vol. 27, no. 11, pp. 2629–2640, Nov. 2019.

[13] R. K. Montoye, E. Hokenek, and S. L. Runyon, “Design of the IBM RISC System/6000 floating-point execution unit,” IBM J. Res. Develop., vol. 34, no. 1, pp. 59–70, Jan. 1990.

[14] S. Mach, D. Rossi, G. Tagliavini, A. Marongiu, and L. Benini, “A transprecision floating-point architecture for energy-efficient embedded computing,” in Proc. IEEE Int. Symp. Circuits Syst. (ISCAS), May 2018, pp. 1–5.

[15] J. Bachrach et al., “Chisel: Constructing hardware in a scala embedded language,” in Proc. 49th Annu. Design Autom. Conf. DAC, 2012, pp. 1212–1221.

[16] J.-M. Müller et al., Handbook of Floating-Point Arithmetic, 2nd ed. Boston, MA, USA: Birkhäuser, 2018.

[17] F. Zaruba, F. Schuiki, S. Mach, and L. Benini, “The floating point trinity: A multi-modal approach to extreme energy-efficiency and performance,” in Proc. 26th IEEE Int. Conf. Electron., Circuits Syst. (ICECS), Nov. 2019, pp. 767–770.

[18] F. de Dinechin and B. Pasca, “Designing custom arithmetic data paths with FloPoCo,” IEEE Des. Test. IEEE Des. Test. Comput., vol. 28, no. 4, pp. 18–27, Jul. 2011.

[19] K. Asanovic et al., “The rocket chip generator,” EECS Dept., Univ. California, Berkeley, Berkeley, CA, USA, Tech. Rep. UCB/EECS-2016- 17, 2016.

[20] K. Asanovic, D. A. Patterson, and C. Celio, “The Berkeley out-of-order machine (boom): An industry-competitive, synthesizable, parameterized RISC-V processor,” Univ. California at Berkeley, Berkeley, CA, USA, Tech. Rep. UCB/EECS-2015-167, 2015.

[21] F. Kaiser, S. Kosnac, and U. Brüning, “Development of a RISC-Vconform fused multiply-add floating-point unit,” Supercomputing Frontiers Innov., vol. 6, no. 2, pp. 64–74, 2019.

[22] P. Zamirai, J. Zhang, C. R. Aberger, and C. De Sa, “Revisiting BFloat16 training,” 2020, arXiv:2010.06192. [Online]. Available: http://arxiv.org/abs/2010.06192

[23] F. Schuiki, M. Schaffner, and L. Benini, “NTX: An energy-efficient streaming accelerator for floating-point generalized reduction workloads in 22 nm FD-SOI,” in Proc. Design, Autom. Test Eur. Conf. Exhib. (DATE), Mar. 2019, pp. 662–667.

[24] J. L. Gustafson, The End of Error: Unum Computing. Boca Raton, FL, USA: CRC Press, 2017.

[25] J. L. Gustafson and I. T. Yonemoto, “Beating floating point at its own game: Posit arithmetic,” Supercomputing Frontiers Innov., vol. 4, no. 2, pp. 71–86, 2017.

[26] F. Glaser, S. Mach, A. Rahimi, F. K. Gurkaynak, Q. Huang, and L. Benini, “An 826 MOPS, 210uW/MHz unum ALU in 65 nm,” in Proc. IEEE Int. Symp. Circuits Syst. (ISCAS), May 2018, pp. 1–5.

[27] A. Bocco, Y. Durand, and F. de Dinechin, “Hardware support for UNUM floating point arithmetic,” in Proc. 13th Conf. Ph.D. Res. Microelectron. Electron. (PRIME), Jun. 2017, pp. 93–96.

[28] M. Gautschi, M. Schaffner, F. K. Gurkaynak, and L. Benini, “4.6 A 65nm CMOS 6.4-to-29.2pJ/FLOP@0.8 V shared logarithmic floating point unit for acceleration of nonlinear function kernels in a tightly coupled processor cluster,” in IEEE Int. Solid-State Circuits Conf. (ISSCC) Dig. Tech. Papers, Jan. 2016, pp. 82–83.

[29] K. Manolopoulos, D. Reisis, and V. A. Chouliaras, “An efficient dualmode floating-point multiply-add fused unit,” in Proc. 17th IEEE Int. Conf. Electron., Circuits Syst., Dec. 2010, pp. 5–8.

[30] V. Arunachalam, A. N. Joseph Raj, N. Hampannavar, and C. B. Bidul, “Efficient dual-precision floating-point fused-multiply-add architecture,” Microprocessors Microsyst., vol. 57, pp. 23–31, Mar. 2018.

[31] H. Zhang, D. Chen, and S.-B. Ko, “Efficient multiple-precision floatingpoint fused multiply-add with mixed-precision support,” IEEE Trans. Comput., vol. 68, no. 7, pp. 1035–1048, Jul. 2019.

[32] H. Kaul et al., “A 1.45GHz 52-to-162GFLOPS/W variable-precision floating-point fused multiply-add unit with certainty tracking in 32nm CMOS,” in IEEE Int. Solid-State Circuits Conf. (ISSCC) Dig. Tech. Papers, Feb. 2012, pp. 182–184.

[33] J. Pu, S. Galal, X. Yang, O. Shacham, and M. Horowitz, “FPMax: A 106GFLOPS/W at 217GFLOPS/mm<sup>2</sup> single-precision FPU, and a 43.7GFLOPS/W at 74.6GFLOPS/mm<sup>2</sup> double-precision FPU, in 28nm UTBB FDSOI,” 2016, arXiv:1606.07852. [Online]. Available: http://arxiv.org/abs/1606.07852

[34] T. M. Bruintjes, K. H. G. Walters, S. H. Gerez, B. Molenkamp, and G. J. M. Smit, “Sabrewing: A lightweight architecture for combined floatingpoint and integer arithmetic,” ACM Trans. Archit. Code Optim., vol. 8, no. 4, pp. 1–22, Jan. 2012.

[35] M. Cavalcante, F. Schuiki, F. Zaruba, M. Schaffner, and L. Benini, “Ara: A 1-GHz+ scalable and energy-efficient RISC-V vector processor with multiprecision floating-point support in 22-nm FD-SOI,” IEEE Trans. Very Large Scale Integr. (VLSI) Syst., vol. 28, no. 2, pp. 530–543, Feb. 2020.

[36] F. Zaruba, F. Schuiki, T. Hoefler, and L. Benini, “Snitch: A tiny pseudo dual-issue processor for area and energy efficient execution of floating-point intensive workloads,” IEEE Trans. Comput., early access, Oct. 7, 2020, doi: 10.1109/TC.2020.3027900.

[37] F. Montagna et al., “A transprecision floating-point cluster for efficient near-sensor data analytics,” 2020, arXiv:2008.12243. [Online]. Available: http://arxiv.org/abs/2008.12243

[38] M. Kovaˇc, D. Reinhardt, O. Jesorsky, M. Traub, J.-M. Denis, and P. Notton, “European processor initiative (EPI)—An approach for a future automotive eHPC semiconductor platform,” in Electronic Components and Systems for Automotive Applications. Cham, Switzerland: Springer, 2019, pp. 185–195.



![](images/cea415a07275f54a58bd01a3bbf2b5a1b617af3a9adc53690ddbdc53d3f65296.jpg)

Stefan Mach 在瑞士苏黎世联邦理工学院（ETHZ）获得学士与硕士学位，目前正在该校攻读博士学位。

自 2017 年起，他担任 ETHZ 集成系统实验室（Integrated Systems Laboratory）的研究助理。他的研究兴趣包括跨精度计算、计算机算术以及能效优先的处理器架构。

![](images/a79224de0c61dce9d10946ed410162f8b8027b56d12d793efb9feaf87a8f60a3.jpg)

Fabian Schuiki 分别于 2014 年和 2016 年在瑞士苏黎世联邦理工学院（ETHZ）获得电气工程学士与硕士学位，目前在 ETHZ 数字电路与系统组攻读博士学位，师从 Luca Benini。

他的研究兴趣包括跨精度计算以及近存计算与存内计算。

![](images/9576b2daf1ece0f94fe8b8e4c9fa2b04ad79b1d793040b07563ff14cbaf903e0.jpg)

Florian Zaruba（IEEE 研究生会员）于 2014 年在维也纳工业大学（TU Wien）获得学士学位，并于 2017 年在瑞士苏黎世联邦理工学院（ETHZ）获得硕士学位，目前在集成系统实验室攻读博士学位。

他的研究兴趣包括超大规模集成电路设计和高性能计算机体系结构。

![](images/0bf2ba435cd8dbcee1a70439fcae486a1870e842b3abd673a54145f7ea56c355.jpg)

Luca Benini（IEEE 会士）现任瑞士苏黎世联邦理工学院（ETHZ）数字电路与系统讲席教授。他同时也是意大利博洛尼亚大学（Università di Bologna）的正教授。

他已发表或合著超过 1000 篇同行评审论文和五部著作。他的研究兴趣涵盖从嵌入式到高性能的能效计算系统设计。

Benini 教授是 ACM 会士、欧洲科学院成员。他曾获得 2016 年 IEEE CAS Mac Van Valkenburg 奖和 2020 年 EDAA 成就奖。