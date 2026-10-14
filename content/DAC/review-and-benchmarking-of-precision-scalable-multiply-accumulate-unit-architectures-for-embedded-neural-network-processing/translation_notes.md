# Review and Benchmarking of Precision-Scalable Multiply-Accumulate Unit Architectures for Embedded Neural-Network Processing 原文翻译

# 面向嵌入式神经网络处理的精度可缩放乘加单元架构综述与基准评测

Vincent Camus , Member, IEEE, Linyan Mei , Student Member, IEEE, Christian Enz , Fellow, IEEE, and Marian Verhelst , Senior Member, IEEE

摘要—— 当前深度学习的发展趋势带来了巨大的计算需求，每次推理需要数十亿次乘加（Multiply-Accumulate, MAC）运算。幸运的是，降低精度已被证明能带来巨大收益且对精度影响很小，为在移动设备和物联网节点上进行处理铺平了道路。为此，近年来已提出多种针对神经网络优化的精度可缩放 MAC 架构。然而，由于这些架构采用不同工艺技术和性能目标实现，人们一直难以理解它们之间的差异，也难以对其相对优势做出公正评判。为克服这一问题，本文详尽综述了最先进的精度可缩放 MAC 架构，并将其统一到一个新的分类体系中。随后，这些不同的拓扑结构在 28 nm 商用 CMOS 工艺下、在广泛的性能目标范围内、以 2 到 8 比特的精度进行了全面的基准评测。电路在每种精度下以及在实际用例中的联合表现均得到了分析，突显了架构和可缩放性在能耗、吞吐量、面积和带宽方面的影响，旨在理解降低神经网络处理计算成本的关键趋势。

索引词— ASIC，深度神经网络，精度可缩放电路，可配置电路，MAC，乘加单元。

## I. 引言

嵌入式深度学习因广泛的应用前景和巨大的潜在市场，如今受到大量关注。然而，拥抱这一边缘智能时代的主要挑战来自于嵌入式设备（通常由电池供电）有限的能量预算与计算密集型深度学习算法（需要数十亿次乘加（MAC）运算和数据搬运）之间的供需差距。

为了缓解这种不平衡关系，研究人员在不同抽象层次上探索了许多方法。在算法层面，研究者引入了能够保持原始算法精度的硬件友好模型 [1], [2]。在系统架构层面，数据流方案和存储层次结构得到了广泛研究，以优化数据搬运和存储 [3]–[5]。最后，在电路层面，人们提出了新的拓扑结构，通过利用数据局部性或错误容忍能力来超越传统设计，提升能耗或性能 [6]–[8]。

在这些技术中，降低精度计算已被证明能带来巨大收益，且对网络精度的影响很低或可以忽略 [9], [10]。研究表明，最优精度不仅在不同神经网络之间有所差异，在单个神经网络内部也可能不同 [11], [12]，不同层之间甚至不同通道之间都存在差异。这催生了面向移动设备和物联网节点中嵌入式深度神经网络（DNN）处理的超高效运行时精度可缩放神经网络处理器的新趋势。

相应地，近年来已引入了许多运行时精度可缩放的 MAC 架构，它们或采用高度并行化的方式构建 [13]–[16]，或采用串行方式构建 [17]–[21]。然而，出于以下几个原因，人们一直难以评估它们的效率或选择某种拓扑结构。首先，它们采用不同的工艺技术实现，位宽或可缩放级别各不相同，或者被集成到完全不同的系统中。其次，很少有工作将其性能与基准设计（即不具备可缩放性的 MAC 单元）进行对比评估，从而清楚展示可配置性带来的开销。最后，许多设计基于特定的（ad-hoc）可缩放技术，缺乏系统化的方法，也缺乏对不同实现之间哪些可缩放原理是共通的、哪些是不同的清晰分析。

本工作旨在通过以下贡献来克服这些问题：

提出一个新的分类体系，对所有现有的精度可缩放 MAC 架构进行归类，揭示其设计模式，并将 MAC 级别的设计决策与阵列级别和算法级别的选择关联起来（第 II 节）。

对最先进的精度可配置 MAC 架构进行详尽且图文并茂的综述，以帮助理解它们的电路拓扑和功能差异。（第 III 节）。

针对每种 MAC 架构，在能耗、面积、带宽和吞吐量方面对精度缩放及其开销进行详细分析（第 IV-V 节）。

在广泛的性能目标范围内，对标称电源电压以及动态电压频率调节（DVFS）下的所有可缩放精度 MAC 单元进行对比研究（第 VI 节）。

![](images/782c787250d2b794e073af734d38943e45fa7cc78089804140d60ad3a2dd32e1.jpg)  
图1：（a）全连接和（b）卷积计算的循环表示。

针对三种实际使用案例进行研究，假设低精度运算的比例不同，探索其对最优架构选择的影响（第 VII 节）。

源代码和补充材料可在以下网址获取：https://github.com/vincent-camus/benchmarkingprecision-scalable-mac-units。

## II. 精度可缩放性的数据流影响

本节首先介绍两种数据流可缩放选项：分开求和（Sum Apart, SA）与合并求和（Sum Together, ST），及其在算法级、处理单元（PE）阵列级和 PE 级的影响，表明运行时精度可缩放性与神经网络数据流的考量紧密交织。随后，本节将最先进的精度可缩放 MAC 架构映射到这些类别中，并提出一个通用的 MAC 分类体系。

## A. 算法级的 SA 与 ST

分开求和（Sum Apart, SA）与合并求和（Sum Together, ST）的概念最初由 Mei 等人 [16] 在 PE 级别提出，用于描述子字并行计算累加的两种对立方式：SA 将并行生成的乘积保持分离，而 ST 将它们相加以形成单一的输出结果。这些概念可用于区分神经网络工作负载的算法级特征。

图 1 以嵌套循环的形式展示了全连接层和卷积层 DNN 的计算。出现在 MAC 运算左侧的索引（b、k、y 和 x）表示其所有相关的部分结果被累加并存储到不同的输出变量中，对应于 SA 类型的累加。相反，未出现在输出中的索引（c、fy 和 fx）表示这些部分结果沿这些维度被合并累加，对应于 ST 类型的累加。所有循环被划分为 SA 或 ST 两种类型。

## B. PE 阵列级别的 SA 与 ST

许多最先进的深度学习处理器依赖于多维 PE 阵列来支持卷积或矩阵乘法。此类阵列的架构与所针对的数据流（空间展开方案）[3]紧密相关，并且同样可以沿每个阵列维度被归类为 SA 或 ST 行为。

例如，一个 2D PE 阵列可以被归类为三种类型：SA-SA、SA-ST 或 ST-ST，如图 2 所示。

![](images/5aea1ee7bcee16a3719c23288be8995e0c1125ebc158bb2fbadf6747e64dcbaa.jpg)  
图2：三类二维PE阵列。

因此，SA-SA 阵列支持沿其两个几何维度对两个 SA 类型循环进行空间展开，而 SA-ST 阵列则对来自每种循环各展开一个，最后 ST-ST 阵列展开两个 ST 类型循环。选择 SA 还是 ST 阵列结构决定了阵列中的累加岛。

这使得神经网络加速器可以被划分为三种不同的类别：SA-SA 阵列在每个累加岛中仅包含单个 PE，也被称为输出驻留阵列，通常跨时钟周期累加结果 [22], [23]。SA-ST 阵列在每个累加岛中包含一维 PE 向量，其结果要么以脉动方式相加 [24], [25]，要么通过单周期累加器相加 [26], [27]。ST-ST 阵列将所有 PE 集成到单个累加岛中，例如采用二维累加加法树 [28], [29]、脉动阵列，或一维累加加法树加脉动阵列。

## C. 精度可扩展 MAC 单元级别的 SA 与 ST

当将计算精度缩减至全精度运算的一小部分时，空间精度可扩展的 MAC 通常会将 PE 细分为若干个并行的更低分辨率的 PE。如此一来，每个名义上的 PE 本身就变成了一个 PE 阵列，从而沿 SA 或 ST 维度提供了更多的空间循环展开机会。文献中已提出许多精度可扩展的 MAC 架构，它们都可以沿着这一 SA 与 ST 概念进行归类。

在全精度模式下，SA 与 ST MAC 的行为完全相同：每个时钟周期只产生一个结果。在缩减精度模式下，会并行产生多个低精度结果。SA 与 ST MAC 之间的主要区别在于，SA MAC 将这些子乘积分开保存，从而在多个输出寄存器中产生多个结果，而 ST MAC 则将它们相加以获得单个累加结果，因而只需要一个寄存器。

在现有的神经网络计算模式中，存在一条隐式规则：如果多个乘法共享一个操作数，它们的乘积便不能相加。换言之，只有属于不同输出值的乘积才有机会共享输入值（无论是权重还是激活值）。这一算法约束构成了 SA 与 ST MAC 之间输入输出带宽权衡的基础。这将在第 III 节中更详细地阐述。

## D. 分类体系

基于上述 ST 与 SA 概念，表 I 给出了本文余下部分所使用的精度可扩展 MAC 架构的完整分类体系。

在纵向维度上，所有精度可扩展 MAC 架构首先被划分为空间精度缩放技术和时间精度缩放技术。在空间精度可配置类别中，MAC 在降低精度模式下空间地拆分为若干个低精度算术运算器。它们进一步使用 ST 与 SA 原则进行归类。时间精度可扩展 MAC 类别则具有完全不同的工作原理。它们通过重复的加法-移位操作的时间迭代来执行乘法。降低精度的操作仅需减少时间迭代的次数即可实现。基于时间的 MAC 进一步依据每次迭代串行输入的数据位宽被分为位串行与多位串行两组。总而言之，降低输入操作数的精度有助于空间可重构 MAC 并行执行更多 MAC 运算，而有助于时间可重构 MAC 减少完成一次 MAC 运算所需的总时钟周期数。

表 I
TAXONOMY OF PRECISION-SCALABLE MAC ARCHITECTURES
<table><tr><td rowspan=1 colspan=2>架构类型</td><td rowspan=1 colspan=1>1D 可扩展</td><td rowspan=1 colspan=1>2D 可扩展</td><td rowspan=1 colspan=1>2D 对称可扩展</td></tr><tr><td rowspan=2 colspan=1>空间</td><td rowspan=1 colspan=1>分开求和</td><td rowspan=1 colspan=1>1D D&amp;C SA(DNPU [13])</td><td rowspan=1 colspan=1>2D D&amp;C SA</td><td rowspan=1 colspan=1>子字并行 SA(DVAFS [15])</td></tr><tr><td rowspan=1 colspan=1>合并求和</td><td rowspan=1 colspan=1>1D D&amp;C ST</td><td rowspan=1 colspan=1>2D D&amp;C ST(BitFusion [14])</td><td rowspan=1 colspan=1>子字并行 ST(ST [16])</td></tr><tr><td rowspan=2 colspan=1>时间</td><td rowspan=1 colspan=1>位串行</td><td rowspan=1 colspan=1>1D 位串行(UNPU [17])</td><td rowspan=1 colspan=1>2D 位串行(LOOM [19])</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>多位串行</td><td rowspan=1 colspan=1>1D 多位串行(Multibit serial [18])</td><td rowspan=1 colspan=1>2D 多位串行(LOOM [19])</td></tr></table>

在横向维度上，架构依据其输入（权重和激活值）缩放特性被划分为三个子类型。“1D 可扩展”表示只有一个输入操作数能够缩减至较低位宽，而另一个始终保持在全精度。“2D 可扩展”MAC 打破了这一约束，使两个输入操作数都能独立缩放。最后，“2D 对称可扩展”MAC 允许两个输入操作数都进行缩放，但只能缩放至相同的精度。注意，在 2D 基于时间的 MAC 中，两个操作数天然相互独立，因此最后一列为空。

表 I 将现有的最先进精度可扩展 MAC 架构映射到这一新分类体系上，并提出了统一的新命名。正如下一节将阐明的，所有分治架构都采用自底向上的设计方法：其乘法器逻辑由若干个独立的、相同的子块组合而成，这些子块始终处于激活状态。它们的可配置性体现在互连和移位-加法逻辑上。相反，子字并行架构则源于自顶向下的设计理念：其单块全精度乘法器通过选择性门控来在降低精度模式下复用现有的算术单元。

注意，MAC 单元本质上是一种 2D 架构（具有激活值和权重两个输入操作数）。通过为每个维度选择表 I 中所列的不同精度可扩展方法，可以创建出许多混合精度可扩展的 MAC 单元，例如 SA-ST MAC 架构或时间-空间混合 MAC 架构。然而，本文专注于讨论和比较表 I 中所列的纯架构类别。

## III. 可扩展 MAC 架构综述

本节是对表 I 中所列所有精度可扩展 MAC 架构的教程。文中将充分讨论它们的工作原理、相似性与差异性，并配以图示来说明它们在不同模式下的计算状态，假设输出累加具有 4 位的寄存器余量。

![](images/c91112ef4020b1f5af63da015f5c4a2ab3ca2de6c0a1bdb40028dc5eb09eca5d.jpg)  
图3：数据门控传统MAC，分别执行一个8×8位全精度、一个4×4位对称缩放或一个2×8位仅权重缩放操作。

## A. 数据门控的常规 MAC

本研究的基线是一种数据门控的常规 MAC 单元，如 [11] 所述。当应用缩放精度时，仅使用 MSB（高位）进行计算，而 LSB（低位）保持为零。因此，开关活动得以降低。由于仅经过 MSB 的关键路径更短，在相同吞吐量下可以动态提升频率或降低电源电压。

这一点在图 3 中得到了展示，从左至右依次为使用 8b 全精度、4b 对称缩放和 2b 仅权重缩放的情况。在以降低精度模式计算时，乘法器和累加器逻辑的部分电路通过将输入数据置零而被门控，如阴影区域所示。然而，由于该 MAC 不嵌入任何可配置性特性（例如选择性时钟门控），即便没有数据到达其 LSB，所有寄存器仍然持续接收时钟信号。

## B. 一维分治 SA（DNPU）

图 4 展示了由 Shin 等人 [13] 提出的深度神经网络处理单元（Deep Neural Processing Unit, DNPU）的 MAC 架构，在新分类法中被归类为一维 D&C SA MAC 单元。

其 8b×8b 乘法器由四个 2b×8b 子乘法器和可配置的移位加法逻辑块构成。其输入中只有一个在功能上可扩展，因此具有一维可扩展性。更具体地说，上方输入是子字可扩展的，而左侧的 8b 输入在所有模式下被视为单一操作数，并被所有子乘法器块共享。通过在降低精度模式下门控移位加法逻辑，可以并行生成 2 个或 4 个结果并分别保存，这解释了其 SA（分开累加）的本质。

请注意，与其他 SA 型 MAC 一样，每个子字输出寄存器都需要各自的累加余量。因此，与只需要一次该余量的时间型（temporal）和 ST 型 MAC 相比，SA 型 MAC 在余量较大时会受到更严重的影响。

![](images/5c92b7fdf3177c0164c3d820add2722ec004220cda21bb3f971384b05c9d93b5.jpg)  
图 5. 配置为每周期执行一次 8b×8b、两次 4b×8b 或四次 2b×8b 运算的一维 D&C ST MAC 中的仅权重精度缩放。

## C. 一维分治 ST

新分类法揭示了一种一维 D&C ST MAC。该 MAC 同样仅具有一维可扩展性。然而，在降低精度时，一维 D&C ST MAC 不是将并行生成的结果分开保存，而是将它们累加在一起。如图 5 所示，可配置移位加法逻辑中的中间加法器并未被旁路，而是被复用于 ST 累加。

尽管仅产生一个结果使得输出带宽有所降低，但当精度下降时，输入带宽却急剧增加，因为其需要各不相同的 8 位激活值。这源于这样一个事实：神经网络计算内核可以在面向不同输出的计算之间共享输入，但无法在累加到同一输出的各结果之间共享被乘数。

## D. 二维分治 SA

新引入的分类法衍生出了一种二维 D&C SA 架构。其基线 8b×8b 乘法器可由 16 个 2b×2b 乘法器连同可配置的移位加法逻辑块实现，二者共同使其能够在一个周期内执行一次 8b×8b、两次 4b×8b、四次 4b×4b、四次 2b×8b、八次 2b×4b 或十六次 2b×2b 乘法。图 6 展示了其四种工作状态。

在降低精度模式下，SA 原理再次导致输出带宽爆炸，例如在图 6（右下）的二维 D&C SA 2b×2b 模式中清晰可见。该架构的第二个缺点是适用性受限。二维 D&C SA 单元仅支持卷积类工作负载或批量工作负载，因为 batch-1 的全连接神经网络层无法在两个操作数之间共享数据。

![](images/9a3634d6e5422771810567207e7824546989b663264ce15e8197c5900823811c.jpg)  
图 6. 配置为每周期执行一次 8b×8b、四次 4b×4b、四次 2b×8b 或十六次 2b×2b 运算的二维 D&C SA MAC 中的精度缩放。

## E. 二维分治 ST（BitFusion）

BitFusion 由 Sharma 等人 [14] 提出，是一种二维 D&C ST MAC 单元。与前面的二维 D&C SA 架构类似，它能够在一个时钟周期内执行一次 8b×8b、两次 4b×8b、四次 4b×4b、四次 2b×8b、八次 2b×4b 或十六次 2b×2b 乘法。图 7 展示了其四种计算模式。二维 D&C ST 不将并行生成的乘积分开保存，而是将其相加，从而避免了输出带宽爆炸。然而，其结果是输入共享变得不可能，从而在降低精度模式下提高了所需的输入带宽。

在由 Ryu 等人 [21] 提出的 BitFusion 变体 BitBlade 中，移位逻辑被从每个 MAC 单元中抽离出来，并由 MAC 阵列中不同 MAC 的十六个 2b×2b 子乘法器共享，以降低可配置性开销。然而，该工作仅关注阵列级技术，此外，这种阵列级优化也适用于许多其他 MAC 架构。因此，BitBlade 未被纳入本基准测试研究。

## F. 子字并行 SA（DVAFS）

由 Moons 等人 [23] 开发的所谓动态电压-精度-频率缩放（Dynamic Voltage-Accuracy-Frequency Scaling, DVAFS）MAC，在所提出的分类法中可归类为子字并行 SA MAC。

与前面讨论的所有精度可扩展 MAC 类型不同，SWP SA 是一种二维对称可扩展架构，遵循自顶向下的设计方法。其功能性 8b 乘法器不再通过以可配置互连融合若干较小的乘法器来实现，而是利用阵列乘法器的互连模式，并有选择性地门控/激活部分算术逻辑，以在一个时钟周期内执行一次 8b×8b、两次 4b×4b 或四次 2b×2b MAC 运算，如图 8 所示。由于 SA 原理及其所需的寄存器余量复制，需要大型输出寄存器。

![](images/82322eb826f993a6239b414122afe03b0dbcdc335dcc7cd5471182ecc8964baf.jpg)  
图 7. 配置为每周期执行一次 8b×8b、四次 4b×4b、四次 2b×8b 或十六次 2b×2b 运算的二维 D&C ST（BitFusion）MAC 中的精度缩放。

请注意，在降低精度模式下，阵列乘法器中会出现三类区域，由不同的阴影区域表示。标有"X"的蓝色块是处于活动状态的乘法逻辑。灰色条带表示完全被门控的逻辑。最重要的是，浅蓝色条带标示了那些不用于算术运算、但为将来自活动乘法逻辑的中间结果传播到收集输出的阵列乘法器右下角而处于活动状态的单元，因此这些区域无法被完全门控。

## G. 子字并行 ST（ST）

由 Mei 等人 [16] 提出的 SWP MAC 单元的 Sum Together（ST）版本同样是一种基于阵列乘法器的二维对称可扩展架构。但与 SWP SA 不同，SWP ST 通过将阵列乘法器激活为相反的对角模式来将所有子字结果相加，如图 9 所示。这种配置节省了加法逻辑，因为它利用乘法器阵列单元隐式地执行加法。无法被完全门控的传播区域随着精度降低而缩小，而关键路径则大致保持不变，这与 SWP SA 恰好相反。

![](images/f52bad6b769c29aae7f301a2ac45350617444086de537a205e4c93bb50cee890.jpg)  
图 9. 配置为每周期执行一次 8b×8b、两次 4b×4b 或四次 2b×2b 运算的 SWP ST MAC 中的对称精度缩放。

值得指出的是，与其他 ST 架构相比，SWP ST 在各精度模式间保持输入带宽不变，代价是在精度下降时部分门控其算术块。因此，ST 型 MAC 在输入带宽一致性与高硬件利用率之间存在权衡。

## H. 一维位串行设计（UNPU）

位串行设计近来备受关注，其中包括 Lee 等人提出的统一神经处理单元（Unified Neural Processing Unit，UNPU）[17]，以及 Ueyoshi 等人提出的 QUEST 对数量化 3D 堆叠推理引擎 [20]。事实上，位串行的操作数输入方式天然支持完全可变的位精度。

在本研究考察的设计中，UNPU 的位串行 MAC 以 1 位迭代的方式接收权重，而激活值则保持全精度并以并行方式输入。如图 10 所示，该设计因此被归类为一维位串行架构。可以通过数据门控（data gating）实现对激活值的缩放。

本研究选用的实现方案采用右移时序乘法器，因为与左移设计相比，它所需的第一级加法器更小，从而避免了较长的进位传播和符号位扩展。

在文献 [18] 中，原始的一维位串行概念已被扩展到一维多位串行设计。图 11 展示了一维 4 位串行 MAC 的示例，其中权重每次输入 4 位。该方案完成一次 8 位计算仅需 2 个时钟周期，因而降低了时钟树和寄存器的能耗。可以通过门控多余的位来获得更低的精度，如图 11 右侧所示。本综述涵盖了一维 2 位串行和 4 位串行两种 MAC。

## I. 二维位串行设计（LOOM）

![](images/2b67c24cd1b2ee61f4ff532fbaf25e2c470213b0572d94eb4f279b77f8a26a57.jpg)

图10：逐位串行MAC的权重精度缩放，分别配置为8×8、4×8、2×8位运算。

![](images/9ae75524eeb411a2c38c4e051fe0cfc96f4580137b3a9e91610177e40ccb5ff6.jpg)

图11：4位串行MAC的权重精度缩放，配置为8×8、4×8及由4×8门控实现的2×8位运算。

由 Sharify 等人设计的 LOOM [19] 将一维位串行方法扩展为一种二维可扩展架构。该二维可扩展设计以位串行方式输入两个输入操作数，如图 12 所示。如此一来，图 10 中的按位 AND 逻辑被进一步简化为单个 AND 门，但完成一次 MAC 运算所需的时钟周期数增加为激活值精度与权重精度的乘积。

由于两个输入操作数均需逐位读取，且需要遍历所有输入位组合，因此需要更复杂的输入调度和控制逻辑。图 13 展示了本工作中使用的 4b 4b 运算调度。

注意，输入读取模式并不规则且包含重复。例如，在该 4b 4b 场景下，权重逐位读取的序列为：w<sub>0</sub>, w<sub>1</sub>, w<sub>0</sub>, w<sub>2</sub>, w<sub>1</sub>, w<sub>0</sub>, w<sub>3</sub>, w<sub>2</sub>, w<sub>1</sub>, w<sub>0</sub>, . . . 。此外，与每个时钟周期执行一次右移的一维位串行单元不同，二维位串行单元对部分积的移位是不规则的，如图 13 中红色箭头所示。

二维位串行实现同样可以采用多位处理方式。图 14 展示了在三种不同精度模式下的二维 4 位串行设计。通过为两个操作数每次各输入 4 位，计算周期数大幅降低。注意，由于基本算术模块是一个 4b 乘法器，精度缩放最低只能降到 4 位。通过 LSB 数据门控，可以再次实现低于 4 位的缩放。本综述涵盖了二维 2 位串行和 4 位串行两种 MAC。

## IV. 设计与基准测试方法

本文余下部分将对上述所有精度可扩展 MAC 架构分别进行评估和比较研究。为此，本节描述用于实现和基准测试这些电路的方法，涵盖 5 种精度模式：8b 8b 全精度、2b 2b 和 4b 4b 对称缩放，以及 2b 8b 和 4b 8b 仅权重缩放。

![](images/55c2556a9881084a59c99f6f2e1e133524de0efa7549642b861363b3d7d19ed1.jpg)  
图12：二维串行MAC的对称精度缩放，分别为8×8、4×4、2×2位运算。

![](images/bd77063d0aaed539e4d58083fb3a85969fc1d645dd1222b4b77e557b555d5d00.jpg)  
图13：二维逐位串行电路在4×4位模式下的逐位输入调度。

![](images/8811880f1e36a769d4780f83816cb0144c40d106e11dfc58c2ded46ee5ff7427.jpg)  
图14：二维4位串行MAC的对称缩放，配置为8×8、4×4或通过4×4位门控实现的2×2位操作。

## A. 设计考量与假设

为确保本研究的公平性，所有 MAC 架构均设计有等同的特性与优化。

它们均针对 8b 全精度基线采用两段流水线结构。为支持不同的精度模式，其累加寄存器按图 15 所示进行了划分和门控。任何精度模式下的累加余量（accumulation headroom）均为 4b。对于 SA MAC（图 15a），输出被分开更新和存储；而对于 ST 和基于时序的可扩展 MAC，输出则存入单一未划分的寄存器（图 15b）。

由于绝大多数嵌入式神经网络在层间使用仅产生正值的 ReLU 激活函数，所有设计均假设采用有符号权重和无符号激活值的表示方式。不过，若考虑全有符号表示，这一选择对架构之间的相对比较影响应当很小。

输入寄存器以及可在多个 PE 之间共享的开销（例如控制逻辑或有限状态机）不计入面积和内部功耗的报告。

为使本架构研究尽可能通用，任何设计均未考虑单独的优化技术或特性。例如，LUT 加速 [13] 或稀疏运算保护（sparse-operation guarding）[30] 均未实现。这些正交方法通常可以惠及多种架构。

<table><tr><td rowspan=1 colspan=1>4b</td><td rowspan=1 colspan=4>16b</td><td rowspan=1 colspan=3>12b</td></tr><tr><td rowspan=1 colspan=1>4b</td><td rowspan=1 colspan=2>8b</td><td rowspan=1 colspan=1>4b</td><td rowspan=1 colspan=1>4b</td><td rowspan=1 colspan=2>8b</td><td rowspan=1 colspan=1>4b</td></tr><tr><td rowspan=1 colspan=1>4b</td><td rowspan=1 colspan=1>4b</td><td rowspan=1 colspan=1>4b</td><td rowspan=1 colspan=1>4b</td><td rowspan=1 colspan=1>4b</td><td rowspan=1 colspan=1>4b</td><td rowspan=1 colspan=1>4b</td><td rowspan=1 colspan=1>4b</td></tr></table>

![图15：SA与ST寄存器布局完整原图](images/figure-15-complete.png)

图15完整图中的浅色区为累加预留位，粉色为乘积，斜线为门控区域。下面保留解析器提取的ST局部图：

![](images/8a133c7c9b8d2dbaa3f49003026aa41a634755b387345c8ff3753b7c262cc3c8.jpg)  
图15：（a）SWP SA和（b）SWP ST的寄存器设置。

## B. 设计空间

每种架构均实现了多个版本，变化的是可扩展级别，即可扩展粒度。例如，1 级可扩展设计在设计上支持从 8b 缩放至 4b，而 2b 模式则通过对 4b 模式进行数据门控来实现。数据门控从 LSB 到 MSB 依次施加，以避免在进位传播和符号位扩展上消耗不必要的能量。2 级可扩展设计则直接在设计上支持缩放至 4b 和 2b。由于二值计算（1b 精度）需要不同的电路和神经网络实现，本研究未将其纳入考虑。

由于各种 MAC 架构的最优工作频率差异很大，本研究在 600 MHz 到 4 GHz 的宽泛范围内探索了多种时钟目标。为使每种模式都能找到其最优工作点，对每种模式单独施加了约束扫描（constraint sweep）。

## C. 物理实现与时序分析

各 MAC 架构的性能通过在商用 28 nm 低漏电工艺、典型工艺角、标称电源电压 1 V 条件下进行综合与仿真来评估。电路由抽象级 SystemVerilog 描述使用 Cadence Genus 以高努力编译选项进行综合。综合与功耗估算采用保守的功耗模型。

模式控制信号的延迟未被优化，因为假设相同精度会连续使用多个周期（例如处理完整的 DNN 层）。

为了在降低精度时强制缩短关键路径的时序扩展，进行了多模式时序优化，以针对每种缩放场景优化速度。因此，电路延迟在各模式下被独立约束和测量（即使该单元在设计上不允许缩放，例如传统 MAC 在数据门控输入时仍会接收多个约束以优化较短的关键路径）。对于每种精度模式，已知的静态信号和未使用的寄存器都会被如此声明，以防止与工作负载无关的静态时序分析（STA）在它们身上不必要地消耗资源。

## D. 功耗估算

功耗通过在每种模式下以最高工作频率进行的综合后仿真来估算，仿真使用 ModelSim 仿真器和 Cadence 功耗提取工具。这种基于开关活动的估算非常重要，因为根据精度模式的不同，被门控或未使用的逻辑会导致动态功耗的巨大差异。

每种模式的仿真由 10,000 次 MAC 操作组成，其中每 50 次操作对累加寄存器复位一次。仿真使用一组高斯分布的随机数，这些随机数可代表针对 CIFAR-10 训练的量化 DNN。

输入寄存器以及可在阵列内多个 MAC 之间共享的开销不包含在所报告的功耗结果中。

## E. DVFS

动态电压频率调节（DVFS）为功耗-频率优化提供了额外的维度，使工作点能够动态地移动到更接近各模式最优点，从而进一步提高能效。

本研究对 DVFS 对电路的影响进行了建模，在电压从 1 V 扫描至 0.8 V 的过程中评估每种模式的吞吐量与能耗。为此，0.8 V 与 1 V 特性化库之间的延迟使用 Sakurai-Newton 模型 [31] 进行估算，而能耗则由先前的门级仿真进行二次插值得到。

## V. 详细分析

本节针对每种架构在每种可缩放级别下的一个代表性综合实例评估其电路特性。在本次详细分析中，仅选取能耗-延迟-面积积最低的电路。

图 16-19 展示了各电路的存储带宽、单次操作能耗和硅面积的分解。左侧子图展示对称缩放场景，右侧子图展示仅权重缩放场景。三元组柱状条区分了每个实例的精度场景（8、4 或 2 位）。

## A. 带宽

图 16a-b 展示了不同 MAC 单元在各模式下每时钟周期所需的存储带宽。

这展示了输入与输出带宽之间的内部权衡。SA MAC 在设计上并行产生多个独立结果，因此当精度降低时（即对于 1 级可缩放电路降至 4 位，对于 2 级可缩放电路降至 2 位），其输出带宽增加。然而，由于大多数神经网络允许在不同输出之间进行数据复用，输入可以在子计算之间共享，从而带来较低的输入带宽。

相反，ST MAC 仅存储一个结果，即多个低精度乘法之和，因此其输出带宽相对较小。然而，由于这些乘积是同一输出元素的一部分，它们无法共享相同的输入。这导致其输入带宽大幅增加。例如，为了在 2b 2b 模式下保持其 16 个算术块全部处于忙碌状态，2D D&C ST 的输入带宽急剧膨胀。这种高度不稳定的带宽可能会使其存储器接口及其在芯片中的集成变得复杂。

最后，基于时间复用的 MAC 单元的带宽要小得多，这对于存储带宽受限的系统而言，相对基于空间复用的设计可能是巨大优势。然而，由于其计算能力也较低，该结果应相对于计算吞吐量来考量。

## B. 每次操作的带宽

图 17a-b 因此展示了每个 MAC 的每次操作带宽，该指标由上述带宽（位/时钟周期）和计算能力（操作/时钟周期）推导得出。

a) 对称缩放
![](images/6e46151590b38512f720e237b3f6b2f6925062ddc21a10fde7b950818ca68500.jpg)

b) 仅权重缩放
![](images/ae43040e269a3a58c4059fd54da11a60a0aee02adcb9a43dc142cacc7b2e223b.jpg)
图 16. 精度可缩放 MAC 单元在对称（a）与仅权重缩放场景下每时钟周期带宽的柱状图

a) 对称缩放
![](images/25e9a938d7c54417e9fa48fea2c293fd4ffbe11874cb15000e2acb0228ff798c.jpg)

b) 仅权重缩放
![](images/6457c600bcfb2ad1c3bb936ee8d64e91d0b84d2f500458914e68aead20fbb5b2.jpg)
图 17. 精度可缩放 MAC 单元在对称与仅权重缩放场景下每次操作带宽的柱状图。

a) 对称缩放
![](images/3641f8a906f7667b67037631855a46a954c8ad6ba85677fec6a72b1a31eee5aa.jpg)

b) 仅权重缩放
![](images/8b9515bd76acaaf215c50fd5f4585275610df5fc84faaecaf542302b4f2ba4c0.jpg)

图 18. 精度可缩放 MAC 单元在对称与仅权重缩放场景下归一化电路吞吐量的柱状图。
![](images/9b810bafaba48a27fb8435a7989b4c83b4f0a594f2cf9de21e96f28d7143294b.jpg)
图 19. 精度可缩放 MAC 单元归一化面积的柱状图。

![](images/94c7dfaac6672980ae12f76a2f5c68f50c988fe2a0a2b331a754a04de5b9fe46.jpg)

正如预期的那样，在全精度下，所有 MAC 的每次操作带宽相同。随着精度降低，ST 类 MAC 相比其他类型表现出优势，尤其是 1D 和 2D D&C ST。事实上，其算术块保持完全利用，这使其计算能力维持在较高水平，而其输出带宽仍然很低，从而抵消了增加的输入带宽。

对于对称缩放，SWP MAC 因其算术逻辑未被利用而失去优势，导致其计算能力相对较低。SWP ST 略优于 SWP SA，因为在计算能力相同的情况下，其带宽需求更低。对于非对称缩放，SWP MAC 仅受益于数据门控，其表现与基线电路类似。

在时间缩放的 MAC 单元中，低带宽与低计算能力相互抵消，最终其每次操作带宽与基线电路相当。

## C. 吞吐量评估

图 18a-b 比较了不同架构在全精度模式下相对于数据门控（data-gated）常规 MAC 的实测吞吐量（已归一化）。

在 8b 和 4b 精度下，大多数设计的 1 级和 2 级变体表现出相近的吞吐量。这一结果表明，吞吐量不受可扩展级别的影响，这与面积和能耗的情况相反。因此，追求高性能的系统可以放心地考虑嵌入多种精度模式。

尽管采用了多模式优化，1 级设计从 4b 降至 2b 时通过数据门控获得的速度提升微乎其微。相反，可扩展级别为所有设计带来了接近二次方的性能提升。2 级可扩展设计在 2b 精度下达到了极高的吞吐量倍数。其中，2D D&C SA 在对称缩放下凭借其 16 个并行子计算，达到了基准吞吐量的 14.5 倍。其对对称与不对称缩放的双重兼容性，使其在仅权重缩放场景中也能获得排名第二的良好表现，达到基准吞吐量的 3.1 倍。

有趣的是，2 级 1D D&C SA 尽管未针对仅权重和对称缩放进行专门优化，却在这两种场景下都表现出良好的通用性。它在对称和不对称场景下分别达到了基准吞吐量的 4.6 倍和 3.5 倍。这得益于其第一级加法器，该加法器在所有模式下都处于关键路径上，因此对其优化可以使所有模式共同受益。而在 SWP 设计中，由于关键路径在不同模式间发生变化，乘法器硬件必须针对不同目标进行协同优化。其速度出色的另一个原因在于其 SA 累加级仅包含较短的加法器，因而速度更快。

## D. 面积分解

图 19 展示了在 28 nm CMOS 工艺下综合得到的不同 MAC 单元的面积分解，已相对于数据门控的常规 MAC 电路面积归一化。

不出所料，所有基于子计算并行性的可扩展单元都需要为可配置性付出开销。其中，1D 和 2D D&C 方法因采用定制化的内部结构而面积开销最大，2D D&C 设计的面积最高可达常规 MAC 的 4.4 倍（不含第 IV-A 节所述的输入寄存器）。

相比之下，SWP MAC 通过复用冗余算术单元进行子字并行计算，从而缓解了开销。SWP ST 电路尤为高效，1 级和 2 级可扩展性分别仅带来 10% 到 18% 的开销。

串行设计是唯一一种面积小于常规乘法器的设计类型，可为面积受限的系统在 MAC 电路上节省高达 40% 的面积。

关于时序面积（图中深色柱），D&C SA 设计比其他设计需要多得多的寄存器。这是由于其较宽的不对称子积（其中一个操作数保持全精度），以及 2D D&C SA 中子积数量的二次方增长。虽然同为 SA 类型，SWP SA 以牺牲吞吐量为代价，保持了较低且增长缓慢的时序面积。

## E. 全精度下的能耗开销

图 20-27 展示了每种架构在精度缩放时每次运算的能耗分解。左子图为对称缩放场景，右子图为仅权重缩放场景。所有能耗值均相对于同一个全精度数据门控常规 MAC（以黑色实线表示）归一化。

使用可扩展设计进行全精度计算总是伴随一定的能耗损失。对于 1D D&C 和 SWP MAC（图 20-21、24-25），在 1 级和 2 级可扩展性下，8b 计算的能耗开销分别约为 20% 到 40%。对于 2D D&C 架构（图 22-23），该开销分别增加至 52% 和 94%。

串行设计（图 26-27）在全精度下需要更多的能耗，因为其每次计算需要多个时钟周期，从而将功耗分散到了时钟树中。在全精度下，1D 位串行 MAC 每次运算的能耗是数据门控方式的 3.3 倍，而 2D 位串行则高达 14 倍。

令人欣慰的是，1D 多位串行设计带来的能耗损失较低：2 位串行 MAC 的能耗为基准的 2.2 倍，同时其设计本身还能够将精度向下缩放至 2 位；4 位串行 MAC 则将代价降低至数据门控方式的 1.5 倍。

对于几乎所有电路，2 级 MAC 在全精度下的能耗都高于其 1 级版本。唯一例外的是 1D D&C ST 和 SWP ST 架构，其第二级可扩展性几乎不比第一级带来更多开销。这与其更简单的 ST 式累加以及乘法逻辑的高效复用密切相关。

## F. 能耗缩放

图 20–27 可用于评估每种架构的缩放效率。

总体而言，通过简单的数据门控同时降低权重和激活值的精度（左侧子图），可实现随比特精度线性变化的能耗节省。相比之下，所有可缩放 MAC 在 8b 到 4b 精度之间表现出更陡的斜率，这意味着它们随比特精度以超线性方式节省能耗。在低于 4 比特时，1 级可缩放电路（包括 1D 和 2D 4-bit 串行 MAC）只能通过数据门控来缩放精度，斜率回到基线的水平。尽管可缩放性较弱，但由于开销更低，这些设计相比 2 级可缩放 MAC 仍能展现出可观的能耗节省。

![](images/d0a00c6e072a0b350d612195876ceedc533a7576237a3e0b83524b50755b13da.jpg)

![](images/9b9cc48924cb73e6d2b1f2cf3d734197af47c32d7c93fe88f9cb9836440f23a1.jpg)

图 20. 1D D&C SA MAC（DNPU）[13] 中对称缩放（a）与仅权重缩放（b）的归一化能耗/操作。  
![](images/b94cd9e61a42c5ab09aa4126543f53b629599be37c464105eff571e22028f3dc.jpg)

![](images/c14832109a5225419c28e97c09f7ad9865b2eb42766c05470998135fd46355be.jpg)  
图 21. 1D D&C ST MAC 中对称缩放（a）与仅权重缩放（b）的归一化能耗/操作。

![](images/90dbb7f79ba144f9bf7d948e6849018960aaa43a2673c2778bc5f5b21d0f67ce.jpg)

![](images/0e48d059bc618234b520291fc3b22c82599cb622bde929087602e9544bd878f6.jpg)  
图 22. 2D D&C SA MAC 中对称缩放（a）与仅权重缩放（b）的归一化能耗/操作。  
图 23. 2D D&C ST MAC（BitFusion）[14] 中对称缩放（a）与仅权重缩放（b）的归一化能耗/操作。

图 24. SWP SA MAC（DVAFS）[15] 中对称缩放（a）与仅权重缩放（b）的归一化能耗/操作。  
![](images/f40113b40ade64f733db63fb99ebaf0c5379eaee04a1d25d33b0e6d8de1b8ca2.jpg)

![](images/9b93de5147ef0d846bb24359613b6a4b12df6f28d9e1f37c4a7e3dc13d6deebb.jpg)

![](images/fd9e3365948f24b8a38f0e318cc43740d310702349e1180662ecdbdb05d79f4f.jpg)

![](images/25ff735963ef525f0ec9d316ec37b9878dc7eb0ff1ce69623eaca0c5919a2380.jpg)  
图 25. SWP ST MAC（ST）[16] 中对称缩放（a）与仅权重缩放（b）的归一化能耗/操作。

![](images/b8f7b750864176c876c7b960de0daab521d76ffb83b212414caac595e67f3a7d.jpg)

![](images/684d3fd249ab92aa0a87b908d93658014ce7b21cbed4085c097583e51d864155.jpg)

![](images/d415006d79682428ae0e78a004cefec56e1f38f45aa8ce47de4beeb41a35c92e.jpg)

![](images/3bf2d92b5078decf14ace82e5a88940743760dc245cfdaa5bdc9465e9b039dbb.jpg)  
图 26. 1D 串行 [17] 与多比特串行 [18] MAC 中对称缩放（a）与仅权重缩放（b）的归一化能耗/操作。注意坐标尺度。

![](images/65fc569076941848cdbfcf0710c7df5c1b6c7837ae9d186e01b681cfeb54e521.jpg)

![](images/47a64d7a4ea6e3fb3cd46862e885e43919696037042154d45c8d6892fa86fa7b.jpg)  
图 27. 2D 串行与多比特串行 MAC（LOOM）[19] 中对称缩放（a）与仅权重缩放（b）的归一化能耗/操作。注意坐标尺度。

当保持全精度激活值（右侧子图）时，能耗节省无一例外地低得多。两种 SWP 架构（图 24–25）只能通过数据门控来缩放能耗，因此不适合仅权重缩放。尽管其架构为两种场景而设计，2D D&C（图 22–23）和 2D 串行（图 27）MAC 在仅权重缩放方面同样显得低效。只有 1D D&C

ST MAC（图 21）被证明在两种缩放场景下都高度高效，其次是 1D 4-bit 串行（图 26）MAC 电路，但差距较大。

1D D&C SA 单元（图 20）的斜率是空间展开架构中最低的之一。尽管专为仅权重缩放而构建，2 级 1D D&C SA 电路在 4b 模式下的能耗甚至高于数据门控，并且在 2b 模式下仅节省与其 1 级变体同样多的能耗（9%）。这表明 1D D&C SA 架构在能耗效率方面的缩放表现很差。相反，1D D&C ST 架构（图 21）高度高效，并证明在增加额外缩放级别后仍能维持其效率。

在空间展开架构中，2D D&C ST MAC（图 23）和 SWP ST MAC（图 25）在对称降低精度时具有最陡的能耗下降。它们在 2b 精度下带来相当的节省（1 级可缩放设计为 48%，2 级可缩放设计为 68%）。当使用 8b 精度时，SWP ST MAC 因开销更低而更受青睐，而当需要非对称缩放时，则应优先选择 2D D&C ST MAC。

## VI. 比较研究

本节从每次操作的能耗以及单位面积的吞吐量两方面，对所有 MAC 架构进行整体比较，涵盖 1V 标称电源电压以及 DVFS 条件下的情况。

## A. 标称电压下可缩放 MAC 的比较

图 28 展示了每种精度可缩放 MAC 设计在各精度场景下的可行实现与 Pareto 前沿，并在标称电源电压下扫掠了广泛的频率约束范围。电路以每次操作的能耗和单位面积的吞吐量进行比较。最优的电路位于各子图的右下角。

在 8b 精度下（图 28c），数据门控无疑是最高效的，能够实现最高的单位面积吞吐量，其次是 1 级（亮色虚线），然后是因关键路径更长而受影响的 2 级（暗色点线）可缩放设计。

在对称缩放精度时（图 28a–b），2D D&C ST 和 SWP ST 架构在每次操作的能耗方面大幅领先其他架构。2D D&C SA 电路在吞吐量-面积性能方面遥遥领先，但由于其庞大的非共享寄存器，能耗效率欠佳。

注意，在 4b 精度及以上时，1 级可缩放电路始终是最佳折中方案，但在 2b 精度时这一趋势发生逆转，这可以通过左图与中间子图之间亮色曲线与暗色曲线的倒置看出。

当仅将权重精度降至 2b 时（图 28d），2 级 1D D&C ST 架构在能耗方面最优，而其配套的 1D D&C SA 则在吞吐量-面积效率方面表现出色。在 4b 精度下（图 28e），1 级 1D D&C ST 和 1D 2-bit 串行在能耗方面最佳，但相对基线的增益有限（最多 20%）。通过旁路内部加法，1D D&C SA 在吞吐量方面占优，而 1D 2-bit 串行则受益于更小的电路面积。

## B. DVFS 下可缩放 MAC 的比较

图 29 展示了在 0.8 至 1 V 之间进行 DVFS 时，各精度可缩放设计的比较与 Pareto 前沿。

各设计之间的相对比较与标称电压下相当类似。然而，所有设计的 Pareto 前沿均明显大于标称电压下的情形，尤其是在可实现的吞吐量-面积解的范围方面。这一点在 SWP ST 架构上尤为明显。DVFS 提供了一个平滑的自由度，使电路能够在目标吞吐量或能耗预算内运行于最优点。

## A. 引言与方法论

最佳的可扩展精度电路既不是在降低精度时功耗最低的电路，也不是开销最小的电路，更不是能耗缩放最陡峭的电路。最佳的是最能优化其应用需求的电路。因此，本节针对 MAC 单元的实际使用场景进行对比研究，而非针对每种精度模式逐一分析。此类研究至关重要，因为它与应用层建立了直接联系，并且通过将许多结果合并为单一权衡，简化了分析。

通过定义 MAC 在每种精度模式下执行运算的百分比，对三个案例研究进行了评估。为简单起见，仅设定全精度运算的比例，而 4b 与 2b 运算的比例被假定相等。在这些比例下对能效和吞吐效率进行加权，以重新绘制全局 Pareto 前沿。

图 30 展示了全精度运算分别占 33%、20% 和 5% 的案例研究（为清晰起见，2D 1-bit 串行 MAC 由于其次优性而被省略）。此处仅考虑标称电压，因为在 1 V 下的比较结果在 DVFS 下依然成立。

## B. 均等使用

首先，当均匀使用所有精度模式时（左侧子图），采用简单数据门控的传统 MAC 接近最高效的架构，同时在吞吐量和面积方面也是遥遥领先的。事实上，无论缩放精度运算多么高效，由于其速度慢得多、能耗高得多，8b 运算在能耗预算中占据绝对主导地位。

请注意，即使在此使用场景中 2b 和 4b 运算占计算总量的 66%，由于这些运算在缩放精度下通常快 2 到 15 倍（参见 V-C 节），电路停留在这些精度模式的时间远短于执行 8b 运算的时间。

对于对称缩放场景，传统 MAC 单元略逊于 1-level SWP SA 架构以及 1-level 和 2-level SWP ST 电路。另一方面，对于仅权重缩放，传统单元在能效上依然优于 1-level 1D D&C SA 和 ST MAC 电路。

## C. 20% 全精度计算

幸运的是，DNN 已被证明对更高比例的缩放精度计算具有韧性。将全精度运算的比例降至 20%（中间子图）后，对于对称缩放场景，可扩展 MAC 便值得投入精力以降低能耗。在这一精度比例下，SWP SA 和 ST 设计在能效方面表现出色，相比数据门控可节省高达 20% 的能耗。

在对称缩放下，1-level SWP SA 和 ST 电路持续超越传统 MAC。总体而言，1-level 可扩展设计的性能优于其对应的 2-level 设计，唯一的例外是 2-level SWP ST 电路表现更为突出。

然而，对于仅权重缩放场景，可扩展 MAC 在能耗方面基本仍然低效。2-level 1D D&C ST 架构勉强优于传统 MAC，但代价是面积-吞吐能力明显变差。

![](images/71c8e588c24d3b43632981c20f23716ed283a613c8e8f125b8f354026b68b497.jpg)

![](images/2e16e85943c1dc7be615a9c6a4233e4138e1c50798faf4dfe830c113b0a7aad6.jpg)

![](images/cfea69eb420f368c2045dd3612d07659263f76602bcd0b9bdc54b52d55645e98.jpg)

![](images/9b7f648bd0d51556d6f55e0375facec214892d92cf954c73f8164c359afbe27b.jpg)

![](images/21d3cbfe508b420b1b9d70a2f3b88d63392e5d1411263b579d8ecfbf804733e5.jpg)  
→Data gating -1-level 1D D&C SA ……2-Ievel 1D D&C SA DNPU [13] -1-level 1D D&C ST .....2-level 1D D&C ST --1-level 2D D&C SA ……2-level 2D D&C SA -1-level 2D D&C ST BitFusion [14] …2-level 2D D&C ST --1-level SWP SA DVAFS [15] ……2-Ievel SWP SA -1-Ievel SWP ST ……2-leyel SWP ST ST [16] ...-1D 1-bit serial UNPU [17] --1D 2-bit serial Multibit serial [18] - 1D 4-bit serial ……2D 1-bit serial --2D 2-bit serial LOOM [19] -2D 4-bit serial

图28：28nm CMOS、1V下综合的MAC架构，以每操作能量和单位面积吞吐比较。  
![](images/f4a2029cb0147e0160545a0f1106bbe2bf0b36a3be7f7d2f62cdae3ac2dc4c66.jpg)

![](images/a187cb845e8076b747f328a3174e0c66f30fdf7bfe281daec0f7c03026831dd1.jpg)

![](images/43f5f68b3cf251a699f6c86a93fae71759810b0df8534e8b1c0461324f0a590d.jpg)

![](images/d57a2c85c5dc38bbcd6c656b7dfda2d2eb712116ebbb89955c5ebbee7ed9c1ca.jpg)

![](images/6aa84bb9ecda542adbb123e58e0f4a0c8fe667b9ca1bd8a61b563ed095f94c11.jpg)  
图29：28nm CMOS、动态电压频率缩放下的MAC每操作能量与单位面积吞吐比较。

Data gating   
-1-level 1D D&C SA   
.2-level 1D D&C SA DNPU [13]   
-1-level 1D D&C ST   
....2-level 1D D&C ST   
--1-leveL2D.D&C SA   
…2-level 2D D&C SA   
-1-level 2D D&C ST BitFusion [14]   
…2-level 2D D&C ST   
DVAFS [15]   
--1-level SWP ST   
……2-level SWP ST ST [16]   
...-1D 1-bit serial UNPU [17]   
---1D 2-bit serial Multibit serial [18]   
- 1D 4-bit serial   
……2D 1-bit serial   
--2D 2-bit serial LOOM [19]   
-·2D 4-bit serial

Data gating -1-level 1D D&C ST -1-level 2D D&C SA1-level 2D D&C ST --1-level SWP SA1-Ievel SWP ST ©1D 1-bit serial 2D 1-bit serial - 1-level 1D D&C SA 2-level 1D D&C ST 2-level 2D D&C SA ·2-level 2D D&C ST …2-level SWP SA 2-level SWP ST --1D 2-bit serial →2D 2-bit serial 2-level 1D D&C SA -01D 4-bit serial -2D 4-bit serial

![](images/d26675dff5ae1024015d257eef30d14c33076c7fafcb680d1624bed35eccc29c.jpg)

![](images/f9991385463af6c65f3791ca779d00cbfa0555fffa143d8d709a855233798418.jpg)

![](images/efc94e5e11552564b0a4605fe270392c10526650a23792db7c3de1053ad351a3.jpg)

![](images/ad51e84bf1d510b7f7d39c4e420a37c8d3af7274bf2abd955e86beae03fefd1b.jpg)

![](images/59c5072536f7cf3aeac4d7000791c18d411ac09bb9d5d1903a1c0fe14cab1421.jpg)

![](images/6cbb443a0f3ecbe15480fd0a06067a49d7fedca4cb0a9f716b8c6d41bc2f4d9d.jpg)  
图30：8位操作占33%、20%、5%时，MAC的整体每操作能量与单位面积吞吐。

## D. 5% 全精度计算

在真实世界的 DNN 中，一些低复杂度应用可以使用更少比例的全精度运算 [14]，例如仅 5% 左右。第三个使用场景即采用这一 8b 运算比例（右侧子图）。在这一 8b 运算比例下，SWP ST 和 2D D&C ST 在对称场景中大幅超越数据门控，分别高出 32% 和 22%。

最终，一些可扩展 MAC 单元此时在仅权重缩放场景中也带来了收益，但能耗增益较低，2-level 1D D&C ST 最高为 15%，其次是 1-level 1D D&C ST 和 1D 4-bit 串行 MAC（8%）。

尽管 1D D&C SA 在对称和仅权重两种场景下的能耗都与数据门控基本相当，但它在吞吐量/面积方面是最佳架构，相比数据门控带来 10% 至 30% 的更高性能。

即便在这一水平上，即 95% 的运算由 2b 和 4b 精度分摊，也没有其他 MAC 单元能够展现出相比简单数据门控 MAC 的优势。特别地，尽管近期串行设计流行，除 1D 4-bit 串行外，它们在所有场景中都被证明是低效的。

这些结果证实了设计者应将资源集中于能带来最大收益之处的重要性，而不是落入“实现并优化到最低层级”的陷阱——后者对整体系统可能只有有限甚至递减的回报。这种设计权衡上的策略，在电路层面呼应了 Gene Amdahl 定律 [32]，应当成为设计者最普遍遵循的原则之一。

## VIII. 结论

本工作提出了一种新的精度可扩展 MAC 架构分类法，依据多个标准对其进行归类：展开类型（空间展开或时间展开）、展开的维度（1D、2D 或 2D 对称），以及对于基于空间展开的设计，其累加类型（Sum Together 或 Sum Apart）。这一新的分类不仅对所有现有架构进行了归类，还揭示了新的设计模式，可以为设计者在系统级和算法级上选择合适类型的处理单元提供洞察。除该分类法外，本文还对当前最先进及进一步的架构进行了详尽的调研，以清晰地理解它们的基本原理、特性、联系与差异。

随后，本文对这些架构进行了基准测试与比较。其中，所有电路均采用 28 nm 商用 CMOS 工艺实现，覆盖了广泛的性能目标区间，精度范围为 2 至 8 位。该研究从能量、吞吐量、带宽和面积等方面，对这些可扩展的 MAC 单元进行了理论与数值上的深入分析，旨在理解通过精度缩放来降低神经网络处理中计算成本的关键趋势。

这项对比研究的结果表明，2D D&C ST（BitFusion）[14] 和 SWP ST（ST）[16] 在对称缩放场景下具有最高的能效，而 1D D&C ST 和 1D 4 位串行 [18] 则在权重缩放场景下表现最佳。此外，1D D&C SA（DNPU）[13] 和 2D D&C SA 在所有缩放场景下均以高吞吐量胜出，但它们与 2D D&C ST（BitFusion）一样，都存在带宽需求变化范围大的问题。尽管近期出现了 1D [17], [18] 和 2D [19], [20] 串行设计的趋势，但这些设计在吞吐量和能效两方面均受到严重制约。

最后，本比较通过在每种精度下以固定比例的运算为约束的三个实际使用案例研究进行了总结，揭示了全精度计算即使在占比很小的情况下，也会对整体能耗预算产生巨大影响。在对称缩放场景中，尽管 2D D&C ST（BitFusion）在低精度计算上单独来看更为高效，但由于其在全精度模式下开销更高，在任何使用案例中都被 SWP ST（ST）所超越。然而，对于仅权重的场景，精度可配置设计大多效果不佳，只有在全精度运算占比非常低的情况下，两级 1D D&C ST 才变得有利。

深度学习应用的兴起——这些应用已被证明对使用缩放精度具有极强的鲁棒性——催生了巨大的研究热潮，并推动了这一新研究方向的转变。在与深度学习处理相关的无数技术和设计的涌现之中，本工作总结并厘清了精度可扩展 MAC 电路方面的进展，以弥合与软件社区之间的鸿沟，并将嵌入式神经网络处理推向成熟阶段。未来的研究将把这些 MAC 级别的理解纳入更高的抽象层次，最终逐步构建至应用层面。

## REFERENCES

[1] W. Wen, C. Wu, Y. Wang, Y. Chen, and H. Li, “Learning structured sparsity in deep neural networks,” in Proc. 30th Int. Conf. Neural Inf. Process. Syst. (NIPS), 2016, pp. 2082–2090.

[2] J. Yu, A. Lukefahr, D. Palframan, G. Dasika, R. Das, and S. Mahlke, “Scalpel: Customizing dnn pruning to the underlying hardware parallelism,” ACM SIGARCH Comput. Archit. News, vol. 45, no. 2, pp. 548–560, 2017.

[3] X. Yang et al., “DNN dataflow choice is overrated,” Sep. 2018, arXiv:1809.04070. [Online]. Available: https://arxiv.org/abs/1809.04070

[4] H. Kwon, P. Chatarasi, M. Pellauer, V. Sarkar, and T. Krishna, “A datacentric approach for modeling and estimating efficiency of dataflows for accelerator design,” May 2018, arXiv:1805.02566. [Online]. Available: https://arxiv.org/abs/1805.02566

[5] A. Parashar et al., “Timeloop: A systematic approach to DNN accelerator evaluation,” in Proc. IEEE Int. Symp. Perform. Anal. Syst. Softw. (ISPASS), Mar. 2019, pp. 304–315.

[6] V. Camus, M. Cacciotti, J. Schlachter, and C. Enz, “Design of approximate circuits by fabrication of false timing paths: The carry cutback adder,” IEEE J. Emerg. Sel. Topics Circuits Syst., vol. 8, no. 4, pp. 746–757, Dec. 2018.

[7] H. Jiang, C. Liu, L. Liu, F. Lombardi, and J. Han, “A review, classification, and comparative evaluation of approximate arithmetic circuits,” ACM J. Emerg. Technol. Comput. Syst., vol. 13, no. 4, pp. 60:1–60:34, Aug. 2017.

[8] L. Yang and B. Murmann, “SRAM voltage scaling for energy-efficient convolutional neural networks,” in Proc. 18th Int. Symp. Qual. Electron. Design (ISQED), Mar. 2017, pp. 7–12.

[9] I. Hubara et al., “Quantized neural networks: Training neural networks with low precision weights and activations,” J. Mach. Learn. Res., vol. 18, no. 1, pp. 6869–6898, Jan. 2017. [Online]. Available: https://dl.acm.org/citation.cfm?id=3122009.3242044

[10] L. Cavigelli and L. Benini, “Origami: A 803-GOp/s/W convolutional network accelerator,” in Proc. IEEE Trans. Circuits Syst. Video Technol. (TCSVT), Nov. 2017, pp. 2461–2475.

[11] B. Moons, B. De Brabandere, L. Van Gool, and M. Verhelst, “Energyefficient ConvNets through approximate computing,” in Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV), Mar. 2016, pp. 1–8.

[12] C. Sakr and N. Shanbhag, “An analytical method to determine minimum per-layer precision of deep neural networks,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), Apr. 2018, pp. 1090–1094.

[13] D. Shin, J. Lee, J. Lee, and H.-J. Yoo, “DNPU: An 8.1TOPS/W reconfigurable CNN-RNN processor for general-purpose deep neural networks,” in IEEE Int. Solid-State Circuits Conf. (ISSCC) Dig. Tech. Papers, Feb. 2017, pp. 240–241.

[14] H. Sharma et al., “Bit fusion: Bit-level dynamically composable architecture for accelerating deep neural networks,” in Proc. 45th IEEE Int. Symp. Comput. Archit. (ISCA), Jun. 2018, pp. 764–775.

[15] B. Moons, R. Uytterhoeven, W. Dehaene, and M. Verhelst, “DVAFS: Trading computational accuracy for energy through dynamic-voltageaccuracy-frequency-scaling,” in Proc. Desing Automat. Test Eur. Conf. Exhib. (DATE), Mar. 2017, pp. 488–493.

[16] L. Mei et al., “Sub-word parallel precision-scalable MAC engines for efficient embedded DNN inference,” in Proc. IEEE 1st Int. Conf. Artif. Intell. Circuits Syst. (AICAS), Mar. 2019, pp. 6–10.

[17] J. Lee, C. Kim, S. Kang, D. Shin, S. Kim, and H.-J. Yoo, “UNPU: A 50.6TOPS/W unified deep neural network accelerator with 1b-to-16b fully-variable weight bit-precision,” in IEEE Int. Solid-State Circuits Conf. (ISSCC) Dig. Tech. Papers, Feb. 2018, pp. 218–220.

[18] V. Camus, C. Enz, and M. Verhelst, “Survey of precision-scalable multiply-accumulate units for neural-network processing,” in Proc. IEEE 1st Int. Conf. Artif. Intell. Circuits Syst. (AICAS), Mar. 2019, pp. 57–61.

[19] S. Sharify, A. D. Lascorz, K. Siu, P. Judd, and A. Moshovos, “Loom: Exploiting weight and activation precisions to accelerate convolutional neural networks,” in Proc. ACM/EDAC/IEEE Design Autom. Conf. (DAC), Jun. 2018, pp. 20:1–20:6.

[20] K. Ueyoshi et al., “QUEST: A 7.49 TOPS multi-purpose log-quantized DNN inference engine stacked on 96 MB 3D SRAM using inductivecoupling technology in 40 nm CMOS,” in IEEE Int. Solid-State Circuits Conf. (ISSCC) Dig. Tech. Papers, Feb. 2018, pp. 216–218.

[21] S. Ryu, H. Kim, W. Yi, and J.-J. Kim, “BitBlade: Area and energyefficient precision-scalable neural network accelerator with bitwise summation,” in Proc. 56th Annu. Design Autom. Conf. (DAC), 2019, pp. 84:1–84:6.

[22] Z. Du et al., “ShiDianNao: Shifting vision processing closer to the sensor,” in Proc. ACM/IEEE Int. Symp. Comput. Archit. (ISCA), Jun. 2015, pp. 92–104.

[23] B. Moons, R. Uytterhoeven, W. Dehaene, and M. Verhelst, “Envision: A 0.26-to-10 TOPS/W subword-parallel dynamic-voltage-accuracyfrequency-scalable convolutional neural network processor in 28 nm FDSOI,” in IEEE Int. Solid-State Circuits Conf. (ISSCC) Dig. Tech. Papers, Feb. 2017, pp. 246–247.

[24] N. P. Jouppi et al., “In-datacenter performance analysis of a tensor processing unit,” ACM SIGARCH Comput. Archit. News, vol. 45, no. 2, pp. 1–12, 2017.

[25] Y. H. Chen, T. Krishna, J. S. Emer, and V. Sze, “Eyeriss: An energyefficient reconfigurable accelerator for deep convolutional neural networks,” IEEE J. Solid-State Circuits, vol. 52, no. 1, pp. 127–138, Jan. 2017.

[26] T. Chen et al., “DianNao: A small-footprint high-throughput accelerator for ubiquitous machine-learning,” ACM SIGPLAN Notices, vol. 49, no. 4, pp. 269–284, Feb. 2014.

[27] C. Zhang, P. Li, G. Sun, Y. Guan, B. Xiao, and J. Cong, “Optimizing FPGA-based accelerator design for deep convolutional neural networks,” in Proc. ACM/SIGDA Int. Symp. Field-Program. Gate Arrays, 2015, pp. 161–170.

[28] J. Qiu et al., “Going deeper with embedded FPGA platform for convolutional neural network,” in Proc. ACM/SIGDA Int. Symp. Field-Program. Gate Arrays (FPGA), 2016, pp. 26–35.

[29] M. Song, J. Zhang, H. Chen, and T. Li, “Towards efficient microarchitectural design for accelerating unsupervised GAN-based deep learning,” in Proc. IEEE Int. Symp. High Perform. Comput. Archit. (HPCA), Feb. 2018, pp. 66–77.

[30] B. Moons and M. Verhelst, “A 0.3–2.6 TOPS/W precision-scalable processor for real-time large-scale ConvNets,” in Proc. IEEE Symp. VLSI Circuits (VLSI-Circuits), Jun. 2016, pp. 1–2.

[31] T. Sakurai and A. R. Newton, “Alpha-power law MOSFET model and its applications to CMOS inverter delay and other formulas,” IEEE J. Solid-State Circuits, vol. 25, no. 2, pp. 584–594, Apr. 1990.

[32] G. M. Amdahl, “Validity of the single processor approach to achieving large scale computing capabilities,” in Proc. AFIPS Spring Joint Comput. Conf., 1967, pp. 483–485.



![](images/885e5f08a8017052ddfe6a5ef690840cea5d2262c27777b47cd4a1a384630427.jpg)

Vincent Camus（S'15–M'19）于 2019 年在瑞士洛桑联邦理工学院（EPFL）获得博士学位。

他于 2013 年加入瑞士洛桑联邦理工学院集成电路实验室（ICLAB）。2018 年，他在比利时鲁汶大学 ESAT-MICAS 实验室担任访问学者。他目前是瑞士马林市斯沃琪集团研发有限公司（The Swatch Group Research and Development Ltd.）的研发数字设计与软件工程师。他已合作发表约 20 篇科学论文或专利，并参与了多个会议和公共科学活动的组织工作。他的研究兴趣包括低功耗数字电路、电子设计自动化、容错与精度可伸缩计算，以及高能效的嵌入式深度神经网络。

![](images/bd3c26b5890d4e6140fa5f7fea5c8e1812f3ca0b695ad8054bcd862670413a58.jpg)

Linyan Mei（S'18）于 2016 年在北京理工大学（BIT）获得电子科学与技术专业学士学位，并于 2018 年在比利时鲁汶大学（KU Leuven）获得电气工程硕士学位。她目前正在鲁汶大学 ESAT-MICAS 实验室攻读博士学位，研究方向为嵌入式机器学习处理器的数字设计。

她在硕士二年级期间曾在比利时鲁汶的 imec 担任实习生。她的研究兴趣包括高能效深度神经网络加速器、算法-硬件协同设计，以及精度可伸缩计算。

![](images/1e3cf6c5670d9be17c5ed9c9df791b52c315fab40b5b06723e39e1b57db3bb69.jpg)

Christian Enz（S'83–M'84–SM'11–F'19）于 1989 年在瑞士洛桑联邦理工学院（EPFL）获得博士学位。

在 2013 年之前，他担任瑞士电子与微技术中心（CSEM）副总裁，并负责集成与无线系统部门。在加入 CSEM 之前，他曾是 Conexant（前身为 Rockwell Semiconductor Systems，位于美国加州纽波特比奇）的首席高级工程师，负责面向射频应用的 MOS 晶体管建模与表征。他目前是 EPFL 的教授，同时担任微工程研究所所长和集成电路实验室负责人。他的专业技术领域包括低功耗模拟与射频集成电路、无线传感器网络以及半导体器件建模。他是 EKV MOS 晶体管模型的开发者。他已合作发表 250 余篇科学论文，并在众多会议上做过报告。他是瑞士工程科学院的个人会员。他曾于 2012 年至 2014 年当选 IEEE 固态电路学会（SSCS）AdCom 成员，目前担任 IEEE SSCS 瑞士分会主席。

![](images/718c1c598b5f7cd75220e8860d14455bf96d73f292984298ab57e7655a62d4d9.jpg)

Marian Verhelst（SM'13）于 2008 年在比利时鲁汶大学（KU Leuven）获得博士学位。

她于 2005 年夏季在加州大学伯克利分校伯克利无线研究中心担任访问学者，并于 2008 年至 2011 年在位于美国俄勒冈州希尔斯伯勒的 Intel Labs 担任研究科学家。她目前是鲁汶大学电气工程系 MICAS 实验室的副教授。她的研究方向聚焦于嵌入式机器学习、硬件加速器、自适应电路与系统、传感器融合，以及低功耗边缘计算处理。她是 DATE 和 ISSCC 执行委员会成员，担任 AICAS 2020 和 tinyML 2020 的 TPC 联合主席，以及 DATE 和 ESSCIRC 的 TPC 成员。她是 SSCS 杰出讲师，曾担任比利时青年科学院成员、弗拉芒政府 STEM 顾问委员会成员，以及 TVLSI、TCAS-II 和 JSSC 的副编辑。她目前持有欧盟享有盛誉的 ERC 启动资助，并于 2016 年成为比利时皇家学院获奖者。