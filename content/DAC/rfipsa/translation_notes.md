# RFIPSA: A Reconfigurable FIP-based Systolic Array for Various Matrix-Intensive Workloads 原文翻译

# RFIPSA：面向多种矩阵密集型工作负载的基于 FIP 的可重构脉动阵列

Zongfan Wu 华南理工大学 中国广州

Junyi Mai 华南理工大学 中国广州

Dong Jiang 华南理工大学 中国广州

Longyuan Kang 华南理工大学 中国广州

Weijie Chen 华南理工大学 中国广州

Enyi Yao<sup>∗</sup> 华南理工大学 中国广州

## 摘要

科学计算和人工智能日益增长的需求要求矩阵计算具备高吞吐量、灵活性和高效率。传统脉动阵列在处理稠密、稀疏和卷积工作负载时，存在计算密度有限、稀疏数据处理效率低和内存效率低等问题。本文提出一种基于快速内积（FIP）算法的可重构脉动阵列加速器，在减少硬件资源消耗的同时，支持稠密和稀疏两种计算模式。为进一步加速稀疏计算，本文提出一种新型稀疏编码方案，使加速器能够以极小的存储开销识别并跳过全零输入向量。此外，采用内存高效卷积算法的硬件处理模块最大限度地减少了卷积工作负载中的数据冗余和带宽需求。该原型采用 28nm CMOS 工艺实现，核心面积为 0.31mm<sup>2</sup>，工作频率为 500MHz，面积减少了 21.3%，SpGEMM 和 SpMM 相对于 GEMM 分别实现了最高 4.4× 和 5.1× 的加速比。

## 关键词

脉动阵列，稀疏性，硬件加速，矩阵乘法

## 1 引言

科学计算和人工智能（AI）的快速发展加剧了对灵活、高效和高吞吐量计算系统的需求。作为一种基础计算内核，通用矩阵乘法（GEMM）仍然是决定系统性能的主要因素 [1]。脉动阵列凭借其高并行度和低内存带宽需求，已在众多硬件实现中被证明能够高效执行 GEMM [2], [3], [4], [5], [6]。现代工作负载正变得日益复杂和多样化，例如大规模 GEMM、稀疏矩阵乘法（SpMM）、广义 SpMM（SpGEMM）以及卷积。处理这些不同的工作负载给经典脉动架构带来了重大挑战，具体体现在计算密度、稀疏性处理和内存效率方面。

计算密度挑战。通过扩展硬件资源来提升计算性能的传统策略正在接近一项根本性限制，通常称为加速器墙 [7]。超过这一饱和点后，以硬件为中心的优化所带来的收益会逐渐递减。进一步扩大芯片面积会带来难以承受的成本和能耗，而缩小面积又会损害吞吐量。这一困境凸显了算法与架构协同设计的必要性：在代数层面对计算进行重构，通过更少或成本更低的运算获得等价输出，从而在显著提高硬件效率的同时实现相当的性能 [8]。

稀疏性处理效率低。经典脉动阵列具有固定的处理单元（PE）结构和确定性数据流，难以高效处理稀疏数据，因为它们会对零值元素执行冗余且无效的运算 [9]。这导致计算时间浪费和过高的功耗。现有解决方案往往会损害架构的规则性 [10]、引入数值精度损失 [11]，或者依赖专用的稀疏性处理硬件，而这些硬件在稠密计算期间处于空闲状态，降低了整体资源利用率 [12]。因此，高效处理稀疏数据仍然是脉动阵列设计中的一项根本性挑战。

卷积中的内存瓶颈。卷积运算是 GEMM 的一种关键工作负载。然而，其固有的滑动窗口机制使其难以直接在脉动阵列上执行。尽管采用领域专用的脉动阵列结构可以优化卷积计算，但这种调整往往会限制其对其他工作负载的适用性 [13], [14]。广泛采用的 im2col 变换通过将卷积转换为 GEMM 运算解决了这种不兼容问题，但它引入了显著的数据冗余和大量内存开销 [15]。这种冗余会迅速耗尽有限的片上内存资源，并加剧带宽瓶颈。一些替代算法 [16], [17], [18]，例如内存高效卷积（MEC）[18]，通过重组数据流，在 CPU 和 GPU 上展现出了应用前景。然而，如何在 FPGA 和 ASIC 上高效实现这些算法，仍然是一个尚待解决的研究问题。

为应对这些挑战，本文提出一种统一的、基于可重构脉动阵列的加速器，通过算法与硬件协同设计解决计算密度、稀疏性和内存瓶颈问题。主要贡献总结如下：

• 提出一种新型脉动阵列架构，利用快速内积（FIP）算法，无需复杂预处理即可将乘法器用量减少约 50%，提升计算密度和硬件效率。

• 为所提出的脉动阵列引入一种可重构机制，使其能够在稠密和稀疏两种模式下高效运行，最大限度地实现资源复用。

• 提出一种压缩稀疏对角（CSD）稀疏编码方案，在保持存储效率的同时，加速该加速器中的稀疏计算。

• 设计一种基于 MEC 算法的硬件处理模块，通过重组数据访问模式来减少冗余、缓解内存压力并提高整体数据吞吐量。

## 2 背景与相关工作

## 2.1 经典脉动阵列

脉动阵列旨在通过在固定的二维 PE 网络上传输数据来执行矩阵乘法，将计算任务均匀分配到整个阵列中。这种结构在实现高并行度的同时，降低了对外部内存的带宽需求。脉动阵列通常分为三种主要的数据流模式 [19]：

1) 输出驻留（OS）：部分和在单个 PE 内累加，而两个输入矩阵的元素分别沿行和列传播。

2) 输入驻留（IS）：输入矩阵被预先加载，并在 PE 内保持驻留，而权重矩阵和部分和在阵列中传播。

3) 权重驻留（WS）：权重矩阵被预先加载，并在 PE 内保持驻留，而输入矩阵和部分和在阵列中传播。

尽管这些模式的数据流各不相同，但它们都依赖内积累加，如式（1）所示。

$$
c _ {i, j} = \sum_ {k = 1} ^ {K} a _ {i, k} \cdot b _ {k, j}\tag{1}
$$

这里，矩阵 A、B 和 C 的尺寸分别为 M × K、K × N 和 M × N。计算最终结果共需要进行 $M \cdot N \cdot K$ 次乘法。

## 2.2 快速内积算法

FIP 算法 [20] 从数学上证明了可以减少 GEMM 所需的乘法次数，如式 (2) 所示。

$$
\begin{array}{l} c _ {i, j} = \sum_ {k = 1} ^ {K / 2} \left(a _ {i, 2 k - 1} + b _ {2 k, j}\right) \left(a _ {i, 2 k} + b _ {2 k - 1, j}\right) - \alpha_ {i} - \beta_ {j} \\ \alpha_ {i} = \sum_ {j = 1} ^ {K / 2} a _ {i, 2 j - 1} \cdot a _ {i, 2 j}, \quad \beta_ {j} = \sum_ {i = 1} ^ {K / 2} b _ {2 i - 1, j} \cdot b _ {2 i, j} \end{array}\tag{2}
$$

![](images/eb49e352b4c0ad91f7c789f6d607db9be0e5393dfa07d6ff8a2c7e89d49ac4e6.jpg)  
图 1：传统的数据打包方法

![](images/04d9ffce754b431b4f99145e8737309bc50e9a788039adb88d01f7a50e8c1e52.jpg)  
图 2：MEC 算法的卷积映射方案。

与内积相比，FIP 算法将乘法总次数从 $MNK$ 减少到 $(MNK+MK+NK)/2$，而加法次数则从 $MNK-MN$ 增加到 $(3MNK+MK+NK)/2-MN-M-N$。

近期研究探索了 FIP 算法的硬件实现 [8]、[21]。文献 [21] 提出了一种减少乘法器数量的硬件原型，但其工作频率受到限制。文献 [8] 引入了自由流水线 FIP 算法，以缓解频率限制，但这一改进以大量预处理开销为代价，降低了整体硬件效率。

## 2.3 稀疏矩阵乘法优化

对于基于脉动阵列的硬件实现，优化空间主要沿两个方向展开：(1) 对 PE 进行解耦，以增强数据路由的灵活性 [10]、[22]、[23]；(2) 采用数据打包方法来提高数据密度 [12]、[24]、[25]。解耦的脉动阵列架构允许通过专用路由路径直接传输数据，提高了灵活性，但也损害了阵列固有的并行性，并引入了不可忽视的路由开销。相比之下，数据打包技术将稀疏矩阵整合为稠密表示，如图 1 所示。这些方法保持了阵列的规则性，但显著增加了 PE 的资源消耗，导致 GEMM 工作负载的执行效率较低。

尽管压缩稀疏行（CSR）格式能够有效减少稀疏矩阵的存储需求，但它无法识别全零对角线。在脉动阵列中处理全零对角线时，传播延迟会使沿同一对角线排列的输入元素同时到达，从而产生对结果无贡献的向量。文献 [24] 提出了一种增强的稀疏编码方案，能够识别全零对角线，使脉动阵列能够跳过冗余计算，提高计算效率。然而，这种方法引入了可观的内存开销，限制了其在硬件实现中的可扩展性。

![](images/6f57c8ce8e156255fa30a771c1e3933bc148953758fdc8d37a7e04d23fda9ee8.jpg)  
图 3：所提出的 PE 架构。

## 2.4 内存高效卷积算法

MEC 算法 [18] 旨在解决卷积计算中内存开销过大和性能下降的问题。其一个关键优势在于，变换后的数据流能够很好地契合脉动阵列的输入要求，从而实现高效的数据复用。

如图 2 所示，MEC 算法采用一种紧凑的降维方案，通过同时处理多列，而非逐个分解子矩阵，最大限度地减少冗余和内存访问流量。在计算过程中，复用这些列组来执行卷积所需的矩阵乘法。例如，在图 2 中，第一次矩阵计算使用第 1 至第 9 列，第二次使用第 4 至第 12 列，表明两次计算之间复用了六列。这种重组提高了计算吞吐量，对硬件加速尤为有利。

## 3 所提出的 PE 架构

所提出的 PE 采用 INT8 精度，以在精度与能效之间取得折中。通过定量分析，将数学公式映射到相应的硬件实现。基于式 (2) 的传统实现需要两个 8 位加法器、一个 9 位乘法器和一个 32 位加法器。然而，这种结构在稠密计算和稀疏计算中均存在局限。在稠密计算中，需要额外的一个 32 位减法器来处理最终的系数项 $\alpha _ { i }$ 和 $\beta _ { j } .$ 在稀疏计算中，典型的数据打包方案需要使用减法器进行索引比较，带来额外的硬件成本。

为解决这些问题，通过将原来的两个加法器替换为减法器，并改进整体数据流，将式 (2) 改写为式 (3)。这一变换增强了每个 PE 内部的资源共享，使其能够在统一的硬件框架内同时支持稠密运算和稀疏运算，如图 3 所示。

$$
c _ {i, j} = \sum_ {k = 1} ^ {K / 2} \left(a _ {i, 2 k - 1} - b _ {2 k, j}\right) \left(b _ {2 k - 1, j} - a _ {i, 2 k}\right) + \alpha_ {i} + \beta_ {j}\tag{3}
$$

![](images/dc450f25cfe673d57a6cae3d01298ac2c5e6bb059a3fa685df7f7b5864afa4eb.jpg)  
图 4：PE 的四种计算模式。

## 3.1 稠密模式

稠密模式的主要设计目标是在保留传统脉动阵列输入数据流特性的同时，通过最大限度地复用硬件来缩短关键路径。根据式 (3)，每次乘法之前都需要进行两次减法运算。利用脉动流水线中相邻 PE 之间的单周期延迟，提出了一种乘法器共享架构。该设计使相邻 PE 能够以时分方式共享一个乘法器，交替生成乘法结果。

两个减法运算的输出被暂存于一个 32 位寄存器中，如图 3 所示。对来自控制器的 step\_ctrl 信号进行解码，生成精确的 in\_sel 控制信号，确保两个减法结果都在各自指定的周期被送入乘法器。该寄存器不仅提供数据缓冲，还简化了组合逻辑，有效放宽了时序约束。

加法器接收来自乘法器输出和部分和寄存器的输入，不同操作数由 out\_sel 信号动态控制。每次运算后，计算结果立即写回相应的寄存器。在 PE 执行矩阵计算的同时，一个外部乘加（MAC）阵列并行运行，以计算 $\alpha_i$ 和 $\beta _ { j }$ 的值。矩阵计算结束时，PE 获取这些值，并复用加法器将其累加到最终结果中。

## 3.2 稀疏模式

所提出的 PE 旨在支持数据打包稀疏计算，同时保持脉动阵列架构的规则性和紧凑性。每个 PE 接收两对索引–数值输入。Idx\_a 信号通过一个额外的 2 位输入端口提供，而 Idx\_b 则复用现有输入端口以降低硬件成本。在稀疏模式下，PE 内部的全部四个 32 位寄存器均被重新分配用于存储部分和，从而提高稠密和稀疏运算中的寄存器利用率。该架构支持三种不同的打包数据计算模式，如图 4 所示：

1) 索引匹配：使用减法器比较 Idx\_a 和 Idx\_b。输出为零表示索引匹配，并触发匹配成功信号，使乘法器的结果能够累加到相应的部分和寄存器中。

2) 单索引分配：将 $V a l \_ a$ 与 Val\_b 相乘后，根据系统配置，将乘积累加并存储到 $I d x \_ a$ 或 $I d x \_ b ,$ 对应的部分和寄存器中。该模式对稀疏–稠密矩阵乘法工作负载尤为有效。

![](images/49bc59accb519932df0f087587e6f20dee10961cf9a578f956d6399c4203dec6.jpg)  
图 5：所提出的 CSD 格式和零对角线跳过机制。

3) 双索引组合：将 Val\_a 与 Val\_b 相乘后，将两个索引的最低有效位拼接起来，以选择用于累加的目标部分和寄存器。

## 4 压缩稀疏对角线编码

打包数据的稀疏性取决于输入矩阵的固有稀疏性以及参与打包过程的矩阵数量。在稀疏性极高的条件下，打包数据仍可能包含大量零元素（计算冗余）。缓解此类计算冗余的一种经典策略是在检测到零值输入时停用相应的算术单元。虽然这种方法能有效降低功耗，但无法加速计算。

第 II 节介绍的全零对角线跳过机制提供了一种加速计算的方法。为了充分利用这一加速潜力，我们提出了 CSD 编码方案，如图 5 所示。受 CSR 格式中高效检测全零行的启发，CSD 使用 ptr 数组记录沿对角线的元素累计数量，其中第 i 条对角线的元素数量为 $\mathit { p t r } [ i + 1 ] - \mathit { p t r } [ i ]$。注意，最左侧的对角线被指定为第 0 条对角线。此外，还引入了 idx 数组，用于存储非零元素在对角线内的索引。对角线长度各不相同，这给 idx 数组的索引带来了挑战。CSD 编码格式将次对角线左侧的 idx 值定义为正值，右侧的 idx 值定义为负值。这确保了次对角线左侧的索引以第一行为基准，而右侧的索引以最后一行为基准。通过检查 ptr 数组，可以高效识别全零对角线，并在计算过程中跳过它们，从而加速稀疏计算。

## 5 硬件实现

所提出的加速器的整体架构由两个主要子系统组成：数据通路和脉动阵列，如图 6 所示。

## 5.1 脉动阵列

脉动阵列由 8 × 4 个 PE 组成，采用可重构架构，既可作为单个 8 × 8 阵列执行稠密计算，也可划分为两个独立的 4 × 4 子阵列执行稀疏运算。

稠密数据流总体上遵循 OS 模式，但采用了经过优化的输入顺序。为了在不引入额外硬件开销的情况下满足式 (3)，右侧输入矩阵的行索引被重新排序，遵循 $[ 2 , 1 , . . . , 2 k , 2 k - 1 ]$ 的模式，而非按顺序递增。

![](images/f10c85051e73fa1dc92d011460d5466025b9e12ddaaaecda26bb31a3081a3770.jpg)  
图 6：加速器的整体结构。

分析表明，第 i 行中的所有 PE 可以共享系数 $\alpha _ { i }$，第 j 列中的所有 PE 可以共享 $\beta _ { j }$。$\alpha _ { i }$ 和 $\beta _ { j }$ 的计算由脉动阵列中额外配置的八个 MAC 单元负责。这些计算与脉动阵列的运行并行执行，以减少相关延迟。由于输出为方阵（其中 $i = j ) ,$ 第 i 个 MAC 单元负责同时计算 $\alpha_i$ 和 $\beta _ { i } .$ 这两个参数每两个时钟周期更新一次，使 MAC 单元能够以时分方式复用硬件资源，从而显著提高硬件利用效率。

在稀疏模式下，两个 4 × 4 阵列并行运行，保持与 GEMM 模式相同的输入带宽利用率。此外，该架构还实现了全零对角线跳过机制。当一条全零对角线注入脉动阵列时，它会被转换为不产生贡献的零向量。因此，当检测到此类对角线时，会将其旁路并替换为跳过信号，以避免低效处理。当接收到大小为 n 的跳过信号时，位于 (i, j) 的 PE 会停止从其紧邻的上方邻居 (i, j − 1) 接收数据，转而从 $( i , j - 1 - n )$ 获取数据，以保持整个阵列中的数据对齐。为了平衡性能与硬件资源，所提出的设计将最大跳过长度配置为 2。

## 5.2 数据通路

5.2.1 控制器。控制器将接收到的用户指令解码为三种不同的子操作：加载（LD）、执行（EX）和存储（ST），并将它们分发到各自的队列。同一条指令的三个子操作必须严格遵循 LD-EX-ST 的处理顺序，而不同指令的子操作可以并行处理。每个完成的子操作都会被相应地标记。当一条指令的全部三个子操作都处理完毕后，该指令便从队列中移除，从而允许加载一条新指令。

5.2.2 输入缓冲区。输入缓冲区采用双缓冲架构，以实现并行数据传输。一个缓冲区向脉动阵列提供数据，另一个缓冲区预加载后续输入数据，从而有效隐藏内存访问延迟。每个缓冲区的接口均配有一个 8 × 10 位的配置参数，以满足稠密和稀疏工作负载的位宽要求。

![](images/d0f81c55ae4eb6d7d8d6b221bd5102d4f3a6bd52bd806b4d2ad327ef8f0b6346.jpg)

<table><tr><td colspan="2">芯片参数</td></tr><tr><td>工艺</td><td>28nm CMOS</td></tr><tr><td>裸片面积</td><td>0.7225mm $^{2}$ </td></tr><tr><td>核心面积</td><td>0.3136mm $^{2}$ </td></tr><tr><td>焊盘电压</td><td>3.3V</td></tr><tr><td>核心电压</td><td>0.9V</td></tr><tr><td>频率</td><td>500MHz</td></tr><tr><td>功耗</td><td>41.46mW</td></tr><tr><td>峰值吞吐量</td><td>72GOPS</td></tr></table>

图 7：所提出加速器的版图。

可寻址访问进一步支持灵活的数据读取模式，以适应多种计算模式。

5.2.3 卷积映射模块。该模块专为卷积核大小为 3 × 3、步长为 1 的卷积映射而设计。它每三个周期存储输入矩阵中的十个输入元素，并在每个周期提取八个输入元素写入输入缓冲区。假设将输入数据视为左矩阵，则在第一轮写入时，需要写入第 1 至第 9 列。由于 MEC 算法中的数据复用，第 4 至第 9 列会同时写入双缓冲结构中的两个输入缓冲区，以减少写入操作。在后续步骤中，当需要第 4 至第 12 列时，第 4 至第 9 列已准备就绪，因此只需写入最后三列。这三列同样会并发写入两个输入缓冲区。除初始轮次外，这种写入方案可确保输入缓冲区中保留六列数据，仅需写入三列，从而显著减少内存访问操作。FIP 算法要求 k 为偶数（式 (2)）。长度为 9 的卷积运算通过补零扩展至长度 10。

5.2.4 稀疏解码器。稀疏解码器解析所提出的 CSD 格式，并将其转换为脉动阵列所需的输入格式。它依次处理从 ptr[i] 到 ptr[i + 3] 的四个连续指针值。如果 $\begin{array} { r } { p t r [ i ] \neq p t r [ i + 1 ] } \end{array}$，则将相应的对角线数据直接写入输入缓冲区中指定的索引位置。相反，如果 $\begin{array} { r } { p t r [ i ] = p t r [ i + 1 ] } \end{array}$，则跳过计数器递增，模块继续比较 $p t r [ i + 2 ]$ 和 ptr[i + 3]。如果跳过计数器达到 2，且 $\begin{array} { r } { p t r [ i ] = p t r [ i + 3 ] } \end{array}$，则由于最大跳过长度为 2 的约束，无法跳过该全零对角线。在这种情况下，解码器将全零对角线写入缓冲区，并向脉动阵列发出信号，使其采用长度为 2 的跳过操作。该机制在尽量减少冗余计算的同时，保持了稳健的控制逻辑。

5.2.5 后处理模块。后处理模块根据配置设置重构输出矩阵，并提供临时存储能力。它采用两个 8 × 8 × 32 位寄存器文件来执行轻量级矩阵重排，从而减轻主机的处理负担。此外，该模块还集成了一个 ReLU 激活单元和一个 INT32 至 INT8 量化单元。量化单元使用预先配置的缩放因子和偏移量处理 INT32 数据，生成 INT8 输出。这两个单元每个时钟周期可处理 8 个数据。

![](images/cac4462a1083704a016fb8b8033064beca9bb2c6d3152d79e2d1edca44b30b6f.jpg)  
图 8：所提出脉动阵列相对于基线的硬件资源消耗分析。

![](images/cf5b0e1d02473fc1498c0ec5f7bc0a29649197fbe17ce9049c224548595907cc.jpg)  
图 9：不同矩阵稀疏度组合下相对于 GEMM 的加速比。

## 6 实验结果

所提出的加速器使用 Verilog HDL 实现，并采用 28nm CMOS 工艺完成了从逻辑综合到物理实现的完整设计流程。布局后分析表明，核心面积为 0.31mm<sup>2</sup>，目标工作频率为 500MHz，如图 7 所示。

## 6.1 硬件资源消耗分析

我们评估了所提出脉动阵列架构的硬件资源消耗。为进行公平比较，实现了一个基于 OS 的参考脉动阵列，其每个 PE 均采用一个 8 位乘法器和一个 32 位加法器，并配备了用于稀疏运算的附加逻辑。两种设计均在相同条件下采用 28nm CMOS 工艺进行综合。如图 8 所示，在相同的 8 × 8 稠密配置下，与基线相比，所提出阵列的硅片面积减少了 21.3%。通过对不同维度的阵列进行综合，进一步分析了其可扩展性，以探索计算规模与资源利用之间的关系。面积缩减比例最初随规模增大而提高，这是因为用于计算 $\alpha$ 和 $\beta$ 的 MAC 阵列随问题规模线性增长，而非二次增长。然而，超过一定规模后，该比例开始下降，主要原因是设计复杂度增加限制了综合优化。

## 6.2 工作负载评估

6.2.1 加速性能。所提出加速器的性能在稠密和稀疏矩阵乘法任务上进行了评估。使用 Python 生成了稀疏度从 0.5 到 0.95、尺寸从 32 × 32 到 512 × 512 的随机矩阵，每种配置生成 100 个实例，以确保统计可靠性。稀疏矩阵在主机端使用贪心启发式方法进行打包。由于片上缓存容量有限，大规模运算被分解为较小的子矩阵计算：GEMM 使用 M8N8K16，SpMM 和 SpGEMM 使用 M4N4K16。

![](images/b3d638a2cf76ae7524c810b95a3854e4c63e102a05022de99c4d884c9ddc72a8.jpg)  
图 10：不同稀疏格式相较于稠密格式的压缩效率。

![](images/8142149a44149addbc4a3b2bb16c0305ab2767c4955670b95c050c5fabc32417.jpg)  
图 11：MEC 相较于 im2col 算法的存储效率。

如图 9 所示，稀疏–稀疏（SS）模式和稀疏–稠密（SD）模式代表了加速器的两种运行配置。在 SS 模式下，加速比随矩阵稀疏度的增加而提高，但当两个矩阵都变得极度稀疏时，加速比趋于饱和，这主要是由于每个 PE 的寄存器文件容量有限，以及脉动阵列的最大跳过长度为 2。在 SD 模式下，性能主要由打包后的左矩阵决定，而未压缩矩阵中额外的稀疏性可以通过支持更多跳过操作来进一步提高加速比。在所有工作负载中，该加速器在 SS 模式下最高实现了 4.4× 的加速比，在 SD 模式下最高实现了 5.1× 的加速比。

6.2.2 存储效率。使用随机生成的方阵评估了所提出 CSD 格式的压缩效率，矩阵尺寸为 {100, 500, 1000}，稀疏度从 0.5 到 0.98，每种配置取 100 个样本的平均值进行分析。压缩比定义为稠密表示中的元素数量除以稀疏表示中的元素数量。如图 10 所示，与 COO 和 CSXD[24] 格式相比，CSD 编码格式实现了更优的压缩性能。随着矩阵稀疏度的提高，其优势愈加显著，表明 CSD 作为一种适用于脉动阵列的稀疏编码方法具有有效性。

对所提出的、实现 MEC 算法的卷积映射模块的存储效率进行了评估。在本实验中，映射后的数据被直接写回片上缓冲区，从而能够精确测量映射后的数据量。使用 VGG-16 模型中的常见矩阵尺寸评估了该模块的性能，所有矩阵均以填充量为 1 进行预处理。如图 11 所示，基于 MEC 的硬件模块实现了约 60% 的内存使用量降低

表 1：与最先进加速器的比较

<table><tr><td></td><td>[24]</td><td>[25]</td><td>[26]</td><td>本文</td></tr><tr><td>工艺</td><td>7nm</td><td>45nm</td><td>22nm</td><td>28nm</td></tr><tr><td>设计阶段</td><td>综合后</td><td>综合后</td><td>布局布线后</td><td>布局布线后</td></tr><tr><td>规模</td><td> $8 \times 8$ </td><td> $128 \times 128$ </td><td> $64 \times 64$ </td><td> $8 \times 4$ </td></tr><tr><td>频率 (GHz)</td><td>2</td><td>0.4</td><td>1</td><td>0.5</td></tr><tr><td>精度</td><td>INT8</td><td>BF16</td><td>INT8</td><td>INT8</td></tr><tr><td>稀疏支持</td><td>是</td><td>是</td><td>否</td><td>是</td></tr><tr><td>阵列面积 ( $mm^2$ )</td><td>0.01</td><td>116.4</td><td>1</td><td>0.04</td></tr><tr><td>功耗 (W)</td><td>0.045</td><td>22.9</td><td>0.858</td><td>0.041</td></tr><tr><td>峰值吞吐量 (GOPS)</td><td>256</td><td>13107</td><td>8192</td><td>72</td></tr><tr><td>面积效率 ( $GOPS/mm^2$ )</td><td>25600</td><td>112.6</td><td>8192</td><td>1800</td></tr><tr><td>能量效率 (GOPS/W)</td><td>5689</td><td>572.4</td><td>9548</td><td>1737</td></tr><tr><td>FNAE ( $GOP/mm^2$ )</td><td>12800</td><td>281.5</td><td>8192</td><td>3600</td></tr><tr><td>面积效率* ( $GOPS/mm^2$ )</td><td>793.6</td><td>321.7</td><td>5,153</td><td>1800</td></tr><tr><td>能量效率* (GOPS/W)</td><td>2133</td><td>797.3</td><td>8182</td><td>1737</td></tr><tr><td>FNAE* ( $GOP/mm^2$ )</td><td>396.8</td><td>804.3</td><td>5,153</td><td>3600</td></tr></table>

<sup>∗</sup> 面积和功耗使用 DeepScaleTool [27] 归一化至 28nm  
，这是在所有测试的矩阵尺寸下与传统的 im2col 方法 [15] 相比较的结果。这一结果验证了该模块通过高效的数据访问模式缓解内存瓶颈并提高数据吞吐量的有效性。

## 6.3 效率比较

表 1 将所提出的设计与最先进的加速器进行了比较，包括 SPSA [24]、DenSparSA [25] 和 DiP [26]，同时也标明了设计成熟度：SPSA 和 DenSparSA 报告的是综合后（post-Syn）结果，而 DiP 和本文设计均为布局后（post-PnR）实现。为了考虑不同工艺节点和设计阶段之间的频率差异，引入了频率归一化面积效率（FNAE）指标，以进行公平的架构比较，其定义为 GOPS/(GHz · mm<sup>2</sup>)。归一化至 28nm CMOS 工艺后，与 [24] 和 [25] 相比，我们设计的面积效率分别提高了 2.27× 和 5.6×，FNAE 分别提高了 9.07× 和 4.48×。相较于 [26]，性能略低主要是由于增加了对稀疏计算的支持。总体而言，这些结果表明了所提出脉动阵列架构的资源效率。

## 7 结论

本研究提出了一种基于 FIP 算法的新型可重构脉动阵列架构，能够高效支持 GEMM、SpMM 和 SpGEMM 运算。引入了一种轻量级 CSD 编码方案，以极小的存储开销实现零值对角线跳过。所提出的设计采用 28nm CMOS 工艺实现，面积减少了 21.3%，且相较于 GEMM，SpGEMM 和 SpMM 分别实现了最高 4.4× 和 5.1× 的加速比。

## References

[1] Hsiang-Tsung Kung. 1982. Why systolic architecture? Design Research Center, Carnegie-Mellon University Pittsburgh, PA, USA.

[2] Norman P Jouppi, Clif Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, et al. 2017. In-datacenter performance analysis of a tensor processing unit. In Proceedings of the 44th annual international symposium on computer architecture. 1–12.

[3] Fanrong Li, Gang Li, Zitao Mo, Xiangyu He, and Jian Cheng. 2020. FSA: A fine grained systolic accelerator for sparse CNNs. IEEE Transactions on Computer-Aided Design ofIntegrated Circuits and Systems 39, 11 (2020), 3589–3600.

[4] Norman P Jouppi, Doe Hyun Yoon, George Kurian, Sheng Li, Nishant Patil, James Laudon, Clif Young, and David Patterson. 2020. A domain-specific supercomputer for training deep neural networks. Commun. ACM 63, 7 (2020), 67–78.

[5] Liqiang Lu, Yicheng Jin, Hangrui Bi, Zizhang Luo, Peng Li, Tao Wang, and Yun Liang. 2021. Sanger: A co-design framework for enabling sparse attention using reconfigurable architecture. In MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture. 977–991.

[6] Jianchao Yang, Mei Wen, Junzhong Shen, Yasong Cao, Minjin Tang, Renyu Yang, Jiawei Fei, and Chunyuan Zhang. 2022. Bp-im2col: Implicit im2col supporting ai backpropagation on systolic arrays. In 2022 IEEE 40th International Conference on Computer Design (ICCD). IEEE, 415–418.

[7] Adi Fuchs and David Wentzlaf. 2019. The accelerator wall: Limits of chip spe cialization. In 2019 IEEE International Symposium on High Performance Computer Architecture (HPCA). IEEE, 1–14.

[8] Trevor E Pogue and Nicola Nicolici. 2023. Fast inner-product algorithms and architectures for deep neural network accelerators. IEEE Trans. Comput. 73, 2 (2023), 495–509.

[9] Minjin Tang, Mei Wen, Yasong Cao, Junzhong Shen, Jianchao Yang, Jiawei Fei, Yang Guo, and Sheng Liu. 2022. Mentha: Enabling sparse-packing computation on systolic arrays. In Proceedings ofthe 51st International Conference on Parallel Processing. 1–11.

[10] Eric Qin, Ananda Samajdar, Hyoukjun Kwon, Vineet Nadella, Sudarshan Srini vasan, Dipankar Das, Bharat Kaul, and Tushar Krishna. 2020. Sigma: A sparse and irregular gemm accelerator with flexible interconnects for dnn training. In 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA). IEEE, 58–70.

[11] HT Kung, Bradley McDanel, and Sai Qian Zhang. 2019. Packing sparse convo lutional neural networks for eficient systolic array implementations: Column combining under joint optimization. In Proceedings ofthe Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems. 821–834.

[12] Xin He, Subhankar Pal, Aporva Amarnath, Siying Feng, Dong-Hyeon Park, Austin Rovinski, Haojie Ye, Yuhan Chen, Ronald Dreslinski, and Trevor Mudge. 2020. Sparse-TPU: Adapting systolic arrays for sparse matrices. In Proceedings of the 34th ACM international conference on supercomputing. 1–12.

[13] Yu-Hsin Chen, Tushar Krishna, Joel S Emer, and Vivienne Sze. 2016. Eyeriss: An energy-eficient reconfigurable accelerator for deep convolutional neural

networks. IEEE journal ofsolid-state circuits 52, 1 (2016), 127–138.

[14] Yu-Hsin Chen, Tien-Ju Yang, Joel Emer, and Vivienne Sze. 2019. Eyeriss v2: A flexible accelerator for emerging deep neural networks on mobile devices. IEEE Journal on Emerging and Selected Topics in Circuits and Systems 9, 2 (2019), 292–308.

[15] Kumar Chellapilla, Sidd Puri, and Patrice Simard. 2006. High performance convolutional neural networks for document processing. In Tenth international workshop on frontiers in handwriting recognition. Suvisoft.

[16] Hyunsun Park, Dongyoung Kim, Junwhan Ahn, and Sungjoo Yoo. 2016. Zero and data reuse-aware fast convolution for deep neural networks on GPU. In Proceedings of the Eleventh IEEE/ACM/IFIP International Conference on Hardware/Software Codesign and System Synthesis. 1–10

[17] Andrew Lavin and Scott Gray. 2016. Fast algorithms for convolutional neural networks. In Proceedings ofthe IEEE conference on computer vision and pattern recognition. 4013–4021.

[18] Minsik Cho and Daniel Brand. 2017. MEC: Memory-eficient convolution for deep neural network. In International Conference on Machine Learning. PMLR, 815–824.

[19] Rui Xu, Sheng Ma, Yang Guo, and Dongsheng Li. 2023. A survey of design and optimization for systolic array-based DNN accelerators. Comput. Surveys 56, 1 (2023), 1–37.

[20] Shmuel Winograd. 2006. A new algorithm for inner product. IEEE Trans. Comput. 100, 7 (2006), 693–694.

[21] Oscar Gustafsson and Andreas Ehliar. 2013. Low-complexity general FIR filters based on Winograd’s inner product algorithm. In 2013 IEEE International Symposium on Circuits and Systems (ISCAS). IEEE, 93–96.

[22] Kartik Hegde, Hadi Asghari-Moghaddam, Michael Pellauer, Neal Crago, Aamer Jaleel, Edgar Solomonik, Joel Emer, and Christopher W Fletcher. 2019. Extensor: An accelerator for sparse tensor algebra. In Proceedings ofthe 52nd Annual IEEE/ACM International Symposium on Microarchitecture. 319–333.

[23] Angshuman Parashar, Minsoo Rhu, Anurag Mukkara, Antonio Puglielli, Rang harajan Venkatesan, Brucek Khailany, Joel Emer, Stephen W Keckler, and William J Dally. 2017. SCNN: An accelerator for compressed-sparse convolutional neural networks. ACM SIGARCH computer architecture news 45, 2 (2017), 27–40.

[24] Minjin Tang, Mei Wen, Jianchao Yang, Zeyu Xue, and Junzhong Shen. 2024. SPSA: Exploring Sparse-Packing Computation on Systolic Arrays From Scratch. IEEE Transactions on Computer-Aided Design ofIntegrated Circuits and Systems (2024).

[25] Ziheng Wang, Ruiqi Sun, Xin He, Tianrui Ma, and An Zou. 2025. DenSparSA: A Balanced Systolic Array Approach for Dense and Sparse Matrix Multiplication. In 2025 62nd ACM/IEEE Design Automation Conference (DAC). IEEE, 1–7.

[26] Ahmed J Abdelmaksoud, Shady Agwa, and Themis Prodromakis. 2025. Dip: A scalable, energy-eficient systolic array for matrix multiplication acceleration. IEEE Transactions on Circuits and Systems I: Regular Papers (2025).

[27] Satyabrata Sarangi and Bevan Baas. 2021. DeepScaleTool: A tool for the accurate estimation of technology scaling in the deep-submicron era. In 2021 IEEE International Symposium on Circuits and Systems (ISCAS). IEEE, 1–5.

> 中文翻译与解读由 Codex/GPT 根据所提供的论文 PDF 生成。PDF 使用 MinerU 解析，乱码数学符号已对照原文校正；参考文献保留原文。技术总结和通俗讲解属于解读，请与全文译文结合阅读。[原文 DOI](https://doi.org/10.1145/3770743.3804286)。
