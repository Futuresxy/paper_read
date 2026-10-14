# RaPiD: AI Accelerator for Ultra-low Precision Training and Inference 原文翻译

# RaPiD：面向超低精度训练与推理的 AI 加速器

Swagath Venkataramani <sup>∗</sup>, Vijayalakshmi Srinivasan<sup>∗</sup>, Wei Wang<sup>∗</sup>, Sanchari Sen<sup>∗</sup>, Jintao Zhang<sup>∗</sup>, Ankur Agrawal<sup>∗</sup>, Monodeep Kar<sup>∗</sup>, Shubham Jain<sup>∗</sup>, Alberto Mannari<sup>†</sup>, Hoang Tran<sup>∗</sup>, Yulong Li<sup>∗</sup>, Eri Ogawa<sup>‡</sup>, Kazuaki Ishizaki<sup>‡</sup>, Hiroshi Inoue<sup>‡</sup>, Marcel Schaal<sup>∗</sup>, Mauricio Serrano<sup>∗</sup>, Jungwook Choi<sup>∗</sup>, Xiao Sun<sup>∗</sup>, Naigang Wang<sup>∗</sup>, Chia-Yu Chen<sup>∗</sup>, Allison Allain<sup>∗</sup>, James Bonano<sup>¶</sup>, Nianzheng Cao<sup>∗</sup>, Robert Casatuta<sup>k</sup>, Matthew Cohen<sup>∗</sup>, Bruce Fleischer<sup>∗</sup>, Michael Guillorn<sup>∗</sup>, Howard Haynie<sup>∗∗</sup>, Jinwook Jung<sup>∗</sup>, Mingu Kang<sup>∗</sup>, Kyu-hyoun Kim<sup>∗</sup>, Siyu Koswatta<sup>∗</sup>, Saekyu Lee<sup>∗</sup>, Martin Lutz<sup>∗</sup>, Silvia Mueller<sup>§</sup>, Jinwook Oh<sup>∗</sup>, Ashish Ranjan<sup>∗</sup>, Zhibin Ren<sup>∗</sup>, Scot Rider<sup>∗∗</sup>, Kerstin Schelm<sup>§</sup>, Michael Scheuermann<sup>∗</sup>, Joel Silberman<sup>∗</sup>, Jie Yang<sup>∗</sup>, Vidhi Zalani<sup>∗</sup>, Xin Zhang<sup>∗</sup>, Ching Zhou<sup>∗</sup>, Matt Ziegler<sup>∗</sup>, Vinay Shah<sup>††</sup>, Moriyoshi Ohara<sup>‡</sup>, Pong-Fei Lu<sup>∗</sup>, Brian Curran<sup>∗∗</sup>, Sunil Shukla<sup>∗</sup>, Leland Chang<sup>∗</sup>, and Kailash Gopalakrishnan <sup>∗</sup>IBM Research, Yorktown Heights, NY, <sup>†</sup>IBM Research, Zurich, Switzerland, <sup>‡</sup>IBM Research, Tokyo, Japan, <sup>§</sup>IBM, Boeblingen, Germany, <sup>¶</sup>IBM, Austin, TX, <sup>k</sup>IBM, Hopewell Junction, NY, <sup>∗∗</sup>IBM, Poughkeepsie, NY, <sup>††</sup>IBM, Hursley, United Kingdom

摘要——人工智能（AI）工作负载日益普遍且计算需求不断增长，这促使硬件加速器在其执行中得到广泛使用。跨代提升 AI 加速器的性能对于其在商业部署中的成功至关重要。AI 工作负载固有的容错特性为通过精度缩放实现性能与能耗的改进提供了独特的机会。受近期推理与训练精度缩放算法进展的启发，我们设计了 RAPID<sup>1</sup>，这是一款支持多种精度谱系的 4 核 AI 加速器芯片，涵盖 16 位和 8 位浮点以及 4 位和 2 位定点表示。这款采用 7nm EUV 工艺制造的 36mm<sup>2</sup> RAPID 芯片在标称电压下，HFP8 模式下的峰值能效达到 3.5 TFLOPS/W，INT4 模式下达到 16.5 TOPS/W。我们利用与实测结果校准误差在 1% 以内的性能模型，评估了采用 4 位定点表示的 DNN 推理（基于 4 核 1 RAPID 芯片系统）以及采用 8 位浮点表示的 DNN 训练（基于由 4 颗 32 核 RAPID 芯片组成、算力达 768 TFLOPs 的 AI 系统）。结果表明，对于 batch size 为 1 的 INT4 推理，在广泛的应用范围内实现了 3 - 13.5（平均 7）TOPS/W 的能效；对于 mini-batch 为 512 的 FP8 训练，实现了 102 - 588（平均 203）TFLOPS 的持续性能。

关键词——硬件加速，深度神经网络，降低精度

## I. 引言

过去十年间，从移动和物联网边缘设备到云端与数据中心，各类计算平台上运行的工作负载性质发生了范式转变。在海量数据的可得性以及深度神经网络（DNN）深度学习进展的推动下，基于 AI 的应用与服务得到了显著普及，在涉及图像、视频、文本和自然语言的若干具有挑战性的 AI 任务上甚至超越了人类水平 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]。然而，DNN 的高准确率伴随着高昂的计算成本。例如，最先进的图像识别 DNN（如 ResNet50）需要对单张图像执行约 100 亿次数值运算。此外，训练 DNN 模型需要 exa-FLOP 级的算力，并使用规模达数百 GB 的海量训练数据集。如此高的计算/存储/带宽需求，给传统计算平台的能力带来了严峻挑战。

AI 加速器。随着 CMOS 技术缩放及其相关红利似乎趋于放缓，要满足 AI 工作负载的计算需求，并为更复杂、更鲁棒的模型的未来 AI 研究提供动力，就必须在硬件与软件系统栈的各个层面进行创新。业界广泛采用的一种途径是构建配备硬件加速器的 AI 专用系统。AI 工作负载非常适合硬件加速，因为它们是静态数据流图，且 DNN 中的计算可以由少量（几十个）原语来表达。学术界大量的 DNN 专用加速器研究 [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]，以及商用 AI 核心（Google TPU、NVIDIA Tensor Core、Intel NNP 等）[33, 34, 35] 都印证了这一点。

![](images/8b16cc61063f01710d06e5d80c999d549fcfc02b7457df45e9dbc0a136b3891b.jpg)  
图 1：训练与推理的精度缩放路线图

精度缩放。在商业化部署中，跨代际地扩展 AI 加速器的性能是其成功的关键。除了在计算栈各个层面（即工艺节点、众核/异构架构等）扩展性能的传统手段之外，AI 工作负载还为通过精度缩放提升性能/能效提供了独特的机遇。近年来近似计算的进展已成功证明，只要处理得当，便可在不损失准确率的情况下，利用 AI 工作负载固有的容错特性来降低计算过程中数据表示所用的位宽。

如图 1 所示，研究工作已持续将推理和训练的精度需求向下推进。由边缘设备部署驱动的推理精度缩放已将权重和激活值的表示位宽降至低至 2-4 比特 [36, 37, 38, 39, 40, 41, 42, 43]。训练的精度缩放则更具挑战性，因为需要在反向传播步骤中保持梯度的保真度，且表示需具有很大的动态范围。近来，8 比特浮点表示 [44, 45] 已被证明对 DNN 训练是有效的。

支持超低精度的 AI 加速器。鉴于这些算法层面的进展，下一代 AI 加速器必须具备超低精度执行能力——超越训练用的 16 比特浮点和推理用的 8 比特整数。在本工作中，我们提出 RAPID，一种支持从 16 比特浮点到 2 比特定点的一系列精度范围的加速器架构。我们设计并流片了采用 7nm EUV 工艺、工作频率 1.5 GHz 的 4 核 RAPID 芯片。

从宏观角度看，精度缩放具有若干优势，使其适合集成到加速器设计之中。首先，它影响系统设计的方方面面，即同时提升计算效率、降低存储占用和数据带宽需求。其次，它保持了计算的规律性，从而将数据流复杂度和控制开销降到最低。然而，它也引入了新的挑战，需要精心的架构设计。我们在下文中重点介绍 RAPID 的关键特性。

1) 混合（超低）精度支持。RAPID 架构支持 5 种不同精度——16 比特浮点（FP16）、两种具有可编程偏置的 8 比特浮点变体（FP8-fwd 和 FP8-bwd，合称 Hybrid-FP8 (HFP8)）、4 比特定点（INT4）和 2 比特定点（INT2）。我们基于针对训练 [44, 45] 和推理 [42, 46] 的详细算法研究来选择这些精度。值得注意的是，尽管我们以超低精度执行为目标，保留对更高精度的支持仍然至关重要。这是因为，虽然超低精度可应用于大多数计算，但部分特定计算（如第一层和最后一层、捷径路径等）需要高精度以保持准确率 [46]。我们这颗采用 7nm EUV 工艺的 4 核 36mm<sup>2</sup> RAPID AI 芯片在 FP16/HFP8/INT4 精度下分别提供 12/24/96 T(FL)OPS 的峰值算力，并分别实现 1.8/3.5/16.5 T(FL)OPS/W 的能效。

2) 在低精度下同时提升 TOPS 和 TOPS/W。RAPID 的一个关键设计原则是在超低精度下同时提升性能（TOPS）与能效（TOPS/W），以满足实时和电池供电两类部署场景的需求。这意味着，在每种精度下权衡速度与资源供给的同时，按照低精度所节省的功率相应地扩展计算引擎的数量。

3) 端到端应用覆盖。虽然卷积和 GEMM 占 DNN 运算的很大比例，但它们嵌入于大量激活、池化、归一化和数据重排操作之中。在加速器上执行这些操作至关重要，因为加速器与主机之间的数据传输代价可能很高。RAPID 的特殊功能单元（SFU）针对广泛的此类操作同时包含精确版和快速版，并配有用于数据重排的置换引擎。它还使用 32 比特浮点（FP32）单元来执行部分选定操作。

4) 稀疏感知的频率调节。为提升能效，RAPID 的融合乘加（FMA）引擎设计了零值门控逻辑，即当某一乘数为零时触发旁路路径。它还包含一个可由软件控制的电源管理单元，能够快速调节有效时钟频率。在推理场景下，我们利用这些特性来提升稀疏（或剪枝）DNN 模型的性能。具体而言，通过对每层所呈现稀疏度的离线分析，我们估算零值门控所节省的功率，并在该层执行期间通过提升有效时钟频率将这些功率重新投入，以提升性能。

5) 多核扩展。RAPID 包含一个存储邻居接口，支持核间以及核与存储之间的通信与同步。我们的通信协议包含简单的原语，允许在相互重叠的多播生产者-消费者核组之间进行并发数据传输，从而使软件能够有效利用可用带宽。

总而言之，RAPID 架构支持以超低精度同时进行训练和推理。该设计支持向多核扩展，并提供执行端到端 AI 工作负载所需的功能覆盖。在本文中，我们评估 4 核 RAPID 芯片模型的推理能力，并以 4 比特定点为主要关注点。我们还评估了采用 32 核 RAPID 芯片模型的分布式 4 芯片系统在 8 比特浮点下的训练能力。据我们所知，RAPID 是首个支持 8 比特训练和 4 比特推理的混合精度架构工作。

本文其余部分的组织结构如下。第 II 节讨论脉动数据流架构（与近期若干先前工作类似），该架构被用作基线核心架构，并总结了算法近似方面的进展，这些进展是使 RAPID 支持超低精度的基石。第 III 节介绍核心中支持超低精度的关键架构与微架构创新，第 IV 节介绍采用 7nm 工艺的 RAPID 芯片（包含 4 个核心）及其编程模型，并讨论利用按比例扩展的 RAPID 芯片构建的用于训练与推理的拟议系统。第 V 节描述 RAPID 在一组流行的 DNN 基准测试上尤其在超低精度下进行推理与训练的有效性。第 VI 节讨论相关工作，最后第 VII 节对全文进行总结并展望未来工作。

## II. 背景

AI 工作负载是静态数据流图，DNN 中的计算可以用少量（几十个）原语来表达。这催生了许多学术界的原型演示和商用 AI 核心，它们均利用脉动数据流架构 [23, 25, 26, 30, 47, 48, 49]。我们采用这样一种脉动数据流架构作为基线，并对其增强以支持超低精度的训练与推理。

## A. 基线：脉动数据流架构

图 2 展示了基线脉动数据流架构的构建模块。主计算单元包括：一个由处理单元（Processing Elements, PE）组成的 8x8 二维脉动阵列，支持 16 位浮点（FP16）计算，用于执行 DNN 中的卷积和矩阵乘法运算；以及一个一维的特殊功能单元（Special Function Units, SFU）阵列，同时支持 16 位和 32 位浮点计算（FP16 和 FP32），用于执行激活函数、池化、梯度归约和归一化等可能需要更高位精度的操作。

每个 PE 包含一个 8 路 SIMD 乘加（MAC）单元，其操作数来自 PE 的北/西相邻单元或其本地寄存器堆。类似地，MAC 的输出要么发送到南侧相邻的 PE，要么写回本地寄存器堆。由于所采用的典型数据流不需要操作数沿对角线流动，因此同一行的 PE 以脉动方式执行相同的指令序列。每个 SFU 还包含以更高精度运行的 8 路 SIMD MAC，作为一个向量单元使用。

![](images/17741d83dd07fb1c51715dfda6dc153b82b12f269bf9af06c291ba57d5c54c81.jpg)

图 2：基线脉动数据流架构。

一个两层暂存器存储层次结构为 PE 阵列和 SFU 供给数据。L0 暂存器用于沿行方向（X 方向）供给数据。L1 暂存器存储器一侧连接到 L0 存储器和 SFU/PE 阵列的列方向（Y 方向），另一侧与外部存储器接口相连。

为了提供最大的灵活性，基线架构通过将计算和流经各组件的数据流解耦为多个独立的执行线程而完全去中心化。与 access-execute 范式类似，可编程单元位于架构中每条（或每组）链路的端点上，从而对链路中数据的传递顺序进行细粒度控制。例如，为了在 L1 与 L0 之间协调数据流，位于 L1 附近的可编程单元控制从 L1 读取的地址序列并将数据推送到链路上。在接收到数据后，位于 L0 附近的可编程单元决定数据需要存储在 L0 中的位置。SFU 也运行各自的程序。它们可以从任一输入/输出链路及其本地寄存器堆读取/写入数据操作数。

因此，DNN 操作的执行通过多个程序来协调，这些程序可大致分为两类： 数据序列化程序，负责从暂存器存储器加载/存储数据并按顺序将数据供给 PE/SFU； 数据处理程序，定义在输入数据元素上由 PE/SFU 执行的计算集合。为确保正确的功能（例如生产者-消费者依赖关系），该架构使用基于令牌的硬件支持来同步选定的可编程单元。例如，考虑数据从 L1 移动到 L0 随后流式传输到 PE 阵列的情形，向 L0 写入数据的程序与从 L0 读取数据的程序会周期性地同步，以确保写操作先于读操作。

## B. 将训练扩展到 FP16 之外

![](images/39311fbb03d7a5141879ce4f857c567bd08ae74fa39469803a4f5de85faa2f43.jpg)
图 3：为训练提供混合 FP8 精度支持 [45]

降低精度的 DNN 训练更具挑战性，原因在于需要在反向传播步骤中保持梯度的保真度。近来，算法近似方法 [44, 45] 提出了一种新的 Hybrid-FP8（HFP8）数据格式（如图 3 所示）用于 DNN 训练。HFP8 格式涉及使用两种不同的 FP8（符号、指数、尾数）表示——一种具有较低动态范围的表示 (1,4,3) 用于激活值和权重，另一种具有较高动态范围的表示 (1,5,2) 用于误差。前向传播中的两个操作数均使用 FP8(1,4,3)，而反向传播和梯度计算中，一个操作数采用 FP8(1,4,3) 表示，另一个操作数采用 FP8(1,5,2) 表示。除了在前向与反向传播中使用两种不同的指数-尾数位组合之外，HFP8 还要求指数偏置可配置。这使得尽管使用相同数量的指数位，不同的 DNN 层可以具有不同的动态范围。

基于 HFP8 的训练 [44, 45] 已被证明能够在包括图像分类、目标检测、机器翻译和语音在内的广泛应用谱系上，实现与 FP32 表示训练相当的深度学习模型精度。

## C. 将推理扩展到 INT8 之外

推理的精度缩放已在 4 位（INT4）和 2 位（INT2）定点表示上得到成功验证。近来，两种量化技术，即用于激活值的参数化截断激活（PArameterized Clipping acTivation, PACT）[42] 和用于权重的统计感知权重量化（Statistics-aware Weight Binning, SaWB）[46]，已经实现了 4 位（INT4）推理且精度损失可忽略不计，以及 2 位（INT2）推理且精度损失极小（≈2%）。PACT 引入了一种从 ReLU 衍生而来的新激活函数，将超过某一数值的输出截断，从而缩小其取值范围。其关键洞察在于，截断值并非静态固定，而是在模型训练期间针对 DNN 的每一层独立学习得到的。SaWB 利用权重的一阶矩和二阶矩对权重进行量化，同时保留权重分布的形状。PACT 和 SaWB 对模型训练时间几乎没有影响。

## III. 面向超低精度的核心架构

精度缩放的一个关键特性在于它保持了计算的规律性，因为给定张量的所有元素都被等比例缩放到相同的精度。此外，精度缩放还能在执行单元以及内存和互连子系统中节省能耗，因为数据结构的容量需求以及组件之间传输的数据量也随之减少。因此，图 2 所示的基线脉动数据流架构具备适合此类精度缩放 AI 工作负载的组织形式。但是，为了满足超低精度训练和推理的计算与带宽需求，基线架构必须增强以实现以下目标：

• 将整体峰值 TOPs 扩展至与缩放后的精度相匹配的水平

• 平衡 PE 阵列的面积与功耗，以有效利用提升后的峰值 TOPs。

• 通过选择合适的数据流，使操作数在 SIMD 以及 PE 阵列的行/列之间得到有效复用，从而满足流入 PE 阵列的数据的带宽约束。

• 使 PE 阵列持续以 16 位格式输出结果，以便辅助运算能够以更高精度执行，从而保持分类的准确性。

• 平衡各计算单元，使之与高精度激活在 SFU 阵列中的分布以及超低精度卷积和矩阵运算在 PE 阵列中的分布相匹配。

下面我们将介绍添加到基线架构基本构建块中不同组件上的关键架构与微架构创新，以克服上述挑战并实现上述目标。

## A. MPE 阵列：混合精度 PE 阵列

随着我们持续扩展计算的精度，关键挑战之一是使整体峰值 TOPs 与扩展后的精度相匹配。我们接下来讨论混合精度 PE 阵列中的架构与微架构增强，这些增强使得峰值 TOPs 能够随精度扩展而扩展，同时克服带宽、面积和功耗方面的挑战。

1) 同时支持 INT 与 FP 流水线 为了在超低精度下同时支持训练与推理，MPE 阵列必须全面支持混合精度执行，包括第 II 节所讨论的不同数字格式：即 FP16、Hybrid-FP8、INT4 和 INT2。图 4(a) 展示了一个 MPE 的框图。在计算引擎中同时支持浮点与定点运算会增加面积和功耗开销，同时也会在流水线深度和执行延迟方面产生潜在的不匹配。将整数流水线与浮点流水线分离，既解决了处理多种精度的架构复杂性，又为在电路实现上激进地提升能效创造了机会。与基线架构相同，每个 MPE 拥有 8 个 FPU 和 8 个 FXU，分别支持 FP16/HFP8 和 INT4/INT2 格式。

2) HFP8 训练 MPE 的 FPU 流水线使用同一条 128 位数据通路同时支持 FP16 和 HFP8，用于 8 路 SIMD FPU。如图 4(a) 所示，每个 MPE 的 FPU 从北向/西向邻居或其本地寄存器堆接收输入操作数，输入操作数同时向东侧链路传播，而输出则传播给南侧邻居。

我们为支持 HFP8 对 FPU 进行了两项关键增强，以支持对输入操作数采用不同表示，并实现相对于 FP16 的 2 倍峰值 TOPs。

即时转换为自定义 FP8 表示：如图 3 所示，FP8 训练要求训练的反向路径中的矩阵乘法和卷积使用不同 FP8 格式的张量作为输入。然而，这会在面积和功耗两方面增加浮点单元的硬件复杂度。我们增强 FPU 使用自定义的（符号、指数、尾数）格式 (1,5,3)，将 (1,5,2) 和 (1,4,3) 格式的输入操作数即时转换为 (1,5,3) 格式 [50]。

因此，对于 HFP8 训练，权重和激活值以 (1,4,3)（带偏置）或 (1,5,2) 格式存储在存储器和暂存器中，并在水平总线上（对于来自 L0 暂存器的数据）或 FIFO 接口上（对于经由北向链路来自 L1 暂存器的数据）被转换为 9 位 (1,5,3) 格式。

![](images/e92a269a43abce968d90c63d311a2b533d49ea7e76e69659192abd9e249c72c8.jpg)

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>格式</td><td rowspan=1 colspan=1>偏置</td><td rowspan=1 colspan=1>指令</td></tr><tr><td rowspan=1 colspan=1>FP16</td><td rowspan=1 colspan=1>1/6/9</td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>FMA: R = A.B + C（均为 FP16）</td></tr><tr><td rowspan=1 colspan=1>FP8-fwd</td><td rowspan=1 colspan=1>1/4/3</td><td rowspan=1 colspan=1>0-15</td><td rowspan=3 colspan=1>FMMA:R = A1.B1 + A2.B2 + CA1, A2, B1, B2: FP8 R,C: FP16</td></tr><tr><td rowspan=1 colspan=1>FP8-bwd</td><td rowspan=1 colspan=1>1/5/2</td><td rowspan=1 colspan=1>15</td></tr><tr><td rowspan=1 colspan=1>FP9</td><td rowspan=1 colspan=1>1/5/3</td><td rowspan=1 colspan=1>15</td></tr><tr><td rowspan=1 colspan=1>INT4</td><td rowspan=1 colspan=2>二进制补码</td><td rowspan=1 colspan=1>R = ∑8 A. Bi + C（A,B: 8x INT4; R,C: INT16）</td></tr><tr><td rowspan=1 colspan=1>INT2</td><td rowspan=1 colspan=2>权重: [-4, -1, 1, 4]激活: [0,1,2,3]</td><td rowspan=1 colspan=1>R = ∑16 A. Bi + C（A,B: 16x INT2; R,C: INT16）</td></tr></table>

<table><tr><td rowspan=2 colspan=1>FP 与 INT流水线</td><td rowspan=1 colspan=3>TOP/W（归一化）</td><td rowspan=2 colspan=1>面积</td></tr><tr><td rowspan=1 colspan=1>FP16</td><td rowspan=1 colspan=1>FP8</td><td rowspan=1 colspan=1>INT4</td></tr><tr><td rowspan=1 colspan=1>合并</td><td rowspan=1 colspan=1>1.0x</td><td rowspan=1 colspan=1>2.2x</td><td rowspan=1 colspan=1>5.6x</td><td rowspan=1 colspan=1>1.0x</td></tr><tr><td rowspan=1 colspan=1>分离</td><td rowspan=1 colspan=1>1.1x</td><td rowspan=1 colspan=1>2.3x</td><td rowspan=1 colspan=1>10.6x</td><td rowspan=1 colspan=1>1.16x</td></tr></table>

图 4：混合精度处理单元（MPE）的框图与指令格式

MPE 的 ISA 支持乘累加指令（FMMA）的输入张量具有不同的 FP8 格式。MPE 程序根据所选择的数据流为输入操作数使用所需的格式。然而，在一个程序内，操作数的精度保持固定，并被设置在寄存器中，以使硬件能够确定操作数的数据门控宽度。HFP8 数据通路的另一项关键特性是其指数偏置是可编程的。根据正在执行的计算的动态范围，MPE 被配置为相应的指数偏置。

sub-SIMD 分区：在 HFP8 模式下扩展峰值 TOPs 的关键使能技术之一是对 FPU 内的 SIMD 单元进行细粒度分区，在使用相同接口总线宽度的情况下，以与 FP16 相同的功耗实现 2 倍性能。如图 4(a) 所示，在 HFP8 模式下，SIMD MPE 的乘累加指令（FMMA）实现 2 次乘法和 2 次加法。尽管增加了额外的复杂性，但由于采用 4 位乘法器，HFP8 被乘数通路的逻辑深度仍与 FP16 相当。

借助图 4(b) 所示的 ISA 扩展以及上文所述的微架构增强，MPE 在 FPU 流水线中同时支持 HFP8 和 FP16。

为了保持端到端精度，辅助运算在 FP16 和 HFP8 训练中都需要更高的精度。由于 FP16 和 HFP8 模式均产生 FP16 结果，因此 FP16 与 HFP8 的 FPU 计算路径在加法器处合并，如图 4(a) 所示。

最后，HFP8 训练还使用基于分块的累加 [51]，以分层方式累加部分和，以保持中间和的保真度。SFU 被用于实现 MPE 所产生的部分和（FP16/INT16）的基于分块的累加。

3) INT4/INT2 推理 如上所述，我们对 MPE 进行了增强，使其拥有分离的 FPU 和 FXU 流水线。此外，由于 INT4/INT2 引擎仅针对 DNN 推理，更多电路级的优化变得可行，从而降低面积和功耗。

INT4 与 INT2 流水线的双重泵送（Double pumping）：我们对解耦的 FPU 与 FXU 单元进行的面积和功耗分析揭示了将 INT4 和 INT2 引擎在 FXU 内翻倍的机会。如图 4(c) 所总结的，增加独立的 INT 流水线带来约 16% 的面积开销，但 INT4 流水线的功耗仅为 FP16 流水线的 0.3 倍，因此能够在 MPE 中将 INT4 和 INT2 计算引擎翻倍。因此，每个 FXU 拥有 8 个 INT4（16 个 INT2）乘累加引擎。

操作数复用：Sub-SIMD + 跨列：由于每个 8 路 SIMD 单元在一个周期内完成 8 次 INT4（以及 16 次 INT2）乘累加操作，我们将 SIMD MAC 单元的数据通路宽度加倍至 256 位，并增强了架构以允许在 MPE 中用单条 MAC 指令访问 2 个寄存器（256 位）。这样做还降低了累加的能耗成本，尤其是随着精度降低乘法器的成本随之降低的情况下。

如图 4(a) 所示，8-way SIMD FXU 支持 4 位和 2 位整数 MAC 运算，产生与 MPE 之间 128 位数据通路相匹配的 16 位整数结果。高效的数据流映射需要在 L0 带宽约束与 FXU 内部的操作数复用之间取得平衡。例如，在改进的权重驻留（weight-stationary）数据流中，来自 West 链路的 8 个 INT4 输入操作数 B（32 位）被 8-way 子 SIMD 的每一通路复用，同时还沿 MPE 的列进行传播，总计在 64 次乘加运算中被复用。

4) MPE 阵列中的卷积与 GEMM 数据流 图 5 展示了我们用于在 MPE 阵列中映射卷积和矩阵乘法的一个数据流示例及简化伪代码。卷积层可以用 7 个维度来表示：输入和输出通道（Ci 和 Co）、特征图尺寸（H×W）、卷积核尺寸（Ki×Kj）以及 mini-batch（N）。数据流设计包括：（i）定义沿 MPE 阵列的行、列、SIMD 通路和本地寄存器堆（LRF）进行空间映射的维度，随后（ii）确定哪些数据结构沿行、列流动、哪些驻留在 LRF 中，以及元素被访问的顺序。

![](images/1e8ac9b8c5157e19c4053be3068e6ac97c28216ea4eb82c628cca5d68a05312f.jpg)  
Figure 5: 卷积数据流

我们采用了一种新颖的权重驻留（weight-stationary）数据流，它是基于以下约束推导出来的，以缩小数据流的选择范围：（i）在 batch size 低至 1 时仍能实现高利用率，这将 batch size 从空间维度中排除；（ii）避免跨行或跨列通信，这意味着 H×W 不能作为空间映射维度；（iii）当工作负载维度不是硬件维度的整数倍时，尽量减少由分段挖掘（stripmining）带来的残差效应，这意味着 Ki×Kj 不是好的选择，因为它们通常是较小的素数。因此，我们选择输入通道（Ci）和输出通道（Co）作为空间维度。此外，为避免硬件中的跨 SIMD 归约，我们将 Co 沿列和 SIMD 映射，将 Ci 沿行和 LRF 映射。在脉动（systolic）数据流中，沿某一行馈入的数据操作数需要被所有列复用，反之亦然。这意味着，沿列进行空间映射的维度应当与沿行流动的数据结构无关。因此，我们将输入数据结构沿行流动，将输出数据结构沿列流动。权重驻留在 LRF 中，保存与该 MPE（基于其在阵列中的位置）所处理的输入和输出通道相对应的元素，对应于给定的 Ki×Kj。

图 5 中包含一段简化伪代码，展示了我们对元素排序的循环序列。由于权重需要在计算开始前分块加载（block-load）到 LRF 中，因此为了获得高利用率，需要最大化分块加载之间的间隔。因此，我们使用 H×W 和 N 作为最内层循环，二者都会复用已分块加载的权重。我们选择 Ki×Kj 作为下一层循环，因为在该循环的所有迭代中，暂存器（scratchpad）中相同的输入/输出数据体都可以被复用。最后，我们放置 Ci 和 Co 循环，由于其循环计数较小，它们在硬件中被空间映射。所选数据流在几乎所有卷积层上都能实现高利用率，只有第一层例外，因为其 Ci 较小。它也可用于全连接层（此时 H×W 和 $K i \times K j$ 为 1），但在 batch size 较小时需要频繁的分块加载。

## B. SFU 阵列：全谱系的激活函数

特殊功能单元（SFU）同时包含精确版与快速版，覆盖一系列 FP16 非线性激活函数以及更高精度的 FP32 运算。DNN 运算前向与反向阶段所涉及的激活函数（包括 ReLU、Leaky-ReLU、Sigmoid 和 Tanh）、归一化函数以及池化运算均使用 SFU。此外，sqrt、exponent、ln、Tanh、Sigmoid 和倒数等函数通过近似方式实现。SFU 阵列还用于实现 shuffle、permute 和 transpose 等运算，这些运算用于训练的更新阶段。

随着我们扩展 MPE 阵列的精度和峰值 TOPs，在 SFU 中执行辅助运算所花费的总周期数占比开始增长。这促使我们将 SFU 阵列数量加倍，以在超低精度卷积和矩阵运算的计算时间与更高精度的辅助运算之间保持平衡。

## C. 稀疏感知的零值门控与频率调节

为了在支持超低精度时实现高 TOPs/W，AI 核心架构包含了时钟门控 FPU 流水线以节省能耗的机制，同时还包括稀疏感知的频率调节，以在功率限制内最大化 TOPs。

1) 节能：零值门控支持。输入操作数中相当大比例的零值带来了通过不对零操作数进行计算来节省功耗的机会。MPE 支持在乘数为零时跳过整个 FPU 流水线，并直接将加数传递到结果。

2) 稀疏感知的频率调节。在给定 DNN 的不同层之间，我们观察到权重的稀疏性存在显著差异。由于权重的稀疏性分布在推理过程中是静态的，RAPID 利用软硬件协同设计来调节功率，以在功率限制内最大化 TOPs。与涉及昂贵电压调节和 PLL 环路的基于 DVFS 的功率调制不同，这里采用由软件引导的时钟调节，在单个时钟周期时间内调制功率。

作为硅特性表征的一部分，我们测量了功率（包括动态和静态功率）随电压变化的函数关系，并确定了可接受电压范围内的频率。我们扩展了这一特性表征，通过改变模型中稀疏度的程度，推导出各工作点（电压、频率）下时钟调节的停顿率。通过使用在标称电压和频率下工作时的功率限制，我们确定了给定稀疏度下的有效频率，并用其推导出该工作点处的停顿率。

如图 6 所示，图编译器分析 DNN 所有层的权重稀疏性，并以硅特性表征数据作为输入，确定各层的调节级别，从而将功率推至最接近功率包络的水平。由于这可以在编译期间完成，其开销不会出现在推理的关键路径上，并且对于给定的 DNN，该开销可在多次推理间摊销。

RAPID 包含一个片上功率控制模块，通过时钟边沿跳过（clock-edge skipping），使用编译器推荐的调节率来降低整体执行时间，同时在功率和热限制内运行。

![](images/58664889e3c55bae4401de6e1c7d9cb927cedc6411922b0b10ee7a8c8817f325.jpg)  
图6：感知工作负载的功率节流。

D. 采用 2 个 corelet 的超低精度核心
![](images/0651006c3ac4bf15d7e053cc65464166cf76c0a33b6d6387e684aa345b9235a3.jpg)  
图7：AI核心由两个corelet及共享L1暂存器组成。

为了最大化对 L1 便笺存储器中数据的重用，并利用超低精度带来的容量需求降低，AI 核心将两组 MPE 阵列、SFU 阵列和 L0 便笺存储器（每组称为一个 corelet）与一个共享的 2MB L1 便笺存储器相结合。如图 7 所示，共享的 L1 便笺存储器通过每个 corelet 各自独立的可编程 load/store 单元进行通信，带宽为每周期 128 字节。L1 便笺存储器 2MB 的大容量使得中间输出能够保留在片上，尤其是在超低精度推理的使用场景中。同样，从 L1 到 2 个 corelet 的带宽经过平衡，以满足不同精度下数据流的速度与带宽需求匹配。例如，在改进的权重驻留数据流下，MPE 的 INT4 计算仍然仅消耗每周期 128 字节可用 L1 带宽中的 $5 / 8 ^ { t h }$。当精度缩放到 INT2 时，每个周期都有部分和被写入 L1 便笺存储器，达到可用 L1 带宽的极限。

如图 7 所示，每个配备 MPE 阵列、SFU 阵列以及两层便笺存储器层级的 AI 核心，具备用作单核心边缘加速器所需的全部特性，同时也可作为由多核心组成的更大系统的基本构建模块。

## E. 核心间及与内存的数据通信

为了与外部内存通信并能够扩展到多核心芯片，我们采用了双向环形互连，每个方向的带宽为每周期 128 字节，用于核心与内存之间的数据通信。每个核心配备一个可编程的 Memory-Neighbor Interface（MNI）单元，通过环形接口单元（RIU）促进与内存和邻居核心的数据通信。我们启用了异步时钟域跨越支持，允许环形互连与核心以不同频率运行。

图 8 展示了 MNI 的概览，以及使用独立的可编程 load 和 store 单元（MNI-LU 和 MNI-SU）在核心与内存之间传递请求和数据返回的流程。由于 DNN 中的数据访问模式既是静态的又是规则的，数据获取延迟可以通过在 L1 便笺存储器中对数据进行双缓冲并与核心中的计算在时间上重叠而被有效隐藏。编译器对程序循环进行分块和分片，通过平衡便笺存储器容量和可用带宽来指导数据获取/存储的粒度。

每个 load/store 请求都被分配一个唯一的标识标签，该标签作为 MNI-SU 和 MNI-LU 分别支持的 send/receive 原语执行的一部分而生成。通过 load 和 store 队列，MNI 支持向邻居和内存发起多个未完成请求，并且 MNI-LU 允许乱序数据返回，因为待写入的本地便笺存储器地址会被记录在 load 队列中。一旦达到允许的未完成请求上限，MNI-LU 和 SU 程序便会停顿。为了利用双向环形带宽，MNI-LU 被设计为在任意周期内可接收多达 2 个数据返回，并利用 L1 便笺存储器的分组架构以及 RIU 中的预留管理策略来避免 L1 便笺存储器中的分组冲突。

![](images/34636736525e27a0e695cbde54f6378494d1f85aca11d8bcbd2d5de5e28a366b.jpg)  
Figure 8: Multi-cast Support in Memory-Neighbor Interface DNN 工作负载展现出高度并行性，允许对工作进行空间划分并在多个核心之间共享数据 [52]。这种数据共享行为通过在 MNI 的 load 和 store 单元的 ISA 及硬件中支持多播通信来加以利用。为支持多播数据传输，编译器通过为参与核心分配共同的标识标签，生成 MNI-SU 和 MNI-LU 的 send 和 receive 原语。如图 8 所示，从核心 A 到核心 B 和 C 的多播数据传输要求每个消费者使用共同的标识标签并向核心 A 发起单独的 Recv 请求（步骤 1 和 2），同时指定参与消费者的数量。与此独立地，核心 A 的程序包含一个匹配的 Send 多播数据操作，带有相同的标识标签以及消费者数量（列表）（步骤 3），从而支持扩展到大量核心。

如图 8 中所强调的（步骤 4 至 6），核心 A 的 MNI-SU 包含“请求聚合”的硬件支持。在接收到所有参与消费者的请求后，核心 A 的 MNI-SU 动态构建消费者列表，从便笺存储器读取数据（步骤 7），并连同共同的标识标签和消费者列表一起发送到环形互连上。外部内存接口中类似的“请求聚合”支持使得多个核心的 MNI-LU 能够请求从内存以多播传输的形式发送共享数据。

## IV. 用于训练与推理系统的 RAPID 芯片

图 9 展示了 4 核 RAPID 芯片 [50] 的架构，该芯片支持 5 种不同的数据格式：FP16 (1,6,9)、FP8(1,4,3)（带可编程偏置）、FP8(1,5,2)、INT4 和 INT2，以充分利用训练与量化推理领域算法近似方面的突破性进展。

![](images/f6f8bfbd330d8dac31134a4098322fa8f6b51d09f4d619822aa8c436c50db4ea.jpg)  
Figure 9: 4 核 RAPID 架构

RAPID 芯片由 4 个核心组成，连接至双向的顺时针（CW）和逆时针（CCW）环。环与核心运行在异步时钟域中，各自具有独立的核心 PLL 和环 PLL，以便在计算与数据搬运之间最优地平衡功耗与性能。各核心通过内存/邻居接口以及跨越异步边界的环接口单元相互通信并与内存通信。为了实现向更大规模系统的扩展，多个环通过芯片管理单元（CMU）相连，CMU 既可以在单个 RAPID 芯片内闭合环路，也可以将多个芯片连接起来构成众核系统。此外，RAPID 芯片还包含功耗控制模块，通过时钟边沿跳过实现负载感知的降频调节，从而充分利用芯片的功耗预算以获得最大的应用性能。

![](images/a4e0ec8b95db988d9597a64b95e984dcbfaa97914f5dd24cb5f045af3e43b78b.jpg)  
Figure 10: 4 核 RAPID 芯片：采用 7nm EUV 工艺，面积 36mm<sup>2</sup>

图 10 展示了采用 7nm EUV 工艺制造的 36mm<sup>2</sup> RAPID 芯片，在标称电压下，该芯片在 HFP8 下可达到 3.5 TFLOPS/W，在 INT4 下可达到 16.5 TOPS/W。

## A. 推理与训练系统

由于 RAPID 芯片架构设计为可扩展至大量核心，因此可以构建多核/多芯片的推理与训练系统。

![](images/aa8cd46b7b2cc78f539e661e5774c1bda102dd54a10b5beee943a20a7d736b9e.jpg)  
Figure 11: AI 训练系统：4 芯片 32 核心

为了评估不同架构配置下的推理与训练性能，我们开发了 RAPID 芯片的详细性能模型，并将其校准到与实测结果相差不超过 1% 的精度 [50]。

在本工作中，我们研究使用 4 核 RAPID 芯片模型在 1.5 GHz 下实现 96 TOPs 的 INT4 推理性能。图 11 展示了 HFP8 训练系统模型，该系统使用 4 颗 RAPID 芯片（每颗 32 核、1.5 GHz），达到 760 TOPs，各芯片通过高带宽互连相连，以在训练的更新阶段交换梯度与权重。

## B. 软件架构

构建并部署端到端的 AI 系统远不止设计加速器核心这一项工作。AI 系统必须在一系列多样化的关键需求之间取得平衡：(i) 在各类工作负载上提供高持续性能与处理效率， 能够灵活适应快速演进的未来工作负载， 无缝集成到现有 AI 软件生态中，同时保持终端用户的生产力。

图 12 展示了该 AI 芯片端到端软件栈的高层概览。我们采用一组编译时与执行时的扩展 [53]，它们可以以插件形式接入现有框架。这既利用了深度学习框架的现有能力，又能实现针对加速器的激进性能优化。

![](images/b981af69c688cb41ca0830b18bd8ea794d8d27569a1155799e52306f83406947.jpg)  
Figure 12: 用于推理与训练的端到端软件栈

该软件栈的关键组件包括：1) 图编译器，它自动识别如何在 AI 芯片上最优地执行给定的 DNN 图，并构建程序二进制文件；2) 执行运行时，它在 AI 芯片上触发并管理计算与数据搬运操作的执行。

作为编译过程的一部分，系统会进行系统化的设计空间探索，重点关注图优化、scratchpad 管理以及向 AI 芯片各核心的工作分配。该设计空间探索由一个以带宽为中心的 AI 芯片功耗-性能解析模型引导，该模型有助于剪枝搜索空间，从而识别出将算子映射到 AI 芯片上的有利选择。该性能模型已通过与硬件实测结果进行验证，因此可用于在训练和推理场景下研究具有多核心与多芯片的规模化 AI 系统。

## V. 实验结果

在本节中，我们介绍实验方法，并总结基于 RAPID 芯片架构的推理与训练系统所达到的性能与计算效率。

## A. 实验方法

性能估计。为了评估不同架构配置下的性能，我们开发了 RAPID 芯片的详细性能模型，并将其校准到与 [50] 的实测结果相差不超过 1% 的精度。不同 DNN 算子（如卷积、ReLU 等）的功耗通过硅片实测获得，并结合各组件（MPE 阵列、SFU、scratchpad 等）的预计利用率来估算计算效率（TOPS/W）。

系统配置。对于推理，我们研究第 IV 节所述的 4 核 RAPID 芯片，连接到带宽为 200 GBps 的外部 DDR 内存。对于训练，我们考虑一个由 4 颗扩展版 RAPID 芯片（第 IV-A 节）构成的分布式系统，每颗芯片包含 32 个核心、64MB 分布式 L1 scratchpad，并连接到带宽为 400 GBps 的高带宽内存（HBM）。芯片间互连带宽为 128 GBps。我们还将展示推理与训练系统分别扩展时的性能敏感性分析。

基准测试。我们使用来自多个应用领域的 11 个最先进的 DNN 基准测试：(i) 图像分类——在 ImageNet 数据集上训练的 VGG16、Resnet50、InceptionV3、InceptionV4、MobileNetV1， 目标检测——在 COCO 数据集上训练的 SSD300、YoloV3、YoloV3-Tiny， 自然语言处理——在 WMT14 英德数据集上训练的 BERT（序列长度 = 384）、在 PennTreeBank (PTB) 数据集上训练的 2 层 LSTM，以及 语音——在 SWB300 数据集上训练的 4 层双向 LSTM。

实验设置与基线。在实验中，推理采用 batch size 为 1，训练采用 mini-batch size 为 512。为了公平比较，我们使用 RAPID 上的 FP16 实现（系统配置完全相同）作为基线，以报告在更低精度下所取得的相对提升。基于 MLPerf [54] 报告的性能数据，在按工艺和功耗/面积归一化后，我们的 FP16 基线与其他加速器设计相比具有很强的竞争力。此外，我们还提供了在更低精度下所达到的绝对推理延迟和训练吞吐量（每秒输入数）。对于推理，我们将研究范围限制在 FP8-fwd 和 INT4 精度，而将 INT2 实现留作未来工作，因为如第 II 节所述，INT2 模型仍会导致约 2% 的精度损失。另外，我们仅在 FP16 精度剪枝的 DNN 上下文中研究稀疏感知频率调节的收益，因为将剪枝与超低精度相结合仍是一个处于快速发展阶段的研究领域。

## B. 推理性能与效率

图 13 展示了 4 核 RAPID 芯片在不同基准测试上以 FP8（1-4-3 格式）和 INT4 精度实现的推理延迟（阴影等高线）。与 RAPID 上的 FP16 基线相比，FP8 和 INT4 实现分别取得了 1.2×-1.9×（平均 1.55×）和 1.4×-4.2×（平均 2.8×）的端到端性能（柱状图）提升。低精度下的加速主要受限于以 FP16 执行的操作比例，即第一层和最后一层、激活函数、归一化和池化操作等。具有计算密集型卷积层的图像分类和目标检测基准测试获得了最佳提升，而具有精简卷积和较大比例辅助操作的移动网络获益最少。

图 14 展示了在 FP8 和 INT4 精度下实现的持续计算效率（TOPS/W）（阴影等高线）。得益于 RAPID 的微架构/电路设计，TOPS/W 在不同精度间具有较强的可扩展性——FP8 实现在各基准测试中实现了 1.4-4.68（平均 3.16）TOPS/W，而 INT4 实现在全国范围内取得了 3-13.5（平均

![](images/bdf3a858a7b49b8a0f849270410411ef9a6764adafa9572b362f93b6bdc7f067.jpg)  
图 13：使用 4 核 RAPID 芯片的每秒分类数

7）TOPS/W 的成绩。这相当于相比 FP16 基线（柱状图）分别取得了 1.6× 和 3.6× 的提升。

![](images/2c50e68c6f743aa787aa0c61bf722d716a819a8bcf3ce81f42a86482f74561ea.jpg)  
图 14：4 核 RAPID 芯片上的持续 TOPS/W

结果表明，即使 batch size 为 1，RAPID 芯片在 FP8 和 INT4 精度下进行推理时也能同时实现高持续 TOPS 和 TOPS/W。

## C. 训练吞吐量

图 15 展示了由 4 颗 RAPID 芯片组成的训练系统（第 IV 节）在各基准测试上以 FP16 和 Hybrid-FP8 两种精度进行训练的吞吐量（即每秒训练的输入数量）。比较 FP16 与 HFP8，加速比介于 1.1×-2× 之间（平均 1.4×）。与推理不同，较大的 mini-batch 使得在卷积和 GEMM 操作中降低精度时仍能实现高核心利用率。然而，训练的整体加速比略低于推理，原因有二： 训练会为梯度归约和权重广播产生额外的片外通信开销； 训练是访存密集型的，因为前向传播过程中产生的激活值需要被保留，以便在反向传播期间计算权重梯度。

## D. 感知稀疏性的降频方案的收益

我们现在展示第 III-C 节所述的感知稀疏性的 zero-grating 与频率降频（frequency throttling）方案所带来的性能提升。在给定功率预算下，图 16(a) 展示了基于硅片测量结果推导出的 4 核 RAPID 芯片在不同稀疏度水平下降频频率的应用比例。我们在推理场景中应用该方案，使用公开可用的稀疏（或剪枝）模型，因为降频程度可以在编译时确定。为此，我们考虑了多个基准测试的剪枝版本 [55, 56, 57, 58]。所使用的剪枝模型采用 FP16 精度；将剪枝与低精度相结合仍是一个不断发展的研究领域，我们希望将其作为未来工作的一部分进行探索。图 16(b) 展示了各基准测试的平均稀疏度，以及与不使用感知稀疏性降频的基线相比所取得的加速比。平均稀疏度（跨层和网络）在 50%-80% 之间变化，且精度损失可忽略不计。相应地，通过选择正确的工作频率，我们取得了 1.1×-1.7×（平均 1.3×）的性能提升。

![](images/d046d7e35ba14c5fa8a0929d3c61a4b899f59dd9834e54312bafe84bc15eb083.jpg)  
图 15：4 芯片 RAPID 训练系统的吞吐量

![](images/d343fc138188d4652d9504bfc0168ecb8856f43649a0341d55fe6ead67bfd2e6.jpg)

![](images/6eecdef40e7cde4748778d0f3fb88c352ea7b0a873dcd41232ae5b0676b2ccd9.jpg)  
图 16：感知稀疏性降频带来的性能收益

## E. 性能分解分析

我们现在展示影响 RAPID 端到端性能的关键因素。对于 INT4 推理，图 17 将计算周期分解为四个关键类别，即 Conv/GEMM、Conv/GEMM 开销、量化和辅助操作。第一类由在 MPE 阵列上执行并充分利用 RAPID 芯片全部计算能力的 Conv/GEMM 操作构成。注意，虽然大多数层以 INT4 精度执行，但为了保持精度，仍有一小部分层可能需要以 FP16 执行。第二类涵盖了 Conv/GEMM 执行期间产生的开销。这些开销来自多种因素，包括由于给定工作负载维度对 MPE 阵列的时空利用不足而导致的数据流低效，以及分配给每个核/corelet 的工作不均衡等。第三类包括在 FP16 ⇔ INT4 之间转换数据所需执行的额外量化和缩放操作。鉴于 MPE 阵列在低精度下的高吞吐量，这一开销变得不容忽视，尤其是当激活值尺寸较大时。最后一类包括其他辅助操作（激活函数、批归一化等）的成本，这些操作在 SFU 中以 FP16 执行。

![](images/d115782c8bf9491646434416d9674926b46a3f8c9fec58038fd8f270d03f01e8.jpg)  
图 17：INT4 推理的计算周期分解

我们观察到，各基准测试在每个类别上消耗的计算周期方面相当异构。诸如 inception3/4、Tiny-yolov3 和 LSTM 等 DNN，由于其工作负载维度与 MPE 阵列维度不完全匹配，会在 CONV/GEMM 操作期间产生开销。具有大激活值尺寸的卷积网络会带来量化开销，而移动网络（MobileNetv1、Tiny-yolov3）则包含最多的辅助操作。平均而言，Conv/GEMM 占据了 50% 的计算周期，而 Conv/GEMM 开销、量化和辅助操作分别占 14%、17% 和 19%。

## F. 推理/训练系统扩展性

在本节中，我们展示在扩展推理和训练系统规模时所取得的加速比。对于推理系统，我们增加芯片中的核心数量，并展示 INT4 精度下的加速比；对于训练系统，我们增加系统中的芯片数量，并展示 HFP8 精度下的加速比。

如图 18(a) 所示，即使 mini-batch 为 1，随着核心数量从 1 扩展到 32，性能也随之提升。计算密集型基准测试（如 VGG16、Resnet50、Yolov3、SSD300）即使扩展到 32 核仍能获得性能提升。对于那些要么由辅助操作主导（MobileNetv1）、要么由于 INT4 引擎的高 TOPs 而受访存停顿主导的基准测试，随着核心数量的增加，加速比会出现饱和，尤其是因为我们在扩展系统中核心数量的同时固定了外部带宽。

类似地，图 18(b) 展示了随着系统中的芯片总数从 1 增加到 32、芯片间带宽为 128 GBps 时，HFP8 训练的加速比。这些研究采用了数据并行，因此需要通信梯度以及训练更新阶段的权重。HFP8 减少了权重的通信开销，因为前向传播仅使用 8 位权重，且每颗芯片并发计算其拥有的那部分权重的更新量，并仅将更新后的 8 位权重传送给邻居。

![](images/a00e04f28e86d27eaa267ceab650014fd79367790a292f5568f03d920f2857a9.jpg)  
图 18：推理与训练的性能扩展性

## VI. 相关工作

提升 AI 工作负载在不同硬件平台上的运行效率是一个活跃的研究课题。我们在此描述在 CPU、GPU、加速器以及商用 AI 芯片上加速 AI 工作负载的相关研究工作。

基于 CPU 的技术。在 CPU 上加速 AI 工作负载的技术包括使用优化的线性代数库 [59]、面向多核的高效并行化技术 [60, 61]，以及高效的数据布局与批处理（batching）[62]。近期的一些工作还提出了编译器、ISA 和微架构层面的优化，以利用 AI 工作负载的某些特性，例如稀疏性（sparsity）[63, 64, 65]。

基于 GPU 的技术。在 GPU 上加速 AI 工作负载的研究工作主要聚焦于数据/模型/流水线并行化技术 [66]、内存管理 [67] 以及局部性感知的设备放置 [68, 69]。与 CPU 类似，一些工作也探索了利用激活值和权重中的稀疏性 [70]。

硬件加速器技术。专用硬件是满足 AI 工作负载计算需求的关键。认识到这一点后，业界已提出大量加速器方案，范围涵盖低功耗 ASIC / FPGA / CGRA 核 [21, 22, 23, 24, 26, 32, 47, 71, 72, 73, 74, 75, 76, 77, 78] 到大规模系统 [29, 30, 31, 79, 80, 81]。这些架构利用稠密算术阵列、异构处理单元（tile）、低精度数据表示，有时还借助动态硬件重配置，展现出令人瞩目的峰值处理能力。近期的工作还探索了利用激活值和权重中的稀疏性 [25, 27, 82]、3D 内存技术 [28, 83]、比特串行（bit-serial）架构 [84, 85] 以及存内计算 [86, 87, 88, 89, 90, 91] 来进一步提升效率。

商用 AI 芯片。专用 AI 加速器的巨大成功进一步推动了其商用化设计进程。其中包括 Google TPU、NVIDIA Tensor Core、Intel NNP 等 [33, 34, 35]。

对于所有这些 AI 加速器方案而言，跨代际扩展性能是其在商业部署中取得成功的关键。由于 AI 工作负载为通过精度缩放（precision scaling）实现性能/能效提升提供了独特的机遇，我们利用这一手段设计了 RAPID，以支持一种能够进行 8 位训练和 4 位推理的混合精度架构。

## VII. 结论

我们提出了一种支持超低精度训练与推理的 4 核 AI 芯片——RAPID 的设计 [50]。RAPID 支持混合精度执行，涵盖多种数值格式，包括 FP16、Hybrid-FP8、INT4 和 INT2。RAPID 提供了广泛的工作负载覆盖（CNN、LSTM 和 transformer），并可扩展至多核和多芯片。基于 7nm 工艺的 4 核 RAPID 芯片的流片测试结果表明，其在 FP8 模式下达到 3.5 TFLOPS/W，在 INT4 模式下达到 16.5 TOPS/W。利用与实测结果校准误差在 1% 以内的性能模型，我们评估了由 4 颗 RAPID 芯片组成的 768 TOPS AI 系统的 FP8 训练性能，以及单颗 RAPID 芯片系统的 INT4 推理性能。结果显示，批大小（batch size）为 1 时的 INT4 推理可实现 3 - 13.5（平均 7）TOPS/W，而小批量（mini-batch）为 512 的 FP8 训练在广泛的应用范围内可实现 102 - 588（平均 203）TFLOPS 的持续性能。作为未来工作，我们计划研究 RAPID 的 INT2 性能以及面向超低精度的稀疏感知功耗调节（sparsity-aware power throttling）。

## REFERENCES

[1] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, page 2012.

[2] Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, and Yann Lecun. Overfeat: Integrated recognition, localization and detection using convolutional networks. http://arxiv.org/abs/1312.6229.

[3] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014.

[4] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015.

[5] C. Szegedy et. al. Going deeper with convolutions. In Proc. CVPR, 2015.

[6] A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar, and L. Fei-Fei. Large-scale video classification with convolutional neural networks. In 2014 IEEE Conference on Computer Vision and Pattern Recognition, pages 1725–1732, June 2014.

[7] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C. Berg. Ssd: Single shot multibox detector. Lecture Notes in Computer Science, page 21–37, 2016.

[8] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications, 2017.

[9] Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, and Kurt Keutzer. Squeezenet: Alexnet-level accuracy with 50x fewer parameters and ¡0.5mb model size, 2016.

[10] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision, 2015.

[11] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alex Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning, 2016.

[12] Gao Huang, Zhuang Liu, and Kilian Q. Weinberger. Densely connected convolutional networks. CoRR, abs/1608.06993, 2016.

[13] Y. Wu. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv, 2016.

[14] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, pages 6000–6010, USA, 2017. Curran Associates Inc.

[15] G. Hinton et. al. Deep neural networks for acoustic modeling in speech recognition. Signal Processing Magazine, 2012.

[16] X. Zhang et. al. Text understanding from scratch. arXiv, 2015.

[17] S. Venugopalan et. al. Sequence to sequence - video to text. In Proc. ICCV, 2015.

[18] A. Hannun et. al. Deep speech: Scaling up end-to-end speech recognition. arXiv, 2014.

[19] Barret Zoph and Quoc V. Le. Neural architecture search with reinforcement learning. 2017.

[20] X. Zhang, X. Zhou, M. Lin, and J. Sun. Shufflenet: An extremely efficient convolutional neural network for mobile devices. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6848–6856, June 2018.

[21] Srimat Chakradhar, Murugan Sankaradas, Venkata Jakkula, and Srihari Cadambi. A dynamically configurable coprocessor for convolutional neural networks. SIGARCH Comput. Archit. News, 38(3):247–257, June 2010.

[22] C. Farabet, B. Martini, B. Corda, P. Akselrod, E. Culurciello, and Y. LeCun. Neuflow: A runtime reconfigurable dataflow processor for vision. In CVPR 2011 WORKSHOPS, pages 109–116, June 2011.

[23] Tianshi Chen, Zidong Du, Ninghui Sun, Jia Wang, Chengyong Wu, Yunji Chen, and Olivier Temam. Diannao: A small-footprint highthroughput accelerator for ubiquitous machine-learning. In Proceedings of the 19th International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS ’14, pages 269–284, New York, NY, USA, 2014. ACM.

[24] Yu-Hsin Chen, Joel Emer, and Vivienne Sze. Eyeriss: A spatial architecture for energy-efficient dataflow for convolutional neural networks. In Proceedings of the 43rd International Symposium on Computer Architecture, ISCA ’16, pages 367–379, Piscataway, NJ, USA, 2016. IEEE Press.

[25] S. Han, X. Liu, H. Mao, J. Pu, A. Pedram, M. A. Horowitz, and W. J. Dally. Eie: Efficient inference engine on compressed deep neural network. In 2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA), pages 243–254, June 2016.

[26] Brandon Reagen, Paul Whatmough, Robert Adolf, Saketh Rama, Hyunkwang Lee, Sae Kyu Lee, Jose Miguel Hern´ andez-Lobato, Gu-´ Yeon Wei, and David Brooks. Minerva: Enabling low-power, highlyaccurate deep neural network accelerators. In Proceedings of the 43rd International Symposium on Computer Architecture, ISCA ’16, pages 267–278, Piscataway, NJ, USA, 2016. IEEE Press.

[27] Jorge Albericio, Patrick Judd, Tayler Hetherington, Tor Aamodt, Natalie Enright Jerger, and Andreas Moshovos. Cnvlutin: Ineffectualneuron-free deep neural network computing. In Proceedings of the 43rd International Symposium on Computer Architecture, ISCA ’16, pages 1–13, Piscataway, NJ, USA, 2016. IEEE Press.

[28] D. Kim, J. Kung, S. Chai, S. Yalamanchili, and S. Mukhopadhyay. Neurocube: A programmable digital neuromorphic architecture with high-density 3d memory. In 2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA), pages 380–392, June 2016.

[29] Abhinandan Majumdar, Srihari Cadambi, Michela Becchi, Srimat T. Chakradhar, and Hans Peter Graf. A massively parallel, energy efficient programmable accelerator for learning and classification. ACM Trans. Archit. Code Optim., 9(1):6:1–6:30, March 2012.

[30] Norman P. Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, Rick Boyle, Pierre-luc Cantin, Clifford Chao, Chris Clark, Jeremy Coriell, Mike Daley, Matt Dau, Jeffrey Dean, Ben Gelb, Tara Vazir Ghaemmaghami, Rajendra Gottipati, William Gulland, Robert Hagmann, C. Richard Ho, Doug Hogberg, John Hu, Robert Hundt, Dan Hurt, Julian Ibarz, Aaron Jaffey, Alek Jaworski, Alexander Kaplan, Harshit Khaitan, Andy Koch, Naveen Kumar, Steve Lacy, James Laudon, James Law, Diemthu Le, Chris Leary, Zhuyuan Liu, Kyle Lucke, Alan Lundin, Gordon MacKean, Adriana Maggiore, Maire Mahony, Kieran Miller, Rahul Nagarajan, Ravi Narayanaswami, Ray Ni, Kathy Nix, Thomas Norrie, Mark Omernick, Narayana Penukonda, Andy Phelps, Jonathan Ross, Matt Ross, Amir Salek, Emad Samadiani, Chris Severn, Gregory Sizikov, Matthew Snelham, Jed Souter, Dan Steinberg, Andy Swing, Mercedes Tan, Gregory Thorson, Bo Tian, Horia Toma, Erick Tuttle, Vijay Vasudevan, Richard Walter, Walter Wang, Eric Wilcox, and Doe Hyun Yoon. In-datacenter performance

analysis of a tensor processing unit. In Proceedings of the 44th International Symposium on Computer Architecture, ISCA ’17, 2017.

[31] S. Venkataramani, A. Ranjan, S. Banerjee, D. Das, S. Avancha, A. Jagannathan, A. Durg, D. Nagaraj, B. Kaul, P. Dubey, and A. Raghunathan. Scaledeep: A scalable compute architecture for learning and evaluating deep networks. In 2017 ACM/IEEE 44th Annual International Symposium on Computer Architecture (ISCA), volume 00, pages 13–26, June 2017.

[32] Swagath Venkataramani, Vinay K. Chippa, Srimat T. Chakradhar, Kaushik Roy, and Anand Raghunathan. Quality programmable vector processors for approximate computing. In Proceedings of the 46th Annual IEEE/ACM International Symposium on Microarchitecture, MICRO-46, pages 1–12, New York, NY, USA, 2013. ACM.

[33] Google supercharges machine learning tasks with tpu custom chip. Google Research blog, 2016.

[34] O. Wechsler, M. Behar, and B. Daga. Spring hill (nnp-i 1000) intel’s data center inference chip. In 2019 IEEE Hot Chips 31 Symposium (HCS), pages 1–12, Aug 2019.

[35] Nvidia tensor cores: https://www.nvidia.com/en-us/datacenter/tensorcore/. NVIDIA blog, 2017.

[36] Chenzhuo Zhu, Song Han, Huizi Mao, and William J. Dally. Trained ternary quantization. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings, 2017.

[37] Jan Achterhold, Jan Mathias Koehler, Anke Schmeink, and Tim Genewein. Variational network quantization. In International Conference on Learning Representations, 2018.

[38] Antonio Polino, Razvan Pascanu, and Dan Alistarh. Model compression via distillation and quantization. arXiv preprint arXiv:1802.05668, 2018.

[39] Lu Hou and James T Kwok. Loss-aware weight quantization of deep networks. arXiv preprint arXiv:1802.08635, 2018.

[40] Eunhyeok Park, Junwhan Ahn, and Sungjoo Yoo. Weighted-entropybased quantization for deep neural networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 7197–7205, 2017.

[41] Zhaowei Cai, Xiaodong He, Jian Sun, and Nuno Vasconcelos. Deep learning with low precision by half-wave gaussian quantization. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5918–5926, 2017.

[42] Jungwook Choi, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srinivasan, and Kailash Gopalakrishnan. Pact: Parameterized clipping activation for quantized neural networks, 2018.

[43] Sangil Jung, Changyong Son, Seohyung Lee, Jinwoo Son, Jae-Joon Han, Youngjun Kwak, Sung Ju Hwang, and Changkyu Choi. Learning to quantize deep networks by optimizing quantization intervals with task loss. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

[44] Naigang Wang, Jungwook Choi, Daniel Brand, Chia-Yu Chen, and Kailash Gopalakrishnan. Training deep neural networks with 8-bit floating point numbers. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, 3-8 December 2018, Montreal, Canada´ , pages 7686–7695, 2018.

[45] Xiao Sun, Jungwook Choi, Chia-Yu Chen, Naigang Wang, Swagath Venkataramani, Vijayalakshmi Srinivasan, Xiaodong Cui, Wei Zhang, and Kailash Gopalakrishnan. Hybrid 8-bit floating point (hfp8) training and inference for deep neural networks. In Advances in Neural Information Processing Systems 32, pages 4901–4910. Curran Associates, Inc., 2019.

[46] Jungwook Choiand, Swagath Venkataramani, Vijayalakshmi Srinivasan, Kailash Gopalakrishnan, Zhuo Wang, and Pierce I-Jen Chuang. Accurate and efficient 2-bit quantized neural network. SysML, 2019.

[47] Y. H. Chen, T. Krishna, J. Emer, and V. Sze. 14.5 eyeriss: An energy-efficient reconfigurable accelerator for deep convolutional neural networks. In 2016 IEEE International Solid-State Circuits Conference (ISSCC), pages 262–263, Jan 2016.

[48] B. Fleischer, S. Shukla, M. Ziegler, J. Silberman, J. Oh, V. Srinivasan, J. Choi, S. Mueller, A. Agrawal, T. Babinsky, N. Cao, C.Y. Chen, P. Chuang, T. Fox, G. Gristede, M. Guillorn, H. Haynie, M. Klaiber, D. Lee, S.Lo, G. Maier, M. Scheuermann, S. Venkataramani, C. Vezyrtzis, N. Wang, F. Yee, C. Zhou, P. F. Lu, B. Curran, L. Chang,

K. Gopalakrishnan. A scalable multi-teraops deep learning processor core for ai training and inference. In Proc. VLSI Symposium, June 2018.

[49] J. Oh, S. K. Lee, M. Kang, M. Ziegler, J. Silberman, A. Agrawal, S. Venkataramani, B. Fleischer, M. Guillorn, J. Choi, W. Wang, S. Mueller, S. Ben-Yehuda, J. Bonanno, N. Cao, R. Casatuta, C. Y. Chen, M. Cohen, O. Erez, T. Fox, G. Gristede, H. Haynie, V. Ivanov, S. Koswatta, S. H. Lo, M. Lutz, G. Maier, A. Mesh, Y. Nustov, S. Rider, M. Schaal, M. Scheuermann, X. Sun, N. Wang, F. Yee, C. Zhou, V. Shah, B. Curran, V. Srinivasan, P. F. Lu, S. Shukla, K. Gopalakrishnan, and L. Chang. A 3.0 tflops 0.62v scalable processor core for high compute utilization ai training and inference. In 2020 IEEE Symposium on VLSI Circuits, pages 1–2, 2020.

[50] Ankur Agrawal, Sae Kyu Lee, Joel Silberman, Matthew Ziegler, Mingu Kang, Swagath Venkataramani, Nianzheng Cao, Bruce Fleischer, Michael Guillorn, Matthew Cohen, Silvia Mueller, Jinwook Oh, Martin Lutz, Jinwook Jung, Siyu Koswatta, Ching Zhou, Vidhi Zalani, James Bonanno, Robert Casatuta, Chia-Yu Chen, Jungwook Choi, Howard Haynie, Alyssa Herbert, Radhika Jain, Monodeep Kar, Kyu-Hyoun Kim, Yulong Li, Zhibin Ren, Scot Rider, Marcel Schaal, Kerstin Schelm, Michael Scheuermann, Xiao Sun, Hung Tran, Naigang Wang, Wei Wang, Xin Zhang, Vinay Shah, Brian Curran, Vijayalakshmi Srinivasan, Pong-Fei Lu, Sunil Shukla, Leland Chang, and Kailash Gopalakrishnan. 9.1 a 7nm 4-core ai chip with 25.6tflops hybrid fp8 training, 102.4tops int4 inference and workload-aware throttling. In 2021 IEEE International Solid- State Circuits Conference (ISSCC), volume 64, pages 144–146, 2021.

[51] Charbel Sakr, Naigang Wang, Chia-Yu Chen, Jungwook Choi, Ankur Agrawal, Naresh R. Shanbhag, and Kailash Gopalakrishnan. Accumulation bit-width scaling for ultra-low precision training of deep networks. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019, 2019.

[52] Swagath Venkataramani, Vijayalakshmi Srinivasan, Jungwook Choi, Philip Heidelberger, Leland Chang Chang, and Kailash Gopalakrishnan. Memory and interconnect optimizations for peta-scale deep learning systems. In Proceedings of the 26TH IEEE International Conference On High Performance Computing, Data, and Analytics, 2019.

[53] S. Venkataramani, J. Choi, V. Srinivasan, W. Wang, J. Zhang, M. Schaal, M. J. Serrano, K. Ishizaki, H. Inoue, E. Ogawa, M. Ohara, L. Chang, and K. Gopalakrishnan. Deeptools: Compiler and execution runtime extensions for rapid ai accelerator. IEEE Micro, Sep. 2019.

[54] Mlperf benchmark suite: https://mlperf.org.

[55] Michael Zhu and Suyog Gupta. To prune, or not to prune: exploring the efficacy of pruning for model compression. arXiv preprin arXiv:1710.01878, 2017.

[56] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. In Advances in neural information processing systems, pages 1135–1143, 2015.

[57] Pravendra Singh, R Manikandan, Neeraj Matiyali, and Vinay Nam boodiri. Multi-layer pruning framework for compressing single shot multibox detector. In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 1318–1327. IEEE, 2019.

[58] Mitchell A. Gordon, Kevin Duh, and Nicholas Andrews. Compressing bert: Studying the effects of weight pruning on transfer learning. In RepL4NLP@ACL, 2020.

[59] Martin Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, Manjunath Kudlur, Josh Levenberg, Rajat Monga, Sherry Moore, Derek G. Murray, Benoit Steiner, Paul Tucker, Vijay Vasudevan, Pete Warden, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. Tensorflow: A system for large-scale machine learning. In Proc. OSDI, pages 265–283, 2016.

[60] Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Marc’aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, Quoc V. Le, and Andrew Y. Ng. Large scale distributed deep networks. In P. Bartlett, F.c.n. Pereira, C.j.c. Burges, L. Bottou, and K.q. Weinberger, editors, Advances in Neural Information Processing Systems 25, pages 1232–1240. 2012.

[61] Dipankar Das, Sasikanth Avancha, Dheevatsa Mudigere, Karthikeyan Vaidyanathan, Srinivas Sridharan, Dhiraj D. Kalamkar, Bharat Kaul, and Pradeep Dubey. Distributed deep learning using synchronous stochastic gradient descent. CoRR, abs/1602.06709, 2016.

[62] Vincent Vanhoucke, Andrew Senior, and Mark Z. Mao. Improving the speed of neural networks on cpus. In Deep Learning and Unsupervised Feature Learning Workshop, NIPS 2011, 2011.

[63] Sanchari Sen, Shubham Jain, Swagath Venkataramani, and Anand Raghunathan. Sparce: Sparsity aware general-purpose core extensions to accelerate deep neural networks. IEEE Trans. Comput., 68(6):912– 925, June 2019.

[64] Berkin Akin, Zeshan A Chishti, and Alaa R Alameldeen. Zcomp: Reducing dnn cross-layer memory footprint using vector extensions. In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pages 126–138, 2019.

[65] Zhangxiaowen Gong, Houxiang Ji, Christopher W. Fletcher, C. Hughes, Sara S. Baghsorkhi, and J. Torrellas. Save: Sparsity-aware vector engine for accelerating dnn training and inference on cpus. 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pages 796–810, 2020.

[66] Forrest N. Iandola, Khalid Ashraf, Matthew W. Moskewicz, and Kurt Keutzer. Firecaffe: near-linear acceleration of deep neural network training on compute clusters. CoRR, abs/1511.00175, 2015.

[67] Minsoo Rhu, Natalia Gimelshein, Jason Clemons, Arslan Zulfiqar, and Stephen W. Keckler. Virtualizing deep neural networks for memoryefficient neural network design. CoRR, abs/1602.08124, 2016.

[68] Azalia Mirhoseini, Hieu Pham, Quoc Le, Mohammad Norouzi, Samy Bengio, Benoit Steiner, Yuefeng Zhou, Naveen Kumar, Rasmus Larsen, and Jeff Dean. Device placement optimization with reinforcement learning. 2017.

[69] Yuanxiang Gao, Li Chen, and Baochun Li. Spotlight: Optimizing device placement for training deep neural networks. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1662–1670, Stockholmsmassan, Stockholm Sweden, 10–15 Jul 2018. PMLR.

[70] Minsoo Rhu, Mike O’Connor, Niladrish Chatterjee, Jeff Pool, Youngeun Kwon, and Stephen W Keckler. Compressing dma engine: Leveraging activation sparsity for training deep neural networks. In 2018 IEEE International Symposium on High Performance Computer Architecture (HPCA), pages 78–91. IEEE, 2018.

[71] V. Gokhale, J. Jin, A. Dundar, B. Martini, and E. Culurciello. A 240 g-ops/s mobile coprocessor for deep neural networks. In 2014 IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 696–701, June 2014.

[72] S. Eldridge, A. Waterland, M. Seltzer, J. Appavoo, and A. Joshi. Towards general-purpose neural network computing. In 2015 International Conference on Parallel Architecture and Compilation (PACT), pages 99–112, Oct 2015.

[73] Chen Zhang, Peng Li, Guangyu Sun, Yijin Guan, Bingjun Xiao, and Jason Cong. Optimizing fpga-based accelerator design for deep convolutional neural networks. In Proceedings ofthe 2015 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays, FPGA ’15, pages 161–170, New York, NY, USA, 2015. ACM.

[74] Masakazu Tanomoto, Shinya Takamaeda-Yamazaki, Jun Yao, and Yasuhiko Nakashima. A cgra-based approach for accelerating convolutional neural networks. In Proceedings of the 2015 IEEE 9th International Symposium on Embedded Multicore/Many-core Systemson-Chip, MCSOC ’15, pages 73–80, Washington, DC, USA, 2015. IEEE Computer Society.

[75] S. Liu, Z. Du, J. Tao, D. Han, T. Luo, Y. Xie, Y. Chen, and T. Chen. Cambricon: An instruction set architecture for neural networks. In 2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA), pages 393–405, June 2016.

[76] Naveen Suda, Vikas Chandra, Ganesh Dasika, Abinash Mohanty, Yufei Ma, Sarma Vrudhula, Jae-sun Seo, and Yu Cao. Throughput-optimized opencl-based fpga accelerator for large-scale convolutional neural networks. In Proceedings of the 2016 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays, FPGA ’16, page 16–25, New York, NY, USA, 2016. Association for Computing Machinery.

[77] W. Lu, G. Yan, J. Li, S. Gong, Y. Han, and X. Li. Flexflow: A flexible dataflow accelerator architecture for convolutional neural networks. In 2017 IEEE International Symposium on High Performance Computer Architecture (HPCA), pages 553–564, Feb 2017.

[78] Mateja Putic, Swagath Venkataramani, Schuyler Eldridge, Alper

Buyuktosunoglu, Pradip Bose, and Mircea Stan. Dyhard-dnn: Even more dnn acceleration with dynamic hardware reconfiguration. In Proceedings of the 55th Annual Design Automation Conference, DAC ’18, New York, NY, USA, 2018. Association for Computing Machinery.

[79] Yunji Chen, Tao Luo, Shaoli Liu, Shijin Zhang, Liqiang He, Jia Wang, Ling Li, Tianshi Chen, Zhiwei Xu, Ninghui Sun, and Olivier Temam. Dadiannao: A machine-learning supercomputer. In Proceedings of the 47th Annual IEEE/ACM International Symposium on Microarchitecture, MICRO-47, pages 609–622, Washington, DC, USA, 2014. IEEE Computer Society.

[80] J. Fowers, K. Ovtcharov, M. Papamichael, T. Massengill, M. Liu, D. Lo, S. Alkalay, M. Haselman, L. Adams, M. Ghandi, S. Heil, P. Patel, A. Sapek, G. Weisz, L. Woods, S. Lanka, S. K. Reinhardt, A. M. Caulfield, E. S. Chung, and D. Burger. A configurable cloudscale dnn processor for real-time ai. In 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), pages 1–14, June 2018.

[81] Yakun Sophia Shao, Jason Clemons, Rangharajan Venkatesan, Brian Zimmer, Matthew Fojtik, Nan Jiang, Ben Keller, Alicia Klinefelter, Nathaniel Pinckney, Priyanka Raina, and et al. Simba: Scaling deeplearning inference with multi-chip-module-based architecture. In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, MICRO ’52, page 14–27, New York, NY, USA, 2019. Association for Computing Machinery.

[82] Angshuman Parashar, Minsoo Rhu, Anurag Mukkara, Antonio Puglielli, Rangharajan Venkatesan, Brucek Khailany, Joel S. Emer, Stephen W. Keckler, and William J. Dally. SCNN: an accelerator for compressedsparse convolutional neural networks. CoRR, abs/1708.04485, 2017.

[83] Mingyu Gao, Jing Pu, Xuan Yang, Mark Horowitz, and Christos Kozyrakis. Tetris: Scalable and efficient neural network acceleration with 3d memory. In Proceedings of the Twenty-Second International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS ’17, page 751–764, New York, NY, USA, 2017. Association for Computing Machinery.

[84] P. Judd, J. Albericio, T. Hetherington, T. M. Aamodt, and A. Moshovos. Stripes: Bit-serial deep neural network computing. In 2016 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pages 1–12, Oct 2016.

[85] H. Sharma, J. Park, N. Suda, L. Lai, B. Chau, V. Chandra, and H. Esmaeilzadeh. Bit fusion: Bit-level dynamically composable architecture for accelerating deep neural network. In 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), pages 764–775, June 2018.

[86] A. Shafiee, A. Nag, N. Muralimanohar, R. Balasubramonian, J. P. Strachan, M. Hu, R. S. Williams, and V. Srikumar. Isaac: A convolutional neural network accelerator with in-situ analog arithmetic in crossbars. In 2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA), pages 14–26, June 2016.

[87] Ping Chi, Shuangchen Li, Cong Xu, Tao Zhang, Jishen Zhao, Yongpan Liu, Yu Wang, and Yuan Xie. Prime: A novel processing-in-memory architecture for neural network computation in reram-based main memory. SIGARCH Comput. Archit. News, 44(3):27–39, June 2016.

[88] L. Song, X. Qian, H. Li, and Y. Chen. Pipelayer: A pipelined rerambased accelerator for deep learning. In 2017 IEEE International Symposium on High Performance Computer Architecture (HPCA), pages 541–552, Feb 2017.

[89] J. Zhang, Z. Wang, and N. Verma. In-memory computation of a machine-learning classifier in a standard 6t sram array. IEEE Journal of Solid-State Circuits, 52(4):915–924, April 2017.

[90] Aayush Ankit, Izzat El Hajj, Sai Rahul Chalamalasetti, Geoffrey Ndu, Martin Foltin, R. Stanley Williams, Paolo Faraboschi, Wenmei W Hwu, John Paul Strachan, Kaushik Roy, and et al. Puma: A programmable ultra-efficient memristor-based accelerator for machine learning inference. In Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS ’19, page 715–731, New York, NY, USA, 2019. Association for Computing Machinery.

[91] Shubham Jain, Sumeet Kumar Gupta, and Anand Raghunathan. Timdnn: Ternary in-memory accelerator for deep neural networks, 2019.