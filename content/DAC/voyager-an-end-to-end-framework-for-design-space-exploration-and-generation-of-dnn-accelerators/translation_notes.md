# Voyager: An End-to-End Framework for Design-Space Exploration and Generation of DNN Accelerators 原文翻译

# Voyager：DNN 加速器设计空间探索与生成的端到端框架

Kartik Prabhu — 斯坦福大学，美国斯坦福，kprabhu7@stanford.edu  
Zhouhua Xie — 斯坦福大学，美国斯坦福，xzh015@stanford.edu  
Ammar Ali Ratnani — 斯坦福大学，美国斯坦福，aratnani@stanford.edu  
Jeffrey Yu — 斯坦福大学，美国斯坦福，jeffreyy@stanford.edu  
Abigail Aleshire — 斯坦福大学，美国斯坦福，abbya@stanford.edu  
Xinyuan Allen Pan — 斯坦福大学，美国斯坦福，allpan@stanford.edu  
Zihan Chen — 斯坦福大学，美国斯坦福，zihan410@stanford.edu  
Priyanka Raina — 斯坦福大学，美国斯坦福，praina@stanford.edu

## 摘要

深度神经网络（DNN）在计算机视觉、自然语言处理等领域取得先进表现，但高效运行这些计算密集模型需要专用硬件。加速器设计耗时、依赖大量人工，难以扩展到多个设计点。已有自动生成工作通常参数化有限，无法生成高性能、可流片设计，对多种数据类型及量化方案支持不足，也缺少集成的端到端编译器。

本文提出基于高层次综合（HLS）的 Voyager，支持快速设计空间探索和 DNN 加速器生成。它可配置工艺节点、时钟和规模，包括 PE 数、片上缓冲大小和外部存储带宽。相比已有工作，支持更多数据类型和量化方案：内置任意长度浮点、posit、整数，以及用户自定义格式，均可采用逐张量缩放和微缩放量化。基于 PyTorch 的编译器将神经网络端到端高效映射到生成硬件，支持量化、算子融合和分块。

我们在先进视觉、语言模型上评估。Voyager 支持不同类型与量化方案的全数据集准确率评估和快速设计探索。生成设计跨模型和规模利用率最高 99.8%；相对已有生成器，延迟最多降低 61%，面积最多降低 56%。相比手工优化加速器，性能相当，而设计和工作负载映射自动化程度更高。

## 1 引言

DNN 广泛成功，但需要大量计算、存储资源。虽有许多加速器，高效硬件设计仍是工程密集过程；模型快速演化，使为每种新模型开发定制加速器越来越不现实。

已有生成器 [4,13,15] 自动化了部分设计，却常有四项不足：（1）缺少全面设计空间探索；（2）不能生成可流片系统；（3）缺少微缩放等量化支持；（4）缺少把高层 DNN 映射到生成硬件的稳健编译器。

Voyager 是端到端 HLS 框架。如图 1，给定数据类型、缩放因子类型/粒度、计算并行度和缓冲大小等架构参数，框架生成高性能、可流片 RTL，并提供协同设计编译器，将 PyTorch 模型高效映射到硬件。

主要贡献如下：

- 端到端 HLS 设计空间探索及神经网络加速器生成框架，针对模型集合、工艺和性能约束优化设计。支持不同 PE 数、片上缓冲和片外带宽，生成高度可配置数据流，包括可编程循环分块、排序、展开与融合。

![](images/fc061dfdcf10cc94a2a1f41cf5569805dbe2178a0a7116058440a6d56960ce71.jpg)

图 1：Voyager 端到端生成、评估神经网络加速器。围绕模板构建 HLS 流程，根据架构参数生成目标时钟、工艺的 RTL。C++ 模型快速评估量化准确率，RTL 仿真与综合提供运行时间、能耗和面积估计。

- 支持比已有工作更广的类型和量化。除浮点、posit、整数及粗粒度缩放外，还生成混合精度、多精度、自定义类型，并支持这些类型的微缩放。可评估微缩放相对粗粒度方案的成本收益。不同于仅用伪量化评估准确率的微缩放研究，HLS 同时生成硬件和与硬件逐比特一致的快速量化推理结果，准确描述累加及非线性的数据类型。
- 编译器将 PyTorch 映射到任意生成实例，支持量化、硬件感知融合，并集成扩展版 Interstellar [19] 寻找最优调度，包括循环分块、排序、展开和融合。

我们生成加速器运行图像分类、语言理解及生成模型。微缩放整数推理准确率在 FP32 的 1% 以内，面积相比低精度浮点最多降低 23%。相同硬件配置下，相对 Gemmini、NVDLA 等生成器面积最多降低 56%、时间最多降低 61%；相对手工优化加速器，性能最多改善 48%，同时提供更强自动化。

## 2 相关工作


DNN 加速器设计空间探索已有许多架构建模工具。Interstellar [19]、Timeloop [11]、MAESTRO [8]、ZigZag [10] 生成映射并用解析成本模型评估。Interstellar 以粗粒度存储访问和计算能耗模型估能量，用启发式剪枝；Timeloop 类似，但支持穷举或随机采样。MAESTRO 聚焦固定数据流的调度搜索；ZigZag 增加架构模板灵活性，扩展有效映射空间。它们有助探索，但不生成硬件。

Tandem Processor [5]、DNNBuilder [20]、MAERI [9]、NVDLA [13]、MAGNet [15]、Gemmini [4] 进一步生成可综合硬件，架构、配置和软件支持各不相同。

Tandem 针对非 GEMM，与 GEMM 单元组合实现端到端模型执行。软件管理暂存器和定制 ISA 避免传统处理器地址计算、循环控制开销，编译器把 ONNX 降低到 ISA。

DNNBuilder 为 FPGA 生成流水架构，每级对应一网络层，因此设计固定于单模型；Voyager 等其他生成器则产生可运行多模型的较通用加速器。

MAERI 采用灵活互连：MAC 经树形网络连接，支持广泛数据流，可配置 MAC 数和 SRAM 大小，生成不同规模，但软件栈较弱。

表 1：与已有 DNN 加速器生成器比较。

| 特性 | Tandem [5] | Builder [20] | MAERI [9] | NVDLA [13] | MAGNet [15] | Gemmini [4] | Voyager |
|---|---|---|---|---|---|---|---|
| 数据类型 | 未说明 | 未说明 | 未说明 | 整数/浮点 | 整数 | 整数/浮点 | 整数/浮点/Posit/自定义 |
| 多精度 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 微缩放 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 抽象层 | RTL Verilog | RTL | RTL Verilog | RTL Verilog | HLS C++ | RTL Chisel | HLS C++ |
| 软件支持 | ONNX | 自定义 | 自定义 | Caffe | 自定义 | ONNX/C | PyTorch |
| SoC 集成 | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |

NVDLA 用卷积、激活、池化、归一化、重塑专用模块支持广泛算子，可配置类型、MAC 阵列、接口宽度和缓冲。各模块独立设定大小，选择包含或移除；软件将 Caffe 编译为硬件指令。

MAGNet 采用 HLS 模板：PE 阵列连接全局缓冲，各 PE 集成向量 MAC、本地缓冲和 ReLU、池化、缩放、量化后处理。可配置 PE 数、缓冲、向量通道数、向量宽度和整数精度，但缺少软件栈，需要人工映射各层。

Gemmini 用 Chisel 生成 PE、暂存及累加存储、池化和激活单元，支持脉动阵列及向量引擎；可配置阵列、缓冲和 im2col、池化、转置等可选模块，支持任意精度整数和浮点，软件采用 ONNX 模型规范。

相比这些工作，Voyager 以更高灵活性支持端到端神经网络，包括新兴类型及低于 8 位时越来越重要的微缩放。已有生成器的软件支持不足，需要人工把模型转成定制格式再映射；Voyager 的 PyTorch 编译器自动量化、融合、调度，以较少人工实现高效运行。

## 3 Voyager 加速器模板

图 2 的灵活模板包含卷积/GEMM 矩阵单元，以及处理非 GEMM 的可编程向量单元，两者各有定制指令集。

### 3.1 矩阵单元

矩阵单元为权重驻留脉动阵列，各 PE 执行 MAC。输入通道沿纵向展开，输出通道沿横向展开；输入激活从左到右流动，部分和从上到下传播。权重预加载到 PE，在一个块计算期间保持不动。

为保持利用率并避免加载权重停顿，每个 PE 配三重缓冲寄存器，以列向移位链互连。一块计算时，下一块权重可同步移入，实现块级流水。离开阵列的部分和可选地加偏置，再写入专用累加缓冲。

可编程地址生成器从主存取输入、权重块，存入独立双缓冲，重叠访问与计算，支持运行时可配置数据流。地址生成器精细控制 L1、L2 循环顺序与边界；输入、权重控制器还通过寄存器缓冲支持原位转置和排列，使转置与 GEMM 融合。


### 3.2 向量单元

$N$ 路多阶段向量单元执行 ReLU、GeLU 等非线性激活，残差相加等逐元素运算，softmax、层归一化等复杂函数，以及量化/反量化。它包括多阶段运算流水、求和/最大值归约单元、累加器和三个独立主存地址生成器。

为减少存储访问，矩阵输出可直接进入向量单元，实现矩阵与向量融合。但紧耦合时，向量单元若受存储访问限制并施加反压，会使矩阵停顿。Voyager 可生成双缓冲累加区：矩阵先写缓冲，向量独立读取，同时矩阵计算下一块。

![](images/439be48b3c302341e3e1a5440cee6f7f0756b8ae9ac45d2754650151473022f5.jpg)

图 2：矩阵单元含输入、权重读取器及缓冲、累加缓冲和脉动阵列；向量单元为逐元素与归约多级流水。

图 2 右侧每级执行逐元素运算或原样转发。先为层归一化、softmax 等常用核心操作构造数据流图，再合并为完整流水，尽量复用算术单元而不增加延迟。输出可送往存储、累加器或归约器；累加、归约结果又可反馈为后续操作数，或写回存储。

非线性用可配置多项式近似模块支持。不同于每函数一套查表或固定电路，采用单个可编程单元。编译时，各函数以七段二次自由结点样条近似，分段边界、系数编码到指令；运行时按输入选段求多项式，兼顾准确率与硬件效率。

图 3 以三次遍历映射 softmax。先流式读取张量到归约器，求归约维度最大值并写回；再读取原张量和最大值，逐元素减最大值、求指数、归约求归一化分母并写回；最后读取原张量、最大值、和，执行减最大值、指数、除以和，得到概率并写回。

![](images/9e1f65d3f46eacda48064b1d4e0e45e55ff4c73d31a7fb0d5548067e6d6e0647.jpg)

图 3：向量单元上的三遍 softmax 映射。

### 3.3 用户定义扩展

用户可添加矩阵、向量单元不能高效处理的专用计算块。深度卷积（DWC）比普通卷积数据复用低，脉动阵列效率较差。图 4 的专用 DWC 包括输入/权重读取器、输入行缓冲，以及可配置数量的 MAC 树，每树归约一个 3×3 滤波器。更大滤波器由编译器切片、填充为多个 3×3，最终在向量单元累加。硬件优化常见 3×3，其他尺寸由编译器支持。

这些扩展能无缝接入基线架构，扩大高效支持的工作负载。

### 3.4 SoC 集成

图 5 中 Voyager 通过 Chipyard [1] 集成到 SoC。Chipyard 提供 RISC-V 核、TileLink 互连生成器、存储及外设等完整基础设施。

生成设计作为存储映射 I/O 外设集成。主 CPU 经控制寄存器配置矩阵、向量操作；专用轻量 TileLink 端口连接系统总线，访问片上 L2 和片外 DRAM。因此不仅可独立评估，也兼容先进 SoC 流程和全系统评估。

![](images/f6a58d90997025878dbaddf36e3a4e8ab775d1afeb31587cd282afaec5d566ac.jpg)

图 4：用户扩展示例——可配置 MAC 树的深度卷积单元，输出直接进入向量单元。

![](images/d54c75bcfe80fa28bf191b8a751880327c383235b4485d482eec283c569ef2a0.jpg)

图 5：用 Chipyard 集成 Voyager SoC。

![](images/a6b3dc3c1b6b320627475a1c83cabfff03d342a4a2abc8581115f5d96b7fb25e.jpg)

图 6：DNN 加速器设计空间。

## 4 硬件生成器参数化

图 6 从资源分配、资源类型、调度三个轴暴露设计空间。各轴对应软硬件不同方面，参数影响性能、功耗、面积和准确率。为保持灵活性及效率，利用 C++ 元编程；HLS 只在编译时工作，因此使用能够编译时专用化的特性，尤其模板。

### 4.1 资源分配：计算/存储规模

脉动阵列维度可按计算需求配置，输入、权重、累加缓冲大小独立调整，主存带宽也可配置。它们直接影响性能、面积、功耗，按延迟、能量等约束选择。更大阵列降低延迟但增加面积功耗；更大片上缓冲提高复用、减少主存访问，却增加缓冲访问能耗与面积 [19]。设计探索需平衡这些取舍，以满足系统目标。


### 4.2 资源类型：数据类型与量化

用户可生成针对工作负载类别的类型、量化变体，例如标准 BFloat16、逐通道 INT8、微缩放 FP6。

代码清单 1：通用处理单元模块（保留原文示意代码，移除排版行号）。

~~~cpp
template<typename InputType, typename WeightType,
         typename PsumType>
SC_MODULE(ProcessingElement) {
  void compute(InputType input, WeightType weight,
               PsumType& psum) {
    psum = input.mac(weight, psum);
  }
};
SC_MODULE(SystolicArray) {
  ProcessingElement<int8, int8, int32> pe[16][16];
};
~~~

**数据类型参数化。** C++ 模板让类、函数以类型为参数；编译器替换类型并生成具体版本。清单 1 的 PE 对输入、权重、部分和三种类型参数化，执行三操作数 MAC；SystolicArray 以具体类型实例化 PE 阵列。

默认提供整数、浮点、posit [6] 三类，各按自身语义实现算术、逻辑，并按位宽、指数大小、有无符号等参数模板化（清单 2）。各类型实现 mac，支持无缝替换。

用户新增类型只需定义满足接口的类，包括 mac 及其他算术逻辑。该机制可支持向量化 PE（清单 3）和基于码本的 NormalFloat4 [3] 等特殊格式（清单 4）。

还支持运行时选择类型的多精度设计：敏感层用高精度，其他层降低精度提升效率。

代码清单 2：数据类型类的示意实现。

~~~cpp
template<int Width, bool Signed>
class Int {
  ac_int<Width, Signed> value;
  Int mac(Int weight, Int psum) {
    return value * weight.value + psum.value;
  }
};
template<int MantissaBits, int ExponentBits>
class FP {
  ac_std_float<MantissaBits, ExponentBits> value;
  FP mac(FP weight, FP psum) {
    return value.fma(weight, psum);
  }
};
template<int Width, int EsBits>
class Posit {
  ac_int<Width, false> bits;
  Posit mac(Posit weight, Posit psum) {
  }
};
~~~

代码清单 3：向量化 PE 示例。

~~~cpp
template<typename T, int Width>
class Vector {
  T data[Width];
  Vector mac(Vector weight, Vector psum) {
    for (int i = 0; i < Width; i++) {
      data[i].mac(weight[i], psum[i]);
    }
  }
};
~~~

代码清单 4：新类型示例。

~~~cpp
class NormalFloat4 {
  ac_int<4, false> bits;
  static const FP codebook[16];
  NormalFloat4 mac(NormalFloat4 weight, NormalFloat4 psum) {
    for (int i = 0; i < Width; i++) {
      codebook[bits].mac(codebook[weight.bits],
                         codebook[psum.bits]);
    }
  }
};
~~~

Voyager 使用可变参数模板与部分特化，让同一模块支持多类型。例如 InputController 从 L2 读取输入块，以支持类型元组实例化（清单 5），生成专用读取逻辑，运行时选择。编译器在指令中编码每层精度。

代码清单 5：单类型或多类型输入控制器。

~~~cpp
template<typename InputTypes>
struct InputController;
template<typename... InputTypes>
struct InputController<std::tuple<InputTypes...>> { ... };

// 以单类型实例化
InputController<std::tuple<FP8>> inputController;
// 或多类型
InputController<std::tuple<FP16, FP8, FP4>> inputController;
~~~

![](images/2f843e44881f8460020992f23c06709170b616384e2f7b65afd69f041c3a5d87.jpg)

图 7：在矩阵单元加入反量化逻辑，生成支持微缩放的加速器。

**量化方案参数化。** 支持逐张量、逐通道等准确率—效率取舍。向量输出有量化阶段，读取器有反量化硬件，可与前一操作融合。

微缩放 [12] 沿归约维度分块缩放，改善低精度 GEMM 准确率，需要高效 GEMM 和逐输出块缩放因子生成的专用硬件。图 7 在矩阵单元加入反量化：输入、权重存低精度整数，如 INT8，各配 E8M0 scale；阵列以较高精度整数，如 INT32 累加，再用输入、权重 scale 乘积将部分和反量化到 BF16 等浮点，和之前部分和相加后存累加缓冲。

~~~cpp
// 对 HLS 而言不理想的编码方式
for (int ic = 0; ic < IC_loop_bound; ic++) {
  for (int oy = 0; oy < OY_loop_bound; oy++) {
    for (int ox = 0; ox < OX_loop_bound; ox++) {
      int address = oy * OX_loop_bound * IC_loop_bound
                  + ox * IC_loop_bound + ic;
    }
  }
}
// 改进编码：预计算常量并用 break 终止
int IC_last = IC_loop_bound - 1;
int OY_last = OY_loop_bound - 1;
int OX_last = OX_loop_bound - 1;
int OY_stride = OX_loop_bound * IC_loop_bound;
for (int ic = 0;; ic++) {
  for (int oy = 0;; oy++) {
    for (int ox = 0;; ox++) {
      int address = oy * OY_stride + ox * IC_loop_bound + ic;
      if (ox == OX_last) { break; }
    }
    if (oy == OY_last) { break; }
  }
  if (ic == IC_last) { break; }
}
~~~

图 8：编码方式显著影响 HLS RTL。上方标准嵌套循环改写为下方预计算及 break 版本。


### 4.3 调度：循环变换

前两轴决定生成什么硬件，调度轴决定如何映射计算。生成设计具有运行时可配置循环计数器、地址生成器，控制数据搬移和访问，支持广泛循环顺序及分块。调度影响复用，继而影响能效、延迟。第 5.5 节介绍选择和降低策略。

![](images/00121b64d04859384076922d1089ee939c72b5ddadfc7b31a929bc85df0dab93.jpg)

图 9：编译器经量化、硬件相关融合/分块，把 PyTorch 降低为矩阵、向量 ISA 指令。

### 4.4 HLS C++ 编码优化

编码方式显著影响 HLS RTL 面积、时序及工具时间。地址生成器用嵌套 for 循环，循环体计算地址。图 8 的两种优化为：（1）预计算常量，把边界、步距等移出循环，缩短流水依赖链、改善时序，也减少调度分析的算术操作；（2）展平循环控制，改为无限循环及显式 break，避免 HLS 在头部判断边界时引入的最后额外迭代。多层循环下，内层加速影响显著。

## 5 Voyager 编译器

编译器与模板协同设计，把高层模型变为专用 ISA 的优化指令。图 9 从 PyTorch 经量化、硬件感知优化、调度降低。以下用卷积后接 ReLU 的简单模型展示流程。

![](images/cc99be2aa9fa4cc00267ff62585f12d640c2577bd252b16a297a5c855c17d7c7.jpg)

图 10：前端用 PyTorch 2 Export 提取静态图。上方原模型降低为下方 ATen 中间表示，供后续编译阶段使用。

### 5.1 前端

前端把 PyTorch 转成连接高层模型与 ISA 的 IR，用 PT2E [2] 提取静态计算图。选择 PyTorch 是因其广泛应用和 torchvision [14]、Hugging Face Transformers [18] 生态；现有模型只需少量修改即可导入。

### 5.2 量化

多数 ML 框架原生支持 PTQ、QAT，但类型、方案有限，缺少新量化技术。因此加入自定义框架，精细控制权重、激活、偏置量化。

![](images/00052443726f95c7265a66674f581880fceed90252331dff244f4cbab1ea4dc0.jpg)

图 11：用户指定量化配置（上），工具插入量化算子得到量化感知图（下），在浮点模型中模拟量化执行。

量化器为节点标注量化意图并插入伪量化。

**数据类型。** 支持任意位宽整数、浮点、posit、NormalFloat；用户可定义把 BFloat16 映射到自定义类型的伪量化函数。

**量化方案。** PyTorch 的逐张量、逐通道缩放常用于 INT8、FP8，但低于 8 位时难保准确率。微缩放又称块量化，在低位宽下明显改善准确率；块越小通常越准，但每参数位数和计算开销增加。据作者所知，这是首个支持完全可配置块大小及类型微缩放的编译器。

**量化粒度。** 依据硬件约束和性能目标逐层指定类型与方案，对混合精度尤为重要：不敏感层可降精度，减小模型。


### 5.3 融合

支持量化、运算、重塑三类融合。

**量化融合。** 量化后把伪量化替换成硬件可执行 quantize/dequantize。quantize 缩放浮点张量再转目标类型，dequantize 通过相应 scale 除法还原浮点；scale 来自样本校准统计。支持更高硬件效率的二次幂 scale，以及更高准确率的一般浮点 scale。直接量化（如 scale=1）删除冗余反量化。为整数 GEMM，将反量化延迟至 GEMM 后，提高能效。

![](images/fa1198bdee566e46d8ffc747f8a1fa9e9195f6287a5c1458ff30168f7b793710.jpg)

图 12：自注意力的 reshape 与 matmul 融合。

![](images/f6d9c9e5639e7263c4e392d0b75024d3de5e6fc5f963a2306d53394541528edd.jpg)

图 13：量化、运算节点融合；卷积—反量化—ReLU 合并为单个运算。

**运算融合。** 将多个小操作合成一个计算核，避免中间写回，提高利用率。支持矩阵与后续逐元素向量操作融合，及多个向量操作融合。根据向量流水结构，识别硬件支持的顺序模式并替换为融合指令，图 13 为例。

**重塑融合。** 把 reshape、dequantize 与使用者融合。地址生成器在访问时直接转置、排列轴，消除中间张量、降低延迟与存储。Transformer 多头注意力中，Q/K/V 投影排列以分离 head 维，K 转置，输出再排列组合；图 12 融合线性投影、排列和转置，减少访问与延迟。

### 5.4 加速器 IR 生成

图 14 的硬件 IR 同时表示高层操作及类型、形状、地址等低层细节。Tensor 类似 PyTorch 张量，但加入形状、类型、地址及可选 permutation 字段，支持隐式转置。

PyTorch 操作分为：（1）卷积/GEMM 矩阵；（2）ReLU、GeLU、残差相加等逐元素；（3）LayerNorm、softmax 等复杂归约；（4）permute、view、transpose 等重塑。节点映射为第 1、3、4 类 IR；融合节点的第 2 类以同块后处理列表表示，例如卷积、残差、ReLU 表为带融合逐元素变换的矩阵操作。

### 5.5 调度

循环分块、顺序、融合形成很大调度空间，各方案在存储层次产生不同复用；加强一种复用会减少另一种机会。扩展 Interstellar [19] 为生成硬件高效调度。

~~~text
name: "conv2d_fused"
op_list: [{
  name: "conv2d"
  target: "conv2d"
  input:
    node: "quantize_symmetric"
    shape: [128,128,32]
    dtype: "int8"
    address: 18688
  weight:
    node: "conv2d_weight"
    shape: [3,3,32,64]
    dtype: "int8"
    address: 0
  bias:
    node: "conv2d_bias"
    shape: [64]
    dtype: "int32"
    address: 18432
}, {
  name: "dequantize_default"
  target: "dequantize"
  dtype: "int24"
  input:
    node: "conv2d"
    shape: [128,128,64]
    dtype: "int32"
  scale:
    node: "conv2d_scale"
    shape: [1]
    dtype: "bfloat16"
}, {
  name: "relu"
  target: "relu"
  input:
    node: "dequantize_default"
    shape: [128,128,64]
    dtype: "bfloat16"
}]
output:
  node: "conv2d_fused"
  shape: [128,128,64]
  dtype: "bfloat16"
  address: 542976
~~~

图 14：IR 包含操作、张量元数据及地址。

~~~python
# 原文循环调度伪代码
# L2 层
for oy1 in range(16):
  for ox1 in range(4):
    for oc2 in range(2):
      # 累加缓冲层
      for ic1 in range(2):
        for oc1 in range(2):
          for fx in range(3):
            for fy in range(3):
              for oy0 in range(8):
                for ox0 in range(32):
                  # 脉动阵列层
                  parallel_for ic0 in range(16):
                    parallel_for oc0 in range(16):
                      # 计算形状
                      input_height = 16*8
                      input_width = 4*32
                      # 计算索引
                      oy = oy1*8 + oy0
                      ox = ox1*32 + ox0
                      oc = oc2*2*16 + oc1*16 + oc0
                      ic = ic1*16 + ic0
                      # 调整填充
                      in_y = oy - 1
                      in_x = ox - 1
                      # 加偏置
                      if ic == 0 and fx == 0 and fy == 0:
                        output[oy,ox,oc] = bias[oc]
                      # 执行 MAC
                      if (0 <= in_y < input_height) and (0 <= in_x < input_width):
                        output[oy,ox,oc] += input[in_y,in_x,ic] * weight[fy,fx,ic,oc]


                      # 应用 ReLU
                      if ic == 31 and fy == 2 and fx == 2:
                        output[oy,ox,oc] = max(output[oy,ox,oc],0)
~~~

图 15：扩展 Interstellar 生成的循环调度示例。

Interstellar 的粗粒度解析模型统计各级访问数，乘硬件查表的每访问能量，用启发式高效搜索。我们：（1）把空间限制为 Voyager 硬件可执行调度；（2）增加计入硬件特征的运行时间模型。

性能分解为两部分：（1）L1 块计算时间，取决于块大小和阵列计算单元数。权重入阵列每行一周期，PE 内缓冲使下一块加载与当前计算重叠。因此每块时间取计算时间与权重加载时间的最大值，再乘脉动阵列块数；（2）L1 存储访问时间，输入、权重进入 L1 的耗时由带宽决定。

总时间取 L1 计算、访问最大值，再乘 L2 块数。没有双缓冲累加区时，还将向量计算时间加入最大值，计入反压；更深层次逐级重复。这样快速准确地预测硬件运行时间。图 15 为多层卷积循环示例。

### 5.6 加速器指令生成

依据 IR 和 Interstellar 调度生成硬件指令。矩阵指令编码边界、顺序、步距，循环直接对应地址生成器配置；向量指令还配置各级功能单元、累加、归约、量化/反量化，从而映射多样及融合操作。

图 16 中 base address 为主存张量地址；loops 编码两级循环边界；各 *_loop_index 把维度对应到 loops 索引。

~~~cpp
struct MatrixUnitInst {
  uint64_t input_base_address = 18688;
  uint64_t weight_base_address = 0;
  uint64_t bias_base_address = 18432;
  uint16_t loops[2][6] = {
    {16,4,2,1},       // L2 层
    {2,2,3,3,8,32}    // 累加缓冲层
  };
  uint3_t x_loop_index[2] = {1,5};
  uint3_t y_loop_index[2] = {0,4};
  uint3_t reduction_loop_index[2] = {3,0};
  uint3_t output_channel_loop_index[2] = {2,1};
  uint3_t fx_index = 2;
  uint3_t fy_index = 3;
};
~~~

图 16：矩阵指令子集，配置地址、边界及循环顺序。

表 2：融合、精细性能建模及双缓冲改善运行时间。

| 比较组 | 方案 | 时间 | 原文改善比例 |
|---|---|---:|---:|
| 融合 | 基线 | 2.67 ms | — |
| 融合 | 加排列与转置融合 | 2.65 ms | 0.75% |
| 融合 | 再加残差 | 2.61 ms | 2.30% |
| 融合 | 再加 NoNorm | 2.37 ms | 12.66% |
| 调度模型 | 仅能量 | 2.44 ms | — |
| 调度模型 | 粗粒度性能 | 2.44 ms | — |
| 调度模型 | 精细性能 | 2.37 ms | 2.86% |
| 缓冲 | 无双缓冲 | 2.37 ms | — |
| 缓冲 | 双缓冲 | 2.29 ms | 3.37% |

## 6 结果

评估跨应用 DNN、多设计点。Catapult HLS 生成 Verilog，Synopsys VCS 做周期精确仿真，Design Compiler 逻辑综合。MAC 利用率为理想周期/实际周期；理想周期等于层 MAC 数/硬件 MAC 单元数。6.1 分析优化，6.2 探索空间，6.3、6.4 比较生成器及手工设计。

### 6.1 Voyager 内部优化

采用 TSMC 16 nm、32×32 E4M3，在 MobileBERT-TINY 上评估。

**编码优化。** 未优化最高400MHz、面积1.16mm²；优化后最高1GHz、面积1.41mm²。同为400MHz时优化面积1.18mm²，仅增1.7%。

**算子融合。** 表2显示逐步融合减少访问，最终时间改善12.66%。

**调度。** 精细性能模型相对仅能量/粗粒度策略，时间最多降低约2.9%，说明准确模型有益。

**双缓冲。** 矩阵和向量紧耦合时，向量存储瓶颈产生反压；可选累加双缓冲解耦，时间降低约3%（表2）。


### 6.2 使用 Voyager 探索设计空间

展示在不同 DNN 上探索：（1）类型、量化（资源类型）；（2）加速器大小（资源分配）。均采用编译器最优调度。

**类型与量化。** 逐比特准确 C++ 模型可在完整数据集推理，评估不同 DNN、格式和量化的硬件级准确率。表3涵盖FP32、BF16、E4M3、Posit8、INT8、MXINT8，预训练模型来自torchvision [14]、timm [17]、transformers [18]。微缩放整数准确率可与浮点相当，并支持跨模型评估，但可能增加局部面积。表5为16nm、1GHz面积；32×32的MXINT8因输入/权重scale使地址控制比INT8多43%，但较小累加缓冲、向量单元使总面积仍低5.64%，较小脉动阵列使其比E4M3低11.6%。

表3：不同类型的视觉/NLP准确率。视觉任务为ImageNet分类；NLP为SST-2情感分类，序列长度128。

| 类别 | 模型 | 参数量 | FP32 | BF16 | E4M3 | Posit8 | INT8 | MXINT8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 视觉 | ResNet-18 | 11.7M | 71.5% | 70.0% | 69.2% | 69.1% | 71.5% | 70.7% |
| 视觉 | ResNet-50 | 25.6M | 80.4% | 81.1% | 78.8% | 79.4% | 78.7% | 79.8% |
| 视觉 | ViT Base | 86M | 84.1% | 84.3% | 83.8% | 84.0% | 75.4% | 84.0% |
| 视觉 | MobileNetV2 | 3.5M | 71.1% | 70.1% | 62.5% | 63.8% | 66.0% | 71.3% |
| NLP | MobileBERT-TINY | 15M | 90.8% | 90.7% | 90.6% | 90.4% | 90.4% | 91.1% |
| NLP | BERT Base | 110M | 93.2% | 93.0% | 93.1% | 92.8% | 92.4% | 93.1% |

表4：不同规模E4M3设计的时间与利用率。M为百万周期，K为千周期。

| 模型/指标 | 8×8，48KB | 8×16，80KB | 16×16，96KB | 16×32，160KB | 32×32，192KB | 32×64，320KB | 64×64，384KB |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 周期 | 31.76M | 15.88M | 7.90M | 3.96M | 2.07M | 1.04M | 660K |
| ResNet-18 MAC利用率% | 90.9 | 90.9 | 93.0 | 92.9 | 91.9 | 91.4 | 76.8 |
| ResNet-50 周期 | 67.70M | 33.86M | 17.00M | 8.51M | 4.40M | 2.21M | 1.30M |
| ResNet-50 MAC利用率% | 95.4 | 95.4 | 96.2 | 96.1 | 94.9 | 94.5 | 82.1 |
| ViT Base 周期 | 326.19M | 163.14M | 84.37M | 42.24M | 22.74M | 12.44M | 7.53M |
| ViT Base MAC利用率% | 99.4 | 99.4 | 99.3 | 99.2 | 98.0 | 96.5 | 88.0 |
| MobileNetV2 周期 | 7.373M | 3.760M | 2.010M | 1.099M | 659K | 534K | 407K |
| MobileNetV2 MAC利用率% | 86.7 | 84.9 | 83.7 | 79.3 | 78.0 | 58.6 | 52.5 |
| MobileBERT-TINY 周期，序列128 | 27.91M | 14.00M | 7.71M | 3.91M | 2.37M | 1.28M | 916K |
| MobileBERT-TINY MAC利用率% | 98.3 | 98.0 | 95.1 | 93.8 | 86.2 | 82.8 | 67.3 |
| BERT Base 周期，序列128 | 179.17M | 89.63M | 46.07M | 23.09M | 12.24M | 6.48M | 3.80M |
| BERT Base MAC利用率% | 99.8 | 99.8 | 99.4 | 99.2 | 97.9 | 96.2 | 90.7 |
| LLaMA3.2 1B预填充周期，序列512 | — | — | — | — | 688.0M | — | — |
| LLaMA MAC利用率% | — | — | — | — | 99.8 | — | — |

**加速器大小。** 表4显示随阵列规模扩展，运行时间近乎线性降低，硬件利用率仍高。

### 6.3 与已有生成器比较

表6比较Gemmini、NVDLA；匹配已有工艺、类型、MAC数、频率。Gemmini面积低28%，但Voyager运行时间低61%；相对NVDLA，面积低35.91%–56.48%，时间低20.84%–53.15%。


### 6.4 与手工优化设计比较

比较CNN加速器Simba [16,21,22] 与向量缩放量化（VSQ）Transformer加速器 [7]，尽量匹配参数。手工设计报告布局后结果，因此把Voyager面积乘1.25以计入布局布线开销。

Simba比较同为TSMC16nm，MAC、类型、缓冲相匹配（表7）。Simba频率更高，但电压远高于我们表征的库，难直接比较时间。按周期，Voyager的ResNet-50为440万，对Simba830万；利用率95%对48%。即使频率低，绝对时间仍好约5%。

VSQ比较中因无5nm访问条件，在7nm生成相同MAC数设计。较旧工艺仍可同频运行；BERT为13.4ms，对VSQ11.4ms。但VSQ不加速全部模型，缺少重塑、转置、残差相加、LayerNorm，而Voyager包含它们。两者MAC利用率均98%。这些比较说明自动生成设计可匹配或超越手工设计性能、利用率，同时增强自动化。

表5：16nm、目标1GHz的综合后面积（mm²）。

| 阵列 | 类型 | 脉动阵列 | SRAM | 地址/控制 | 向量单元 | 总计 |
|---|---|---:|---:|---:|---:|---:|
| 8×8 | E4M3 | 0.048 | 0.085 | 0.062 | 0.142 | 0.336 |
| 8×8 | Posit8 | 0.052 | 0.085 | 0.064 | 0.145 | 0.346 |
| 8×8 | INT8 | 0.027 | 0.111 | 0.064 | 0.148 | 0.350 |
| 8×8 | MXINT8 | 0.031 | 0.095 | 0.095 | 0.135 | 0.355 |
| 16×16 | E4M3 | 0.192 | 0.109 | 0.096 | 0.237 | 0.633 |
| 16×16 | Posit8 | 0.210 | 0.109 | 0.101 | 0.242 | 0.661 |
| 16×16 | INT8 | 0.109 | 0.211 | 0.098 | 0.250 | 0.667 |
| 16×16 | MXINT8 | 0.123 | 0.123 | 0.161 | 0.224 | 0.632 |
| 32×32 | E4M3 | 0.769 | 0.217 | 0.199 | 0.442 | 1.627 |
| 32×32 | Posit8 | 0.846 | 0.217 | 0.215 | 0.450 | 1.728 |
| 32×32 | INT8 | 0.440 | 0.422 | 0.198 | 0.465 | 1.525 |
| 32×32 | MXINT8 | 0.501 | 0.243 | 0.283 | 0.412 | 1.438 |
| 64×64 | E4M3 | 3.104 | 0.434 | 0.453 | 0.825 | 4.817 |
| 64×64 | Posit8 | 3.399 | 0.434 | 0.506 | 0.836 | 5.175 |
| 64×64 | INT8 | 1.783 | 0.844 | 0.432 | 0.869 | 3.928 |
| 64×64 | MXINT8 | 2.021 | 0.484 | 0.565 | 0.752 | 3.821 |

表6：Intel16上的Gemmini与TSMC16上的NVDLA比较。全部INT8、1GHz；面积为标准单元面积，Gemmini比较仅计算阵列，NVDLA比较完整设计。

| 基线 | MAC数 | 基线面积（µm²） | Voyager面积（µm²） | 基线ResNet-50时间（ms） | Voyager时间（ms） |
|---|---:|---:|---:|---:|---:|
| Gemmini | 256 | 116,000 | 160,314 | 43.58 | 17.00 |
| NVDLA | 2048 | 3,300,000 | 2,114,852 | 3.72 | 2.21 |
| NVDLA | 1024 | 1,800,000 | 1,102,659 | 6.54 | 4.40 |
| NVDLA | 512 | 1,400,000 | 852,504 | 10.75 | 8.51 |
| NVDLA | 256 | 1,000,000 | 456,468 | 21.74 | 17.00 |
| NVDLA | 128 | 840,000 | 394,511 | 50.00 | 33.86 |
| NVDLA | 64 | 550,000 | 239,519 | 136.99 | 67.70 |
| NVDLA | 32 | 510,000 | 221,943 | 277.78 | 130.13 |

## 7 结论

Voyager 是端到端DNN加速器生成框架，HLS生成器广泛配置类型及量化，PyTorch编译器支持量化、融合、分块，高效映射模型。两者支持快速设计探索，并在全数据集运行量化模型，精确估计准确率。生成设计跨模型、规模保持高利用率及性能；相对已有生成器，同配置面积最多降低56%、时间降低61%；相对手工加速器，时间最多降低48%，人工更少。这证明高性能DNN硬件不必依赖传统的大量手工设计及映射。

表7：与Simba及4位VSQ比较。

| 指标 | Simba | Voyager | VSQ | Voyager |
|---|---:|---:|---:|---:|
| MAC数 | 1024 | 1024 | 1024 | 1024 |
| 类型 | INT8 | INT8 | INT4 | INT4 |
| 工艺 | 16nm | 16nm | 5nm | 7nm |
| 频率GHz | 1.797 | 1.0 | 0.909 | 0.909 |
| 电压V | 1.1 | 0.8 | 0.67 | 0.75 |
| 面积mm² | 3.1 | 1.8 | 0.153 | 0.57 |
| ResNet-50周期 | 8.3M | 4.4M | 不适用 | 不适用 |
| ResNet-50时间ms | 4.64 | 4.40 | 不适用 | 不适用 |
| BERT周期 | 不适用 | 不适用 | 10.4M | 12.2M |
| BERT时间ms | 不适用 | 不适用 | 11.4 | 13.4 |
| MAC利用率 | 48% | 95% | 98% | 98% |

## 8 致谢

研究获得DARPA青年教师奖、COCOSYS SRC JUMP2.0中心、Amazon、Apple、Siemens及PORTAL合作计划资助。


## References

[1] Alon Amid, David Biancolin, Abraham Gonzalez, Daniel Grubb, Sagar Karandikar, Harrison Liew, Albert Magyar, Howard Mao, Albert Ou, Nathan Pemberton, Paul Rigge, Colin Schmidt, John Wright, Jerry Zhao, Yakun Sophia Shao, Krste Asanović, and Borivoje Nikolić. Chip yard: Integrated design, simulation, and implementation framework for custom SoCs. IEEE Micro, 40(4):10–21, July 2020.

[2] Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai Lu, C. K. Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark Saroufim, Marcos Yukio Siraichi, Helen Suk, Shunting Zhang, Michael Suo, Phil Tillet, Xu Zhao, Eikan Wang, Keren Zhou, Richard Zou, Xiaodong Wang, Ajit Mathews, William Wen, Gregory Chanan, Peng Wu, and Soumith Chintala. PyTorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages

and Operating Systems, Volume 2, ASPLOS ’24, page 929–947, New York, NY, USA, 2024. Association for Computing Machinery.

[3] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Eficient finetuning of quantized LLMs. Advances in Neural Information Processing Systems, 36:10088–10115, 2023.

[4] Hasan Genc, Seah Kim, Alon Amid, Ameer Haj-Ali, Vighnesh Iyer, Pranav Prakash, Jerry Zhao, Daniel Grubb, Harrison Liew, Howard Mao, Albert Ou, Colin Schmidt, Samuel Stefl, John Wright, Ion Sto ica, Jonathan Ragan-Kelley, Krste Asanovic, Borivoje Nikolic, and Yakun Sophia Shao. Gemmini: Enabling systematic deep-learning architecture evaluation via full-stack integration. In Proceedings ofthe 58th Annual Design Automation Conference (DAC), 2021.

[5] Soroush Ghodrati, Sean Kinzer, Hanyang Xu, Rohan Mahapatra, Yoonsung Kim, Byung Hoon Ahn, Dong Kai Wang, Lavanya Karthikeyan, Amir Yazdanbakhsh, Jongse Park, Nam Sung Kim, and Hadi Es maeilzadeh. Tandem processor: Grappling with emerging operators in neural networks. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS ’24, page 1165–1182, New York, NY, USA, 2024. Association for Computing Machinery.

[6] John L Gustafson and Isaac T Yonemoto. Beating floating point at its own game: Posit arithmetic. Supercomputing Frontiers and Innovations, 4(2):71–86, June 2017.

[7] Ben Keller, Rangharajan Venkatesan, Steve Dai, Stephen G. Tell, Brian Zimmer, Charbel Sakr, William J. Dally, C. Thomas Gray, and Brucek Khailany. A 95.6-TOPS/W deep learning inference accelerator with per-vector scaled 4-bit quantization in 5 nm. IEEE Journal ofSolid-State Circuits, 58(4):1129–1141, 2023.

[8] Hyoukjun Kwon, Prasanth Chatarasi, Vivek Sarkar, Tushar Krishna, Michael Pellauer, and Angshuman Parashar. MAESTRO: A datacentric approach to understand reuse, performance, and hardware cost of DNN mappings. IEEE Micro, 40(3):20–29, 2020.

[9] Hyoukjun Kwon, Ananda Samajdar, and Tushar Krishna. MAERI: Enabling flexible dataflow mapping over DNN accelerators via reconfigurable interconnects. SIGPLAN Not., 53(2):461–475, March 2018.

[10] Linyan Mei, Pouya Houshmand, Vikram Jain, Sebastian Giraldo, and Marian Verhelst. ZigZag: Enlarging joint architecture-mapping design space exploration for DNN accelerators. IEEE Transactions on Computers, 70(8):1160–1174, 2021.

[11] Angshuman Parashar, Priyanka Raina, Yakun Sophia Shao, Yu-Hsin Chen, Victor A. Ying, Anurag Mukkara, Rangharajan Venkatesan, Brucek Khailany, Stephen W. Keckler, and Joel Emer. Timeloop: A systematic approach to DNN accelerator evaluation. In 2019 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), pages 304–315, 2019.

[12] Bita Darvish Rouhani, Ritchie Zhao, Ankit More, Mathew Hall, Alireza Khodamoradi, Summer Deng, Dhruv Choudhary, Marius Cornea, Eric Dellinger, Kristof Denolf, Stosic Dusan, Venmugil Elango, Maximil ian Golub, Alexander Heinecke, Phil James-Roxby, Dharmesh Jani, Gaurav Kolhe, Martin Langhammer, Ada Li, Levi Melnick, Maral Mesmakhosroshahi, Andres Rodriguez, Michael Schulte, Rasoul Shafipour, Lei Shao, Michael Siu, Pradeep Dubey, Paulius Micikevicius, Maxim Naumov, Colin Verrilli, Ralph Wittig, Doug Burger, and Eric Chung. Microscaling data formats for deep learning, 2023.

[13] Frans Sijstermans. The NVIDIA deep learning accelerator. In 2018 IEEE Hot Chips 30 Symposium (HCS), Cupertino, CA, USA, August 19-21, 2018. IEEE, 2018.

[14] TorchVision maintainers and contributors. TorchVision: PyTorch’s computer vision library. htps://github.com/pytorch/vision, 2016.

[15] Rangharajan Venkatesan, Yakun Sophia Shao, Miaorong Wang, Jason Clemons, Steve Dai, Matthew Fojtik, Ben Keller, Alicia Klinefelter, Nathaniel Pinckney, Priyanka Raina, Yanqing Zhang, Brian Zimmer, William J. Dally, Joel Emer, Stephen W. Keckler, and Brucek Khailany. MAGNet: A modular accelerator generator for neural networks. In

2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), pages 1–8, 2019.

[16] Rangharajan Venkatesan, Yakun Sophia Shao, Brian Zimmer, Jason Clemons, Matthew Fojtik, Nan Jiang, Ben Keller, Alicia Klinefelter, Nathaniel Ross Pinckney, Priyanka Raina, Stephen G. Tell, Yanqing Zhang, William J. Dally, Joel S. Emer, C. Thomas Gray, Stephen W. Keckler, and Brucek Khailany. A 0.11 pJ/op, 0.32-128 TOPS, scalable multi-chip-module-based deep neural network accelerator designed with a high-productivity VLSI methodology. In 2019 IEEE Hot Chips 31 Symposium (HCS), Cupertino, CA, USA, August 18-20, 2019, pages 1–24. IEEE, 2019.

[17] Ross Wightman. PyTorch image models. htps://github.com/ huggingface/pytorch-image-models, 2019.

[18] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, October 2020. Association for Computational Linguistics.

[19] Xuan Yang, Mingyu Gao, Qiaoyi Liu, Jef Setter, Jing Pu, Ankita Nayak, Steven Bell, Kaidi Cao, Heonjae Ha, Priyanka Raina, Christos Kozyrakis, and Mark Horowitz. Interstellar: Using Halide’s scheduling language to analyze DNN accelerators. In Proceedings of the Twenty-Fifth International Conference on Architectural SupportforProgramming Languages and Operating Systems, ASPLOS ’20, page 369–383, New York, NY, USA, 2020. Association for Computing Machinery.

[20] Xiaofan Zhang, Junsong Wang, Chao Zhu, Yonghua Lin, Jinjun Xiong, Wen-mei Hwu, and Deming Chen. DNNBuilder: An automated tool for building high-performance DNN hardware accelerators for FPGAs. In 2018 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), pages 1–8, 2018.

[21] Brian Zimmer, Rangharajan Venkatesan, Yakun Sophia Shao, Jason Clemons, Matthew Fojtik, Nan Jiang, Ben Keller, Alicia Klinefelter, Nathaniel Pinckney, Priyanka Raina, Stephen G. Tell, Yanqing Zhang, William J. Dally, Joel S. Emer, C. Thomas Gray, Stephen W. Keckler, and Brucek Khailany. A 0.11 pJ/op, 0.32-128 TOPS, scalable multi-chip module-based deep neural network accelerator with ground-reference signaling in 16nm. In 2019 Symposium on VLSI Circuits, pages C300– C301, 2019.

[22] Brian Zimmer, Rangharajan Venkatesan, Yakun Sophia Shao, Jason Clemons, Matthew Fojtik, Nan Jiang, Ben Keller, Alicia Klinefelter, Nathaniel Pinckney, Priyanka Raina, Stephen G. Tell, Yanqing Zhang, William J. Dally, Joel S. Emer, C. Thomas Gray, Stephen W. Keckler, and Brucek Khailany. A 0.32–128 TOPS, scalable multi-chip-module-based deep neural network inference accelerator with ground-referenced signaling in 16 nm. IEEE Journal of Solid-State Circuits, 55(4):920–932, 2020.



## 附录 A：完整运行时间结果

表8、9、10、11、12给出所报告各模型在16×16、32×32设计上的逐层详细时间，以便与相关工作比较。总MAC利用率先分别汇总矩阵操作的理想周期和实际周期，再用理想总周期除以实际总周期。


表 8：ResNet-18 逐层性能与 MAC 利用率。
<table><tr><td rowspan=2 colspan=1>层</td><td rowspan=2 colspan=1>出现次数</td><td rowspan=2 colspan=1>执行单元</td><td rowspan=1 colspan=3>16×16</td><td rowspan=1 colspan=3>32×32</td></tr><tr><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td></tr><tr><td rowspan=1 colspan=1>conv1_fused224×224×3*7×7×3×64</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>460992</td><td rowspan=1 colspan=1>956615</td><td rowspan=1 colspan=1>48.19%</td><td rowspan=1 colspan=1>115248</td><td rowspan=1 colspan=1>246291</td><td rowspan=1 colspan=1>46.79%</td></tr><tr><td rowspan=1 colspan=1>max_pool2d_default112×112×64</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>223641</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>111859</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>quantize_default_156×56×64</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>25163</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>12618</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>layer1_0_conv1_fused56×56×64*3×3×64×64</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451850</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113366</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer1_0_conv2_fused56×56×64*3×3×64×64</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451850</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113366</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer2_0_conv1_fused56×56×64*3×3×64×128</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>225792</td><td rowspan=1 colspan=1>226378</td><td rowspan=1 colspan=1>99.74%</td><td rowspan=1 colspan=1>56448</td><td rowspan=1 colspan=1>57518</td><td rowspan=1 colspan=1>98.14%</td></tr><tr><td rowspan=1 colspan=1>layer2_0_conv2_fused28×28×128 *3×3×128×128</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451870</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113355</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer2_0_downsample_0_fused56×56×64*1×1×64×128</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>25088</td><td rowspan=1 colspan=1>25222</td><td rowspan=1 colspan=1>99.47%</td><td rowspan=1 colspan=1>6272</td><td rowspan=1 colspan=1>6491</td><td rowspan=1 colspan=1>96.62%</td></tr><tr><td rowspan=1 colspan=1>layer2_1_conv1_fused28×28×128*3×3×128×128</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451870</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113355</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer2_1_conv2_fused28×28×128*3×3×128×128</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451870</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113355</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer3_0_conv1_fused28×28×128*3×3×128×256</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>225792</td><td rowspan=1 colspan=1>226378</td><td rowspan=1 colspan=1>99.74%</td><td rowspan=1 colspan=1>56448</td><td rowspan=1 colspan=1>57518</td><td rowspan=1 colspan=1>98.14%</td></tr><tr><td rowspan=1 colspan=1>layer3_0_conv2_fused14×14×256 *3×3×256×256</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451835</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113355</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer3_0_downsample_0_fused28×28×128*1×1×128×256</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>25088</td><td rowspan=1 colspan=1>25222</td><td rowspan=1 colspan=1>99.47%</td><td rowspan=1 colspan=1>6272</td><td rowspan=1 colspan=1>6491</td><td rowspan=1 colspan=1>96.62%</td></tr><tr><td rowspan=1 colspan=1>layer3_1_conv1_fused14×14×256*3×3×256×256</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451835</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113355</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer3_1_conv2_fused14×14×256*3×3×256×256</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451835</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113355</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer4_0_conv1_fused14×14×256*3×3×256×512</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>225792</td><td rowspan=1 colspan=1>226154</td><td rowspan=1 colspan=1>99.84%</td><td rowspan=1 colspan=1>56448</td><td rowspan=1 colspan=1>56907</td><td rowspan=1 colspan=1>99.19%</td></tr><tr><td rowspan=1 colspan=1>layer4_0_conv2_fused7×7×512*3×3×512×512</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451835</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113355</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer4_0_downsample_0_fused14×14×256 *1×1×256×512</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>25088</td><td rowspan=1 colspan=1>25243</td><td rowspan=1 colspan=1>99.39%</td><td rowspan=1 colspan=1>6272</td><td rowspan=1 colspan=1>6491</td><td rowspan=1 colspan=1>96.62%</td></tr><tr><td rowspan=1 colspan=1>layer4_1_conv1_fused7×7×512*3×3×512×512</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451835</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113809</td><td rowspan=1 colspan=1>99.19%</td></tr><tr><td rowspan=1 colspan=1>layer4_1_conv2_fused7×7×512*3×3×512×512</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>452597</td><td rowspan=1 colspan=1>99.78%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113809</td><td rowspan=1 colspan=1>99.19%</td></tr><tr><td rowspan=1 colspan=1>adaptive_avg_pool2d_default7×7×512</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3212</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1643</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>fc1×512*1000×512</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>66038</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>33581</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>总计</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7904048</td><td rowspan=1 colspan=1>93.0%</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2071975</td><td rowspan=1 colspan=1>91.9%</td></tr></table>

表 9：ResNet-50 逐层性能与 MAC 利用率。
<table><tr><td rowspan=2 colspan=1>层</td><td rowspan=2 colspan=1>出现次数</td><td rowspan=2 colspan=1>执行单元</td><td rowspan=1 colspan=3>16×16</td><td rowspan=1 colspan=3>32×32</td></tr><tr><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td></tr><tr><td rowspan=1 colspan=1>quantize_default_156×56×64</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>25163</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>12618</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>conv1_fused224×224×3 *7×7×3×64</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>460992</td><td rowspan=1 colspan=1>956615</td><td rowspan=1 colspan=1>48.19%</td><td rowspan=1 colspan=1>115248</td><td rowspan=1 colspan=1>246291</td><td rowspan=1 colspan=1>46.79%</td></tr><tr><td rowspan=1 colspan=1>max_pool2d_default112×112×64</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>223641</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>111859</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>layer1_0_conv1_fused56×56×64 *1×1×64×64</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50310</td><td rowspan=1 colspan=1>99.73%</td><td rowspan=1 colspan=1>12544</td><td rowspan=1 colspan=1>12763</td><td rowspan=1 colspan=1>98.28%</td></tr><tr><td rowspan=1 colspan=1>layer1_0_conv2_fused56×56×64*3×3×64×64</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451850</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113366</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer1_0_conv3_fused56×56×64*1×1×64×256</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>200838</td><td rowspan=1 colspan=1>99.93%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50395</td><td rowspan=1 colspan=1>99.56%</td></tr><tr><td rowspan=1 colspan=1>layer1_0_downsample_0_fused56×56×64*1×1×64×256</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>200838</td><td rowspan=1 colspan=1>99.93%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50395</td><td rowspan=1 colspan=1>99.56%</td></tr><tr><td rowspan=1 colspan=1>layer1_1_conv1_fused56×56×256 *1×1×256×64</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>200838</td><td rowspan=1 colspan=1>99.93%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50395</td><td rowspan=1 colspan=1>99.56%</td></tr><tr><td rowspan=1 colspan=1>layer1_1_conv2_fused56×56×64*3×3×64×64</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451870</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113355</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer1_1_conv3_fused56×56×64*1×1×64×256</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>200838</td><td rowspan=1 colspan=1>99.93%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50395</td><td rowspan=1 colspan=1>99.56%</td></tr><tr><td rowspan=1 colspan=1>layer2_0_conv1_fused56×56×256*1×1×256×128</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>401408</td><td rowspan=1 colspan=1>401542</td><td rowspan=1 colspan=1>99.97%</td><td rowspan=1 colspan=1>100352</td><td rowspan=1 colspan=1>100571</td><td rowspan=1 colspan=1>99.78%</td></tr><tr><td rowspan=1 colspan=1>layer2_0_conv2_fused28×28×128*3×3×128×128</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>452170</td><td rowspan=1 colspan=1>99.87%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113966</td><td rowspan=1 colspan=1>99.06%</td></tr><tr><td rowspan=1 colspan=1>layer2_0_conv3_fused28×28×128*1×1×128×512</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>200838</td><td rowspan=1 colspan=1>99.93%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50395</td><td rowspan=1 colspan=1>99.56%</td></tr><tr><td rowspan=1 colspan=1>layer2_0_downsample_0_fused56×56×256*1×1×256×512</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>401408</td><td rowspan=1 colspan=1>401542</td><td rowspan=1 colspan=1>99.97%</td><td rowspan=1 colspan=1>100352</td><td rowspan=1 colspan=1>100571</td><td rowspan=1 colspan=1>99.78%</td></tr><tr><td rowspan=1 colspan=1>layer2_1_conv1_fused28×28×512*1×1×512×128</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>200838</td><td rowspan=1 colspan=1>99.93%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50395</td><td rowspan=1 colspan=1>99.56%</td></tr><tr><td rowspan=1 colspan=1>layer2_1_conv2_fused28×28×128*3×3×128×128</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451870</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113355</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer2_1_conv3_fused28×28×128*1×1×128×512</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>200838</td><td rowspan=1 colspan=1>99.93%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50395</td><td rowspan=1 colspan=1>99.56%</td></tr><tr><td rowspan=1 colspan=1>layer3_0_conv1_fused28×28×512*1×1×512×256</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>401408</td><td rowspan=1 colspan=1>401542</td><td rowspan=1 colspan=1>99.97%</td><td rowspan=1 colspan=1>100352</td><td rowspan=1 colspan=1>100571</td><td rowspan=1 colspan=1>99.78%</td></tr><tr><td rowspan=1 colspan=1>layer3_0_conv2_fused14×14×256*3×3×256×256</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>452170</td><td rowspan=1 colspan=1>99.87%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113966</td><td rowspan=1 colspan=1>99.06%</td></tr><tr><td rowspan=1 colspan=1>layer3_0_conv3_fused14×14×256*1×1×256×1024</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>200838</td><td rowspan=1 colspan=1>99.93%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50395</td><td rowspan=1 colspan=1>99.56%</td></tr><tr><td rowspan=1 colspan=1>layer3_0_downsample_0_fused28×28×512*1×1×512×1024</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>401408</td><td rowspan=1 colspan=1>401542</td><td rowspan=1 colspan=1>99.97%</td><td rowspan=1 colspan=1>100352</td><td rowspan=1 colspan=1>100571</td><td rowspan=1 colspan=1>99.78%</td></tr><tr><td rowspan=1 colspan=1>layer3_1_conv1_fused14×14×1024*1×1×1024×256</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>200838</td><td rowspan=1 colspan=1>99.93%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50395</td><td rowspan=1 colspan=1>99.56%</td></tr><tr><td rowspan=1 colspan=1>layer3_1_conv2_fused14×14×256 *3×3×256×256</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451835</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113355</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=9>下页续表</td></tr></table>

表 9：ResNet-50 逐层性能与 MAC 利用率。
<table><tr><td rowspan=2 colspan=1>层</td><td rowspan=2 colspan=1>出现次数</td><td rowspan=2 colspan=1>执行单元</td><td rowspan=1 colspan=3>16×16</td><td rowspan=1 colspan=3>32×32</td></tr><tr><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td></tr><tr><td rowspan=1 colspan=1>layer3_1_conv3_fused14×14×256*1×1×256×1024</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>200838</td><td rowspan=1 colspan=1>99.93%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50395</td><td rowspan=1 colspan=1>99.56%</td></tr><tr><td rowspan=1 colspan=1>layer4_0_conv1_fused14×14×1024*1×1×1024×512</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>401408</td><td rowspan=1 colspan=1>401542</td><td rowspan=1 colspan=1>99.97%</td><td rowspan=1 colspan=1>100352</td><td rowspan=1 colspan=1>100571</td><td rowspan=1 colspan=1>99.78%</td></tr><tr><td rowspan=1 colspan=1>layer4_0_conv2_fused7×7×512 *3×3×512×512</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451946</td><td rowspan=1 colspan=1>99.92%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113355</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer4_0_conv3_fused7×7×512*1×1×512×2048</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>200859</td><td rowspan=1 colspan=1>99.92%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50395</td><td rowspan=1 colspan=1>99.56%</td></tr><tr><td rowspan=1 colspan=1>layer4_0_downsample_0_fused14×14×1024*1×1×1024×2048</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>401408</td><td rowspan=1 colspan=1>401563</td><td rowspan=1 colspan=1>99.96%</td><td rowspan=1 colspan=1>100352</td><td rowspan=1 colspan=1>100571</td><td rowspan=1 colspan=1>99.78%</td></tr><tr><td rowspan=1 colspan=1>layer4_1_conv1_fused7×7×2048*1×1×2048×512</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>200859</td><td rowspan=1 colspan=1>99.92%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50395</td><td rowspan=1 colspan=1>99.56%</td></tr><tr><td rowspan=1 colspan=1>layer4_1_conv2_fused7×7×512 *3×3×512×512</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>451835</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>113355</td><td rowspan=1 colspan=1>99.59%</td></tr><tr><td rowspan=1 colspan=1>layer4_1_conv3_fused7×7×512*1×1×512×2048</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>200859</td><td rowspan=1 colspan=1>99.92%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>50395</td><td rowspan=1 colspan=1>99.56%</td></tr><tr><td rowspan=1 colspan=1>layer4_2_conv3_fused7×7×512*1×1×512×2048</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>200704</td><td rowspan=1 colspan=1>205493</td><td rowspan=1 colspan=1>97.67%</td><td rowspan=1 colspan=1>50176</td><td rowspan=1 colspan=1>52145</td><td rowspan=1 colspan=1>96.22%</td></tr><tr><td rowspan=1 colspan=1>adaptive_avg_pool2d_default7×7×2048</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>12620</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>6347</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>fc1×2048 *1000×2048</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>259574</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>131885</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>总计</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>16896078</td><td rowspan=1 colspan=1>96.2%</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4378064</td><td rowspan=1 colspan=1>94.9%</td></tr></table>

表 10：ViT 逐层性能与 MAC 利用率。
<table><tr><td rowspan=2 colspan=1>层</td><td rowspan=2 colspan=1>出现次数</td><td rowspan=2 colspan=1>执行单元</td><td rowspan=1 colspan=3>16×16</td><td rowspan=1 colspan=3>32×32</td></tr><tr><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td></tr><tr><td rowspan=1 colspan=1>add197×768</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>18989</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>9533</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>add_scalar1×768</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>173</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>125</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>classifier1×768*768×1000</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>98294</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>49965</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>layer_norm_24224×768</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>86336</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>43326</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>layer_norm_fused224×768</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>86334</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>43323</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>matmul_24_1_fused224×64 *64×256</td><td rowspan=1 colspan=1>144</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>12544</td><td rowspan=1 colspan=1>14500</td><td rowspan=1 colspan=1>86.51%</td><td rowspan=1 colspan=1>3136</td><td rowspan=1 colspan=1>4439</td><td rowspan=1 colspan=1>70.65%</td></tr><tr><td rowspan=1 colspan=1>matmul_36_1_fused224×224 *224×64</td><td rowspan=1 colspan=1>144</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>12544</td><td rowspan=1 colspan=1>12678</td><td rowspan=1 colspan=1>99.09%</td><td rowspan=1 colspan=1>3136</td><td rowspan=1 colspan=1>3362</td><td rowspan=1 colspan=1>93.28%</td></tr><tr><td rowspan=1 colspan=1>quantize_default_7224×768</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>21579</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10826</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=9>下页续表</td></tr></table>

表 10：ViT 逐层性能与 MAC 利用率。
<table><tr><td rowspan=2 colspan=1>层</td><td rowspan=2 colspan=1>出现次数</td><td rowspan=2 colspan=1>执行单元</td><td rowspan=1 colspan=3>16×16</td><td rowspan=1 colspan=3>32×32</td></tr><tr><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td></tr><tr><td rowspan=1 colspan=1>softmax_12_fused224×256</td><td rowspan=1 colspan=1>144</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>19057</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>9646</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>patch_embeddings_projection224×224×3 *16×16×3×768</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>451584</td><td rowspan=1 colspan=1>603197</td><td rowspan=1 colspan=1>74.87%</td><td rowspan=1 colspan=1>112896</td><td rowspan=1 colspan=1>172323</td><td rowspan=1 colspan=1>65.51%</td></tr><tr><td rowspan=1 colspan=1>attention_key_fused224×768 *768×768</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>516096</td><td rowspan=1 colspan=1>516230</td><td rowspan=1 colspan=1>99.97%</td><td rowspan=1 colspan=1>129024</td><td rowspan=1 colspan=1>129250</td><td rowspan=1 colspan=1>99.83%</td></tr><tr><td rowspan=1 colspan=1>attention_output_dense_fused_012×224×64*768×768</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>516096</td><td rowspan=1 colspan=1>525447</td><td rowspan=1 colspan=1>98.22%</td><td rowspan=1 colspan=1>129024</td><td rowspan=1 colspan=1>134528</td><td rowspan=1 colspan=1>95.91%</td></tr><tr><td rowspan=1 colspan=1>intermediate_dense_fused224×768 *768×3072</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>2064384</td><td rowspan=1 colspan=1>2064930</td><td rowspan=1 colspan=1>99.97%</td><td rowspan=1 colspan=1>516096</td><td rowspan=1 colspan=1>519608</td><td rowspan=1 colspan=1>99.32%</td></tr><tr><td rowspan=1 colspan=1>output_dense_fused224×3072*3072×768</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>2064384</td><td rowspan=1 colspan=1>2064930</td><td rowspan=1 colspan=1>99.97%</td><td rowspan=1 colspan=1>516096</td><td rowspan=1 colspan=1>519608</td><td rowspan=1 colspan=1>99.32%</td></tr><tr><td rowspan=1 colspan=1>attention_output_dense_fused_112×224×64*768×768</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>516096</td><td rowspan=1 colspan=1>516641</td><td rowspan=1 colspan=1>99.90%</td><td rowspan=1 colspan=1>129024</td><td rowspan=1 colspan=1>132536</td><td rowspan=1 colspan=1>97.35%</td></tr><tr><td rowspan=1 colspan=1>总计</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>84369325</td><td rowspan=1 colspan=1>99.3%</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>22741874</td><td rowspan=1 colspan=1>98.0%</td></tr></table>

表 11：BERT-Base 逐层性能与 MAC 利用率。
<table><tr><td rowspan=2 colspan=1>层</td><td rowspan=2 colspan=1>出现次数</td><td rowspan=2 colspan=1>执行单元</td><td rowspan=1 colspan=3>16×16</td><td rowspan=1 colspan=3>32×32</td></tr><tr><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td></tr><tr><td rowspan=1 colspan=1>attention_output_dense_fused128×768*768×768</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>294912</td><td rowspan=1 colspan=1>298537</td><td rowspan=1 colspan=1>98.79%</td><td rowspan=1 colspan=1>73728</td><td rowspan=1 colspan=1>76485</td><td rowspan=1 colspan=1>96.40%</td></tr><tr><td rowspan=1 colspan=1>attention_self_key_fused128×768 *768×768</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>294912</td><td rowspan=1 colspan=1>295053</td><td rowspan=1 colspan=1>99.95%</td><td rowspan=1 colspan=1>73728</td><td rowspan=1 colspan=1>73965</td><td rowspan=1 colspan=1>99.68%</td></tr><tr><td rowspan=1 colspan=1>intermediate_dense_fused128×768*768×3072</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>1179648</td><td rowspan=1 colspan=1>1179790</td><td rowspan=1 colspan=1>99.99%</td><td rowspan=1 colspan=1>294912</td><td rowspan=1 colspan=1>295149</td><td rowspan=1 colspan=1>99.92%</td></tr><tr><td rowspan=1 colspan=1>output_dense_fused128×3072 *3072×768</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>1179648</td><td rowspan=1 colspan=1>1180393</td><td rowspan=1 colspan=1>99.94%</td><td rowspan=1 colspan=1>294912</td><td rowspan=1 colspan=1>297092</td><td rowspan=1 colspan=1>99.26%</td></tr><tr><td rowspan=1 colspan=1>classifier1×768*768×2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>1635</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>1634</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>layer_norm_23128×768</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>49464</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3072</td><td rowspan=1 colspan=1>24886</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>layer_norm_fused128×768</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>49462</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3072</td><td rowspan=1 colspan=1>24883</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>matmul_24_fused128×64*64×128</td><td rowspan=1 colspan=1>144</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>4096</td><td rowspan=1 colspan=1>5135</td><td rowspan=1 colspan=1>79.77%</td><td rowspan=1 colspan=1>1024</td><td rowspan=1 colspan=1>1778</td><td rowspan=1 colspan=1>57.57%</td></tr><tr><td rowspan=1 colspan=1>matmul_36_fused128×128 *128×64</td><td rowspan=1 colspan=1>144</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>4096</td><td rowspan=1 colspan=1>4237</td><td rowspan=1 colspan=1>96.67%</td><td rowspan=1 colspan=1>1024</td><td rowspan=1 colspan=1>1261</td><td rowspan=1 colspan=1>81.18%</td></tr><tr><td rowspan=1 colspan=1>pooler_dense1×768*768×768</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>36864</td><td rowspan=1 colspan=1>74955</td><td rowspan=1 colspan=1>49.17%</td><td rowspan=1 colspan=1>18432</td><td rowspan=1 colspan=1>37491</td><td rowspan=1 colspan=1>49.15%</td></tr><tr><td rowspan=1 colspan=1>quantize_default128×768</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>12361</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>6216</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>softmax_12_fused128×128</td><td rowspan=1 colspan=1>144</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>6379</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3304</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=9>下页续表</td></tr></table>

表 11：BERT-Base 逐层性能与 MAC 利用率。
<table><tr><td rowspan=2 colspan=1>层</td><td rowspan=2 colspan=1>出现次数</td><td rowspan=2 colspan=1>执行单元</td><td rowspan=1 colspan=3>16×16</td><td rowspan=1 colspan=3>32×32</td></tr><tr><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td></tr><tr><td rowspan=1 colspan=1>tanh1×768</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>171</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>123</td><td rowspan=1 colspan=1>—</td></tr><tr><td rowspan=1 colspan=1>总计</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>46070904</td><td rowspan=1 colspan=1>99.4%</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>12243503</td><td rowspan=1 colspan=1>97.9%</td></tr></table>

表 12：MobileBERT-TINY 逐层性能与 MAC 利用率。
<table><tr><td rowspan=2 colspan=1>层</td><td rowspan=2 colspan=1>出现次数</td><td rowspan=2 colspan=1>执行单元</td><td rowspan=1 colspan=3>16×16</td><td rowspan=1 colspan=3>32×32</td></tr><tr><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td></tr><tr><td rowspan=1 colspan=1>add_10128×512</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>8267</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>4171</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>add_4_fused128×128</td><td rowspan=1 colspan=1>63</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2121</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>1096</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>classifier1×512*512×2</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1123</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>1122</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>matmul_2_fused128×32*32×128</td><td rowspan=1 colspan=1>84</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>2048</td><td rowspan=1 colspan=1>3094</td><td rowspan=1 colspan=1>66.18%</td><td rowspan=1 colspan=1>512</td><td rowspan=1 colspan=1>1262</td><td rowspan=1 colspan=1>40.57%</td></tr><tr><td rowspan=1 colspan=1>matmul_6_fused128×128 *128×32</td><td rowspan=1 colspan=1>84</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>2048</td><td rowspan=1 colspan=1>2189</td><td rowspan=1 colspan=1>93.56%</td><td rowspan=1 colspan=1>512</td><td rowspan=1 colspan=1>748</td><td rowspan=1 colspan=1>68.45%</td></tr><tr><td rowspan=1 colspan=1>attention_output_dense_fused4×128×32*128×128</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>8192</td><td rowspan=1 colspan=1>9294</td><td rowspan=1 colspan=1>88.12%</td><td rowspan=1 colspan=1>2048</td><td rowspan=1 colspan=1>2791</td><td rowspan=1 colspan=1>73.38%</td></tr><tr><td rowspan=1 colspan=1>attention_self_query_fused128×128*128×128</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>8192</td><td rowspan=1 colspan=1>8333</td><td rowspan=1 colspan=1>98.31%</td><td rowspan=1 colspan=1>2048</td><td rowspan=1 colspan=1>2285</td><td rowspan=1 colspan=1>89.60%</td></tr><tr><td rowspan=1 colspan=1>attention_self_value_fused128×512*512×128</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>32768</td><td rowspan=1 colspan=1>32908</td><td rowspan=1 colspan=1>99.57%</td><td rowspan=1 colspan=1>8192</td><td rowspan=1 colspan=1>8428</td><td rowspan=1 colspan=1>97.19%</td></tr><tr><td rowspan=1 colspan=1>bottleneck_input_dense_fused128×512*512×128</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>32768</td><td rowspan=1 colspan=1>33802</td><td rowspan=1 colspan=1>96.94%</td><td rowspan=1 colspan=1>8192</td><td rowspan=1 colspan=1>8926</td><td rowspan=1 colspan=1>91.78%</td></tr><tr><td rowspan=1 colspan=1>ffn_intermediate_dense_fused128×128*128×512</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>32768</td><td rowspan=1 colspan=1>32909</td><td rowspan=1 colspan=1>99.57%</td><td rowspan=1 colspan=1>8192</td><td rowspan=1 colspan=1>8429</td><td rowspan=1 colspan=1>97.18%</td></tr><tr><td rowspan=1 colspan=1>ffn_output_dense_fused128×512*512×128</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>32768</td><td rowspan=1 colspan=1>33805</td><td rowspan=1 colspan=1>96.93%</td><td rowspan=1 colspan=1>8192</td><td rowspan=1 colspan=1>8930</td><td rowspan=1 colspan=1>91.77%</td></tr><tr><td rowspan=1 colspan=1>output_bottleneck_dense_fused128×128 *128×512</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>32768</td><td rowspan=1 colspan=1>36750</td><td rowspan=1 colspan=1>89.16%</td><td rowspan=1 colspan=1>8192</td><td rowspan=1 colspan=1>10459</td><td rowspan=1 colspan=1>78.32%</td></tr><tr><td rowspan=1 colspan=1>quantize_default128×512</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>8265</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4168</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>softmax_1_fused128×128</td><td rowspan=1 colspan=1>84</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6379</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3304</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>总计</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>7713636</td><td rowspan=1 colspan=1>95.1%</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2369283</td><td rowspan=1 colspan=1>86.2%</td></tr></table>

表 13：LLaMA 逐层性能与 MAC 利用率。
<table><tr><td rowspan="2">层</td><td rowspan="2">出现次数</td><td rowspan="2">执行单元</td><td colspan="3">32×32</td></tr><tr><td>理想周期</td><td>实际周期</td><td>MAC利用率</td></tr><tr><td>add_39_fused</td><td>128</td><td>向量</td><td>—</td><td>16,456</td><td></td></tr><tr><td>1×8×512×64</td><td></td><td></td><td></td><td>下页续表</td><td></td></tr></table>

表 13：LLaMA 逐层性能与 MAC 利用率。
<table><tr><td rowspan=2 colspan=1>层</td><td rowspan=2 colspan=1>出现次数</td><td rowspan=2 colspan=1>执行单元</td><td rowspan=1 colspan=3>32×32</td></tr><tr><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td></tr><tr><td rowspan=1 colspan=1>layernorm_default_fused1×32×64×512</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>262,451</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>matmul_2_fused512×512*512×64</td><td rowspan=1 colspan=1>512</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>16384</td><td rowspan=1 colspan=1>16754</td><td rowspan=1 colspan=1>97.79%</td></tr><tr><td rowspan=1 colspan=1>matmul 34 fused512×64*64×512</td><td rowspan=1 colspan=1>512</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>16384</td><td rowspan=1 colspan=1>16620</td><td rowspan=1 colspan=1>98.58%</td></tr><tr><td rowspan=1 colspan=1>model_layers_0_mlp_down_proj_fused512×8192 *8192×2048</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>8,388,608</td><td rowspan=1 colspan=1>8,390,496</td><td rowspan=1 colspan=1>99.98%</td></tr><tr><td rowspan=1 colspan=1>model_layers_0_mlp_gate_proj_fused512×2048 *2048×8192</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>16,777,216</td><td rowspan=1 colspan=1>16,780,160</td><td rowspan=1 colspan=1>99.98%</td></tr><tr><td rowspan=1 colspan=1>model_layers_0_mlp_up_proj_fused512×2048 *2048×8192</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>8,388,608</td><td rowspan=1 colspan=1>8,391,520</td><td rowspan=1 colspan=1>99.79%</td></tr><tr><td rowspan=1 colspan=1>model_layers_0_self_attn_k_proj_fused512×2048 *2048×512</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>524,288</td><td rowspan=1 colspan=1>524,656</td><td rowspan=1 colspan=1>99.93%</td></tr><tr><td rowspan=1 colspan=1>model_layers_0_self_attn_o_proj_fused512×2048 *2048×2048</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>2,097,152</td><td rowspan=1 colspan=1>2,097,520</td><td rowspan=1 colspan=1>99.98%</td></tr><tr><td rowspan=1 colspan=1>model_layers_0_self_attn_q_proj_fused512×2048 *2048×2048</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>2,097,152</td><td rowspan=1 colspan=1>2,097,520</td><td rowspan=1 colspan=1>99.98%</td></tr><tr><td rowspan=1 colspan=1>mul 2 fused512×2048 *2048×2048</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65608</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>mul_4_fused512×2048 *2048×2048</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>16456</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>permute_default_1_fused4×1×8×512×64</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>16459</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>permute_default_fused512×2048 *2048×2048</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65608</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>quantize_default32×64×512</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65608</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>slice_132×512×64-&gt; 32×64×512</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>32843</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>slice 2 fused32×512×64-&gt; 512×32×64</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>32843</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>slice_332×512×64-&gt;32×64×512</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>8267</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>slice 4 fused32×512×64-&gt; 512×32×64</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>8267</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>softmax_1_fused512×512</td><td rowspan=1 colspan=1>512</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>49384</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>lm_head512×2048 *2048×126976</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>8,126,464</td><td rowspan=1 colspan=1>8,127,936</td><td rowspan=1 colspan=1>99.98%</td></tr><tr><td rowspan=1 colspan=1>总计</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>687,950,352</td><td rowspan=1 colspan=1>99.8%</td></tr></table>

表 14：MobileNetV2 逐层性能与 MAC 利用率。
<table><tr><td rowspan=2 colspan=1>层</td><td rowspan=2 colspan=1>出现次数</td><td rowspan=2 colspan=1>执行单元</td><td rowspan=1 colspan=3>16×16</td><td rowspan=1 colspan=3>32×32</td></tr><tr><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td></tr><tr><td rowspan=1 colspan=1>features_0_0_fused224×224×3 *3×3×3×32</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>42336</td><td rowspan=1 colspan=1>466179</td><td rowspan=1 colspan=1>9.08%</td><td rowspan=1 colspan=1>10584</td><td rowspan=1 colspan=1>117505</td><td rowspan=1 colspan=1>9.00%</td></tr><tr><td rowspan=1 colspan=1>features_1_conv_0_0_fused112×112×32*3×3×1×32</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>深度卷积</td><td rowspan=1 colspan=1>25088</td><td rowspan=1 colspan=1>26794</td><td rowspan=1 colspan=1>93.63%</td><td rowspan=1 colspan=1>12544</td><td rowspan=1 colspan=1>13460</td><td rowspan=1 colspan=1>93.19%</td></tr><tr><td rowspan=1 colspan=1>features_1_conv_1_fused112×112×16*1×1×32×16</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>25088</td><td rowspan=1 colspan=1>25224</td><td rowspan=1 colspan=1>99.46%</td><td rowspan=1 colspan=1>6272</td><td rowspan=1 colspan=1>12765</td><td rowspan=1 colspan=1>49.13%</td></tr><tr><td rowspan=1 colspan=1>features_2_conv_0_0_fused112×112×16*1×1×16×96</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>75264</td><td rowspan=1 colspan=1>75400</td><td rowspan=1 colspan=1>99.82%</td><td rowspan=1 colspan=1>18816</td><td rowspan=1 colspan=1>37853</td><td rowspan=1 colspan=1>49.71%</td></tr><tr><td rowspan=1 colspan=1>features_2_conv_1_0_fused112×112×96 *3×3×1×96</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>深度卷积</td><td rowspan=1 colspan=1>18816</td><td rowspan=1 colspan=1>80155</td><td rowspan=1 colspan=1>23.47%</td><td rowspan=1 colspan=1>9408</td><td rowspan=1 colspan=1>40153</td><td rowspan=1 colspan=1>23.43%</td></tr><tr><td rowspan=1 colspan=1>features_2_conv_2_fused56×56×96 *1×1×96×24</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>28224</td><td rowspan=1 colspan=1>37768</td><td rowspan=1 colspan=1>74.73%</td><td rowspan=1 colspan=1>7056</td><td rowspan=1 colspan=1>9629</td><td rowspan=1 colspan=1>73.28%</td></tr><tr><td rowspan=1 colspan=1>features_3_conv_0_0_fused56×56×24*1×1×24×144</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>42336</td><td rowspan=1 colspan=1>56584</td><td rowspan=1 colspan=1>74.82%</td><td rowspan=1 colspan=1>10584</td><td rowspan=1 colspan=1>15901</td><td rowspan=1 colspan=1>66.56%</td></tr><tr><td rowspan=1 colspan=1>features_3_conv_1_0_fused56×56×144*3×3×1×144</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>深度卷积</td><td rowspan=1 colspan=1>28224</td><td rowspan=1 colspan=1>31018</td><td rowspan=1 colspan=1>90.99%</td><td rowspan=1 colspan=1>14112</td><td rowspan=1 colspan=1>17354</td><td rowspan=1 colspan=1>81.32%</td></tr><tr><td rowspan=1 colspan=1>features_3_conv_2_fused56×56×144*1×1×144×24</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>42336</td><td rowspan=1 colspan=1>56584</td><td rowspan=1 colspan=1>74.82%</td><td rowspan=1 colspan=1>10584</td><td rowspan=1 colspan=1>15901</td><td rowspan=1 colspan=1>66.56%</td></tr><tr><td rowspan=1 colspan=1>features_4_conv_1_0_fused56×56×144*3×3×1×144</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>深度卷积</td><td rowspan=1 colspan=1>7056</td><td rowspan=1 colspan=1>30997</td><td rowspan=1 colspan=1>22.76%</td><td rowspan=1 colspan=1>3528</td><td rowspan=1 colspan=1>17333</td><td rowspan=1 colspan=1>20.35%</td></tr><tr><td rowspan=1 colspan=1>features_4_conv_2_fused28×28×144*1×1×144×32</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>14112</td><td rowspan=1 colspan=1>14248</td><td rowspan=1 colspan=1>99.05%</td><td rowspan=1 colspan=1>3528</td><td rowspan=1 colspan=1>4141</td><td rowspan=1 colspan=1>85.20%</td></tr><tr><td rowspan=1 colspan=1>features_5_conv_0_0_fused28×28×32 *1×1×32×192</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>18816</td><td rowspan=1 colspan=1>18952</td><td rowspan=1 colspan=1>99.28%</td><td rowspan=1 colspan=1>4704</td><td rowspan=1 colspan=1>4925</td><td rowspan=1 colspan=1>95.51%</td></tr><tr><td rowspan=1 colspan=1>features_5_conv_1_0_fused28×28×192*3×3×1×192</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>深度卷积</td><td rowspan=1 colspan=1>9408</td><td rowspan=1 colspan=1>10726</td><td rowspan=1 colspan=1>87.71%</td><td rowspan=1 colspan=1>4704</td><td rowspan=1 colspan=1>5506</td><td rowspan=1 colspan=1>85.43%</td></tr><tr><td rowspan=1 colspan=1>features_5_conv_2_fused28×28×192*1×1×192×32</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>18816</td><td rowspan=1 colspan=1>18952</td><td rowspan=1 colspan=1>99.28%</td><td rowspan=1 colspan=1>4704</td><td rowspan=1 colspan=1>4925</td><td rowspan=1 colspan=1>95.51%</td></tr><tr><td rowspan=1 colspan=1>features_7_conv_1_0_fused28×28×192*3×3×1×192</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>深度卷积</td><td rowspan=1 colspan=1>2352</td><td rowspan=1 colspan=1>10695</td><td rowspan=1 colspan=1>21.99%</td><td rowspan=1 colspan=1>1176</td><td rowspan=1 colspan=1>5475</td><td rowspan=1 colspan=1>21.48%</td></tr><tr><td rowspan=1 colspan=1>features_7_conv_2_fused14×14×192*1×1×192×64</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>9408</td><td rowspan=1 colspan=1>9544</td><td rowspan=1 colspan=1>98.58%</td><td rowspan=1 colspan=1>2352</td><td rowspan=1 colspan=1>2573</td><td rowspan=1 colspan=1>91.41%</td></tr><tr><td rowspan=1 colspan=1>features 8 conv 0 0 fused14×14×64*1×1×64×384</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>18816</td><td rowspan=1 colspan=1>18952</td><td rowspan=1 colspan=1>99.28%</td><td rowspan=1 colspan=1>4704</td><td rowspan=1 colspan=1>4925</td><td rowspan=1 colspan=1>95.51%</td></tr><tr><td rowspan=1 colspan=1>features_8_conv_1_0_fused14×14×384*3×3×1×384</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>深度卷积</td><td rowspan=1 colspan=1>4704</td><td rowspan=1 colspan=1>6238</td><td rowspan=1 colspan=1>75.41%</td><td rowspan=1 colspan=1>2352</td><td rowspan=1 colspan=1>3358</td><td rowspan=1 colspan=1>70.04%</td></tr><tr><td rowspan=1 colspan=1>features_8_conv_2_fused14×14×384*1×1×384×64</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>18816</td><td rowspan=1 colspan=1>18952</td><td rowspan=1 colspan=1>99.28%</td><td rowspan=1 colspan=1>4704</td><td rowspan=1 colspan=1>4925</td><td rowspan=1 colspan=1>95.51%</td></tr><tr><td rowspan=1 colspan=1>features_11_conv_2_fused14×14×384*1×1×384×96</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>28224</td><td rowspan=1 colspan=1>28360</td><td rowspan=1 colspan=1>99.52%</td><td rowspan=1 colspan=1>7056</td><td rowspan=1 colspan=1>7277</td><td rowspan=1 colspan=1>96.96%</td></tr><tr><td rowspan=1 colspan=1>features_12_conv_0_0_fused14×14×96 *1×1×96×576</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>42336</td><td rowspan=1 colspan=1>42472</td><td rowspan=1 colspan=1>99.68%</td><td rowspan=1 colspan=1>10584</td><td rowspan=1 colspan=1>10805</td><td rowspan=1 colspan=1>97.95%</td></tr><tr><td rowspan=1 colspan=1>features_12_conv_1_0_fused14×14×576*3×3×1×576</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>深度卷积</td><td rowspan=1 colspan=1>7056</td><td rowspan=1 colspan=1>9310</td><td rowspan=1 colspan=1>75.79%</td><td rowspan=1 colspan=1>3528</td><td rowspan=1 colspan=1>4990</td><td rowspan=1 colspan=1>70.70%</td></tr><tr><td rowspan=1 colspan=1>features_12_conv_2_fused14×14×576 *1×1×576×96</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>42336</td><td rowspan=1 colspan=1>42472</td><td rowspan=1 colspan=1>99.68%</td><td rowspan=1 colspan=1>10584</td><td rowspan=1 colspan=1>10805</td><td rowspan=1 colspan=1>97.95%</td></tr><tr><td rowspan=1 colspan=9>下页续表</td></tr></table>

表 14：MobileNetV2 逐层性能与 MAC 利用率。
<table><tr><td rowspan=2 colspan=1>层</td><td rowspan=2 colspan=1>出现次数</td><td rowspan=2 colspan=1>执行单元</td><td rowspan=1 colspan=3>16×16</td><td rowspan=1 colspan=3>32×32</td></tr><tr><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td><td rowspan=1 colspan=1>理想周期</td><td rowspan=1 colspan=1>实际周期</td><td rowspan=1 colspan=1>MAC利用率</td></tr><tr><td rowspan=1 colspan=1>features_14_conv_1_0_fused14×14×576*3×3×1×576</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>深度卷积</td><td rowspan=1 colspan=1>1764</td><td rowspan=1 colspan=1>9293</td><td rowspan=1 colspan=1>18.98%</td><td rowspan=1 colspan=1>882</td><td rowspan=1 colspan=1>4973</td><td rowspan=1 colspan=1>17.73%</td></tr><tr><td rowspan=1 colspan=1>features 14 conv 2 fused7×7×576 *1×1×576×160</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>17640</td><td rowspan=1 colspan=1>17797</td><td rowspan=1 colspan=1>99.12%</td><td rowspan=1 colspan=1>4410</td><td rowspan=1 colspan=1>4631</td><td rowspan=1 colspan=1>95.23%</td></tr><tr><td rowspan=1 colspan=1>features_15_conv_0_0_fused7×7×160 *1×1×160×960</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>29400</td><td rowspan=1 colspan=1>29557</td><td rowspan=1 colspan=1>99.47%</td><td rowspan=1 colspan=1>7350</td><td rowspan=1 colspan=1>7571</td><td rowspan=1 colspan=1>97.08%</td></tr><tr><td rowspan=1 colspan=1>features_15_conv_1_0_fused7×7×960 *3×3×1×960</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>深度卷积</td><td rowspan=1 colspan=1>2940</td><td rowspan=1 colspan=1>5374</td><td rowspan=1 colspan=1>54.71%</td><td rowspan=1 colspan=1>1470</td><td rowspan=1 colspan=1>3214</td><td rowspan=1 colspan=1>45.74%</td></tr><tr><td rowspan=1 colspan=1>features_15_conv_2_fused7×7×960*1×1×960×160</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>29400</td><td rowspan=1 colspan=1>29557</td><td rowspan=1 colspan=1>99.47%</td><td rowspan=1 colspan=1>7350</td><td rowspan=1 colspan=1>7571</td><td rowspan=1 colspan=1>97.08%</td></tr><tr><td rowspan=1 colspan=1>features_17_conv_2_fused7×7×960 *1×1×960×320</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>58800</td><td rowspan=1 colspan=1>58957</td><td rowspan=1 colspan=1>99.73%</td><td rowspan=1 colspan=1>14700</td><td rowspan=1 colspan=1>14921</td><td rowspan=1 colspan=1>98.52%</td></tr><tr><td rowspan=1 colspan=1>features_18_0_fused7×7×320*1×1×320×1280</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>矩阵</td><td rowspan=1 colspan=1>78400</td><td rowspan=1 colspan=1>78659</td><td rowspan=1 colspan=1>99.67%</td><td rowspan=1 colspan=1>19600</td><td rowspan=1 colspan=1>19923</td><td rowspan=1 colspan=1>98.38%</td></tr><tr><td rowspan=1 colspan=1>adaptive_avg_pool2d_default7×7×1280</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7914</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3993</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>classifier_11280 *1280×1000</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>向量</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>162867</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>82731</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>总计</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>2010336</td><td rowspan=1 colspan=1>83.7%</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1>659439</td><td rowspan=1 colspan=1>78.0%</td></tr></table>