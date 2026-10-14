# FPnew: An Open-Source Multiformat Floating-Point Unit Architecture for Energy-Proportional Transprecision Computing 论文解析

## 0. 论文基本信息

**作者 (Authors)**: Stefan Mach, Fabian Schuiki, Florian Zaruba, Luca Benini

**发表期刊/会议 (Journal/Conference)**: IEEE Transactions on Very Large Scale Integration (VLSI) Systems

**发表年份 (Publication Year)**: 2021

**研究机构 (Affiliations)**: Swiss Federal Institute of Technology Zürich (ETH Zürich), Università di Bologna

---

## 1. 摘要

**目的**

- 应对 Moore 定律放缓与功耗墙带来的能效挑战，推动 **transprecision computing（跨精度/细粒度可调精度计算）** 范式在通用处理器中的落地。
- 设计 **FPnew**：一个开源、高度可配置的 **TP-FPU（Transprecision Floating-Point Unit）** 架构，支持任意 IEEE 754-2008 兼容 FP 格式。
- 扩展 RISC-V ISA，使 FP8、FP16、FP16alt（bfloat16）等非标准窄格式、SIMD 向量化及 multiformat 运算可在高级语言与编译器层面被直接使用。
- 通过 32 位嵌入式核心与 64 位应用级核心的流片验证，评估该架构的**能效比例性**与综合性能。

---

**方法**

**架构设计（FPnew）**
- 采用层级化架构：**FPU Top Level → Operation Group Blocks → Format Slices → Functional Units**。
- 四大操作组：
  - **ADDMUL**：加法、乘法、FMA（单路径架构，1/2 ulp 精度，支持 multiformat FMA，即 `dst_fmt fma(src_fmt, src_fmt, dst_fmt)`）。
  - **DIVSQRT**：迭代式非恢复除法/开方，每周期产出 3 位尾数。
  - **COMP**：比较与位操作。
  - **CONV**：任意 FP 格式间及 FP↔整数转换，支持 cast-and-pack 动态打包向量。
- Format Slices 可配置为 **parallel（每格式独立切片、独立流水线、可时钟门控）** 或 **merged（共享硬件、多格式合一）** 两种模式。
- 支持格式宽度非 2 的幂的任意 ；SIMD lane 数按 `k = ⌊w_fpu / w_f⌋`（parallel）或基于最小格式（merged）确定。
- 全局采用 **valid-ready 握手协议**，实现细粒度时钟门控（可“压缩”流水线气泡）与粗粒度模块关断，无需预知操作延迟。

**ISA 扩展与编译器支持**
- 新增格式：**FP64、FP32、FP16、FP16alt (8,7)、FP8 (5,2)**。
- 指令分三类：
  - **scalar**：复用标准 "F" 编码映射至新格式。
  - **vectorial**：SIMD 与 vector-scalar 混合操作（标量操作数广播复制至全部 lane），向量转换提供取低/高半区指令。
  - **auxiliary**：expanding FMA（fmacex）与 cast-and-pack 指令。
- 采用 brown-field 编码（FP16 用保留编码、FP8 复用 "Q" 编码、FP16alt 借用保留舍入模式），并集成进 **RISC-V GCC 工具链**（如 `float8` C 类型、intrinsic 函数）。

**硬件实现**

| 平台 | 核心 | w_fpu | 关键配置 |
|---|---|---|---|
| PULPissimo | RI5CY（32 位，RV32IMFC） | 32 bit | ADDMUL 采用 merged multiformat 切片，全单周期；DIVSQRT 禁用 |
| Kosmodrom | Ariane（64 位，RV64GC） | 64 bit | ADDMUL 采用 parallel 切片，格式相关流水深度（FP64 4 级 / FP8 2 级） |

- 两个平台均基于 **GLOBALFOUNDRIES 22FDX（22 nm FD-SOI）** 完成完整 place & route。
- Kosmodrom 芯片：>96% FPU 寄存器经自动时钟门控插入，利用 retiming 优化流水线寄存器位置；深度探索了 cell mix（74% LVT + 26% SLVT）。
- 测试方法：合成负载注入随机正态分布 FP 值（限制无溢出，保证最高翻转活动），配合 post-layout 校准仿真进行能耗分项归因。

---

**结果**

**RI5CY + TP-FPU（PULPissimo）**
- 核心面积 147 kGE，其中 FPU 占 69 kGE（47%）；ADDMUL 占 FPU 面积 76%。
- 相比仅 FP32 的基准：核心面积 +29%、静态功耗 +37%；但系统级仅 +0.7% / +0.9%（存储器主导）。
- FP32 FMA 执行单元能耗 3.9 pJ，与 32 位整数 MAC（1.0 pJ）在系统级差距微小（22.2 vs 21.2 pJ/cycle），证明 MCU 场景下 FP 运算成本可负担。

**应用案例（element-wise 乘积累加）**

| 实现 | 精度（正确位数） | 核心相对能耗 | 系统相对能耗 |
|---|---|---|---|
| 纯 FP16 | 9 bit | 0.60 | 0.63 |
| 纯 FP32 | 22 bit | 1.00 | 1.00 |
| FP16 乘 + FP32 加 | 19 bit | 1.16 | 1.03 |
| SIMD FP16 + FP32 加 | 19 bit | 0.97 | 0.75 |
| **expanding FMA (fmacex)** | **22 bit** | **0.63** | **0.63** |

- 最优方案（fmacex）：达到与纯 FP16 相同的 3 指令/元素性能与 FP32 全精度，相比 FP32 基准**加速 1.67×、系统能耗降低 37%**。

**Ariane + TP-FPU（Kosmodrom 流片实测）**
- FPU 面积 247 kGE（其中 FMA 单元 160 kGE）；相比仅 "F"+"D" 的 Ariane，处理器面积 +9.3%、静态能耗 +11.1%。
- 标称条件下（0.8 V, 923 MHz）实测硅片性能与能效（FMA）：

| 格式 | 性能 | 能效 |
|---|---|---|
| FP64 (scalar) | 1.85 Gflop/s | 74.83 Gflop/sW |
| FP32 (scalar) | 1.85 Gflop/s | 211.66 Gflop/sW |
| FP8 (scalar) | 1.85 Gflop/s | 786.30 Gflop/sW |
| FP8 (vector, 2/8) | **14.83 Gflop/s** | **1244.78 Gflop/sW** |

- **能效比例性**：FP64 FMA 26.7 pJ；缩窄格式相对下一级大格式的标量 FMA 能耗节省为 65% / 48% / 54% / 49%（依次递缩）；SIMD FMA 按 datum 计的能耗收益达 60% / 66% / 58%，**优于直接比例**（得益于独立切片 + 空闲 lane 门控）。
- 例外：CONV 块（merged 切片）格式减半时能耗仅降 30%–35%，比例性弱于 parallel ADDMUL。
- 电压/频率缩放（0.425–1.2 V）：峰值 **25.33 Gflop/s @ FP8**、**2.95 Tflop/sW @ FP8**；FP64 为 3.17 Gflop/s / 178 Gflop/sW。
- 核心级视角：FP64 FMA 中 FPU 仅占核心能耗 39%，FP8 标量仅使整体核心能耗降 38%（核心侧开销显著）；SIMD 向量化可大幅摊薄该开销（FP8 vector FMA 再降 6.2×）。
- 对比 SOTA（同 22 nm 工艺 Kaiser et al. 的 FP64 FMA）：本设计整 FPU 能效高 49%；相比其他 multimode FMA 架构，FP8 相对 FP64 能效达 **16.6×**，叠加电压缩放后可再增 2.3×。

![](images/abb01a4121dcc57ed75b45d837b40a5b5cbf247d589ab31e7213a7a78cf7898b.jpg)

---

**结论**

- FPnew 证明了通过 **parallel 切片 + 格式专属流水深度 + 细/粗粒度时钟门控** 的架构取舍（以面积换能效），可实现显著优于共享数据通路方案的**能量比例性**，窄格式收益呈超比例放大。
- 开源特性带来广泛生态影响：已被 **Ara**（RISC-V 向量处理器）、**Snitch**、GreenWaves **GAP9** 及 **European Processor Initiative (EPI)** 等项目采用。
- 额外的面积（9.3%）与静态功耗（11.1%）代价，相比获得的性能与能效增益（最高 2.95 Tflop/sW、25.33 Gflop/s）是可接受且划算的。
- 局限与方向：CONV 类 merged 切片的比例性仍有优化空间；应用级核心中非 FPU 开销部分抵消 transprecision 收益，需依赖 SIMD 向量化摊薄；未来可针对不同格式探索差异化的 FMA 微架构。

---

## 2. 背景知识与核心贡献

**研究背景**

- 摩尔定律放缓与功耗墙限制，使能效成为硬件设计的终极指标，推动计算范式向**transprecision computing（跨精度计算）**转型。
- 机器学习等新兴负载在数据中心、HPC 到 IoT 领域均要求在恒定或降低的功耗预算下持续提升算力。
- IEEE 754 标准浮点具备舍入模式、优雅下溢和无穷表示，数值鲁棒性与动态范围远优于 fixed-point，但传统 CPU/GPU 的精度调节仅限于 **FP64/FP32**。
- 专用加速器领域出现 FP 格式的“寒武纪大爆发”：
  - Intel Nervana 的 **Flexpoint**
  - Microsoft Brainwave 的 **9-bit floats**
  - Google TPU 的 **bfloat16**
  - NVIDIA 的 **TF32**
- RISC-V 作为开放、可扩展的 ISA，为引入非标准 FP 格式扩展提供了理想载体。

**研究动机**

- 现有 FPU 无法同时满足以下需求：
  - 支持**任意位宽、任意格式**的 IEEE 754 二进制浮点运算；
  - 具备 **SIMD 向量化**与**多格式混合运算**（multiformat FMA）能力；
  - 实现高度**能量比例性**，即窄格式操作能耗随位宽降低而超比例下降；
  - 灵活适配不同目标平台（嵌入式 MCU、应用级 CPU、加速器）的流水线深度与工艺流程。
- 缺乏面向 transprecision 的 **ISA 扩展与编译器支持**，导致软硬件栈脱节。

**核心贡献**

- **FPnew 架构设计**：
  - 开源 SystemVerilog 实现的高度可配置 TP-FPU（Transprecision FPU）；
  - 四大操作块：**ADDMUL**（加/乘/FMA）、**DIVSQRT**（除法/开方）、**COMP**（比较/位操作）、**CONV**（格式转换）；
  - 每块支持 parallel（格式专用）或 merged（多格式共享）slice，格式位宽不限 2 的幂；
  - 有意**增加面积换取能效与能量比例性**，利用 clock gating 静默未激活分支。
- **RISC-V ISA 扩展**：
  - 新增 **FP16、FP16alt (bfloat16)、FP8** 三种格式及全部对应操作；
  - 支持 SIMD 向量指令、vector-scalar 变体、cast-and-pack、expanding FMA（扩精度乘累加）；
  - 扩展 GCC 工具链，C 语言可直接使用 `float8`、`float16` 等原生类型。
- **RI5CY（32 位嵌入式核）集成**：
  - 混合精度应用实现 **1.67× 加速**，端到端精度不损失，系统功耗降低 **37%**。
- **Ariane（64 位应用级核）集成与流片验证**：
  - 采用 GLOBALFOUNDRIES **22FDX** 工艺，首个支持五种 FP 格式的完整 TP-FPU 硅实现；
  - 关键实测指标：

| 格式 | 峰值性能 | 能量效率 |
|---|---|---|
| FP64 | 3.17 Gflop/s | 178 Gflop/sW |
| FP32 | 6.33 Gflop/s | 473 Gflop/sW |
| FP16 | 12.67 Gflop/s | 1.18 Tflop/sW |
| FP16alt | 12.67 Gflop/s | 1.38 Tflop/sW |
| FP8 | 25.33 Gflop/s | **2.95 Tflop/sW** |

- 工作电压范围 **0.45–1.2 V**，处理器面积仅增加 **9.3%**、静态功耗增加 **11.1%**，却在灵活性与能效上全面超越已发表的 FPU 设计。

![](images/b8de613683eb42b1c9317afc0db8f813fd63de2c8e39b10b4772271f14cc6114.jpg)

---

## 3. 核心技术和实现细节

### 0. 技术架构概览

本文提出 **FPnew**，一个高度可配置的**开源 transprecision 浮点运算单元 (TP-FPU)** 架构，采用 SystemVerilog 编写，遵循 IEEE 754-2008 标准，支持任意数量的自定义 FP 格式，并可灵活集成到不同处理器系统中。

---

**顶层架构 (FPU Top Level)**

- 单元宽度 **w_fpu** 参数化，每周期最多接收 **3 个 FP 操作数** 输入，配合控制信号决定操作类型与格式
- 输入操作数被路由至 **4 个操作组块 (Operation Group Blocks)** 之一，结果经仲裁器输出
- 由于任一时钟周期内仅一个操作组块可接收新数据，可对未使用分支采用 **clock gating 与 datapath gating**，消除伪翻转功耗

![](images/6d90228bbe42e0d4ce69d481fe9d9b6362fc5581b0e7cfc0a4672b711decd75d.jpg) *Fig. 2. Datapath block diagram of the TP-FPU with its underlying levels of hierarchy. It supports multiple FP formats, and the datapath width is configurable.*

**四大操作组块 (Operation Group Blocks)**

- **ADDMUL**：处理加法、乘法与 FMA 操作，采用单路径架构的全流水 FMA 单元，支持多格式混合精度操作，乘法在 **src_fmt** 执行、加法在 **dst_fmt** 执行，对应 C 风格原型 `dst_fmt fma(src_fmt, src_fmt, dst_fmt)`
- **DIVSQRT**：采用迭代式非恢复除法器计算除法与平方根，每周期生成 **3 位尾数位**，迭代次数可覆盖为少于正确舍入所需次数，实现精度换吞吐
- **COMP**：包含比较器及选择与位操作逻辑，执行比较、符号操作与 FP 分类
- **CONV**：多格式转换单元，支持任意两种 FP 格式间转换、整数与 FP 格式间互转，以及 **cast-and-pack** 动态向量构建

---

**格式切片 (Format Slices) 机制**

- 每个操作组块可包含多个切片，实现方式为 **parallel (并行)** 或 **merged (合并)** 两种
  - **Parallel 切片**：每种 FP 格式独立实现，拥有独立内部流水线，可针对各格式优化延迟与能耗；非活动切片可被 clock-gate，代价是面积增加
  - **Merged 切片**：多种格式共享硬件降低面积，但小格式复用过尺寸数据路径，引入能耗与延迟开销，且所有格式共享同一流水线深度

**SIMD 向量数据路径**

- 满足 $w_{fpu} \geq 2 \times w_f$ 时，切片可承载多向量 **lane**
- Parallel 模式下：$k_{parallel} = \lfloor w_{fpu} / w_f \rfloor$，每 lane 宽度等于格式宽度
- Merged 模式下：lane 数由最小支持格式决定，各 lane 位宽可不同，需支持不同格式集合
- 依据当前操作为 scalar 或 vectored，仅使用一个或多个 lane 计算，未使用 lane 被静默 (silenced)
- **CONV 块的 merged 切片**需更复杂的数据分发与收集机制，支持运行时动态向量创建

---

**控制流 (Unit Control Flow)**

- 仅允许一个操作进入 FPU，但不同延迟路径的结果可能在同一周期到达输出
- 所有切片与块的结果通过 **公平轮询仲裁** 合并
- 采用简单同步 **valid-ready 握手协议**，覆盖内部层级与外部接口
- 数据无需预知操作延迟即可穿越 FPU，支持以下优化：
  - 基于 handshake 信号的**细粒度 clock gating**，在各流水级内部静默未使用部分，让数据“追赶”流水线头部，消除气泡
  - **粗粒度 clock gating**：流水线中无有效数据时，可关闭整个操作组块或整个 TP-FPU

---

**配置与参数化**

- 通过 SystemVerilog **package 与 instance 参数** 实现充分可配置性
- 可自定义任意指数/尾码位数的格式，**不限于 2 的幂次位宽**
- 每个格式与操作组块可独立配置为 parallel、merged 或完全禁用
- SIMD 向量可全局启用，条件为 $w_f \leq w_{fpu}/2$；各格式与操作组块的**流水线级数独立可配**
- 所有实例参数在综合时固定为硬件，同一 FPnew RTL 无需修改即可生成多个不同配置实例，本工作中所有实现均基于默认 package 参数化得到

---

**系统级集成架构**

本文将 TP-FPU 集成到两个开源 RISC-V 处理器核，展示架构的伸缩性：

| 集成平台 | 核心特性 | FPU 配置 |
|---------|---------|---------|
| **RI5CY** (32-bit, PULPissimo SoC) | 嵌入式低功耗核，复用通用寄存器堆，支持 FP16/FP16alt SIMD 及 cast-and-pack、expanding FMA | w_fpu = 32 bit，ADDMUL 为 merged 多格式切片，DIVSQRT 禁用，全单周期操作 |
| **Ariane** (64-bit, Kosmodrom 芯片) | 应用级核，流片于 **GLOBALFOUNDRIES 22FDX** 工艺，支持 FP64/FP32/FP16/FP16alt/FP8 五种格式及 2/4/8-way SIMD | w_fpu = 64 bit，ADDMUL 为 parallel 切片（延迟 2–4 周期格式相关），DIVSQRT 为迭代式 merged 实现 |

![](images/4494004d09a481386945fccb9ada10b46eff5d767a66033e06ffa76bf642bb90.jpg) *Fig. 3. Floorplan and block diagram of the PULPissimo SoC (without pad frame) using RI5CY Core with TP-FPU.*

---

**核心设计理念总结**

- 以**面积换能耗比例性**：在先进工艺节点面积成本降低的背景下，采用 parallel 独立数据路径，实现**超比例的能耗收益**（如 FP8 FMA 相比 FP64 能耗降低 16.6 倍，远超 8 倍位宽缩减的直接比例）
- **全栈贯通**：配合 RISC-V ISA 扩展与 GCC 编译器支持，从硬件到软件构建完整 transprecision 计算生态
- **开源可扩展**：设计者可自由提取、替换功能单元或添加新操作组块，FPnew 已被 **Ara、Snitch、GAP9、EPI** 等项目复用

### 1. FPnew：高度可配置的多格式跨精度FPU架构

**核心定位**

FPnew 是一个以 SystemVerilog 编写、完全开源的**transprecision floating-point unit (TP-FPU)** IP 核。其设计哲学是在单一硬件单元内支持任意数量、任意位宽的 IEEE 754-2008 二进制 FP 格式，通过**精细粒度的格式切分与动态时钟门控**实现能量比例性——即能耗随实际使用的数值精度等比例甚至超比例地降低，而非传统 FPU 中“小格式复用宽通路导致的能量浪费”。该设计被集成到 32-bit RI5CY 与 64-bit Ariane 两款 RISC-V 处理器中，并在 GLOBALFOUNDRIES 22FDX 工艺完成**流片实测**。

---

**一、FP 格式编码基础与设计需求**

FPnew 遵循 IEEE 754-2008 二进制格式规范，任意格式由三个字段构成：

- **Sign (1 bit)** + **Exponent (w_exp bit, 无符号)** + **Mantissa (w_mant bit, 无符号二进制小数)**
- 格式总宽：**w_f = 1 + w_exp + w_mant**，指数偏置：**bias = 2^(w_exp-1) − 1**
- 编码值解析规则：
  - 常规值：v = (−1)^s × 2^(exp−bias) × 1.mant
  - exp = 0 时进入**非规格化数**：v = (−1)^s × 2^(1−bias) × 0.mant
  - exp = 全 1 且 mant = 0 表示 **±∞**；mant ≠ 0 表示 **NaN**

设计阶段确立的五项关键需求：

- **FP Format Encoding**：不限制格式宽度为 2 的幂，支持任意 的组合，同时保持 IEEE 754 标准兼容（含 denormal、rounding mode、infinity 处理），这是与 FloPoCo 等非标准实现的本质区别。
- **Operations 完整性**：覆盖 IEEE 754 规定的算术、比较、转换三类操作；核心是 **FMA (fused multiply-add)**，即单次舍入的 (a×b)+c；还支持**multiformat FMA**——乘法用 src_fmt、加法用 dst_fmt，匹配张量加速器中常见的混合精度模式。
- **SIMD Vectors**：不能靠复制多个窄 FPU 拼宽通路（那样操作被限制在同一种窄格式），必须原生支持向量化的**格式间转换**，实现运行时动态精度调整。
- **Variable Pipeline Depths**：不硬编码流水级数，每种格式、每个操作块的流水寄存器数量独立可配——GPU 可容忍长延迟追求小面积，嵌入式 MCU 则需要单周期完成。
- **Design Tool Flow**：选择 SystemVerilog (IEEE 1800) 而非 Chisel，保证商用 EDA 流程兼容；代码风格鼓励推断式描述，让综合器自动选择 FPGA DSP slice 或 ASIC Wallace-Tree 乘法器，并支持自动 clock gating 与 retiming。

---

**二、层次化架构：四层结构解析**

![](images/6d90228bbe42e0d4ce69d481fe9d9b6362fc5581b0e7cfc0a4672b711decd75d.jpg) *Fig. 2. Datapath block diagram of the TP-FPU with its underlying levels of hierarchy. It supports multiple FP formats, and the datapath width is configurable.*

**1) FPU Top Level（顶层）**

- 每时钟周期最多进入 **3 个 FP 操作数** + 控制信号（操作类型、格式选择），输出 1 个 FP 结果 + IEEE 754-2008 status flags。
- 输入输出位宽由参数 **w_fpu**（unit width）决定。
- 操作数被路由至四个**操作组块**之一；由于任一时钟周期仅一个组块接收新数据，可对未使用分支实施**数据通路门控与时钟门控**，消除伪翻转活动。
- 组块输出经**公平轮询仲裁器**汇聚到单元输出。

**2) Operation Group Blocks（操作组块）**

四个独立数据通路组：

- **ADDMUL**：加法、乘法、FMA（全流水）
- **DIVSQRT**：除法与开方（迭代式）
- **COMP**：比较与位操作
- **CONV**：FP 格式间转换、FP↔整数转换

每个组块内部可容纳多个 format slice，部署方式二选一：

- **Parallel（并行切分）**：每个 slice 只承载一种格式，各自拥有独立流水线，路径延迟与时延可独立优化；未激活的 slice 被时钟门控静默。代价是面积增加，但在先进工艺下面积相对廉价。这是实现**能量比例性**的关键——FP8 操作只翻转 FP8 通路，FP64 通路完全静默。
- **Merged（合并切分）**：多格式共享同一数据通路，降低面积但牺牲灵活性；小格式被迫复用过宽的数据通路，带来能量与时延开销。FPnew 在 CONV 块中采用 merged 方案（转换操作需要跨格式天然融合的通路）。

**3) Format Slices（格式切片）与 SIMD Lane 组织**

当 w_fpu ≥ 2 × w_f 时可构造 SIMD 向量数据通路：

- **Parallel 情况**：lane 数 k_parallel = ⌊w_fpu / w_f⌋，每 lane 宽度 w_lane = w_f，所有 lane 是同一功能单元的重复实例。
- **Merged 情况**：lane 数由最小支持格式决定，k_merged = ⌊w_fpu / Min(w_f)⌋；各 lane 位宽可以不同，每个 lane 需支持不同的格式子集。CONV 块的 merged slice 需要更复杂的数据分发/收集机制，因为输入输出向量长度可能不同（如 2×FP16 转 4×FP8 时目的向量元素数翻倍）。
- 支持将两个**标量 FP 操作数 cast 后动态打包**进向量元素，即 cast-and-pack。
- 标量操作只激活一条 lane，其余 lane 被静默。

**4) Functional Units（功能单元）**

- **ADDMUL**：采用**single-path 架构**的全流水 IEEE 754-2008 合规 FMA，结果精度 1/2 ulp；multiformat FMA 用于 merged slice，映射 C 风格原型 `dst_fmt fma(src_fmt, src_fmt, dst_fmt)`。
- **DIVSQRT**：**迭代式非恢复除法器**，每时钟周期产出 **3 个尾数位**，作为 merged slice 实现；迭代次数可被配置为少于正确舍入所需次数，以吞吐换精度——这是硬件层面的 transprecision 旋钮。FP64 除法需 21 周期，FP8 仅 4 周期（3 bit/迭代 + 恒定 3 周期前后处理）。
- **COMP**：比较器 + 选择逻辑 + 位操作，执行比较、符号操作、FP 分类。
- **CONV**：multiformat 转换单元，支持任意两种 FP 格式之间、整数→FP、FP→整数转换。

**5) Unit Control Flow（控制流）**

- 内部与外部接口均使用**同步 valid-ready 握手协议**。
- 数据可在无先验时延知识的情况下穿越 FPU；基于握手信号的**细粒度时钟门控**可逐流水级实施，让数据“追赶”被阻塞的流水线头部，主动挤出 pipeline bubble。
- **粗粒度门控**在流水线无有效数据时直接关闭整个操作组或整个 TP-FPU。

---

**三、配置参数体系与 ISA 扩展**

**配置机制**

- SystemVerilog package 定义任意数量的自定义格式（任意 w_exp / w_mant），不限于 2 的幂宽度。
- 每个操作块 × 每种格式可独立选择：parallel / merged / **disabled**。
- SIMD 全局使能条件：w_f ≤ w_fpu/2；w_fpu 可远大于最大格式宽度（面向向量加速器场景）。
- 所有参数在综合时固化；同一份 RTL 可实例化出多个不同配置。所有实现均从默认 package 参数化而来，无需修改 RTL。

**支持的五种格式**

| 格式 | 描述 | (w_exp, w_mant) | 总宽 |
|---|---|---|---|
| **FP64** | IEEE 754 binary64 | (11, 52) | 64 bit |
| **FP32** | IEEE 754 binary32 | (8, 23) | 32 bit |
| **FP16** | IEEE 754 binary16 | (5, 10) | 16 bit |
| **FP16alt** | 即 **bfloat16** | (8, 7) | 16 bit |
| **FP8** | 自定义 mini-float | (5, 2) | 8 bit |

**RISC-V ISA 扩展（三类指令）**

- **Scalar 指令**：将 "F" 扩展的全部操作（算术、比较、转换、数据搬运）映射到新格式；全格式间转换支持运行时精度缩放。
- **Vectorial 指令**：对所有窄于 FLEN 的格式提供 SIMD 版本；额外提供 **vector-scalar 变体**（标量操作数复制到所有 lane），使得 SIMD 矩阵乘无需在内存中转置矩阵。向量转换需特殊处理长度失配——提供取向量**下半部/上半部**的独立指令。
- **Auxiliary 指令**：
  - **fmacex（expanding FMA）**：乘法在窄格式、累加在宽格式，映射到 merged FMA slice 的多精度能力。
  - **cast-and-pack**：转换两个标量并打包进结果向量元素。

**指令编码策略（brown-field 扩展）**

- 标量指令复用标准 FP 编码：FP16 使用保留 format 编码；FP8 复用 quad-precision "Q" 扩展的编码空间（因目标平台不支持 128-bit FP）；FP16alt 编码为 FP16 + 保留 rounding mode。
- 向量扩展利用整数 opcode 空间中大量未用区域（与 RI5CY DSP 扩展同思路）。
- 编译器支持：格式以 `float8`、`float16` 等 C 类型进入 RISC-V GCC 工具链，运算符透明映射；向量操作经 GCC vector extensions 自动推断；fmacex / cast-and-pack 等不可推断操作通过 **compiler intrinsic** 调用。

---

**四、两种处理器集成方案对比**

**RI5CY（32-bit 嵌入式核，PULPissimo SoC）**

- 核心特性：RV32IMFC，4 级顺序流水；FP 数据可复用通用寄存器堆以省面积，并借此利用整数通路的 DSP SIMD 洗牌功能。
- 关键电路修改：解码器扩展；load/store 单元改造预存符号扩展电路实现**NaN-boxing 的 one-extension**；不在运算单元强制 NaN-boxing 检查，以便按需将 SIMD 数据当标量处理。
- FPU 配置（w_fpu = 32 bit，全部单周期，目标 MCU 级 150–370 MHz 时钟）：

| 格式 | ADDMUL | DIVSQRT | COMP | CONV |
|---|---|---|---|---|
| FP32 | merged (1,1) | **disabled** | parallel (1,1) | merged (1,2) |
| FP16 | merged (1,2) | disabled | parallel (1,2) | merged (1,2) |
| FP16alt | merged (1,2) | disabled | parallel (1,2) | merged (1,2) |

- **DIVSQRT 整块禁用**是细粒度可配置性的直接展示。ADDMUL 用 merged multiformat slice 支撑 FP16[alt]↔FP32 多精度操作；CONV 的两条 32-bit lane 支撑双 FP32 操作数 cast-and-pack。

![](images/4494004d09a481386945fccb9ada10b46eff5d767a66033e06ffa76bf642bb90.jpg) *Fig. 3. Floorplan and block diagram of the PULPissimo SoC (without pad frame) using RI5CY Core with TP-FPU.*

![](images/5e240019b12d46adfeac90ab30ac1ece1fe18bd2525fdff28ebc445d168790ee.jpg) *Fig. 4. Area distribution of the PULPissimo SoC and the RI5CY core (in kGE, $1 \mathrm { G E } \approx 0 . 1 9 9 \mu \mathrm { m } ^ { 2 } )$*

- 面积/能耗影响：RI5CY 核总面积 **147 kGE**，TP-FPU 占 **69 kGE (47%)**；其中 ADDMUL 占 FPU 面积 76%，CONV 18%，COMP 4%。对比纯 FP32 版本：核面积 +29%、静态能耗 +37%（FPU 长路径使用更多高驱动短栅晶体管：20 nm 与 28 nm 晶体管比例高出其余核 2.4 倍）。但 SoC 系统层面（含 512 kB 内存，占系统面积 92%/漏电 96%），增量仅 0.7% 面积、0.9% 静态能耗——证明 FP 计算在 MCU 语境下也是可负担的。

**Ariane（64-bit 应用级核，Kosmodrom 流片）**

![](images/543257c357148523a27985f40ac667204ed0600a781e36277eb591540134688c.jpg) *Fig. 5. Die micrograph and block diagram of entire Kosmodrom chip showing the placement of the two Ariane core macros with TP-FPU.*

- 核心特性：RV64GC，6 级部分乱序，带 I/D cache、RAS、BHT、BTB；两颗功能相同的 Ariane（不同 cell library）共享 1.25 MB L2。
- FPU 配置（w_fpu = 64 bit，ADD_MUL 采用**全 parallel** 以实现格式相关延迟，目标约 1 GHz）：

| 格式 | ADDMUL | DIVSQRT | COMP | CONV |
|---|---|---|---|---|
| FP64 | parallel (4,1) | merged (21,1*) | parallel (1,1) | merged (2,2*) |
| FP32 | parallel (3,2) | merged (11,0) | parallel (1,2) | merged (2,0) |
| FP16 | parallel (3,4) | merged (7,0) | parallel (1,4) | merged (2,2*) |
| FP16alt | parallel (3,4) | merged (6,0) | parallel (1,4) | merged (2,0) |
| FP8 | parallel (2,8) | merged (4,0) | parallel (1,8) | merged (2,4) |

- 设计要点：FP8 FMA 延迟仅 2 周期 vs FP64 的 4 周期——**窄格式同时获得更低的能耗与更短的延迟**；DIVSQRT 不支持 SIMD 以省面积（除法/开方在优化代码中出现频率低，长延迟可接受）；CONV 保留两条 64-bit lane 用于 FP64 cast-and-pack，另有 2×16 bit 和 4×8 bit lane 覆盖全部转换组合。
- 后端实现：22FDX 8-track LVT 库；最差约束 SSG 0.72 V/125°C 1 GHz + 20% uncertainty；**>96% FPU 寄存器被自动时钟门控**（未门控的仅剩握手 token 和除法/开方 FSM）；流水寄存器位置经综合工具 **retiming** 优化；单元库混合调优为 74% LVT + 26% SLVT、86%/8%/6% 的 28/24/20 nm 晶体管。
- 实测频率：0.96 GHz（最差角）/ **1.29 GHz**（典型角）/ 1.76 GHz（最佳角）。

![](images/b8de613683eb42b1c9317afc0db8f813fd63de2c8e39b10b4772271f14cc6114.jpg)

- 面积：Ariane 核 + TP-FPU 共 **742 kGE**（含 cache 1.4 MGE）；FPU 共 **247 kGE**，其中 FMA 单元 160 kGE、CONV 47 kGE、DIVSQRT 19 kGE、COMP 13 kGE。对比仅支持标量 FP32/FP64 的版本：面积 +9.3%、静态能耗 +11.1%——以两位数百分比成本换取三种新格式 + 最高 8× SIMD 吞吐提升。

---

**五、硅片实测：能量比例性验证**

**测量方法**：合成负载逐指令施压（正态分布随机值，约束无溢出，制造数据通路最差翻转场景），满流水峰值性能下测量；核级能耗通过矩阵乘 + 校准后布局布线仿真分层归因。测量条件：TT 角 0.8 V/25°C，实测 923 MHz。

**标量算术指令的能耗阶梯**

![](images/abb01a4121dcc57ed75b45d837b40a5b5cbf247d589ab31e7213a7a78cf7898b.jpg)

- FP64 FMA `fmadd.d` 消耗 **26.7 pJ**；FP32 较其省 65%；继续缩小到 FP16/FP16alt/FP8 分别再省 48%/54%/49%。
- FP16alt 比 FP16 便宜 12%（7-bit 尾数乘法器 vs 10-bit）。
- 乘法链路每级缩小收益：65%/47%/52%/47%；加法：53%/47%/57%/47%；比较：38%/34%/35%/22%——**至少直接比例**。

**SIMD 向量指令的超比例收益**

- 直觉上 64-bit 数据通路被填满后各格式能耗应趋同，实测却相反：从 FP32 SIMD FMA `vfmac.s` 的 20.0 pJ 出发，FP16/FP16alt/FP8 逐级再省 20%/31%/20%。
- 原因：各格式拥有**独立执行单元，空闲 slice 被时钟门控**——共享数据通路方案无法达到的效果。
- 计入吞吐提升后（每指令处理更多数据），SIMD FMA 的**单数据项能耗**相对上一级格式降低 60%/66%/58%，显著优于直接比例。

**转换指令：merged 架构的能耗短板**

- 向大格式转换更便宜（输入只有部分通路翻转、输出尾数补零）；向小格式转换更贵（宽输入引起动态翻转）。
- 输入输出格式同时减半时（fcvt.s.d → fcvt.h.s → fcvt.b.h）：从 7.0 pJ 出发仅降 30%/35%——远差于算术操作，因为 CONV 是 merged 单元，未用通路难以关断。
- 向量转换部分恢复比例性：FP32/FP16 → FP16/FP8 便宜 9.5%。
- **cast-and-pack**：双 FP64 转换+打包仅比单次标量转换贵约 30%；若无此指令需两次标量 cast + 一次打包（ISA 中本不存在）。
- FP→整数转换增益有限（FP16alt/int32 vs FP32/int32 仅省 25%）；FP8/int8 向量转换因与其他格式共享 lane 甚至可能更贵。

**汇总（nominal 条件）**

| 指标 | 范围 |
|---|---|
| 标量 FMA 能耗 | 2.5–26.7 pJ |
| SIMD FMA 单数据项能耗 | 1.6–10.0 pJ |
| 标量 FP-FP 转换 | 7.0–26.7 pJ |
| 向量 FP-FP 转换 | 2.2–4.9 pJ |
| FMA 性能 | 1.85–14.83 Gflop/s |
| FMA 能效 | 75–1245 Gflop/sW |

**电压/频率缩放（0.425–1.2 V 实测峰值）**

![](images/301c1670dd641475fb49613bb18dc1e69f93629aa971752c0f930b99ddae1d71.jpg)

| 格式 | 峰值性能 | 峰值能效 |
|---|---|---|
| FP64 | 3.17 Gflop/s | 178 Gflop/sW |
| FP32 | 6.33 Gflop/s | 473 Gflop/sW |
| FP16 | 12.67 Gflop/s | 1.18 Tflop/sW |
| FP16alt | 12.67 Gflop/s | 1.38 Tflop/sW |
| FP8 | **25.33 Gflop/s** | **2.95 Tflop/sW** |

**核级视角**

![](images/cc4906d526440f24318c651b9600898cd2931531d51a6042c60b5c6adef3509f.jpg)

- FP64 FMA 中 FPU 仅占 Ariane 核总能耗的 39%，内存操作能耗与之相当。
- FP8 FMA 的 FPU 能耗比 FP64 低 10.5 倍，但核总能耗只降 38%——**核侧开销（取指、译码、寄存器堆、流水线控制）稀释了 FPU 级节能**，标量 transprecision 在应用级核中不足以释放全部节能潜力。
- SIMD 是解药：FP8 FMA 以向量形式执行时，总核能耗再降 6.2 倍——每次摊销核开销的数据项更多。

---

**六、应用案例：点积累加核**

![](images/b4e908ea25231826328373ac67326b6c0786958ab153f0465fcd5c95ba259a42.jpg) *Fig. 11. RISC-V assembly implementations of two iterations of the loop body in Fig. 10 (gray).*

任务为两输入流元素乘积累加（信号处理/SVM 常见核），输入 FP16、累加器可选 FP16 或 FP32，五种实现路径对比：

| 实现 | 精确位数 | 相对误差 | 核能耗 | 系统能耗 |
|---|---|---|---|---|
| a) 纯 FP16 标量 | 9 bit | 2.7×10⁻³ | 0.60 | 0.63 |
| b) 全 FP32 | 22 bit | 2.0×10⁻⁷ | 1.00 (基准) | 1.00 (基准) |
| c) FP16 乘 + FP32 加 | 19 bit | 1.6×10⁻⁶ | 1.16 | 1.03 |
| d) c) + SIMD 加速 | 19 bit | 1.6×10⁻⁶ | 0.97 | 0.75 |
| e) **fmacex 扩展 FMA** | 22 bit | 2.0×10⁻⁷ | **0.63** | **0.63** |

- 关键结论：
  - a) 累加舍入误差累积使结果仅 9 bit 正确，远差于 FP16 可表达的 12 bit。
  - c) 的“省能”策略适得其反：FP16 乘法比其替代的转换更耗能。
  - **e) 是最优解**：乘法在 FP16（快且省能），累加在 FP32（保精度），三条指令完成——精度与全 FP32 完全一致，性能与纯 FP16 相同，相对 FP32 基准**核与系统能耗均降 37%**，转成 FP16 后精度比纯 FP16 高 **14.6 倍**。这正是 multiformat FMA 硬件 + ISA 扩展协同的收益。
- 编程模型：`float16`/`float8` 类型与运算符由编译器透明处理；向量操作经 GCC vector extensions 自动生成 d)；fmacex/cast-and-pack 需 intrinsic 调用生成 e)。

---

**七、与 SOA 多模式 FMA 架构的对比**

| 工作 | 工艺/电压 | 格式 | 相对 FP64 能效倍数 |
|---|---|---|---|
| **本工作**（整 FPU 实测） | 22 nm / 0.8 V | FP64→FP8 向量 | **16.6×**（FP8 向量） |
| Kaiser et al. [21] | 22 nm / 0.8 V | 仅 FP64 | 基准（比本工作低 49%） |
| Manolopoulos [29] | 130 nm / 1.2 V | FP64/FP32 | 2.0× |
| Arunachalam [30] | 130 nm / 1.2 V | FP64/FP32 | 2.0× |
| Zhang [31] | 90 nm / 1 V | FP64/FP32/FP16 | 4.0× |
| Kaul [32] | 32 nm / 1.05 V | FP32/FP20/FP14 | 3.1× |
| Pu (FPMax) [33] | 28 nm / 0.8 V | FP64/FP32 独立 | ~2.5× |

- 竞品架构因重度**硬件共享**（同一 FMA 数据通路）被迫统一延迟、小格式无标量支持、能量仅直接比例；FPnew 的 parallel slice 方法使能效增益随格式缩小**逐级放大**（8 倍宽度缩减换来 16.6 倍能效），再叠加电压缩放还可再乘 2.3 倍。
- Kaul [32] 的 FP20/FP14 为“FP32 尾数截断 + 不确定性追踪重算”方案，中间格式不对应任何 IEEE 标准；FPMax [33] 的 FP32/FP64 独立实例能效比例性略低于 FPnew 的完整 FPU，进一步验证了**逐格式独立数据通路**在 FPU 全局尺度上的价值。

---

**八、生态影响与总结**

- 开源 + 可综合 SystemVerilog 使 FPnew 可直接嵌入新系统而无需额外 IP 授权；层次化架构允许外部功能单元（如 FloPoCo、Hardfloat 生成模块）替换内部单元。
- 已落地的下游应用：**Ara** 向量处理器（16 条 64-bit lane）、**Snitch** 伪双发射核（FP64 + SIMD FP32）、Montagna 等的多核 cluster 配置探索、GreenWaves 商用 IoT 处理器 **GAP9**、以及 **EPI (European Processor Initiative)** 项目的 FP 硬件。
- 核心贡献链：**并行格式切分 + 逐级时钟门控**带来超越直接比例的能量效率；**可截断的迭代除法器**与**可配置流水深度**提供硬件级精度/性能旋钮；**fmacex / cast-and-pack** 指令将多精度混合运算暴露给软件栈；最终在 22 nm FD-SOI 硅片上以 9.3% 面积、11.1% 静态能耗的代价，换取 **2.95 Tflop/sW 峰值能效与 25.33 Gflop/s 峰值性能**（FP8 SIMD），确立了对既有 SOA 在灵活性与效率上的双重领先。

### 2. 以面积换能量比例性的并行切片设计机制

**核心观点**

FPnew 的核心设计哲学是一种**以面积换能量比例性**的架构决策：通过为每种 FP 格式（FP64/FP32/FP16/FP16alt/FP8）配置**独立的并行格式切片**，并配合**细粒度 clock gating**，使得窄格式操作时未被激活的运算单元完全静默，从而实现超越“直接比例性”的能量缩放——即数据宽度减半时，单数据项能耗降幅**超过 50%**。这与传统 SOA 多模 FMA 采用**共享数据通路以压缩面积**的做法形成根本对立。

---

**一、并行切片 vs 合并切片的实现原理**

论文在 TP-FPU 的 **operation group block**（ADDMUL/DIVSQRT/COMP/CONV）内部定义了两种切片组织方式：

- **Parallel Slices（并行切片）**：
  - 每个切片**独占一种 FP 格式**，拥有独立的完整数据通路与独立流水线。
  - Lane 数量与宽度计算规则：
    - `k_parallel = ⌊w_fpu / w_f⌋`
    - `w_lane,parallel = w_f`
  - 关键收益：每种格式可**独立优化路径延迟与流水线深度**，例如 Ariane 配置中 FP64 ADDMUL 为 4 周期、FP32 为 3 周期、FP8 仅需 2 周期（见 Table II）。
  - 关键代价：**电路面积增加**——不同格式的 FMA 单元被物理复制，Ariane 中 247 kGE 的 FPU 总面积中有 **160 kGE（约 65%）来自多格式 FMA 单元**。

- **Merged Slices（合并切片）**：
  - 多种格式**共享同一硬件数据通路**，lane 数由最小格式决定：`k_merged = ⌊w_fpu / Min⌋`。
  - 面积更小，但小格式被迫复用**过尺寸数据通路**，导致**能量与延迟开销**（同一流水线深度、难以关闭未用部分）。

**二、能量比例性的来源：时钟门控与静态隔离**

- **硬件层面**：
  - 每个并行切片是独立的运算单元实例，**未参与当前运算的切片可被 clock-gated 和 datapath-gated**，消除伪开关活动。
  - Ariane 硅实现中，**超过 96% 的 FPU 寄存器被自动时钟门控**，未门控的仅剩 handshake token 与控制除法/开方的有限状态机。
  - 顶层采用**fair round-robin arbitration** 汇聚来自不同延迟路径的切片输出，配合 **valid-ready handshake**，流水线气泡可被“弹跳式”压缩，进一步允许**逐流水级的细粒度门控**。

- **架构层面（关键反直觉设计）**：
  - 论文明确指出：“Unlike SOA designs, **we increase the circuit area to achieve high energy proportionality and efficiency**"。
  - 在先进工艺节点（22 nm FD-SOI）中，**晶体管面积相对廉价，而能量昂贵**——面积增加换取的是每个窄格式操作**真实的、可门控的窄数据通路**，而非共享宽通路的“部分利用”。
  - 粗粒度门控：无有效数据时甚至可以关闭整个 operation group 或整个 TP-FPU。

**三、实测能量缩放数据（Kosmodrom 芯片，22FDX，0.8 V，25 °C）**

| 操作类型 | FP64 → FP32 | FP32 → FP16 | FP16 → FP16alt | FP16alt → FP8 |
|---|---|---|---|---|
| 标量 FMA 能耗节省 | 65% | 48% | 54% | 49% |
| SIMD FMA 单指令能耗节省 | — | 20% | 31% | 20% |
| **SIMD FMA 单数据项能耗节省** | — | **60%** | **66%** | **58%** |
| SIMD 乘法（单数据项） | — | 21% | 32% | 21% |
| SIMD 加法（单数据项） | — | 20% | 31% | 19% |

- **超越比例性的量化**：FP8 相对 FP64 宽度缩减 8 倍，能量效率增益达 **16.6×**（Table IV），远超 8× 的直接比例预期；再叠加电压缩放（VFS）后增益可进一步放大至 **2.3×** 以上。
- **峰值指标**：FP8 SIMD 达 **1244.78 Gflop/sW**（标称 0.8 V）、电压缩放后最高 **2.95 Tflop/sW**；FP64 标量为 74.83 Gflop/sW，**同工艺下比 Kaiser et al. 的专用 FP64 FMA 高出 49% 能量效率**。

**四、对照实验：合并切片为何缩放不佳**

论文用 **CONV block**（采用 merged slice）作为天然对照组，验证并行切片机制的价值：

- 标量 FP-FP 转换（如 fcvt.s.d, fcvt.h.s, fcvt.b.h）中，格式宽度整体减半仅带来 **30%–35%** 能耗下降，**远劣于直接比例性**。
- 原因：合并单元中**未使用的宽通路部分难以被关闭**，动态开关活动仍波及大部分转换电路。
- FP-Integer 向量转换中，甚至出现 FP8/int8 转换比大格式**更耗能**的倒挂现象（与其它格式的 lane 共享所致）。
- 对照结论：**并行切片 + 独立单元门控**是获得能量超比例缩放的必要条件，而非 merely 好的工艺或门控策略。

---

**五、在整体系统中的作用与输入输出关系**

- **单元接口**：每周期最多 3 个 FP 操作数输入 + 操作/格式控制信号 → 经路由分发至 4 个 operation group → 各格式切片内多个 lane 并行计算 → 结果与 IEEE 754-2008 status flags 输出，由 arbiter 公平仲裁返回。
- **SIMD 数据通路构造**：当 `w_f ≤ w_fpu/2` 时启用，Ariane 的 `w_fpu = 64 bit` 下：
  - FP32 → 2 lanes；FP16/FP16alt → 4 lanes；FP8 → 8 lanes（Table II 中 ADDMUL 分别的 L/T 为 3/2、3/4、2/8）。
- **面积代价核算**：
  - RI5CY（32 位核）：TP-FPU 占核面积 **47%（69 kGE）**，相对纯 FP32 版本面积 +29%。
  - Ariane（64 位核）：相对仅支持 "F"+"D" 标量扩展的版本，面积仅 **+9.3%**、静态能量 **+11.1%**——代价换来了 FP 吞吐最高 8 倍提升与 16.6× 能量效率缩放。
- **端到端收益（RI5CY 案例研究）**：使用 **expanding FMA**（FP16 乘法 + FP32 累加的 multiformat 操作，由 merged multiformat FMA 切片支持）+ SIMD，在保持 FP32 精度的前提下实现 **1.67× 加速**与 **37% 系统级节能**。

---

**六、设计决策的边界条件**

- **适用前提**：该机制依赖**工艺缩放后面积成本占比低**的场景（如 22 nm FD-SOI）；在存储器占主导的系统（PULPissimo 中 memory 占 92% 面积/96% 漏电）中，FPU 面积增幅在系统层面被稀释至 **<1%**。
- **不适用于热路径的权衡**：DIVSQRT 因迭代式非恢复除法器天然为 merged slice（节省面积），论文接受其**非比例性能量缩放**，理由是除法/开方在性能优化代码中出现频率低。
- **Core 级天花板**：应用级核（Ariane）中 FPU 仅占总核能耗的 39%（FP64 FMA 时），故纯标量 transprecision 的核级节能被稀释至 38%；**SIMD 向量化是放大该机制收益的关键杠杆**（FP8 FMA 向量化后再降低 6.2× 核级单数据项能耗）。

**总结**

该机制的本质是把“**精度选择**”从软件层面的数据类型映射，下沉为硬件层面的“**激活哪条物理数据通路**”：并行切片使每种 FP 格式拥有专属的、尺寸匹配的运算单元，clock gating 使未选中通路近零功耗，独立流水线深度使窄格式同时获得延迟收益。面积是显式付出的代价，回报是 16.6× 的能量效率缩放空间和 8 倍吞吐提升——这正是论文标题中 **energy-proportional transprecision computing** 的硬件根基。

### 3. RISC-V跨精度ISA扩展与编译器支持

**核心定位：ISA 扩展是软件栈与 TP-FPU 硬件能力之间的桥梁**

FPnew 的硬件价值必须通过 ISA 层暴露给软件才能兑现。作者利用 RISC-V 开放、可扩展的架构特性，构建了一条从 **C/C++ 源码 → GCC 编译器 → 自定义指令编码 → TP-FPU 硬件执行单元** 的完整纵向通路，使 transprecision（跨精度）计算不再依赖汇编级手工调优，而成为**原生编程语言能力**。

---

**一、扩展的五种 FP 格式定义**

- 在标准 "F"（binary32）与 "D"（binary64）扩展之外，新增三种窄格式，全部遵循 **IEEE 754-2008 二进制编码原则**（符号位 + 无符号指数 + 隐含前导 1 的尾数二进制小数）：

| 格式 | (指数位， 尾数位) | 总宽 | 来源 |
|---|---|---|---|
| **FP64** (binary64) | (11, 52) | 64 bit | RISC-V "D" 标准扩展 |
| **FP32** (binary32) | (8, 23) | 32 bit | RISC-V "F" 标准扩展 |
| **FP16** (binary16) | (5, 10) | 16 bit | IEEE 754 半精度 |
| **FP16alt** (bfloat16) | (8, 7) | 16 bit | 自定义半精度，同 Google TPU **BF16** 编码 |
| **FP8** (minifloat) | (5, 2) | 8 bit | 自定义 quarter-precision |

- 所有格式数据在架构层面与标准 RISC-V FP 格式**同质处理**：
  - 支持 **denormals（次正规数）**、**NaN**、全部标准**舍入模式**；
  - 窄格式数值在宽 FP 寄存器中遵循 RISC-V **NaN-boxing** 规则——高位未用比特必须置为逻辑 1；
  - 指数偏置遵循通用公式：bias = 2^(wexp−1) − 1。

- **FP16alt 与 FP16 的取舍**：FP16alt 保留 FP32 的 8 位指数（动态范围一致），牺牲尾数精度。硅片实测显示 **FP16alt 运算比 FP16 能耗低约 12%**，源于更小的尾数乘法器，这正是机器学习场景偏好 BF16 的硬件佐证。

---

**二、指令集扩展的三层结构**

指令扩展分为 **scalar（标量）**、**vectorial（向量化）**、**auxiliary（辅助）** 三组，逐层释放 TP-FPU 的硬件能力。

**1. Scalar 标量指令**

- 将 "F" 标准扩展的全部操作（算术、比较、转换、数据搬运）**逐一映射**到 FP16、FP16alt、FP8 上；
- 关键新增：**所有支持格式之间的两两转换指令**，使运行时可以按需动态调节精度，这是 transprecision 应用的核心原语；
- 精度调整完全在 FP 寄存器内完成，无需绕道内存。

**2. Vectorial 向量化指令**

- 启用条件：格式宽度 < FP 寄存器宽度（**FLEN**）。例如 FLEN=32 的核可打包 **2×FP16** 向量；FLEN=64 的核可打包 2/4/8-way SIMD 向量；
- 包含标量指令的全部向量化版本，并额外提供 **vector-scalar 变体**：标量操作数被复制到向量的每个 lane，从而支持 SIMD 矩阵乘法**无需在内存中转置矩阵**；
- **向量转换的不对称处理**（因转换前后向量长度可能改变）：
  - FLEN 32 下，2×FP16 转 FP8 时目标仅 16 bit（2 元素），只能填充 32 bit 目的寄存器的一半；
  - 反向 4×FP8 转 FP16 会产生 64 bit 结果，超出寄存器容量；
  - 解决方案：提供**独立指令分别使用向量的低半或高半**做转换，保证任意方向的精度伸缩都有确定语义。

**3. Auxiliary 辅助指令**

- **Expanding FMA（扩展乘加）**：乘法在窄的 **src_fmt** 源格式完成、加法在宽的 **dst_fmt** 目的格式完成，对应 C 原型 `dst_fmt fma(src_fmt, src_fmt, dst_fmt)`，直接映射到 TP-FPU **merged FMA slice** 的多格式硬件，一次完成“乘 + 扩展 + 累加”；
- **Cast-and-pack（转换并打包）**：利用向量化转换硬件，将两个标量操作数转换后打包为结果向量的两个元素，**运行时动态构造 SIMD 向量**；若 ISA 中无此功能，需两条标量 cast 加一条不可用的 pack 操作才能等效。

![](images/6d90228bbe42e0d4ce69d481fe9d9b6362fc5581b0e7cfc0a4672b711decd75d.jpg) *Fig. 2. Datapath block diagram of the TP-FPU with its underlying levels of hierarchy. It supports multiple FP formats, and the datapath width is configurable.*

- 上图（Fig. 2）展示的分层架构（operation group blocks → format slices → vector lanes → functional units）正是 ISA 扩展的直接硬件落点：标量/向量指令经仲裁后路由至 ADDMUL、DIVSQRT、COMP、CONV 四个操作组，未用分支通过 clock gating 静默。

---

**三、指令编码策略：Brown-field（棕地）扩展**

作者刻意避免新开 opcode 空间，采用**复用既有编码**的 brown-field 方案：

- **Scalar 指令**：因只引入新格式，直接复用标准 RISC-V FP 指令编码并适配：
  - **FP16** 使用一个 reserved format encoding 标识；
  - **FP8** 复用 quad 精度标准扩展 "Q" 的编码位（作者不 targeting 任何支持 128-bit FP 的处理器，故该空间空闲）；
  - **FP16alt** 编码为 FP16，但通过指令字中的**一个 reserved rounding mode 位**区分——同一套编码空间承载两种 16-bit 格式；
- **Vectorial 指令**：占用 integer opcode space 中大片未用区域，编码手法与 RI5CY 已有的 **DSP 扩展**一致；
- **Auxiliary 指令**：按操作对象是标量还是向量，分别落入未用的 FP 或 integer opcode space。

该策略的工程价值：
- 硬件解码器改动最小（decoder 只需扩展识别新编码）；
- 与既有工具链、调试器、模拟器的兼容性摩擦最小。

**四、核侧微架构适配**

- **RI5CY（32-bit，RV32IMFC）**：
  - 可选取消独立 FP 寄存器堆，将 FP 数据放入通用寄存器堆，借道整数路径上已有的 **DSP SIMD shuffle 功能**；
  - Load/store unit 修改预存的符号扩展电路，实现窄 FP 数据的**one-extend**（高位补 1），满足 NaN-boxing；
  - 有意**不在运算单元中强制校验 NaN-boxing**，以便在需要时把 SIMD 数据当作标量处理；
- **Ariane（64-bit，RV64GC）**：
  - 同样替换 FPU、扩展 decoder、扩展 load/store 电路；新增控制逻辑非时序关键，相对整个核的资源开销**可忽略**；
  - FLEN=64，FP64 无 SIMD 向量（w_fpu = 64 bit = FP64 宽度），其余四种格式全部支持 SIMD。

两核的 TP-FPU 配置对比（体现“同一 RTL、参数化衍生”）：

| 维度 | RI5CY (w_fpu=32) | Ariane (w_fpu=64) |
|---|---|---|
| 支持格式 | FP32/FP16/FP16alt | FP64/FP32/FP16/FP16alt/FP8 |
| ADDMUL | merged 单周期，多格式 FMA | **parallel** 分格式独立流水（2–4 级） |
| DIVSQRT | disabled（展示可裁剪性） | 迭代式 divider，每周期 3 位尾数 |
| Expanding FMA | 支持 | 不支持 |
| 时序目标 | 单周期完成（MCU 级） | ~1 GHz（每格式独立配流水级数） |

---

**五、编译器支持：三层使用模型**

格式与操作被加入 **RISC-V GCC 工具链**，形成由浅入深的三层编程模型：

- **第一层：透明类型支持**
  - 新增 **float16、float16alt、float8** 原生 C 类型；
  - 编译器透明处理标量运算，程序员像使用 `float` 一样使用窄类型，编译器自动选择对应指令（如生成 Fig. 11(a)/(b) 的代码）；
- **第二层：编译器自动向量化**
  - 向量化操作通过 **GCC vector extensions** 推断生成；
  - 编译器可直接从普通 C 代码（Fig. 10(b)）产出 SIMD 代码（Fig. 11(d)），无需程序员手写 intrinsic；
- **第三层：Intrinsic 手工调优**
  - FMA、cast-and-pack、expanding FMA 等**无法干净映射到高级语言运算符语义**的操作，通过编译器 **intrinsic 函数**调用（Fig. 10(c) 中的 `macex f16(sum, a[i], b[i])`），产出最优汇编（Fig. 11(e)）；
  - 作者明确指出：即使非 transprecision 代码，性能优化的 FP 代码也常需低层手工调优——这不是该扩展的特有缺陷，而是 FP 编程的普遍现实。

![](images/b4e908ea25231826328373ac67326b6c0786958ab153f0465fcd5c95ba259a42.jpg) *Fig. 11. RISC-V assembly implementations of two iterations of the loop body in Fig. 10 (gray).*

---

**六、端到端效果验证：点积累加案例**

输入流为 FP16、累加结果可为 FP16 或 FP32 的元素积求和内核，五种实现路径的实测数据：

| 实现 | 正确位数 | 相对误差 | 核心/系统能耗（归一化） |
|---|---|---|---|
| (a) 纯 FP16 | 9 bit | 2.7×10⁻³ | 0.60 / 0.63 |
| (b) 纯 FP32（基线） | 22 bit | 2.0×10⁻⁷ | 1.00 / 1.00 |
| (c) FP16 乘 + FP32 加 | 19 bit | 1.6×10⁻⁶ | 1.16 / 1.03 |
| (d) (c) + SIMD 加速 | 19 bit | 1.6×10⁻⁶ | 0.97 / 0.75 |
| (e) Expanding FMA (fmacex) | 22 bit | 2.0×10⁻⁷ | **0.63 / 0.63** |

关键结论：
- (a) 舍入误差累积导致精度崩塌（9 bit vs FP16 可达的 12 bit）；
- (c) 反直觉地**比 FP32 更耗能**：FP16 乘法的能耗反而高于它所替代的 cast；
- **(e) 是最优解**：`fmacex` 一条指令同时完成乘法与格式扩展累加，取得与 (a) 相同的 3 指令性能、与 相同的 22-bit 精度，结果转 FP16 后精度比纯 FP16 **高 14.6 倍**，相较 FP32 基线**核心与系统能耗均降 37%**（摘要中 1.67× 加速与 37% 系统能耗降低即源于此）。

---

**七、与商业 ISA 及标准向量的对比定位**

- **x86 AVX-512**：当时不支持非标准 FP 格式；后续 Cooper Lake 加入 BF16 时也提供了 cast-and-pack 与 expanding dot-product——与本文扩展思路殊途同归，印证了设计前瞻性；
- **ARM NEON**：通过不同 addressing view 隐式处理多格式寄存器融合，编码一致性优于本文“多编码切分向量”的方式，但本文方案无需独立 SIMD 寄存器堆；
- **ARM SVE**：BF16 仅作**存储格式**，计算时隐式转 FP32——本质是“窄存储、宽计算”，与 FPnew 的**窄计算、原生节能**路线不同；
- **RISC-V "V" 标准向量扩展草案**：Cray 式变长向量语义，其 widening FMA 与本文 `fmacex` 机制等价，但限制目标格式宽度必须恰为源格式的 2 倍，灵活性不及 FPnew 的任意格式组合；
- **MVE 式 beat 化执行**：可通过简单发射 2 或 4 条 SIMD 指令序列在本文扩展上低成本复现。

---

**总结**

该 ISA 扩展的核心设计哲学可归纳为三点：其一，**编码上最大限度 brown-field 复用**（reserved encoding、Q 扩展空间、rounding mode 复用），换取工具链与解码器的低摩擦；其二，**指令语义与 TP-FPU 微架构一一对应**——cast-and-pack 映射 CONV 块的双 lane 结构、fmacex 映射 merged FMA slice 的多格式路径、SIMD 指令映射 parallel slice 的多 lane 结构，软件抽象不引入硬件无法兑现的承诺；其三，**编译器分层暴露能力**——从透明类型到自动向量化再到 intrinsic，让不同投入程度的程序员都能分享 transprecision 红利，最终以“精度不损失、能耗降 37%”的实测结果闭环验证了整条软硬件协同路径的有效性。

### 4. 应用级RISC-V核中的全硅实现与跨格式能效实测

**核心观点**

FPnew 作为 TP-FPU 被集成进 64 位应用级 RISC-V 核 **Ariane**，并通过 **GLOBALFOUNDRIES 22FDX**（22 nm FD-SOI）工艺完成全硅流片（芯片代号 **Kosmodrom**），成为首个支持五种 FP 格式（FP64/FP32/FP16/FP16alt/FP8）的完整 transprecision FPU 硅实现。实测表明，该设计在 **0.45–1.2 V** 宽电压范围内实现了领先的能效表现：从 **178 Gflop/sW（FP64）** 到 **2.95 Tflop/sW（FP8 SIMD）**，性能覆盖 **3.2–25.3 Gflop/s**，验证了**并行格式切片 + 细粒度时钟门控**架构带来的超比例能效收益。

![](images/543257c357148523a27985f40ac667204ed0600a781e36277eb591540134688c.jpg) *Fig. 5. Die micrograph and block diagram of entire Kosmodrom chip showing the placement of the two Ariane core macros with TP-FPU.*

---

**一、硅实现架构与芯片组织**

- **Kosmodrom 测试芯片**：单芯片集成**两个功能完全相同的 Ariane + TP-FPU 核**，但采用不同工艺库与目标频率，可独立时钟与独立供电，便于对比测量：
  - 高性能核：**低阈值 8-track cell library**
  - 低功耗核：**7.5-track library**
  - 后续性能/能效分析仅聚焦高性能核
- **共享资源**：两个核共享 **1.25 MB L2 memory**、中断与 debug 等公共外设
- **格式支持**：五种 FP 格式均拥有**专属独立 datapath**，充分利用**格式相关的流水线深度差异化设计**，避免窄格式承受宽格式的时延与能耗惩罚

**FPU 配置（w_fpu = 64 bit）**

| Format | ADDMUL | DIVSQRT | COMP | CONV |
|---|---|---|---|---|
| FP64 | parallel (4,1) | merged (21,1*) | parallel (1,1) | merged (2,2*) |
| FP32 | parallel (3,2) | merged (11,0) | parallel (1,2) | merged (2,0) |
| FP16 | parallel (3,4) | merged (7,0) | parallel (1,4) | merged (2,2*) |
| FP16alt | parallel (3,4) | merged (6,0) | parallel (1,4) | merged (2,0) |
| FP8 | parallel (2,8) | merged (4,0) | parallel (1,8) | merged (2,4) |

*注：括号内为（周期数，lane 数）；带 * 的 merged lane 支持等宽及更窄的所有格式。

---

**二、设计实现流程与物理设计参数**

- **综合**：Synopsys Design Compiler，worst case 约束 **1 GHz（SSG, 0.72 V, 125°C）**，附 **20% clock uncertainty**
- **时钟门控**：自动化 clock gate 插入覆盖 **>96% 的 FPU 寄存器**；仅 handshaking token 与控制除法/开方的有限状态机保持不门控
- **Retiming**：利用综合工具的**寄存器重定时**功能优化全 FPU 流水线寄存器位置，适配目标时序
- **布局布线**：Cadence Innovus，**multimode multicorner flow** 覆盖全部温度与掩膜失准 corner，共 **8 个 corner**
- **Body biasing（体偏置）**：corner trimming 评估中，在 worst 与 typical corner 施加**前向偏置**放松关键路径压力、降低漏电——这是 **FD-SOI 工艺独有的能量调节旋钮**
- **Cell mix 优化**：大面积探索阈值电压与晶体管长度组合，最终 **74% LVT + 26% SLVT**；**86% 28 nm / 8% 24 nm / 6% 20 nm** 晶体管

**实测频率覆盖**

| 工艺角条件 | 电压/偏置 | 频率 |
|---|---|---|
| Worst (SSG, 0.8 V bias, 40/125°C) | 0.72 V | **0.96 GHz** |
| Nominal (TT, 0.45 V bias, 25°C) | 0.8 V | **1.29 GHz** |
| Best (FFG, 0 V bias, 40/125°C) | 0.88 V | **1.76 GHz** |

硅测量在无偏置 nominal 条件下达到 **923 MHz**，后续能效数据均以此为基准。

---

**三、面积开销分析**

![](images/b8de613683eb42b1c9317afc0db8f813fd63de2c8e39b10b4772271f14cc6114.jpg)

- Ariane 核（含 TP-FPU，不含 cache）总面积 **742 kGE**，含 cache 为 **1.4 MGE**
- FPU 总计 **247 kGE**，内部构成：

| 模块 | 面积 | 占 FPU 比例 |
|---|---|---|
| FMA 单元 | 160 kGE | ~65% |
| CONV 转换单元 | 47 kGE | ~19% |
| DIVSQRT 迭代单元 | 19 kGE | ~8% |
| COMP 比较/位操作 | 13 kGE | ~5% |

- 相比仅支持标量 FP32/FP64（"F"+"D" 扩展）的完整 Ariane 核，面积仅增加 **9.3%**，静态能量增加 **11.1%**
- 该代价换取了三个新格式 + SIMD 支持，FP8 向量化下 FP 操作吞吐提升最高 **8 倍**，性价比显著

---

**四、能效测量方法论（输入-输出关系）**

- **输入**：通过合成应用向每条指令注入**随机正态分布 FP 数值**，并约束避免 overflow，制造 datapath 高翻转活动的 **worst case 功耗场景**；测量在**满流水线、FPU 峰值吞吐**状态下进行，保证公平比较
- **辅助手段**：矩阵-矩阵乘法下对核与存储功耗做硅测量；结合**带全层级可见性的 calibrated post-place-and-route 仿真**（TT, 0.8 V, 25°C 典型库）分解各硬件块的相对能量贡献
- **输出**：得到**每条指令 / 每数据项的 pJ 级能耗**、各电压点下的 **Gflop/s 性能与 Gflop/sW 能效**，以及 FPU 能耗占整核能耗的比例关系

![](images/abb01a4121dcc57ed75b45d837b40a5b5cbf247d589ab31e7213a7a78cf7898b.jpg)

---

**五、跨格式能效实测结果：能量比例性**

- **标量运算（ADDMUL 块）**：能量比例性最显著，源于乘法器的主导性影响
  - FP64 FMA **fmadd.d = 26.7 pJ**；FP32 相比 FP64 节省 **65%** 能量
  - 向更窄格式递进的能量节省：FP16 **48%**、FP16alt **54%**、FP8 **49%（相对次大格式）**
  - FP16alt 相比 FP16 便宜 **12%**，得益于更小的尾数乘法器（8 位指数、7 位尾数）
  - 乘法节省 65%/47%/52%/47%，加法节省 53%/47%/57%/47%，比较节省 38%/34%/35%/22%
- **SIMD 向量运算：超比例能效**
  - 直觉上 SIMD 应因完整利用 64-bit datapath 而能耗相近，但实测**持续保持能量比例性**
  - 从 FP32 SIMD FMA **vfmac.s = 20.0 pJ** 出发，向下一级格式的每指令节省为 FP16 **20%**、FP16alt **31%**、FP8 **20%**
  - 折算**每数据项**能量节省达 **60%/66%/58%**（SIMD FMA），**优于直接比例**
  - 根本原因：**各格式独立执行单元分离** + **空闲 slice 时钟门控**，共享 datapath 架构难以企及
- **CONV 块（merged slice）：比例性较弱的反例**
  - FP→FP 标量转换：扩格式更便宜（输入仅部分翻转、输出高位补零）；缩格式更贵（宽输入触发动态切换）
  - 格式宽度减半时能量仅降 **30%–35%**，远逊于并行 slice，因 merged 单元中未用 datapath 部分难以关断
  - SIMD 转换恢复部分比例性：FP32/FP16 → FP16/FP8 转换便宜 **9.5%**
- **Cast-and-pack 指令**：两个 FP64 标量转换并打包进目的向量，能耗仅为单次 FP64 标量转换的 **~130%**；若无此指令需两次标量 cast + 一次 packing（后者 ISA 中并不存在）
- **标称条件下的绝对能耗区间**：
  - 标量 FMA：**2.5–26.7 pJ**；SIMD FMA：**1.6–10.0 pJ/数据项**
  - FP-FP cast：标量 **7.0–26.7 pJ**，向量 **2.2–4.9 pJ**
  - 对应性能 **1.85–14.83 Gflop/s**，能效 **75–1245 Gflop/sW**（923 MHz 下）

---

**六、电压/频率缩放实测**

![](images/301c1670dd641475fb49613bb18dc1e69f93629aa971752c0f930b99ddae1d71.jpg)

- 扫描 **0.425–1.2 V** 供电范围，测量每个电压点的最高频率与对应功耗
- **峰值性能与能效**：

| Format | 峰值性能 | 峰值能效 |
|---|---|---|
| FP64 | 3.17 Gflop/s | **178 Gflop/sW** |
| FP32 | 6.33 Gflop/s | **473 Gflop/sW** |
| FP16 | 12.67 Gflop/s | **1.18 Tflop/sW** |
| FP16alt | 12.67 Gflop/s | **1.38 Tflop/sW** |
| FP8 | **25.33 Gflop/s** | **2.95 Tflop/sW** |

- 关键机制：**电压缩放作为第二层能效放大器**——FP8 相对 FP64 的 16.6 倍架构性能效增益，叠加电压调节后可再放大 **2.3 倍**，形成远超简单 2:1 面积-精度折衷的综合收益

---

**七、核级视角：FPU 节能的边界与 SIMD 的放大作用**

![](images/cc4906d526440f24318c651b9600898cd2931531d51a6042c60b5c6adef3509f.jpg)

- FP64 FMA（能耗最高的 FP 操作）中，FPU 仅占 Ariane 核总能耗的 **39%**，存储操作能耗与之相当
- FP8 FMA 虽在 FPU 层面比 FP64 便宜 **10.5 倍**，但整核能耗仅降 **38%**——核侧开销（取指、译码、寄存器堆、访存）稀释了 transprecision 收益
- **SIMD 向量化是克服该瓶颈的关键**：FP8 FMA 以向量方式执行时，整核每数据项总能耗再降 **6.2 倍**，因核侧开销被多个数据项摊薄
- 结论：在应用级核中，**transprecision + SIMD 组合**才是释放最大能效的正确姿势；纯标量窄格式节能仅在嵌入式小核场景足够

---

**八、与相关工作的 FMA 对比（nominal 条件）**

| 设计 | 工艺 | 格式 | 能效 [Gflop/sW] | 相对增益 |
|---|---|---|---|---|
| **This Work（整 FPU）** | 22 nm, 0.8 V | FP64 标量 | 74.83 | 1.0× 基准 |
| **This Work** | 22 nm | FP32 标量 | 211.66 | 2.8× |
| **This Work** | 22 nm | FP8 向量 (2/8) | **1244.78** | **16.6×** |
| Kaiser et al. | 22 nm | FP64 标量 | 37.88 | 低 49% |
| Manolopoulos et al. | 130 nm | FP32 向量 | 33.05 | 2.0× |
| Zhang et al. | 90 nm | FP16 向量 | 121.76 | 4.0× |
| Kaul et al. | 32 nm | FP14 向量 | 161.11 | 3.1× |
| Pu et al. (FPMax) | 28 nm | FP32 标量 | 110.00 | 2.5× |

- 对比要点：
  - Kaiser et al. 同工艺同节点，且仅支持 FP64（连 RISC-V 必需的 FP32 都不支持），能效反低于本工作整 FPU 做 FP64 FMA
  - **多模式 FMA 设计（[29]–[31]）**：深度硬件共享迫使所有格式共用同一时延、不支持窄格式标量、能量仅达**直接比例**；本工作 FP8 相对 FP64 达 **16.6×**（宽度缩减仅 8×）
  - **Kaul et al.**：以非标准中间格式 + 确定性追踪做迭代精化，其 FP14/FP20 并非 IEEE 754 格式
  - **FPMax**：FP32/FP64 独立 datapath 无共享，其两实例间能量比例性**略低于本工作的完整 FPU**，进一步佐证了格式独立 datapath 在 FPU 整体尺度上的价值

---

**在整体工作中的定位**

- Kosmodrom 硅实测是 FPnew 架构论点的**最终实证闭环**：架构层面提出的“**用面积换能量比例性**”（parallel slices、独立流水线深度、细粒度 clock gating）在真实硅片上兑现为从 178 Gflop/sW 到 2.95 Tflop/sW 的 **16.6 倍跨格式能效跨度**
- 它同时划定了 transprecision 的系统性边界——FPU 内部节能必须配合 **SIMD 向量化**与 **AVFS（自适应电压频率调节）**才能穿透核级开销，传导至端到端系统收益，为后续 Ara（向量处理器）、Snitch、EPI 等基于 FPnew 的设计提供了可复用的架构蓝图与实测基线


---

## 4. 实验方法与实验结果

**实验设置**

论文围绕两个 RISC-V 处理器平台展开验证，形成 **“FPGA 级后端仿真 + 真实流片实测”** 的双层实验体系。

- **平台一：PULPissimo SoC（RI5CY 内核，嵌入式场景）**
  - 采用 **32 位 RV32IMFC** 的 RI5CY 核心，TP-FPU 配置为 **单周期流水**（w_fpu = 32 bit），仅启用 FP32/FP16/FP16alt 三种格式。
  - **DIVSQRT 块被刻意禁用**，以演示架构的细粒度可配置性。
  - 完整走完 **Synopsys Design Compiler 综合 + Cadence Innovus 布局布线** 流程，工艺为 **GLOBALFOUNDRIES 22FDX（22 nm FD-SOI）**，低阈值 8-track 单元库。
  - 约束条件：
    - 最坏情况：SSG, 0.59 V, 125°C，目标 150 MHz，含 20% clock uncertainty；
    - 典型情况：TT, 0.65 V, 25°C，实测达 **370 MHz**，关键路径位于存储器与核心之间的 SoC 互连。

- **平台二：Kosmodrom 芯片（Ariane 内核，应用级场景）**
  - **64 位 RV64GC** Ariane 核心，TP-FPU 支持 **五种格式**（FP64/FP32/FP16/FP16alt/FP8），配置为 **w_fpu = 64 bit**，各格式具备独立流水深度（FP64 FMA 4 级、FP8 FMA 2 级等），目标频率约 1 GHz。
  - 芯片集成 **两个功能相同的 Ariane 核心**（分别使用 8-track 高性能库与 7.5-track 低功耗库），共享 1.25 MB L2 存储。
  - 后端采用 **multimode multicorner（8 corner）** 流程；时钟门控插入覆盖率超过 **96%** 的 FPU 寄存器；流水寄存器位置经 **retiming** 优化。
  - 实测频率范围：最坏 0.96 GHz → 典型 1.29 GHz → 最好 1.76 GHz。

- **能耗测量方法论**
  - 通过合成内核逐条压测指令，输入为**随机正态分布 FP 值**（约束不溢出），构成高翻转活动的**最坏功耗场景**。
  - 芯片实测在 **无偏置典型条件（0.8 V, 0 V bias, 25°C）** 下进行，实测 923 MHz；分层能耗归因通过**带完整层级可见性的校准后布局仿真**获得（TT 库，0.8 V, 25°C）。

![](images/543257c357148523a27985f40ac667204ed0600a781e36277eb591540134688c.jpg) *Fig. 5. Die micrograph and block diagram of entire Kosmodrom chip showing the placement of the two Ariane core macros with TP-FPU.*

---

**结果数据：RI5CY / PULPissimo 系统**

- **面积开销**：
  - RI5CY + TP-FPU 核心总面积 **147 kGE**，其中 FPU 占 **69 kGE（47%）**；ADDMUL 块（merged multiformat FMA）占 FPU 面积的 **76%**。
  - 对比仅支持 FP32 的基线 RI5CY：核心面积增加 **29%**，静态能耗增加 **37%**（能耗增幅高于面积，源于单周期长路径需要更多高驱动短栅单元——20 nm 晶体管用量是核心其余部分的 2.4 倍）。
  - 但在 SoC 层面（总 5.1 MGE，存储器占面积 92%、漏电 96%），面积与静态能耗代价分别仅 **0.7% / 0.9%**。
- **能耗对比**：FP32 FMA 单指令 3.9 pJ vs 整数 MAC 1.0 pJ；系统级（含滤波内核）FP32 为 22.2 pJ/cycle，整数版本 21.2 pJ，性能相当——证明 **FP 计算在 MCU 场景下代价可负担**。

![](images/5e240019b12d46adfeac90ab30ac1ece1fe18bd2525fdff28ebc445d168790ee.jpg) *Fig. 4. Area distribution of the PULPissimo SoC and the RI5CY core (in kGE, $1 \mathrm { G E } \approx 0 . 1 9 9 \mu \mathrm { m } ^ { 2 } )$*

---

**结果数据：Kosmodrom 芯片实测**

- **面积构成**：Ariane 核心总计 **742 kGE**（含 cache 1.4 MGE），FPU 占 **247 kGE**，其中 FMA 单元 160 kGE、CONV 转换单元 47 kGE、迭代除法/开方 19 kGE。相比仅支持标量 FP32/FP64 的基线，面积/静态能耗分别仅增加 **9.3% / 11.1%**——换来三种新格式及最高 **8× SIMD 吞吐提升**。

![](images/b8de613683eb42b1c9317afc0db8f813fd63de2c8e39b10b4772271f14cc6114.jpg)

- **每指令能耗（核心发现：能量比例性）**：

| 操作类别 | FP64 | FP32 | FP16 | FP16alt | FP8 |
|---|---|---|---|---|---|
| 标量 FMA | 26.7 pJ | −65% | −48% | −54% | −49% |
| SIMD FMA（vs 上一档） | — | 20.0 pJ 基准 | −20% | −31% | −20% |
| SIMD FMA（按数据项折算） | — | — | −60% | −66% | −58% |

  - 标量窄格式操作能耗至少**直接成比例下降**；SIMD 向量操作在**满 64-bit 数据通路占用**的情况下仍呈现**超比例能耗增益**——归因于 **parallel slice 架构 + 空闲 slice 时钟门控**，共享数据通路的常规方案难以达到。
  - 标量 FMA 全格式范围 **2.5–26.7 pJ**，SIMD FMA 每数据项 **1.6–10.0 pJ**。

![](images/abb01a4121dcc57ed75b45d837b40a5b5cbf247d589ab31e7213a7a78cf7898b.jpg)

- **例外：CONV 块**。由于采用 **merged slice** 共享数据通路，格式减半时 FP-FP cast 能耗仅下降 **30%–35%**，远差于 ADDMUL 块——反向验证了 **parallel 独立数据通路是能量比例性的关键来源**。
- **cast-and-pack 指令**：两个 FP64 值转换并打包仅比单次标量转换贵约 **30%**，替代了传统方案中不可直接实现的“两次 cast + 打包”指令序列。

- **电压缩放（0.425–1.2 V 实测）**：

| 格式 | 峰值性能 | 峰值能效 |
|---|---|---|
| FP64 | 3.17 Gflop/s | 178 Gflop/sW |
| FP32 | 6.33 Gflop/s | 473 Gflop/sW |
| FP16 | 12.67 Gflop/s | 1.18 Tflop/sW |
| FP16alt | 12.67 Gflop/s | 1.38 Tflop/sW |
| FP8 | **25.33 Gflop/s** | **2.95 Tflop/sW** |

![](images/301c1670dd641475fb49613bb18dc1e69f93629aa971752c0f930b99ddae1d71.jpg)

- **核心级能耗摊销**：FP64 FMA 中 FPU 仅占整核能耗 **39%**；FP8 FMA 虽然 FPU 能耗低 10.5×，整核能耗仅降 **38%**——说明**核心侧开销**稀释了 FPU 级节能。SIMD 向量化显著缓解该问题：FP8 FMA 以向量方式执行时，整核每数据项能耗再降 **6.2×**。

![](images/cc4906d526440f24318c651b9600898cd2931531d51a6042c60b5c6adef3509f.jpg)

- **与 SOTA 多模式 FMA 对比（Table IV）**：同为 22 nm、0.8 V 下，本工作整个 FPU（非仅 FMA 块）在 FP64 FMA 上比 Kaiser et al. 的 RISC-V 专用 FMA 能效高 **49%**；相比 Manolopoulos/Arunachalam/Zhang 等共享数据通路设计，能量比例性从“直接成比例”提升至 FP8 相对 FP64 达 **16.6×**（位宽仅缩小 8×），叠加电压缩放可达 **~38×**。

---

**消融实验：Transprecision 内核案例研究**

论文未采用传统消融框架，但通过 **五段等价汇编实现的横向对比**（Table III + Fig. 11）系统隔离了各类指令/机制的贡献，本质上是面向精度-能耗权衡的消融研究。

- **任务设置**：两 FP16 输入流的逐元素乘累加（点积式累加），n 次迭代，比较最终结果精度、核心与系统能耗。基线为 FP32 全精度实现。
- **五个变体**：
  - **(a) 纯 FP16**：3 条指令/迭代，最短代码；累加舍入误差累积，精度仅 **9 bit 正确位**（相对误差 2.7×10⁻³），能耗最低（0.60/0.63 归一化）但精度不合格。
  - **(b) 纯 FP32 基线**：5 条指令/迭代（含 cast），精度 **22 bit**；核心/系统能耗设为 1.00/1.00 基准。
  - **(c) FP16 乘法 + FP32 累加**：试图以窄格式乘法省能，但精度降至 19 bit，且**核心能耗反而比 FP32 基线高 16%**——证明“局部用窄格式”若无专用指令支持（需额外 cast），可能适得其反。
  - **(d) SIMD 加速 (c) 中的 FP16 部分**：压缩至 3.5 条指令/迭代，核心能耗降 3%、系统能耗降 **25%**——隔离出 SIMD 对系统级（访存减少）的收益。
  - **(e) Expanding FMA（fmacex 混合格式指令）**：乘法在 FP16、加法在 FP32，3 条指令/迭代。
- **关键消融结论**（变体 e，最优解）：
  - 性能与 纯 FP16 版本**完全相同**（同为 3 指令/迭代）；
  - 精度与 全 FP32 版本**完全相同**（22 bit，相对误差 2.0×10⁻⁷）；
  - 结果转回 FP16 后比纯 FP16 版本**精确 14.6×**；
  - 核心/系统能耗相比 FP32 基线降低 **37%**（0.63/0.63）。

![](images/b4e908ea25231826328373ac67326b6c0786958ab153f0465fcd5c95ba259a42.jpg) *Fig. 11. RISC-V assembly implementations of two iterations of the loop body in Fig. 10 (gray).*

- **结构性消融结论**（来自能耗测量）：
  - **Parallel slice vs Merged slice**：ADDMUL（parallel）实现超比例能耗缩放，CONV（merged）仅 30%–35% 缩放——直接归因数据通路共享策略；
  - **DIVSQRT 禁用**：RI5CY 配置中禁用该块，面积/能耗影响被完整隔离，验证配置粒度；
  - **SIMD 开/关**：向量 FMA 相比标量在满数据通路占用下仍获得超比例节能（60%/66%/58% 每数据项），分离执行单元 + 时钟门控是根因；
  - **电压/体偏置调节**：0.45–1.2 V 扫描将 FP8 能效推至 2.95 Tflop/sW，验证 transprecision 与 DVFS 两个旋钮的正交叠加收益。

---

**总结**

- 实验设计采用**双平台递进验证**：PULPissimo 后端仿真覆盖嵌入式成本分析，Kosmodrom 真实流片覆盖应用级性能与能效实测，方法论上通过随机高活动输入构成最坏功耗场景，可信度较高。
- 核心数据支撑三大论点：**能量比例性**（FP8 SIMD FMA 每数据项 0.80 pJ/flop，较 FP64 提升 16.6×）、**适度开销**（整核面积 +9.3%）、**transprecision 实效性**（案例研究中同等精度下能耗 −37%）。
- 消融式分析明确了设计选择与收益的因果关系：parallel slice 与时钟门控是超比例节能的架构根基，merged slice 在 CONV 块中暴露共享数据通路的能耗缩放瓶颈，混合格式 fmacex 指令则是打通“窄格式性能 + 宽格式精度”的关键软件-硬件接口。

---

