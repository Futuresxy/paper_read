# HBQ: Hierarchical Scaling Block Quantization with Hardware-Efficiency-Aware Design for Accurate LLM Inference 图表详解

### Fig. 2. Baseline design for block quantization PE with block size B. (FP2FX: floating point to fixed point converter)

![092ad0346275c93d4b37b21a7d772eb04777921544f7ef8c8755f2eb4c695a5f.jpg](images/092ad0346275c93d4b37b21a7d772eb04777921544f7ef8c8755f2eb4c695a5f.jpg)

- **图像核心内容解析**
    - 该图展示了 **Block Quantization (BQ)** 的基线 **Processing Element (PE)** 微架构设计，针对块大小为 **B** 的量化方案。
    - 整体采用 **四阶段流水线** 结构，实现了从低精度乘法到高精度累加的完整数据通路。

- **详细模块分解**

| 阶段 | 模块名称 | 功能描述 | 关键技术细节 |
| :--- | :--- | :--- | :--- |
| **(1)** | **MUL + FP2FX** | 并行乘法与格式转换 | 处理 **B** 对 **Act/Wgt** (激活值/权重) 的乘法运算；**FP2FX** (Floating Point to Fixed Point Converter) 将浮点乘积转换为定点格式，以降低后续加法树的硬件开销。 |
| **(2)** | **Adder Tree (+)** | 块内归约 (Intra-block Reduction) | 采用 **B-to-1 定点加法树**，将 B 个乘积无损地累加为一个部分和 (Partial Sum)。此阶段保持全精度以避免量化误差在累加过程中放大。 |
| **(3)** | **Dequant** | 反量化 (Dequantization) | 引入 **s_x/s_w** (FP8 缩放因子)，对块内累加结果进行缩放恢复。这是 BQ 区别于普通量化的核心步骤，补偿了低精度元素格式带来的动态范围损失。 |
| **(4)** | **FP Accum** | 跨块浮点累加 | 使用高精度浮点加法器，将当前块的反量化结果与历史部分和进行累加，最终形成通道级输出。 |

- **关键设计洞察**
    - **计算范式**：该设计体现了 **"先低精度计算，后高精度恢复"** 的思想。通过在阶段 (1)-(2) 使用紧凑的定点算术，显著降低了面积和功耗；仅在阶段 (3)-(4) 引入昂贵的浮点操作。
    - **块大小 (Block Size B) 的影响**：参数 **B** 直接决定了并行度。**增大 B** 可以分摊 (Amortize) 阶段 (3) 反量化器和阶段 (4) 浮点累加器的固定硬件开销，从而提高 **每 MAC 的能效比**；但过大的 B 会增加块内动态范围，导致量化误差上升（即 Accuracy-Efficiency Trade-off）。
    - **缩放因子路径**：**s_x/s_w** 绕过了前端的低精度乘法区，直接馈入反量化单元，这符合 BQ 的数学定义：$ \hat{x} = s_b \cdot q^{(b)} $，确保了缩放操作的高精度执行。
    - **基线定位**：作为后续 **HBQ (Hierarchical Block Quantization)** 改进的参照系，该设计采用了单层量化结构。论文后续提出的 HBQ 将在此基础上的阶段 (2) 和 (3) 之间引入 **L2 层级的微块 (Micro-block) 反量化**，以解决大块尺寸带来的精度退化问题。

### 1b0f6c70adfff6ea0e1cae12fb8bcda5f0e2488aa4d930d09945b6905da8aa0b.jpg

![1b0f6c70adfff6ea0e1cae12fb8bcda5f0e2488aa4d930d09945b6905da8aa0b.jpg](images/1b0f6c70adfff6ea0e1cae12fb8bcda5f0e2488aa4d930d09945b6905da8aa0b.jpg)

- **图表基本信息**
    - 标题：**Per Block (B=8) Distribution**（每块 B=8 分布）
    - 图表类型：**量化表示级别 vs. 数据概率密度分布图**
    - 研究对象：**Llama3-8B** 模型的激活值在 Block Size=8 条件下的统计分布
    - 核心指标：**Kurtosis（峰度）= 2.59**，表明该分布相对平坦，接近正态分布，尾部较轻

- **数据格式对比分析**

| 格式类型 | 颜色标识 | 表示级别特征 | QSNR (dB) | 动态范围特点 |
|---------|---------|------------|-----------|-------------|
| **E5M2** | 紫色 | 极度稀疏，仅覆盖极端值区域 | 30.87 | 过宽，大量级别浪费在低概率区域 |
| **E4M3** | 红色 | 较稀疏，覆盖范围仍偏广 | 33.97 | 指数位冗余，精度不足 |
| **E3M4** | 绿色 | 中等密度，开始覆盖主分布区 | 40.67 | 逐渐适配块内窄动态范围 |
| **E2M5** | 橙色 | **高密度**，紧密贴合数据分布 | 45.27 | **最优平衡**，尾数精度高 |
| **INT8** | 蓝色 | 最高密度，均匀覆盖 | 47.61 | 整数格式，无指数开销 |

- **关键视觉发现**
    - **灰色曲线**代表真实数据的概率密度函数（PDF），呈现**单峰近似正态分布**
    - **传统高指数格式（E5M2/E4M3）**的量化点（彩色圆点）主要分布在分布的**尾部低概率区域**，而在数据集中的**峰值区域反而稀疏**，导致严重的量化误差
    - **低指数格式（E2M5）**的量化点**密集分布在概率较高的中心区域**，与灰色曲线高度重合，实现了更高的量化信噪比（QSNR）

- **论文核心论据支撑**
    - 该图直观验证了 **Section IV-B** 的核心观点：在 **Block Quantization (BQ)** 场景下，由于块内动态范围已被显著压缩（Kurtosis 仅 2.59），传统为宽动态范围设计的 FP8 格式（如 E4M3/E5M2）存在**"指数位浪费、尾数位不足"**的结构性缺陷
    - **E2M5 格式**通过将比特从指数位重新分配给尾数位（Significand），在保持足够动态范围的同时大幅提升了有效精度，QSNR 较 E5M2 提升 **14.74 dB**（相对提升 47.7%）
    - 这一发现直接支撑了 HBQ 方法采用 **2-bit exponent** 作为元素格式（Element Format）的设计选择，为后续引入 **Significand Scaling (SIG)** 奠定了理论基础

- **硬件设计启示**
    - 从硬件实现角度，E2M5 的窄动态范围允许使用**紧凑的定点数据通路**（Fixed-point Datapath），其面积和功耗特性接近 INT8，同时保持了优于 E4M3 的量化质量
    - 该图解释了为何 HBQ 能够在**不增加硬件成本**的前提下，通过优化数值格式实现比 NVFP4（基于 E4M3）更高的推理精度

### ede42991881cee6911305ed85dc427c8b2a89098cd98d2d36ad85bca4383f4ae.jpg

![ede42991881cee6911305ed85dc427c8b2a89098cd98d2d36ad85bca4383f4ae.jpg](images/ede42991881cee6911305ed85dc427c8b2a89098cd98d2d36ad85bca4383f4ae.jpg)

针对论文中 **Figure 6(a)** 的详细分析如下：

- **图表标题**: Area vs. Perplexity（面积 vs. 困惑度权衡曲线）
- **坐标轴定义**:
  - **X轴**: Llama3-8B **Perplexity**（困惑度，↓ 越低表示模型精度越高）
  - **Y轴**: **Area per MAC (μm²)**（每个MAC计算单元的硬件面积，↓ 越小表示硬件效率越高）
- **评估模型**: Llama3-8B
- **固定参数**: Weight格式固定为 **FP4** (E2M1)，Scaling格式为 **FP8-scale**

| 标记/颜色 | 含义 | 代表配置 |
|---------|------|---------|
| **棕色圆点 (●)** | PoT-scale (Power-of-Two 缩放) | Block Size = 32 |
| **蓝色方块 (■)** | **FP8-scale** (浮点缩放) | 多种Block Size |
| **A4 / A5 / A8** | 激活值位宽 | 4-bit / 5-bit / 8-bit 激活量化 |
| **NVFP4** | NVIDIA NVFP4 格式基准 | Block Size = 16 |
| **MXFP4** | Microscaling FP4 格式基准 | Block Size = 32 |
| **红色虚线** | WoQ (Weight-only Quantization) 面积基线 | 约 200 μm² |


**1. Pareto最优前沿识别**
- 图表揭示了 **新的Pareto前沿**（左下角方向为"Better"），现有方法（NVFP4、MXFP4）均处于**次优位置**
- **FP8-scale**（蓝色）在相同面积下始终比 **PoT-scale**（棕色）实现**更低的Perplexity**（更高精度）

**2. 激活精度（A4→A5→A8）的边际收益递减**
- **W4A4 → W4A5**: Perplexity显著下降（精度大幅提升），面积仅小幅增加 → **性价比最高**
- **W4A5 → W4A8**: Perplexity改善有限，但面积成本上升明显 → **边际收益递减**
- **结论**: **A5 (5-bit激活)** 是精度与硬件效率的最佳平衡点

**3. Block Size的关键作用**
- 沿每条曲线从右向左移动：**Block Size增大**（从16→64→128）
- **面积持续下降**：大Block Size有效**摊薄了反量化（dequantization）和累加器的硬件开销**
- **Perplexity适度上升**：大Block Size导致块内动态范围增大，量化误差增加
- **核心矛盾**: 这是传统BQ方法的根本限制——**效率与精度的固有 trade-off**

**4. 与现有方法的对比**
- **NVFP4** (B=16): 面积约93 μm²，PPL≈6.88
- **MXFP4** (B=32): 面积约90 μm²，但PPL高达~7.98（精度严重损失）
- **HBQ目标区域** (W4A5, 大Block Size): 面积可降至**60-70 μm²**，同时保持PPL**<6.7**，**全面超越**NVFP4和MXFP4

**5. 与WoQ的跨范式对比**
- **WoQ基线** (红色虚线): 面积约200 μm²，PPL≈6.53 (AWQ)
- **HBQ-A (W4A5)**: 面积仅约**72-87 μm²**（**WoQ的1/3以下**），PPL可达**6.52-6.68**（**接近甚至匹配WoQ精度**）
- **意义**: 首次实现 **"WoQ级别的精度 + BQ级别的硬件效率"**

此DSE结果直接催生了**Hierarchical Block Quantization (HBQ)** 的设计动机：
- 采用**大Block Size (B=128)** 以最大化硬件效率（面积最小化）
- 通过引入**二级量化（L2 SIG scaling）** 补偿大Block Size带来的精度损失
- 选择**A5 (5-bit激活)** 作为最佳精度/效率平衡点
- 最终形成 **HBQ-E**（高效模式）和 **HBQ-A**（高精度模式）两种配置

### b82f8c66da597c11065ff015bea27a95139e0d79258b7642230e26314131ee0d.jpg

![b82f8c66da597c11065ff015bea27a95139e0d79258b7642230e26314131ee0d.jpg](images/b82f8c66da597c11065ff015bea27a95139e0d79258b7642230e26314131ee0d.jpg)

- **图像基本信息**
    - 图表类型：**误差分布密度图** (Error Distribution Mass Plot)
    - 分析对象：**Weight Error Distribution**（权重张量的量化误差分布）
    - 对比条件：块大小 **B=16** vs **B=128**
    - 坐标轴定义：
        - X轴：归一化数值幅度（对数尺度，范围 $10^{-2}$ 至 $10^{0}$）
        - Y轴：**SE Mass**（Squared Error Mass，平方误差质量/概率密度）

- **核心数据解读**

| 配置 | 块大小 (B) | MSE (均方误差) | 分布特征 |
|------|-----------|----------------|----------|
| 基线 | 16 | **2.08×10⁻⁶** | 误差集中在高幅值区域 |
| 大块 | 128 | **2.89×10⁻¹⁶** | 误差仍集中于高幅值区，整体略增 |

- **关键观察与洞察**
    - **误差增幅有限**：将块大小从 16 扩大至 128（**8倍增长**），权重 MSE 仅增加约 **39%**（从 2.08e-6 到 2.89e-6），表明**权重对块大小的敏感度远低于激活值**。
    - **误差空间局部性**：图中明确标注 **"Most error still comes from large magnitude"**（绝大部分误差源于大幅值元素），两条曲线的峰值均出现在 $10^{-1}$ 至 $10^{0}$ 区间。
    - **分布形态稳定性**：相比激活值在增大块尺寸时误差急剧恶化且向小数值扩散（见论文 Fig. 8a 右侧激活图），**权重的误差分布形态保持相对稳定**，未出现显著的尾部扩散。

- **对 HBQ 设计的指导意义**
    - 该分布特征直接催生了 **Significand Scaling (SIG)** 针对权重的差异化策略：由于权重误差集中在高幅值区，HBQ 对权重采用更细粒度的 **SIG₃** 缩放（提供更精细的高幅值分辨率），而非针对激活值的 SIG₁ 策略。
    - 这一观察支持了 **Hierarchical Block Quantization** 的核心论点：**权重与激活具有异质性误差特性**，需要分层、差异化的量化方案，而非统一处理。
    - 从硬件效率角度，该结果证明了采用**大块尺寸 (B=128)** 对权重量化的可行性——精度损失可控，而硬件效率（面积/能量摊销）收益显著。

### e7323aa35c11cec23927019c3dad783d28be13095ae95e15ac2421d8510ca547.jpg

![e7323aa35c11cec23927019c3dad783d28be13095ae95e15ac2421d8510ca547.jpg](images/e7323aa35c11cec23927019c3dad783d28be13095ae95e15ac2421d8510ca547.jpg)

- **图表基本信息**
    - 图表编号：Fig. 8(c)
    - 图表标题：MSE summary for different L2 scaling schemes
    - 图表类型：双 Y 轴折线图，对比不同二级（L2）缩放方案在**大块大小（L1 B=128）**条件下的量化误差表现

- **坐标轴定义**
    
    | 坐标轴 | 物理含义 | 单位/范围 |
    |--------|----------|-----------|
    | X 轴 | L2 缩放方案（从粗粒度到细粒度） | PoT → INT → SIG₁ → SIG₂ → SIG₃ |
    | 左 Y 轴 | **Act/KV MSE**（激活值/KV缓存的均方误差） | ×10⁻³ |
    | 右 Y 轴 | **Wgt MSE**（权重的均方误差） | ×10⁻⁶ |

- **数据系列说明**
    - **红色圆点线 (Act/KV MSE)**：表示激活值和 KV 缓存在不同 L2 缩放方案下的量化误差
    - **黄色方块线 (Wgt MSE)**：表示权重在不同 L2 缩放方案下的量化误差
    - **红色虚线 (L1 B16 Act/KV MSE)**：参考基线，表示使用小块大小（B=16）的一级量化时的激活/KV 误差水平
    - **黄色虚线 (L1 B16 Wgt MSE)**：参考基线，表示使用小块大小（B=16）的一级量化时的权重误差水平

- **关键数据趋势分析**

    **激活值/KV 缓存（Act/KV）误差变化趋势：**
    - **PoT 方案**：误差最高，约为 **1.4×10⁻³**，显著高于 B16 基准线（约 1.2×10⁻³）
    - **INT 方案**：误差降至约 **1.05×10⁻³**，已接近或略低于 B16 基准线
    - **SIG₁ 方案**：达到最低点，约为 **1.0×10⁻³**，**成功恢复到 B16 水平或更低**
    - **SIG₂ / SIG₃ 方案**：误差反而略有回升（SIG₂ 约 1.15×10⁻³，SIG₃ 约 1.45×10⁻³）

    **权重（Wgt）误差变化趋势：**
    - **PoT 方案**：误差最高，约为 **2.5×10⁻⁶**
    - **INT 方案**：降至约 **1.8×10⁻⁶**
    - **SIG₁ 方案**：进一步降至约 **1.65×10⁻⁶**
    - **SIG₂ 方案**：继续降至约 **1.55×10⁻⁶**
    - **SIG₃ 方案**：达到最低点，约为 **1.5×10⁻⁶**，**低于 B16 基准线（约 1.9×10⁻⁶）**

- **核心结论与论文贡献**
    - **异构性验证**：该图直观证明了**激活值和权重的误差分布具有异构性**，需要不同的 L2 缩放策略来优化
    - **Significand Scaling (SIG) 的有效性**：
        - 对于**激活值/KV 缓存**，**SIG₁**（较粗粒度）是最佳选择，能够有效补偿大块大小带来的精度损失
        - 对于**权重**，**SIG₃**（最细粒度）效果最佳，因为权重分布更集中于大幅值区域，需要更精细的缩放粒度
    - **HBQ 设计依据**：这一分析直接支撑了 HBQ 方法中**为权重和激活分别采用不同 SIG 参数**的设计决策（权重用 SIG₂&SIG₃，激活用 SIG₁）
    - **大块大小的可行性证明**：通过引入合适的 L2 SIG 缩放，**可以在使用大块大小（B=128）获得硬件效率优势的同时，维持甚至超越小块大小（B=16）的量化精度**

### (b) HBQ MAC operation

![8c91280dbaa74fe12f0fd8af61d96fdcb3c8d97580f89bea107cda710dad97a4.jpg](images/8c91280dbaa74fe12f0fd8af61d96fdcb3c8d97580f89bea107cda710dad97a4.jpg)

- **图像核心内容解析**：该图为 **HBQ (Hierarchical Block Quantization) MAC Operation** 的硬件数据流架构图，展示了分层块量化在处理单元（PE）中的具体实现方式。这是论文提出的核心硬件创新，用于解决大块尺寸（Large Block Size）带来的精度损失问题。

- **整体架构层次**：该设计采用**两级流水线式量化/反量化结构**，将传统的单级 Block Quantization 扩展为 Hierarchical 形式：
  - **L1 层（外层）**：处理大块（Block Size B = 128），负责主要的效率提升
  - **L2 层（内层）**：处理微块（Micro-block Size μB），负责精度恢复

- **详细数据流分析**：

  | 阶段 | 操作单元 | 输入数据 | 关键信号 | 功能说明 |
  |------|----------|----------|----------|----------|
  | **乘法阶段** | Mul + FP2FX | Act/Wgt (每 μB 个元素为一组) | - | 浮点乘法后转为**定点数**，确保后续低精度运算效率 |
  | **微块内归约** | 加法树 (△) | μB 个乘积结果 | - | 对微块内所有元素进行**无损定点累加** |
  | **L2 反量化** | L2 Dequant | μB psum | **k, c_a, c_w** | 使用 **Significand Scaling** 进行细粒度缩放恢复 |
  | **跨微块归约** | 加法器 (+) | 多个 L2 输出 | - | 将各微块结果合并 |
  | **L1 反量化** | L1 Dequant | B psum | **s_a, s_w** | 使用 **FP8-scale** 进行粗粒度全局缩放 |

- **L2 Dequant 模块细节**（图中绿色框标注区域）：
  - **输入**：微块部分和（μB psum）、2-bit 缩放编码（**c_a** 用于激活，**c_w** 用于权重）、1-bit 方案选择信号（**k**，仅权重需要）
  - **解码逻辑**：根据编码 **c** 和选择信号 **k**，解码出 **Significand Scaling 因子**（α_a, α_w）
  - **数学实现**：$\text{output} = \mu\text{B psum} \times \alpha_a \times \alpha_w$
  - **硬件优势**：由于 α 值域有限（SIG 缩放的特性），此处的乘法可在**窄位宽定点域**高效完成

- **L1 Dequant 模块细节**（图中橙色框标注区域）：
  - **输入**：块部分和（B psum）、FP8 格式的 L1 缩放因子（**s_a**, **s_w**）
  - **转换流程**：**FX2FP**（定点转浮点）→ **FP Mul**（浮点乘法）
  - **输出**：最终的高精度累加结果

- **关键设计创新与硬件效率权衡**：
  - **大块分摊开销**：通过设置 **B = 128**（远大于传统 NVFP4 的 16 或 MX 的 32），将 L1 反量化中的昂贵 **FP8 乘法和浮点累加器** 成本分摊到更多元素上，显著降低 **Area/MAC**
  - **定点域最大化**：在 L1 反量化之前的**所有运算**（乘法、微块内归约、L2 反量化）均保持在**定点数域**，避免过早引入浮点运算开销
  - **L2 的轻量化设计**：采用 **2-bit Significand Scaling**（而非 8-bit 整数或浮点），使 L2 反量化的额外面积开销控制在 **<10%**（HBQ-E 配置下约 9%）

- **与基线 BQ PE 的对比差异**：
  - 传统 BQ（Figure 2）：单级结构，B 个元素直接经乘法→B-to-1 加法树→反量化→FP 累加
  - **HBQ**：引入中间层，变为 **μB-to-1 加法树 → L2 反量化 → (B/μB)-to-1 加法 → L1 反量化 → FP 累加**
  - **本质变化**：用**低成本的 L2 定点反量化**替代了部分**高成本的 L1 浮点运算**，同时允许 B 增大以进一步提升效率

- **实际配置参数对应关系**：
  - **HBQ-E (Efficient)**：B=128, μB=32，适合吞吐量优先场景
  - **HBQ-A (Accurate)**：B=128, μB=8，适合精度敏感场景（如推理任务）
  - 图中 **k* 信号**（L2 scheme select for weight）体现了 HBQ 对权重和激活的**差异化处理**：权重可根据分布特性在 SIG₂ 和 SIG₃ 间动态选择，而激活固定使用 SIG₁

### Fig. 13. Dataflow illustration for HBQ accelerator.

![25c3b99c317a1c59473688b3d5b68aab86326a736db41a007bc290f6a9aeada5.jpg](images/25c3b99c317a1c59473688b3d5b68aab86326a736db41a007bc290f6a9aeada5.jpg)

- **图表核心内容解析**
    - 该图为 **HBQ (Hierarchical Block Quantization) 加速器的数据流示意图 (Dataflow Illustration)**，展示了基于 **Weight-Stationary (WS, 权重静止)** 数据流的 systolic array 架构如何在多维度上调度计算与数据移动。
    - 图表由三大部分组成：**Activation (激活值)**、**Weight/KV (权重与KV缓存)** 和 **Output (输出)**，分别对应 GEMM 运算的三个操作数维度。

- **顶层循环嵌套结构 (Loop Hierarchy)**
    - 图表顶部的伪代码定义了五层循环嵌套，决定了数据访问顺序和复用效率：
    
    | 循环层级 | 循环变量 | 范围 | 功能描述 | 硬件意义 |
    |:---:|:---:|:---:|:---:|:---:|
    | ⑤ | `i` | `[0 : T/Tk]` | **Temporal tiling along token dimension** | 沿序列长度方向的时间分片，处理长序列 |
    | ④ | `j` | `[0 : O/To]` | **Temporal tiling along output-channel dimension** | 沿输出通道方向的时间分片 |
    | ③ | `k` | `[0 : M/B)` | **Reduce over M dim, psum held in psum buffer** | 在输入特征维度 M 上进行分块归约，**部分和 (Partial Sum, Psum) 驻留在片上 Psum Buffer 中** |
    | ② | `l` | `[0 : To/N)` | **Input reuse in input buffer by To times** | 输入数据在 Input Buffer 中被复用 To 次 |
    | ① | `m` | `[0 : Tk)` | **Weight stationary (reuse weight Tk times)** | **权重静止**，每个权重被复用 Tk 次 |

    - **单周期计算量**：每个时钟周期执行 **B × N 次 MAC 操作**（B 为 Block Size，N 为 PE 数量）。

- **分模块数据流详解**

    - **Activation (激活值) 路径 (左图，粉色)**
        - 维度表示：**T (Token/时间维度)** × **M (输入通道维度)**。
        - 数据流向：激活数据从顶部加载到 Input Buffer，按 **To** 大小进行分块 (Tiling)，并在循环 ② 中被复用。
        - 关键机制：采用 **广播式 (Broadcast)** 传输，同一批激活数据同时发送给所有 N 个 PE。图中箭头 ①→②→③ 展示了数据如何流入计算单元并被消费。

    - **Weight/KV (权重与KV缓存) 路径 (中图，绿色)**
        - 维度表示：**M (输入通道)** × **N (PE数量/输出通道)**，内部包含 **Block Size B** 的细分。
        - **核心架构**：这是一个由 N 个 PE 组成的 Systolic Array。每个 PE 内部处理一个大小为 B 的数据块。
        - **Weight-Stationary 特性**：权重（或量化后的 KV Cache）预先加载到 PE 阵列中并保持静止 (Stationary)。在循环 ① 中，同一组权重会被来自不同 Token (Tk 个) 的激活数据重复使用，极大降低了权重数据的搬运功耗 (EMA)。
        - **HBQ 特有的两级计算**：对应论文中的 Hierarchical 设计，PE 内部先进行 **Micro-block (µB)** 内的归约 (Fixed-point Adder Tree)，再进行 L1 Block 级别的缩放 (Dequantization) 与浮点累加。

    - **Output (输出) 路径 (右图，蓝色)**
        - 维度表示：**T (Token/时间维度)** × **O (输出通道维度)**，以 **To** 为 Tile 大小。
        - 数据流向：PE 计算产生的部分和 (Psum) 写入 **Psum Buffer**。
        - **Psum 累加逻辑**：在循环 ③ 中，当遍历完 M 维度的所有 Block 后，Psum Buffer 中的数据完成最终归约并输出。图中箭头 ③→④→⑤ 展示了 Psum 从 Buffer 读出、累加、写回或输出的过程。

- **关键设计洞察与硬件效率关联**
    - **内存层级优化**：该数据流通过 **Input Buffer** (41.5kB) 和 **Psum Buffer** (132kB) 实现了高数据复用。特别是 Psum Buffer 的设计，配合论文提出的 **MXINT8 Partial Sum Quantization** 技术，将 Psum 存储位宽减半，等效于将 Buffer 容量翻倍，从而允许更大的 Tile Size (To, Tk)，直接减少了对外部 DRAM 的访问次数 (EMA Reduction)。
    - **端到端统一性**：由于 HBQ 对 **Weights, Activations, KV Cache** 均采用了统一的低精度格式 (如 W4A5)，Attention 操作中的 $QK^T$ 和 $SV$ 可以直接映射到该数据流中，无需切换数据路径或精度格式，实现了真正的 **End-to-End Low-Precision Inference**。
    - **并行度与吞吐量**：N=32 个 PE 并行工作，每个 PE 处理 B=128 (HBQ-E 配置) 或 B=128/µB=8 (HBQ-A 配置) 的数据块，总吞吐量达到 **4096 MACs/cycle** @ 500MHz。

- **总结**
    - Fig. 13 直观地阐释了 HBQ 加速器如何通过 **Weight-Stationary 数据流**、**多维 Tiling 策略** 以及 **分层 Buffer 管理** 来协同算法 (Hierarchical Quantization) 与硬件架构。这种设计不仅最大化了权重的复用率以降低能耗，还通过 Psum 量化技术缓解了片上存储瓶颈，是 HBQ 实现 **1.6–3.3× 系统能耗降低** 和 **1.5–3.0× 加速比** 的物理基础。

### e8709543265bec91bcdb777cb1d43db2b892cebb70aee537ee9408f88dd97a56.jpg

![e8709543265bec91bcdb777cb1d43db2b892cebb70aee537ee9408f88dd97a56.jpg](images/e8709543265bec91bcdb777cb1d43db2b892cebb70aee537ee9408f88dd97a56.jpg)

- **图表基本信息**
  - 图表编号：Fig. 17(a)
  - 图表类型：折线图（Line Plot）
  - X轴：Frequency (Hz)，范围 500M – 1G Hz
  - Y轴：Area per MAC (μm²)，范围 75 – 225 μm²
  - 研究主题：不同量化方案在**不同工作频率**下的**单MAC单元面积**变化趋势

- **数据系列识别**
  - 图例包含6条曲线，对应不同量化配置：
  
  | 方法 | 配置 | Pipeline阶段数 | 线型 |
  |------|------|---------------|------|
  | MXFP | W4A8 | 2 cycles | 虚线（蓝色，增长最快） |
  | MXFP | W4A8-PL | 3 cycles | 虚线（橙色） |
  | NVFP | W4A5 | 2 cycles | 实线（绿色） |
  | NVFP | W4A5-PL | 3 cycles | 实线（黄色/浅绿） |
  | HBQ-A | 3 cycles | 实线（红色，最低） |
  | HBQ-E | 3 cycles | 实线（深红/棕色） |

- **核心数据分析**
  - **低频区域 (500MHz)**：
    - 所有方法的**初始面积**集中在 **90–125 μm²** 区间
    - HBQ-A 表现最优，面积约 **95–100 μm²**
    - MXFP W4A8 (2-cycle) 起始面积最高，约 **120 μm²**
  
  - **中频区域 (800MHz)**：
    - **面积分化开始显现**
    - MXFP W4A8 急剧上升至 **~135 μm²**
    - HBQ-A/HBQ-E 保持平稳，分别约为 **85 μm²** 和 **100 μm²**
    - NVFP 系列维持在 **105–125 μm²**
  
  - **高频区域 (900MHz – 1GHz)**：
    - **关键差异爆发点**
    - **MXFP W4A8 (2-cycle)** 呈**指数级增长**，1GHz时达到 **~215 μm²**（增长约79%）
    - **MXFP W4A8-PL (3-cycle)** 增长相对温和，1GHz时约 **130 μm²**
    - **HBQ-A** 在1GHz时仅约 **105 μm²**，**面积效率优势显著**
    - **HBQ-E** 在1GHz时约 **120 μm²**

- **关键技术洞察**
  - **Pipeline深度对高频可扩展性的影响**：
    - 2-cycle 设计（如 MXFP W4A8, NVFP W4A5）在高频下因**时序收敛困难**，需要更大的逻辑门尺寸或更宽松的约束，导致面积激增
    - 3-cycle 设计（包括 -PL 版本和 HBQ）通过**插入额外流水线阶段**，有效缓解了关键路径压力，使面积随频率的增长更加线性可控
  
  - **HBQ的架构优势**：
    - 尽管 HBQ 采用 **3-cycle latency**（比基础 BQ 多1个周期），但在 **>800MHz** 的高频域展现出**卓越的面积效率**
    - 在 **100 μm² 的面积预算** 下，HBQ 可实现 **~1.8× 更高的工作频率**（相比 2-cycle BQ）
    - 这验证了 **Hierarchical Block Quantization** 引入的额外 L2 dequantization 逻辑开销，被大块 size (B=128) 带来的**摊销效应**所抵消

- **设计空间含义**
  - 对于**中低频应用**（≤500MHz）：2-cycle BQ（如 NVFP4）具有竞争力，面积差异不大
  - 对于**高性能加速器**（≥800MHz）：**HBQ 是更优选择**，其 3-cycle 流水线设计避免了高频下的面积爆炸
  - **Pareto 最优性**：HBQ 在 **Area-Frequency** 二维空间中构成了新的前沿面，特别是在 **>900MHz** 区域，传统 BQ 方法已无法在合理面积内实现

- **与论文结论的关联**
  - 该图直接支撑了论文 Section VII-C 中关于 **"Operating Frequency and Critical Path"** 的论述
  - 证明了 HBQ 通过 **co-design**（算法-硬件协同设计），不仅在精度-效率权衡上占优，在**物理实现层面**也具备更好的**可扩展性**
  - 为实际 ASIC/FPGA 落地提供了关键的**频率-面积预算**参考依据

### 4a4b8e1f4e930d26b56cd0f98fa67c41b23d22fa6d671f62e2e6464098e931b9.jpg

![4a4b8e1f4e930d26b56cd0f98fa67c41b23d22fa6d671f62e2e6464098e931b9.jpg](images/4a4b8e1f4e930d26b56cd0f98fa67c41b23d22fa6d671f62e2e6464098e931b9.jpg)

- **图表类型与核心指标**
    - 该图为**堆叠柱状图 (Stacked Bar Chart)****，展示在 **SeqLen=4K (序列长度4000)** 条件下，不同量化方法的 **Iso-area Speedup (等面积加速比)**。
    - **Y轴**：加速倍数 (Speedup)，基准值 (Baseline) 为 1.0x。
    - **X轴**：评估模型，包括 **Llama2-7B, Llama3-8B, Llama3.2-3B, Qwen2.5-3B, Qwen2.5-7B** 以及 **GeoMean (几何平均值)**。

- **图例与数据构成**
    - **蓝色 (Linear)**：表示线性投影层 (Projection Layers) 的加速贡献。
    - **绿色 (Attention)**：表示注意力机制 (Attention Mechanism) 的加速贡献。
    - 总高度代表该配置下的**端到端总加速比**。

- **对比方法分析**
    - **NVFP / MXFP**：作为基线 Block Quantization (BQ) 方法（对应 NVFP4 和 MXFP 格式）。
    - **NVFP-HBQ-E / MXFP-HBQ-E**：应用了本文提出的 **HBQ-E (Efficient)** 配置后的加速效果。

- **关键数据观察 (GeoMean 几何平均)**
    
| 配置方案 | 相对加速比 (约值) | 主要增益来源 |
| :--- | :--- | :--- |
| NVFP (Baseline) | **1.00x** | 基准线 |
| NVFP-HBQ-E | **1.50x** | Attention 层显著优化 |
| MXFP | **1.93x** | 整体计算效率提升 |
| MXFP-HBQ-E | **2.47x** | Linear 与 Attention 双重优化 |
| **HBQ-E (最优)** | **3.00x** | **实现最高系统级吞吐量** |

- **深度技术解读**
    - **HBQ 的核心优势**：在**相同的硅片面积 (Iso-area)** 约束下，由于 HBQ 采用了更高效的 **W4A5** 量化位宽及大块尺寸 (Large Block Size)，使得芯片能集成更多的 **MAC 计算单元**。
    - **Attention 层的突破性提升**：观察绿色部分 (Attention)，**HBQ-E 相比传统 BQ 方法在该部分的加速贡献尤为突出**。这是因为 HBQ 实现了对 **KV Cache 的低比特量化 (4-bit)**，使得原本受限于内存带宽的解码阶段 (Decode Phase) 变得更加高效。
    - **长序列场景 (Long Context) 敏感性**：在 SeqLen=4K 的长文本生成场景中，**KV Cache 的内存访问开销占据主导地位**。HBQ 通过压缩 KV Cache 位宽，直接降低了 DRAM 访问延迟和能耗，从而在长序列任务中获得接近 **3 倍的整体速度提升**。

- **结论总结**
    - 该图有力地证明了 **HBQ-E 在实际部署中的系统级优势**：它不仅在 MAC 单元层面提高了面积/能效比 (如论文 Fig. 14 所示)，更重要的是在**完整的 LLM 推理流程**（尤其是包含大量 Attention 操作的长序列生成任务）中，实现了 **1.5× 至 3.0× 的实际运行速度提升**。

