---
title: 多精度算术：乘法、点积与共享数据通路
description: 八篇多精度算术与架构论文的中文全文翻译、技术总结、通俗讲解和图表解读。
---

# 多精度算术：乘法、点积与共享数据通路

本批为用户提供的八篇论文，每篇四类报告，共 32 份；均由 Codex/GPT 翻译/解读。**MPICC 只处理论文，不翻译或发布配套 PPT。**

| 论文 | 版本 | 主要内容 | 阅读入口 |
| --- | --- | --- | --- |
| LowCost-MU | GLSVLSI 2023 | 低精度组合、向量化及浮点/定点乘法资源复用 | [全文翻译](low-cost-multiple-precision-multiplier/translation_notes.md) · [技术总结](low-cost-multiple-precision-multiplier/paper_notes.md) · [通俗讲解](low-cost-multiple-precision-multiplier/ELI5_notes.md) · [图表说明](low-cost-multiple-precision-multiplier/figs_notes.md) |
| Samsung-8K-MAC | ISSCC 2022 | 4nm 移动 SoC 的多模式统一多精度数据通路，ISSCC 会议版 | [全文翻译](samsung-8k-mac-isscc2022/translation_notes.md) · [技术总结](samsung-8k-mac-isscc2022/paper_notes.md) · [通俗讲解](samsung-8k-mac-isscc2022/ELI5_notes.md) · [图表说明](samsung-8k-mac-isscc2022/figs_notes.md) |
| BitFusion | ISCA 2018 | BitBrick 的动态位级组合、分解与阵列数据流 | [全文翻译](bitfusion/translation_notes.md) · [技术总结](bitfusion/paper_notes.md) · [通俗讲解](bitfusion/ELI5_notes.md) · [图表说明](bitfusion/figs_notes.md) |
| Jack-Unit | ISLPED 2025（用户提供作者稿） | CSM 内对齐、二维子字并行与 INT/FP/MX 统一 MAC | [全文翻译](jack-unit/translation_notes.md) · [技术总结](jack-unit/paper_notes.md) · [通俗讲解](jack-unit/ELI5_notes.md) · [图表说明](jack-unit/figs_notes.md) |
| MXDOTP | ASAP 2025（用户提供 arXiv v1） | MXFP8 点积指令、缩放、FP32 累加与流式供数 | [全文翻译](mxdotp/translation_notes.md) · [技术总结](mxdotp/paper_notes.md) · [通俗讲解](mxdotp/ELI5_notes.md) · [图表说明](mxdotp/figs_notes.md) |
| MPICC | ASP-DAC 2025 | 多类型混合精度 MAC、随机舍入与高精度时分仿真；仅论文，不翻译 PPT | [全文翻译](mpicc/translation_notes.md) · [技术总结](mpicc/paper_notes.md) · [通俗讲解](mpicc/ELI5_notes.md) · [图表说明](mpicc/figs_notes.md) |
| Fused-FP8-DOT4 | ARITH 2024 | early/late accumulation、FP32 后端复用及单次舍入 | [全文翻译](fused-fp8-dot4/translation_notes.md) · [技术总结](fused-fp8-dot4/paper_notes.md) · [通俗讲解](fused-fp8-dot4/ELI5_notes.md) · [图表说明](fused-fp8-dot4/figs_notes.md) |
| Rethinking-FP | MLSys 2021（用户提供 arXiv v1） | 基于低位内积的时间分解与浮点对齐/累加成本优化 | [全文翻译](rethinking-floating-point-overheads/translation_notes.md) · [技术总结](rethinking-floating-point-overheads/paper_notes.md) · [通俗讲解](rethinking-floating-point-overheads/ELI5_notes.md) · [图表说明](rethinking-floating-point-overheads/figs_notes.md) |

## 原文与版本说明

### LowCost-MU

Low-Cost Multiple-Precision Multiplication Unit Design For Deep Learning

字体乱码变量、模式信号及面积/功耗单位已按原 PDF 恢复。原文中 FP8 格式命名、mode 控制值及延迟措辞有不一致，译文保留原文，解读中另行指出，不自行改写硬件合同。

[原文入口](https://doi.org/10.1145/3583781.3590269)。

### Samsung-8K-MAC

A Multi-Mode 8K-MAC HW-Utilization-Aware Neural Processing Unit with a Unified Multi-Precision Datapath in 4nm Flagship Mobile SoC

本页是所提供的 ISSCC 2022 三页会议材料，不是 JSSC 2023 的扩展版本；不得将该页码冒充期刊正文页码。

[原文入口](https://doi.org/10.1109/ISSCC42614.2022.9731639)。

### BitFusion

Bit Fusion: Bit-Level Dynamically Composable Architecture for Accelerating Deep Neural Networks

采用用户提供的 ISCA 2018 作者版本。图片与数学表达保留；本文讨论整数位级组合，不将其写成原生浮点三模式容量设计。

[原文入口](https://doi.org/10.1109/ISCA.2018.00069)。

### Jack-Unit

Jack Unit: An Area- and Energy-Efficient Multiply-Accumulate (MAC) Unit Supporting Diverse Data Formats

ISLPED 2025 出版状态由作者实验室确认，技术内容和页码依据用户提供的作者 PDF；原文入口另列 arXiv:2507.04772。

[原文入口](https://arxiv.org/abs/2507.04772)。

### MXDOTP

MXDOTP: A RISC-V ISA Extension for Enabling Microscaling (MX) Floating-Point Dot Products

使用用户提供的 arXiv:2505.13159v1 作者稿，正式会议为 ASAP 2025；实际实现以 MXFP8 为重点，不将标准的所有 MX 类型都视作已实现。

[原文入口](https://doi.org/10.1109/ASAP65064.2025.00021)。

### MPICC

MPICC: Multiple-Precision Inter-Combined MAC Unit with Stochastic Rounding for Ultra-Low-Precision Training

本页仅翻译 MPICC_paper.pdf。按用户要求，MPICC_ppt.pdf 不解析、不翻译、不发布。FP8_H/FP8_L 等字体乱码按论文 PDF 修复。

[原文入口](https://doi.org/10.1145/3658617.3697641)。

### Fused-FP8-DOT4

Fused FP8 4-Way Dot Product With Scaling and FP32 Accumulation

使用用户提供的 ARITH 2024 正式版；与后续期刊扩展分开，不混用页码和实验结果。

[原文入口](https://doi.org/10.1109/ARITH61463.2024.00016)。

### Rethinking-FP

Rethinking Floating Point Overheads for Mixed Precision DNN Accelerators

按用户提供的 arXiv:2101.11748v1 全文翻译（含附录）。论文发表于 MLSys 2021；作者稿与正式会议摘要对 8-bit 对齐的措辞有差异，阅读和引用应注明版本，不静默合并。

[原文入口](https://arxiv.org/abs/2101.11748)。

## 阅读提示

先看 BitFusion 的整数位级组合，再看 Rethinking 的浮点对齐代价；Low-Cost MU、Jack Unit 和三星 NPU 展示不同共享边界；Fused FP8 DOT4、MXDOTP、MPICC 分别讨论融合点积、缩放与指令集供数、随机舍入及混合类型。数值格式、舍入和实现范围需分开核查，不能仅凭支持位宽计算可比吞吐。

[全部论文](../library.md) · [FIGNA / BitMoD](../HPCA/index.md) · [DAC 研究合集](../DAC/index.md)

## 最新补充：MFSA

**MFSA: A Multi-Format Systolic Array with Native Micro Scaling Support**（arXiv:2610.19645v1，2026-10-14 预印本）。共享多格式浮点点积 PE、脉动阵列和在线块缩放重标定（OBSR），覆盖 FP16/BF16、MXFP8 和 MXFP4。

[全文翻译](mfsa/translation_notes.md) · [技术总结](mfsa/paper_notes.md) · [通俗讲解](mfsa/ELI5_notes.md) · [图表说明](mfsa/figs_notes.md)

## 最新补充：MX-like Tensor Core

**MX-like Tensor Core for LLM Inference with NPU Integration via Hardware-Software Co-Design Exploration**（arXiv:2610.26083v1，2026-10-19 预印本）。研究块缩放格式、权重/激活精度与硬件成本的协同探索，包含专用 MAC 模板和 SNAX NPU 集成。

[全文翻译](mx-like-tensor-core/translation_notes.md) · [技术总结](mx-like-tensor-core/paper_notes.md) · [通俗讲解](mx-like-tensor-core/ELI5_notes.md) · [图表说明](mx-like-tensor-core/figs_notes.md)
