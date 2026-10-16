---
title: DAC 混合精度与加速器论文
description: 18 篇论文的研究内容与阅读入口，已上线 18 篇全文翻译及解读。
tags:
  - paper
  - mixed-precision
  - accelerator
---

# DAC 混合精度与加速器论文

本合集收录 DAC 目录中的 18 篇论文，覆盖精度可扩展算术电路、低精度训练与推理、加速器生成和系统仿真。目前已上线 18 篇的全文翻译、技术总结、通俗讲解和图表说明，共 72 份报告。各篇研究内容见下文。

全文翻译按原论文章节组织，保留公式、图片和引用；参考文献保留原文，便于检索。技术总结包含研究背景、核心机制、实验设置和结果分析。

## 阅读入口

| 论文 | 关注问题 | 报告 |
| --- | --- | --- |
| AMPLE | GNN 事件驱动与节点级混合精度 | [全文翻译](ample-event-driven-accelerator-for-mixed-precision-inference-of-graph-neural-networks/translation_notes.md) · [技术总结](ample-event-driven-accelerator-for-mixed-precision-inference-of-graph-neural-networks/paper_notes.md) · [通俗讲解](ample-event-driven-accelerator-for-mixed-precision-inference-of-graph-neural-networks/ELI5_notes.md) · [图表说明](ample-event-driven-accelerator-for-mixed-precision-inference-of-graph-neural-networks/figs_notes.md) |
| 机器人学习中的 MX 硬件 | Microscaling 算术与方形共享指数分组 | [全文翻译](efficient-precision-scalable-hardware-for-microscaling-mx-processing-in-robotics-learning/translation_notes.md) · [技术总结](efficient-precision-scalable-hardware-for-microscaling-mx-processing-in-robotics-learning/paper_notes.md) · [通俗讲解](efficient-precision-scalable-hardware-for-microscaling-mx-processing-in-robotics-learning/ELI5_notes.md) · [图表说明](efficient-precision-scalable-hardware-for-microscaling-mx-processing-in-robotics-learning/figs_notes.md) |
| FPnew | 可配置多格式浮点单元 | [全文翻译](fpnew-an-open-source-multiformat-floating-point-unit-architecture-for-energy-proportional-transprecision-computing/translation_notes.md) · [技术总结](fpnew-an-open-source-multiformat-floating-point-unit-architecture-for-energy-proportional-transprecision-computing/paper_notes.md) · [通俗讲解](fpnew-an-open-source-multiformat-floating-point-unit-architecture-for-energy-proportional-transprecision-computing/ELI5_notes.md) · [图表说明](fpnew-an-open-source-multiformat-floating-point-unit-architecture-for-energy-proportional-transprecision-computing/figs_notes.md) |
| FlexBlock | 多模式块浮点训练加速器 | [全文翻译](flexblock-a-flexible-dnn-training-accelerator-with-multi-mode-block-floating-point-support/translation_notes.md) · [技术总结](flexblock-a-flexible-dnn-training-accelerator-with-multi-mode-block-floating-point-support/paper_notes.md) · [通俗讲解](flexblock-a-flexible-dnn-training-accelerator-with-multi-mode-block-floating-point-support/ELI5_notes.md) · [图表说明](flexblock-a-flexible-dnn-training-accelerator-with-multi-mode-block-floating-point-support/figs_notes.md) |
| Sum-Together 乘法器与高层设计 | 子字并行乘法器与 HLS 设计空间 | [全文翻译](high-level-design-of-precision-scalable-dnn-accelerators-based-on-sum-together-multipliers/translation_notes.md) · [技术总结](high-level-design-of-precision-scalable-dnn-accelerators-based-on-sum-together-multipliers/paper_notes.md) · [通俗讲解](high-level-design-of-precision-scalable-dnn-accelerators-based-on-sum-together-multipliers/ELI5_notes.md) · [图表说明](high-level-design-of-precision-scalable-dnn-accelerators-based-on-sum-together-multipliers/figs_notes.md) |
| RaPiD | 超低精度训练与推理芯片 | [全文翻译](rapid-ai-accelerator-for-ultra-low-precision-training-and-inference/translation_notes.md) · [技术总结](rapid-ai-accelerator-for-ultra-low-precision-training-and-inference/paper_notes.md) · [通俗讲解](rapid-ai-accelerator-for-ultra-low-precision-training-and-inference/ELI5_notes.md) · [图表说明](rapid-ai-accelerator-for-ultra-low-precision-training-and-inference/figs_notes.md) |
| 精度可扩展 MAC 架构综述与基准 | MAC 分类与统一工艺下的比较 | [全文翻译](review-and-benchmarking-of-precision-scalable-multiply-accumulate-unit-architectures-for-embedded-neural-network-processing/translation_notes.md) · [技术总结](review-and-benchmarking-of-precision-scalable-multiply-accumulate-unit-architectures-for-embedded-neural-network-processing/paper_notes.md) · [通俗讲解](review-and-benchmarking-of-precision-scalable-multiply-accumulate-unit-architectures-for-embedded-neural-network-processing/ELI5_notes.md) · [图表说明](review-and-benchmarking-of-precision-scalable-multiply-accumulate-unit-architectures-for-embedded-neural-network-processing/figs_notes.md) |
| SCALE-Sim v3 | 多核脉动阵列与存储系统仿真 | [全文翻译](scale-sim-v3-a-modular-cycle-accurate-systolic-accelerator-simulator-for-end-to-end-system-analysis/translation_notes.md) · [技术总结](scale-sim-v3-a-modular-cycle-accurate-systolic-accelerator-simulator-for-end-to-end-system-analysis/paper_notes.md) · [通俗讲解](scale-sim-v3-a-modular-cycle-accurate-systolic-accelerator-simulator-for-end-to-end-system-analysis/ELI5_notes.md) · [图表说明](scale-sim-v3-a-modular-cycle-accurate-systolic-accelerator-simulator-for-end-to-end-system-analysis/figs_notes.md) |
| SEADA | 混合精度网络与空间架构联合优化 | [全文翻译](seada-an-efficient-methodology-for-optimizing-mixed-precision-dnns-on-multi-precision-spatial-architectures/translation_notes.md) · [技术总结](seada-an-efficient-methodology-for-optimizing-mixed-precision-dnns-on-multi-precision-spatial-architectures/paper_notes.md) · [通俗讲解](seada-an-efficient-methodology-for-optimizing-mixed-precision-dnns-on-multi-precision-spatial-architectures/ELI5_notes.md) · [图表说明](seada-an-efficient-methodology-for-optimizing-mixed-precision-dnns-on-multi-precision-spatial-architectures/figs_notes.md) |
| TransDot | 跨精度点积与 FP32 累加 | [全文翻译](transdot-an-area-efficient-reconfigurable-floating-point-unit-for-trans-precision-dot-product-accumulation-for-fpga-ai-engines/translation_notes.md) · [技术总结](transdot-an-area-efficient-reconfigurable-floating-point-unit-for-trans-precision-dot-product-accumulation-for-fpga-ai-engines/paper_notes.md) · [通俗讲解](transdot-an-area-efficient-reconfigurable-floating-point-unit-for-trans-precision-dot-product-accumulation-for-fpga-ai-engines/ELI5_notes.md) · [图表说明](transdot-an-area-efficient-reconfigurable-floating-point-unit-for-trans-precision-dot-product-accumulation-for-fpga-ai-engines/figs_notes.md) |
| Voyager | DNN 加速器生成与端到端设计空间探索 | [全文翻译](voyager-an-end-to-end-framework-for-design-space-exploration-and-generation-of-dnn-accelerators/translation_notes.md) · [技术总结](voyager-an-end-to-end-framework-for-design-space-exploration-and-generation-of-dnn-accelerators/paper_notes.md) · [通俗讲解](voyager-an-end-to-end-framework-for-design-space-exploration-and-generation-of-dnn-accelerators/ELI5_notes.md) · [图表说明](voyager-an-end-to-end-framework-for-design-space-exploration-and-generation-of-dnn-accelerators/figs_notes.md) |
| XtraMAC | FPGA 上整数与浮点混合精度 MAC | [全文翻译](xtramac-an-efficient-mac-architecture-for-mixed-precision-llm-inference-on-fpga/translation_notes.md) · [技术总结](xtramac-an-efficient-mac-architecture-for-mixed-precision-llm-inference-on-fpga/paper_notes.md) · [通俗讲解](xtramac-an-efficient-mac-architecture-for-mixed-precision-llm-inference-on-fpga/ELI5_notes.md) · [图表说明](xtramac-an-efficient-mac-architecture-for-mixed-precision-llm-inference-on-fpga/figs_notes.md) |
| MixFP4 | NVFP4 逐块 E2M1/E1M2 自适应与张量核协同设计 | [全文翻译](mixfp4-enhancing-nvfp4-with-adaptive-fp4-int4-block-representations/translation_notes.md) · [技术总结](mixfp4-enhancing-nvfp4-with-adaptive-fp4-int4-block-representations/paper_notes.md) · [通俗讲解](mixfp4-enhancing-nvfp4-with-adaptive-fp4-int4-block-representations/ELI5_notes.md) · [图表说明](mixfp4-enhancing-nvfp4-with-adaptive-fp4-int4-block-representations/figs_notes.md) |

| SynGPU | 词元差分位稀疏、寄存器数据复用与位串行 Tensor Core | [全文翻译](syngpu/translation_notes.md) · [技术总结](syngpu/paper_notes.md) · [通俗讲解](syngpu/ELI5_notes.md) · [图表说明](syngpu/figs_notes.md) |
| PacQ | 低比特权重打包与超非对称 GEMM 的 SIMT 数据流 | [全文翻译](pacq/translation_notes.md) · [技术总结](pacq/paper_notes.md) · [通俗讲解](pacq/ELI5_notes.md) · [图表说明](pacq/figs_notes.md) |
| BBAL | 双向块浮点量化与低精度 LLM 加速器 | [全文翻译](bbal/translation_notes.md) · [技术总结](bbal/paper_notes.md) · [通俗讲解](bbal/ELI5_notes.md) · [图表说明](bbal/figs_notes.md) |
| Pareto | 低精度数据格式和 MAC 架构的精度、面积与能耗 Pareto 前沿 | [全文翻译](low-precision-formats-mac-pareto/translation_notes.md) · [技术总结](low-precision-formats-mac-pareto/paper_notes.md) · [通俗讲解](low-precision-formats-mac-pareto/ELI5_notes.md) · [图表说明](low-precision-formats-mac-pareto/figs_notes.md) |
| Revised-MX | RMX 格式、两级量化 RMFQ 与自适应脉动阵列 RMFA | [全文翻译](revised-microscaling-rmfq-rmfa/translation_notes.md) · [技术总结](revised-microscaling-rmfq-rmfa/paper_notes.md) · [通俗讲解](revised-microscaling-rmfq-rmfa/ELI5_notes.md) · [图表说明](revised-microscaling-rmfq-rmfa/figs_notes.md) |

## 阅读路线

- **数值格式与训练**：从 FPnew 的多格式浮点单元开始，再看 RaPiD 的低精度芯片和 FlexBlock 的块浮点训练，最后读 MixFP4 的逐块 FP4/INT4 自适应。
- **算术电路**：先读精度可扩展 MAC 综述，再比较 Sum-Together、TransDot、XtraMAC 与机器人学习中的 MX 硬件。
- **系统设计**：结合 SEADA 的混合精度分配与映射、Voyager 的硬件生成以及 SCALE-Sim v3 的系统仿真。
- **不规则工作负载**：AMPLE 展示 GNN 节点级混合精度与事件驱动调度的结合。

## 各篇研究内容

### AMPLE

面向图神经网络的不规则访存和非均匀节点度分布，将事件驱动调度、异构精度聚合核心和节点预取结合起来，使节点能够独立获取和释放计算资源。

EuroMLSys，2025。 [原文 DOI](https://doi.org/10.1145/3721146.3721963)。

[全文翻译](ample-event-driven-accelerator-for-mixed-precision-inference-of-graph-neural-networks/translation_notes.md) · [技术总结](ample-event-driven-accelerator-for-mixed-precision-inference-of-graph-neural-networks/paper_notes.md) · [通俗讲解](ample-event-driven-accelerator-for-mixed-precision-inference-of-graph-neural-networks/ELI5_notes.md) · [图表说明](ample-event-driven-accelerator-for-mixed-precision-inference-of-graph-neural-networks/figs_notes.md)

### 机器人学习中的 MX 硬件

设计支持六种 MX 数据类型的精度可扩展算术单元，并使用方形共享指数分组处理反向传播中的权重，减少向量分组导致的重复存储与重量化开销。

ISLPED，2025。 [原文 DOI](https://doi.org/10.1109/ISLPED65674.2025.11261796)。

[全文翻译](efficient-precision-scalable-hardware-for-microscaling-mx-processing-in-robotics-learning/translation_notes.md) · [技术总结](efficient-precision-scalable-hardware-for-microscaling-mx-processing-in-robotics-learning/paper_notes.md) · [通俗讲解](efficient-precision-scalable-hardware-for-microscaling-mx-processing-in-robotics-learning/ELI5_notes.md) · [图表说明](efficient-precision-scalable-hardware-for-microscaling-mx-processing-in-robotics-learning/figs_notes.md)

### FPnew

提供可配置的开源浮点单元，支持多种精度、SIMD 和跨格式操作，并通过 RISC-V 处理器集成评估不同精度下的性能与能耗。

IEEE Transactions on Very Large Scale Integration (VLSI) Systems，2021。 [原文 DOI](https://doi.org/10.1109/TVLSI.2020.3044752)。

[全文翻译](fpnew-an-open-source-multiformat-floating-point-unit-architecture-for-energy-proportional-transprecision-computing/translation_notes.md) · [技术总结](fpnew-an-open-source-multiformat-floating-point-unit-architecture-for-energy-proportional-transprecision-computing/paper_notes.md) · [通俗讲解](fpnew-an-open-source-multiformat-floating-point-unit-architecture-for-energy-proportional-transprecision-computing/ELI5_notes.md) · [图表说明](fpnew-an-open-source-multiformat-floating-point-unit-architecture-for-energy-proportional-transprecision-computing/figs_notes.md)

### FlexBlock

支持激活、权重和梯度使用不同的块浮点精度，通过可重构计算映射和动态精度控制，提高不同层类型和精度组合下的训练吞吐与计算单元利用率。

IEEE Transactions on Computers，2023。 [原文 DOI](https://doi.org/10.1109/TC.2023.3253050)。

[全文翻译](flexblock-a-flexible-dnn-training-accelerator-with-multi-mode-block-floating-point-support/translation_notes.md) · [技术总结](flexblock-a-flexible-dnn-training-accelerator-with-multi-mode-block-floating-point-support/paper_notes.md) · [通俗讲解](flexblock-a-flexible-dnn-training-accelerator-with-multi-mode-block-floating-point-support/ELI5_notes.md) · [图表说明](flexblock-a-flexible-dnn-training-accelerator-with-multi-mode-block-floating-point-support/figs_notes.md)

### Sum-Together 乘法器与高层设计

比较 Sum-Together 精度可扩展乘法器，并将其用于卷积、深度卷积和全连接层的 HLS 加速器设计，研究量化、乘法器结构与并行配置之间的取舍。

IEEE Access，2024。 [原文 DOI](https://doi.org/10.1109/ACCESS.2024.3380472)。

[全文翻译](high-level-design-of-precision-scalable-dnn-accelerators-based-on-sum-together-multipliers/translation_notes.md) · [技术总结](high-level-design-of-precision-scalable-dnn-accelerators-based-on-sum-together-multipliers/paper_notes.md) · [通俗讲解](high-level-design-of-precision-scalable-dnn-accelerators-based-on-sum-together-multipliers/ELI5_notes.md) · [图表说明](high-level-design-of-precision-scalable-dnn-accelerators-based-on-sum-together-multipliers/figs_notes.md)

### RaPiD

将浮点训练和低位宽整数推理集成到同一 AI 加速器中，结合实际芯片测量与校准后的系统模型，评估数值精度缩放带来的吞吐和能效收益。

ISCA，2021。 [原文 DOI](https://doi.org/10.1109/ISCA52012.2021.00021)。

[全文翻译](rapid-ai-accelerator-for-ultra-low-precision-training-and-inference/translation_notes.md) · [技术总结](rapid-ai-accelerator-for-ultra-low-precision-training-and-inference/paper_notes.md) · [通俗讲解](rapid-ai-accelerator-for-ultra-low-precision-training-and-inference/ELI5_notes.md) · [图表说明](rapid-ai-accelerator-for-ultra-low-precision-training-and-inference/figs_notes.md)

### 精度可扩展 MAC 架构综述与基准

建立精度可扩展 MAC 架构的分类体系，在统一工艺、精度范围和性能目标下比较不同电路拓扑的面积、能耗和吞吐，解释架构之间的适用条件。

IEEE Journal on Emerging and Selected Topics in Circuits and Systems，2019。 [原文 DOI](https://doi.org/10.1109/JETCAS.2019.2950386)。

[全文翻译](review-and-benchmarking-of-precision-scalable-multiply-accumulate-unit-architectures-for-embedded-neural-network-processing/translation_notes.md) · [技术总结](review-and-benchmarking-of-precision-scalable-multiply-accumulate-unit-architectures-for-embedded-neural-network-processing/paper_notes.md) · [通俗讲解](review-and-benchmarking-of-precision-scalable-multiply-accumulate-unit-architectures-for-embedded-neural-network-processing/ELI5_notes.md) · [图表说明](review-and-benchmarking-of-precision-scalable-multiply-accumulate-unit-architectures-for-embedded-neural-network-processing/figs_notes.md)

### SCALE-Sim v3

在周期级脉动阵列仿真中加入多核、稀疏计算、数据布局、DRAM 和能耗建模，使加速器评估能够覆盖计算与存储系统的相互影响。

ISPASS，2025。 [原文 DOI](https://doi.org/10.1109/ISPASS64960.2025.00026)。

[全文翻译](scale-sim-v3-a-modular-cycle-accurate-systolic-accelerator-simulator-for-end-to-end-system-analysis/translation_notes.md) · [技术总结](scale-sim-v3-a-modular-cycle-accurate-systolic-accelerator-simulator-for-end-to-end-system-analysis/paper_notes.md) · [通俗讲解](scale-sim-v3-a-modular-cycle-accurate-systolic-accelerator-simulator-for-end-to-end-system-analysis/ELI5_notes.md) · [图表说明](scale-sim-v3-a-modular-cycle-accurate-systolic-accelerator-simulator-for-end-to-end-system-analysis/figs_notes.md)

### SEADA

把逐层量化敏感度、工作负载映射和系统成本模型结合起来，评估混合精度分配在多精度空间架构上的真实收益，兼顾模型精度与硬件约束。

ArXiv，2026。 [原文预印本](https://arxiv.org/abs/2606.27884)。

[全文翻译](seada-an-efficient-methodology-for-optimizing-mixed-precision-dnns-on-multi-precision-spatial-architectures/translation_notes.md) · [技术总结](seada-an-efficient-methodology-for-optimizing-mixed-precision-dnns-on-multi-precision-spatial-architectures/paper_notes.md) · [通俗讲解](seada-an-efficient-methodology-for-optimizing-mixed-precision-dnns-on-multi-precision-spatial-architectures/ELI5_notes.md) · [图表说明](seada-an-efficient-methodology-for-optimizing-mixed-precision-dnns-on-multi-precision-spatial-architectures/figs_notes.md)

### TransDot

在共享的可重构浮点数据通路中统一 SIMD FMA 与点积累加，使低精度乘法能够配合 FP32 累加，并分析增加硬件面积与流水级后获得的吞吐收益。

ArXiv，2026。 [原文预印本](https://arxiv.org/abs/2605.07245)。

[全文翻译](transdot-an-area-efficient-reconfigurable-floating-point-unit-for-trans-precision-dot-product-accumulation-for-fpga-ai-engines/translation_notes.md) · [技术总结](transdot-an-area-efficient-reconfigurable-floating-point-unit-for-trans-precision-dot-product-accumulation-for-fpga-ai-engines/paper_notes.md) · [通俗讲解](transdot-an-area-efficient-reconfigurable-floating-point-unit-for-trans-precision-dot-product-accumulation-for-fpga-ai-engines/ELI5_notes.md) · [图表说明](transdot-an-area-efficient-reconfigurable-floating-point-unit-for-trans-precision-dot-product-accumulation-for-fpga-ai-engines/figs_notes.md)

### Voyager

将可配置 HLS 硬件模板、数值格式与量化支持、PyTorch 编译映射和设计空间探索组合成端到端流程，生成并评估不同规模的 DNN 加速器。

ArXiv，2025。 [原文预印本](https://arxiv.org/abs/2509.15205)。

[全文翻译](voyager-an-end-to-end-framework-for-design-space-exploration-and-generation-of-dnn-accelerators/translation_notes.md) · [技术总结](voyager-an-end-to-end-framework-for-design-space-exploration-and-generation-of-dnn-accelerators/paper_notes.md) · [通俗讲解](voyager-an-end-to-end-framework-for-design-space-exploration-and-generation-of-dnn-accelerators/ELI5_notes.md) · [图表说明](voyager-an-end-to-end-framework-for-design-space-exploration-and-generation-of-dnn-accelerators/figs_notes.md)

### XtraMAC

将整数、浮点和混合精度 MAC 转换到共享的整数尾数乘法数据通路，通过动态操作数打包与 DSP 资源共享，支持不同数值类型在运行时切换。

ISCA，2026。 [原文 DOI](https://doi.org/10.1109/ISCA66397.2026.00177)。

[全文翻译](xtramac-an-efficient-mac-architecture-for-mixed-precision-llm-inference-on-fpga/translation_notes.md) · [技术总结](xtramac-an-efficient-mac-architecture-for-mixed-precision-llm-inference-on-fpga/paper_notes.md) · [通俗讲解](xtramac-an-efficient-mac-architecture-for-mixed-precision-llm-inference-on-fpga/ELI5_notes.md) · [图表说明](xtramac-an-efficient-mac-architecture-for-mixed-precision-llm-inference-on-fpga/figs_notes.md)

### MixFP4

让每个 16 元素块在 E2M1 与 E1M2 之间按重建 MSE 选择，复用 E4M3 缩放的符号位保存格式，统一解码为内部 E2M2。实验覆盖 RTN、SmoothQuant/GPTQ/SpinQuant 及 114M/476M 预训练；计算切片面积增加 3.1%、功耗增加 1.5%，尚未测量原生内核的端到端速度。

ICML 2026（PMLR 306，按论文首页），2026。[原文预印本](https://arxiv.org/abs/2605.31035)。

[全文翻译](mixfp4-enhancing-nvfp4-with-adaptive-fp4-int4-block-representations/translation_notes.md) · [技术总结](mixfp4-enhancing-nvfp4-with-adaptive-fp4-int4-block-representations/paper_notes.md) · [通俗讲解](mixfp4-enhancing-nvfp4-with-adaptive-fp4-int4-block-representations/ELI5_notes.md) · [图表说明](mixfp4-enhancing-nvfp4-with-adaptive-fp4-int4-block-representations/figs_notes.md)

## DAC 2025 新增论文

[本批论文目录与处理状态](dac2025.md)

### SynGPU

词元差分位稀疏、寄存器数据复用与位串行 Tensor Core。

[全文翻译](syngpu/translation_notes.md) · [技术总结](syngpu/paper_notes.md) · [通俗讲解](syngpu/ELI5_notes.md) · [图表说明](syngpu/figs_notes.md)

### PacQ

低比特权重打包与超非对称 GEMM 的 SIMT 数据流。

[全文翻译](pacq/translation_notes.md) · [技术总结](pacq/paper_notes.md) · [通俗讲解](pacq/ELI5_notes.md) · [图表说明](pacq/figs_notes.md)

### BBAL

双向块浮点量化与低精度 LLM 加速器。

[全文翻译](bbal/translation_notes.md) · [技术总结](bbal/paper_notes.md) · [通俗讲解](bbal/ELI5_notes.md) · [图表说明](bbal/figs_notes.md)

### Pareto

低精度数据格式和 MAC 架构的精度、面积与能耗 Pareto 前沿。

[全文翻译](low-precision-formats-mac-pareto/translation_notes.md) · [技术总结](low-precision-formats-mac-pareto/paper_notes.md) · [通俗讲解](low-precision-formats-mac-pareto/ELI5_notes.md) · [图表说明](low-precision-formats-mac-pareto/figs_notes.md)

### Revised-MX

RMX 格式、两级量化 RMFQ 与自适应脉动阵列 RMFA。

[全文翻译](revised-microscaling-rmfq-rmfa/translation_notes.md) · [技术总结](revised-microscaling-rmfq-rmfa/paper_notes.md) · [通俗讲解](revised-microscaling-rmfq-rmfa/ELI5_notes.md) · [图表说明](revised-microscaling-rmfq-rmfa/figs_notes.md)
