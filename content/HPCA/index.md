---
title: HPCA：FP–INT 与位串行共享计算
description: FIGNA 与 BitMoD 的全文中文翻译、技术解读和图表说明。
---

# HPCA：FP–INT 与位串行共享计算

两篇原文均已取得，包含全文翻译、技术总结、通俗讲解及图表说明，共八份报告。正文与图表内容使用 Codex/GPT 翻译/解读，参考文献保持原文。

| 论文 | 会议 | 主要内容 | 阅读入口 |
| --- | --- | --- | --- |
| BitMoD | HPCA 2025 | 分组混合数据类型量化与位串行共享计算单元 | [全文翻译](bitmod/translation_notes.md) · [技术总结](bitmod/paper_notes.md) · [通俗讲解](bitmod/ELI5_notes.md) · [图表说明](bitmod/figs_notes.md) |
| FIGNA | HPCA 2024 | 保持 FP–INT GEMM 数值精度的整数单元、对齐与归约设计 | [全文翻译](figna/translation_notes.md) · [技术总结](figna/paper_notes.md) · [通俗讲解](figna/ELI5_notes.md) · [图表说明](figna/figs_notes.md) |

## 建议关注的原文位置

- **FIGNA：**§III-B 的误差界；§III-C 的零权重问题与 0-less 编码；§III-D 的尾数 chunk 分配/位置还原；§III-E 的阵列与 FP32 后累加。对应 PDF 第 5–8 页。数值精度声明应按原文条件审查。
- **BitMoD：**§IV-A/B 的统一位串行项和 PE；§IV-C 的行列广播、输出缓冲和列共享累加器。对应 PDF 第 6–8 页；§V-C、Table X 可核查硬件成本。附录是原文的复现说明，译文中的命令只作材料展示，本次没有执行这些模型实验。

FIGNA 是 FP 激活×INT 权重，BitMoD 是 FP16 激活×低位混合类型权重；都不能直接当作对称 BF16/FP8/FP4 三模式容量配比。

[浏览全部论文](../library.md) · [返回首页](../index.md)
