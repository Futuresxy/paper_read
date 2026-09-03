# Paper Notes

论文阅读笔记公开存档，使用 [Quartz v5](https://quartz.jzhao.xyz/) 渲染，部署在 GitHub Pages。

站点地址：<https://futuresxy.github.io/paper_read/>

## 更新笔记

把 PDF 放入本地 `papers/` 后，运行：

```bash
cd /home/songxy/workspace/paper_read/paper-notes
bash scripts/run_paper_pipeline.sh papers/your-paper.pdf \
  --collection "论文分类" \
  --tags "paper,研究方向" \
  --publish
```

这条命令会依次调用 MinerU、GLM-5.3 和 GLM-5V-Turbo，生成笔记后提交到 GitHub，并触发 GitHub Pages 部署。首次运行前按 [PIPELINE_SETUP.md](PIPELINE_SETUP.md) 填写本地 `.env`。

## 自动生成论文翻译与笔记

仓库已经接入 BlaBlaPaper 流水线，可从本地 PDF 或 GitHub Actions 的公开 PDF URL 自动生成：

- 技术解析 `paper_notes.md`
- 通俗讲解 `ELI5_notes.md`
- 图表详解 `figs_notes.md`
- 全文翻译 `translation_notes.md`

生成内容会自动进入 `content/<分类>/<论文 slug>/`，最终只公开上述四份 Markdown 和 `images/`；PDF、日志、JSON、checkpoint 与解析中间文件不会提交。随后由 Quartz 构建并发布。`BlaBlaPaper/` 生成器已直接收录在本仓库中，不需要手工移动结果文件。完整配置和使用方法见 [PIPELINE_SETUP.md](PIPELINE_SETUP.md)。
