# PDF → MinerU → GLM 翻译笔记 → Quartz → GitHub Pages

这套流水线提供两个入口：

- 本地 PDF：运行 `scripts/run_paper_pipeline.sh`。
- 公开 PDF URL：在 GitHub Actions 页面运行 `Generate and publish paper notes`。

两条路径都会生成四份报告和图片，自动放入 `content/<collection>/<slug>/`。GitHub 入口还会提交生成内容、构建 Quartz 并立即部署 Pages。

## 1. 本地配置文件

真实配置只填写在：

```text
/home/songxy/workspace/paper_read/paper-notes/.env
```

需要填写：

```ini
model_provider=ZhipuAI
model=glm-5.3
model_image=glm-5v-turbo
base_url=https://open.bigmodel.cn/api/paas/v4
wire_api=chat
OPENAI_API_KEY=<你的智谱开放平台 API Key>
MODEL_REASONING_EFFORT=high
MINERU_API_TOKEN=<MinerU Token>
MINERU_BASE_URL=https://mineru.net/api/v4
```

这里沿用变量名 `OPENAI_API_KEY`，只是因为 BlaBlaPaper 使用 OpenAI-compatible 协议；实际填写的是智谱 Key。请使用智谱开放平台按量 API Key，不要使用仅限指定编程工具的 Coding Plan Key。`glm-5.3` 只处理文本且始终启用思考，`glm-5v-turbo` 用于论文图表，两者默认共用同一把 Key。

`.env` 已加入 `.gitignore`。不要用 `git add -f` 提交它。

本地运行：

```bash
cd /home/songxy/workspace/paper_read/paper-notes
mkdir -p papers
# 把 PDF 放到 papers/，例如 papers/attention-is-all-you-need.pdf
bash scripts/run_paper_pipeline.sh papers/attention-is-all-you-need.pdf \
  --collection "LLM Theories" \
  --tags "paper,transformer,attention"
```

脚本会自动创建 Python virtualenv、运行 BlaBlaPaper、导入内容并构建 `public/`。

确认结果后自动 commit 和 push：

```bash
bash scripts/run_paper_pipeline.sh papers/attention-is-all-you-need.pdf \
  --collection "LLM Theories" \
  --tags "paper,transformer,attention" \
  --publish
```

## 2. GitHub Secrets

打开：

```text
GitHub → Futuresxy/paper_read → Settings
→ Secrets and variables → Actions → Secrets → New repository secret
```

添加：

| Secret             | 必填 | 内容                             |
| ------------------ | ---- | -------------------------------- |
| `GLM_API_KEY`      | 是   | 智谱开放平台 API Key             |
| `MINERU_API_TOKEN` | 是   | MinerU API Token                 |
| `IMAGE_API_KEY`    | 否   | 只有图片模型使用不同供应商时填写 |

## 3. GitHub Variables

打开同一页面的 `Variables` 标签，添加：

| Variable                 | 示例                                   | 说明                                 |
| ------------------------ | -------------------------------------- | ------------------------------------ |
| `TEXT_MODEL`             | `glm-5.3`                              | 总结和全文翻译                       |
| `IMAGE_MODEL`            | `glm-5v-turbo`                         | 图表理解，必须支持图片               |
| `LLM_BASE_URL`           | `https://open.bigmodel.cn/api/paas/v4` | 智谱 OpenAI-compatible API 根地址    |
| `LLM_WIRE_API`           | `chat`                                 | 使用 Chat Completions 协议           |
| `MODEL_PROVIDER`         | `ZhipuAI`                              | 仅用于标识供应商                     |
| `MODEL_REASONING_EFFORT` | `high`                                 | GLM-5.3 推理强度：`low`/`high`/`max` |
| `MINERU_BASE_URL`        | `https://mineru.net/api/v4`            | MinerU API 根地址                    |
| `IMAGE_BASE_URL`         | 图片模型供应商地址                     | 可选，独立图片供应商时填写           |
| `IMAGE_WIRE_API`         | `responses` 或 `chat`                  | 可选                                 |

`BlaBlaPaper` 生成器源码已经包含在本仓库的 `BlaBlaPaper/` 目录，无需再配置或下载另一个仓库。

## 4. GitHub Pages

先打开：

```text
Settings → Actions → General → Workflow permissions
```

选择 `Read and write permissions`，让生成工作流可以把新笔记提交回 `main`。

打开：

```text
Settings → Pages → Build and deployment → Source
```

选择：

```text
GitHub Actions
```

同时确认 `quartz.config.yaml`：

```yaml
configuration:
  baseUrl: futuresxy.github.io/paper_read
```

## 5. 从 GitHub 发起论文阅读

这一步是可选的远程入口；本地 PDF 推荐按第 1 节运行。远程运行方法：

1. 找到论文的直接公开 PDF URL，例如 arXiv 的 `/pdf/...` 地址。
2. 打开仓库的 `Actions`。
3. 选择 `Generate and publish paper notes`。
4. 点击 `Run workflow`。
5. 填写：
   - `pdf_url`：直接 PDF URL；
   - `collection`：例如 `ISCA26`、`LLM Theories` 或 `misc`；
   - `tags`：英文逗号分隔。
6. 运行完成后，workflow 会自动提交 Markdown、构建 Quartz 并部署网站。

如果仓库启用了 branch protection，需要允许 GitHub Actions 写入 `main`，或者把
`ingest-paper.yml` 的提交步骤改为创建 Pull Request；默认配置采用直接提交，以实现全自动发布。

注意：GitHub Pages 是静态站点，上传和翻译发生在 GitHub Actions，不是在公开网页浏览器中执行。PDF URL 必须能由 GitHub runner 无登录访问；私有 PDF 请使用本地入口。

## 6. 生成结果

```text
content/<collection>/<slug>/
├── index.md
├── paper_notes.md
├── ELI5_notes.md
├── figs_notes.md
├── translation_notes.md
├── paper.json
└── images/
```

`index.md` 是论文入口页，四份报告分别提供技术解析、通俗讲解、图表详解和全文翻译。`paper.json` 保存元数据和报告 hash，便于追踪生成结果。
