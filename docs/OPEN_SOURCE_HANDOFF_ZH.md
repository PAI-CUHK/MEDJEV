# MEDJEV GitHub Repo 交接说明

这是一个独立于实验工作区的公开仓库草案：

```text
D:\F\29(MEDJEV)\MEDJEV-GitHub
```

## 仓库定位

MEDJEV 是 JEV 启发的临床证据关系判断研究原型。它不输出疾病诊断概率，而是把“记录、陈述、候选语义”组成一个显式查询，返回候选之间的关系概率和校准范围。

`medjev` 负责文本证据；`sleepjev` 是独立的睡眠信号实验扩展。二者共享研究思想，但数据和模型边界分开。

## 已包含内容

- 标准 Python `src/` 目录结构。
- `pyproject.toml`、可选依赖、CLI 入口和构建配置。
- `tests/`：从原工作区迁移的 CPU-safe 测试。
- `examples/validate_query_contract.py`：不需要数据和权重的首个 smoke example。
- `docs/`：架构、安装、开发、数据模型、benchmark、发布清单。
- `.github/workflows/ci.yml`：Python 3.10/3.11/3.12 测试与构建。
- `LICENSE`、`CITATION.cff`、`CHANGELOG.md`、`CONTRIBUTING.md`、`SECURITY.md`。
- `docs/assets/`：MEDJEV logo、JEV 架构图和评测摘要图。

## 明确不包含

没有复制当前实验工作区的 dataset、artifacts、模型权重、缓存、EDF 文件、结果压缩包、服务器路径、SSH 密码或远程 GPU 脚本。

## 参考的通用工程模式

- Pydantic：明确类型、稳定 API 和文档组织。
- FastAPI：清晰包结构、示例、测试和 CI。
- pytest：可扩展的测试组织方式。
- scikit-learn：科研软件的统一接口、examples、benchmark 和长期维护。
- Requests：小型 Python 包的清晰发布边界。
- Ruff：现代静态检查与自动化质量门槛。

本仓库没有复制这些项目的代码，只借鉴公开工程实践。

## 当前状态

这是 alpha 研究软件。首次公开前仍应由项目负责人确认真实作者、仓库 URL、软件许可证、citation 信息和所有外部数据/模型的再分发条款。
