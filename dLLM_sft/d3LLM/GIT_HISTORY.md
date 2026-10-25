# d3LLM 本地 Git 历史说明

本目录是独立于 `efficient-verl-agent` 主仓库的 **嵌套 git 仓库**（主仓库 `.gitignore` 仍忽略 `temp/`）。

## 提交结构

| 提交 | 说明 |
|------|------|
| `baseline: upstream hao-ai-lab/d3LLM main` | 官方 [hao-ai-lab/d3LLM](https://github.com/hao-ai-lab/d3LLM) 浅克隆快照 |
| `local: WebShop SFT and Dream training fixes` | 本地改动（WebShop 数据/训练/评测、Dream `full_lengths` 等） |

## 常用命令

```bash
cd temp/d3llm

# 查看相对官方的本地 diff 汇总
git diff HEAD~1 --stat

# 查看相对官方的完整 diff
git diff HEAD~1

# 只看 Dream 训练脚本改动
git diff HEAD~1 -- d3llm/d3llm_DREAM/distill_2_training/d3llm_dream_train.py

# 与远程官方同步对比（需网络）
git fetch upstream
git diff upstream/main HEAD~1   # baseline 应与 upstream 接近
git diff upstream/main          # 全部本地改动 vs 最新 upstream
```

## 大文件

`cache/`、`*.pkl`、`output_model/` 等已在 `.gitignore` 中排除，不会进入 git 历史。
