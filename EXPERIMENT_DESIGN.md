# TAAC2025 广告推荐项目实验设计

## 1. 项目目标

本项目面向 TAAC2025 腾讯广告算法竞赛数据，目标是完成从官方 baseline 到增强序列推荐模型的复现、对比和消融分析。

实验重点不只是获得一个分数，而是验证以下问题：

1. 官方 baseline 在全量数据上的效果如何。
2. finalist 方案相比官方 baseline 是否有稳定提升。
3. 数据规模从小样本到全量时，模型效果和训练成本如何变化。
4. log-q correction、hard negative curriculum、global negative sampling 等模块分别贡献多少。
5. 本地 parquet 到训练缓存的数据管线是否能提升实验迭代效率。

## 2. 数据设置

本项目使用 Hugging Face 上的 `TAAC2025/TencentGR-1M` 数据集，已下载本地 parquet 并在云端重建训练格式。

### 2.1 全量数据

全量训练数据目录：

```text
data_full/
```

已生成内容包括：

```text
seq.jsonl
seq_offsets.pkl
indexer.pkl
item_feat_dict.json
eval/
```

全量规模：

| 字段 | 数量 |
|---|---:|
| 用户数 | 1,001,845 |
| item 数 | 4,783,154 |
| 最大稀疏特征 121 取值数 | 2,135,891 |
| 本地 eval 用户数 | 4,096 |

全量训练前还需要生成：

```text
data_full/cache/item_freq.csv
data_full/cache/item_last_ts.json
```

这两个文件用于 log-q correction 和负采样候选过滤，只需生成一次。

### 2.2 子集数据

为了降低消融实验成本，使用 100k 用户子集进行模块消融。

建议子集目录：

```text
data_100k/
```

100k 子集应从 `data_full` 或本地 parquet 中稳定生成，保证不同消融实验使用完全相同的数据、相同 eval、相同随机种子。

## 3. 评价指标

本项目统一使用 Top-K 推荐指标：

| 指标 | 含义 |
|---|---|
| HitRate@10 | 真实目标 item 是否出现在推荐 Top10 中 |
| NDCG@10 | 考虑命中 item 排名位置的排序指标 |
| Score | 竞赛综合分数，`0.31 * HitRate@10 + 0.69 * NDCG@10` |
| 训练时间 | 单次实验总训练耗时 |
| 推理时间 | 生成 result.json 所需耗时 |
| 显存峰值 | `nvidia-smi` 观察到的显存峰值 |
| GPU 利用率 | CUDA 利用率均值或观察区间 |

其中 Score 作为主要指标，HitRate@10 和 NDCG@10 用于解释模型是“命中更多”还是“排序更靠前”。

## 4. 总体实验设计

实验分为五类：

1. 官方 baseline 复现：建立标准对照组。
2. Finalist 模型复现：复现增强模型主结果。
3. 数据规模实验：分析 100k 到 full 的效果和成本变化。
4. 消融实验：在 100k 数据上验证关键模块贡献。
5. 数据管线工程优化：验证本地 parquet 和 cache 的复用价值。

整体原则：

- 每组实验只改变一个主要变量。
- 同一对比组保持数据、epoch、batch size、maxlen、hidden units、num blocks、num heads 一致。
- 消融实验不在全量上全部执行，避免 A100 成本过高。
- 全量实验只跑官方 baseline 和 finalist 主模型，用于最终简历结果。

## 5. 实验 1：官方 Baseline 复现

### 设计目的

官方 baseline 是项目的基础对照组，用于衡量增强模型的实际收益。

### 实验设置

| 项目 | 设置 |
|---|---|
| 代码 | `TencentAdvertisingAlgorithmCompetition/baseline_2025` |
| 数据 | `data_full` |
| 多模态特征 | 不使用 `mm_emb` |
| 训练轮数 | 按官方 baseline 推荐配置 |
| 评估 | 本地 eval + result.json |

### 结果记录表

| 实验 | 数据 | HitRate@10 | NDCG@10 | Score | 训练时间 | 推理时间 | 显存峰值 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Official Baseline | full | TBD | TBD | TBD | TBD | TBD | TBD | 官方 SASRec/Transformer baseline |

## 6. 实验 2：Finalist 模型全量复现

### 设计目的

复现增强序列推荐方案，并与官方 baseline 对比。该实验是项目主结果。

### 建议配置

| 参数 | 值 |
|---|---:|
| 数据 | `data_full` |
| batch size | 232 |
| eval batch size | 512 |
| maxlen | 101 |
| hidden units | 128 |
| num blocks | 16 |
| num heads | 8 |
| num epochs | 4 |
| item emb batch size | 8192 |
| GPU | A100/A800 80GB |

### 结果记录表

| 实验 | 数据 | HitRate@10 | NDCG@10 | Score | 训练时间 | 推理时间 | 显存峰值 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Finalist Reproduce | full | TBD | TBD | TBD | TBD | TBD | TBD | 主模型复现 |

### 与官方 baseline 对比

| 模型 | 数据 | HitRate@10 | NDCG@10 | Score | Score 提升 |
|---|---|---:|---:|---:|---:|
| Official Baseline | full | TBD | TBD | TBD | - |
| Finalist Reproduce | full | TBD | TBD | TBD | TBD |

## 7. 实验 3：数据规模实验

### 设计目的

分析训练数据规模对推荐效果、训练时间和资源消耗的影响。

### 实验设置

固定模型结构和训练参数，只改变训练用户数。

建议数据规模：

```text
100k / full
```

其中 100k 用于消融和中等规模验证，full 用于最终结果。

### 结果记录表

| 数据规模 | 用户数 | item 数 | HitRate@10 | NDCG@10 | Score | 训练时间 | 显存峰值 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 100k | 100,000 | TBD | TBD | TBD | TBD | TBD | TBD | 消融实验数据规模 |
| full | 1,001,845 | 4,783,154 | TBD | TBD | TBD | TBD | TBD | 最终主结果 |

### 预期分析

如果数据规模增大后 Score 提升，说明模型从更多用户行为序列中学习到了更稳定的 item 表示和行为转移规律。

如果 100k 到 full 提升有限，需要进一步分析是否受以下因素限制：

- 模型容量不足。
- 训练 epoch 不够。
- 候选集过大导致排序难度提升。
- eval 构造方式与官方评估存在差异。

## 8. 实验 4：100k 消融实验

### 设计目的

消融实验用于验证关键模块的实际贡献。为了控制成本，消融统一在 100k 数据集上执行，不全部跑全量。

### 控制变量

以下配置保持一致：

| 参数 | 值 |
|---|---|
| 数据 | `data_100k` |
| batch size | 与 100k 主模型一致 |
| maxlen | 与 100k 主模型一致 |
| hidden units | 与 100k 主模型一致 |
| num blocks | 与 100k 主模型一致 |
| num heads | 与 100k 主模型一致 |
| num epochs | 与 100k 主模型一致 |
| eval 数据 | 同一份 100k eval |
| 随机种子 | 固定 |

### 消融组设计

| 实验 | 改动 | 验证问题 |
|---|---|---|
| Full Model | 完整模型 | 作为 100k 消融基准 |
| w/o log-q correction | 去掉采样分布校正 | log-q correction 是否缓解热门 item 采样偏差 |
| w/o hard negative curriculum | 固定负样本策略，不随 step 调整 hard negative 难度 | 课程式 hard negative 是否提升排序质量 |
| w/o global negative sampling | 只保留 batch 内负样本或降低全局负样本比例 | 全局负样本是否提升候选区分能力 |

### 结果记录表

| 实验 | 数据 | HitRate@10 | NDCG@10 | Score | 相对 Full 下降 | 训练时间 | 显存峰值 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Full Model | 100k | TBD | TBD | TBD | - | TBD | TBD | 消融基准 |
| w/o log-q correction | 100k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| w/o hard negative curriculum | 100k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| w/o global negative sampling | 100k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 预期结论写法

如果 `w/o log-q correction` 明显下降，可以说明：

```text
热门 item 在 batch 内负采样中被过度采样，log-q correction 能校正采样分布偏差，使 logits 更接近全量 softmax 下的排序目标。
```

如果 `w/o hard negative curriculum` 明显下降，可以说明：

```text
训练中后期逐步引入更难负样本有助于提升模型对相似 item 的区分能力，从而提高 NDCG@10。
```

如果 `w/o global negative sampling` 下降，可以说明：

```text
仅依赖 batch 内负样本覆盖不足，引入全局负样本有助于提升候选空间中的泛化排序能力。
```

## 9. 实验 5：数据管线工程优化

### 设计目的

该实验用于体现工程优化能力，比较不同数据加载方式的成本。

### 对比方案

| 数据管线 | 是否访问 HF | 是否可复用 | 说明 |
|---|---|---|---|
| HF 在线读取 | 是 | 弱 | 依赖网络，请求慢且不稳定 |
| 本地 parquet 转换 | 否 | 中 | 从已下载 parquet 生成训练格式 |
| data_full + cache | 否 | 强 | 训练时直接复用 seq、indexer、item_feat 和统计 cache |

### 结果记录表

| 数据管线 | 数据准备耗时 | 训练启动耗时 | 是否依赖网络 | 磁盘占用 | 备注 |
|---|---:|---:|---|---:|---|
| HF 在线读取 | TBD | TBD | 是 | 低 | 不推荐长期实验 |
| 本地 parquet 转换 | TBD | TBD | 否 | 中 | 适合云端重建 |
| data_full + cache | TBD | TBD | 否 | 高 | 适合反复训练 |

### 工程结论

正式训练阶段应使用：

```text
local parquet -> data_full -> data_full/cache -> train
```

这样训练过程中不再访问 Hugging Face，实验复现性和稳定性更好。

## 11. 实验执行顺序

推荐按以下顺序执行：

1. 云端生成 `data_full`。
2. 生成 `data_full/cache`。
3. 从 `data_full` 生成 `data_100k`。
4. 跑 100k Full Model。
5. 跑 100k 消融实验。
6. 跑官方 baseline full。
7. 跑 finalist full。
8. 汇总所有指标，完成简历和项目报告。

正式训练入口：

```bash
bash scripts/run_taac_train.sh data_full
```

历史兼容入口 `scripts/run_hf_smoke_train.sh` 仅用于旧命令转发，后续实验记录统一使用 `TAAC_*` 环境变量和 `run_taac_train.sh`。

## 12. 最终汇总表

| 类别 | 实验 | 数据 | HitRate@10 | NDCG@10 | Score | 训练时间 | 显存峰值 | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---|
| Baseline | Official Baseline | full | TBD | TBD | TBD | TBD | TBD | 官方对照 |
| Main | Finalist Reproduce | full | TBD | TBD | TBD | TBD | TBD | 主结果 |
| Scale | Finalist | 100k | TBD | TBD | TBD | TBD | TBD | 规模对比 |
| Ablation | Full Model | 100k | TBD | TBD | TBD | TBD | TBD | 消融基准 |
| Ablation | w/o log-q correction | 100k | TBD | TBD | TBD | TBD | TBD | 模块贡献 |
| Ablation | w/o hard negative curriculum | 100k | TBD | TBD | TBD | TBD | TBD | 模块贡献 |
| Ablation | w/o global negative sampling | 100k | TBD | TBD | TBD | TBD | TBD | 模块贡献 |

## 13. 简历表述模板

实验完成后，可将项目写为：

```text
基于 TAAC2025 腾讯广告推荐数据，复现官方 SASRec/Transformer baseline 与 finalist 生成式推荐方案，在百万用户、478 万 item 的全量序列数据上完成 Top-K 推荐训练与评测。构建 local parquet -> seq.jsonl -> cache 的离线数据管线，避免训练阶段重复访问 HF 数据源；在 100k 数据上完成 log-q correction、hard negative curriculum、global negative sampling 等模块消融，分析其对 HitRate@10、NDCG@10 和综合 Score 的贡献。最终在全量数据上将 Score 从 X 提升至 Y，训练吞吐达到 Z it/s，A100 显存峰值约 M GB。
```

其中 `X/Y/Z/M` 等待正式实验结果填入。
