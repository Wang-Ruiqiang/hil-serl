# RL run 索引

每一行的配置都是**从 checkpoint 参数里读出来的**（grounding query 的 slot 数、
是否有 tactile conditioner、mask_encoder 是否存在、action 维度、buffer 的 obs
keys），不是靠目录名或记忆。

"手 attention" 一列是在 08-14 的 400 帧上量的：pick 相位、14×14 token 网格、
cell 阈值 0.04、**只统计落在"是手但不是球"的格子上的质量**（球和手的 mask 在
网格上会重叠，不排除掉的话球的 attention 会被算进手里）。随机基线 0.120。
重跑：`scratchpad/hand_attn2.py`。

| 目录 | 主干 | attention 监督 | RL 输入 mask | 手 attention | 步数 | 结果 |
|---|---|---|---|---|---|---|
| `2026-07-21_resnet_maskhead_maskobs_SUCCESS` | 冻结 ResNet10 + mask2抑制 + mask feature head + raw/head 融合 | 无 ViT query；CGL 训 mask head | ✅ | — | 75k | **成功** |
| `2026-08-13_vit_noattn_maskobs_fail` | ViT 4层，**无 grounding query** | 无 | ✅ | — | 141k | 失败 |
| `2026-08-17a_cnn_v1_maskobs_fail` | task_cnn 4 block + attention_logits，tactile 走 auxiliary encoder | 离线预训练 | ✅ | — | 108k | 失败 |
| `2026-08-17b_cnn_v2_maskobs_fail` | task_cnn v2：stem + 4 block + relation_block×2 + geometry/presence 头 | 离线预训练 | ✅ | — | 76k | 失败 |
| `2026-08-18a_cnntransformer_maskobs_fail` | CNN 7 block + self_attention×2 + geometry/presence | 离线预训练 | ✅ | — | 110k | 失败 |
| `2026-08-18b_resnet_maskhead_tactilecnn_maskobs_unknown` | 同 07-21，但 tactile 换成小 CNN | CGL 训 mask head | ✅ | — | 100k | 未记录 |
| `2026-08-19_vit_attnBallBasket_maskobs_fail` | ViT + grounding query（**1 slot 无条件**） | **只有球和框，没有手** | ✅ | **0.107**（0.89×随机） | 64k | 失败 |
| `2026-08-20a_vit_attnPlusHand_noPhase_maskobs_fail` | ViT + grounding query（**1 slot 无条件**） | 球/框 + 手 0.2 | ✅ | 0.269（2.24×） | 102k | 失败 |
| `2026-08-20b_vit_attnPlusHand_phase_maskobs_SUCCESS` | ViT + grounding query（**2 slot，pick classifier 相位**） | 球/框 + 手 0.2 | ✅ | **0.324**（2.70×） | 134k | **成功** |
| `2026-08-24_vit_gaze_noMaskObs_fail` | ViT + gaze 监督 query | 操作员 gaze 热图 | ❌ | — | ckpt 已删 | 失败 |
| `2026-08-27_vit_gaze40_noMaskObs_fail` | ViT + gaze query（1 slot），08-25 数据 | gaze 热图，无手 | ❌ | — | 90k | 失败 |
| `2026-08-28_vit_gazemaskTactile_noMaskObs_fail` | ViT + gaze 选 mask 的 query（2 slot，**tactile 条件化**） | gaze 选中的球或框，无手 | ❌ | **0.091**（0.76×） | 166k | 失败 |
| `2026-08-31_vit_nohand_attention_ablation` | ViT + grounding query（2 slot，pick classifier 相位），08-25 数据 | **pick→只有球，place→只有框，无手** | ✅ | 待测 | 待跑 | **手 attention 消融** |

## 两条已确认的事实

**1. ViT 主干和 grounding query 在 RL 期间是冻结的。** 把 08-20b 的 RL
checkpoint 和它的预训练 msgpack 逐参数对比：88 个共同参数里 83 个逐字节未变，
只有 `bottleneck_dense`、`bottleneck_ln`、`spatial_learned_embeddings` 这 5 个
被 RL 训过。所以上表的"手 attention"就是预训练给的值，RL 全程没动过它。

**2. 两次失败的手 attention 都低于随机。** 08-19（0.89×）和 08-28（0.76×）分别
来自完全不同的流水线（mask 监督 vs gaze 监督），却得到几乎相同的手 attention，
而唯一成功的 ViT run 是 2.70×。08-20a 卡在 2.24× 仍然失败，说明手的 attention
大概率**必要但不充分**——它的 place 相位框 attention 只有 0.194，相位分离根本没
建立起来，那是另一个独立的失败原因。

## 命名规则

`<日期>_<主干>_<attention监督>_<RL是否输入mask>_<结果>`

同一天多个 run 用 a/b 后缀按时间排序。
