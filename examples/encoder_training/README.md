# Offline pretraining for the `vit-grounded` encoder

Pretrains the exact `ViTImageEncoder` module that HIL-RL loads, so the
checkpoint drops straight into the RL agent.

## Data policy

**Only the demos recorded at t=0 are used** —
`/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick_and_place`
(42 episodes / 10921 frames, 3-class masks already labelled).

Replay buffers from previous RL runs are deliberately **not** used. They were
produced by the method being compared against, so pretraining on them would be
circular and would invite the "why not just do imitation learning" objection.
Task-agnostic ImageNet weights are fine — the ResNet baseline already uses
frozen ImageNet ResNet-10, so giving the ViT the same kind of prior is what
makes the comparison about architecture rather than about pretraining.

Continuing to ground on the **current run's own** replay buffer during RL is
also fine, and happens automatically: the CGL loss runs on every learner step.

## Architecture

```text
128x128 RGB (exactly what the env produces)
-> bilinear upsample to 256x256      (keeps a pretrained-compatible 16x16 patch)
-> patch embedding, 16x16 grid = 256 tokens
-> N Transformer encoder blocks
-> SpatialLearnedEmbeddings readout  (position-aware, same as the ResNet path)
   + grounding-query summary
-> Dense(256) + LayerNorm + tanh
```

The grounding query cross-attends to the patch tokens and exposes its
attention logits. That map is what the CGL mask loss supervises, both here and
during RL.

## Objective

| term | weight | why |
|---|---|---|
| grounding KL (ball, pick-phase only) | 1.0 | identical to the RL-time CGL loss |
| segmentation BCE+Dice (ball/hand/basket) | 1.0 | pretrain-only head; makes the trunk represent all three objects |
| geometry (centers / areas / relations) | 1.0 | forces position into the 256D readout |
| presence | 0.1 | |
| inverse dynamics | **0.5** | the only term forcing control-relevant information into the code |

**No temporal-invariance term.** The old CNN encoder rewarded representations
for *not* changing between adjacent frames, which is the opposite of what the
RL critic needs.

`--frame_stride` defaults to **1**: all 10921 legitimately collected frames,
not the 1/5 subsample the previous encoder used.

## Usage

```bash
# pretrain (from scratch)
bash examples/encoder_training/run_train_encoder.sh

# pretrain from ImageNet weights (requires matching dims, e.g. ViT-S/16
# means --vit_hidden_dim 384 --vit_num_layers 12)
bash examples/encoder_training/run_train_encoder.sh \
    --imagenet_init /path/to/ViT-S_16.npz \
    --vit_hidden_dim 384 --vit_num_layers 12

# check that grounding actually happened before spending robot time
python examples/encoder_training/visualize_encoder.py \
    --run_dir examples/encoder_training/runs/tennis_ball_pick_and_place_vit_grounded
```

Checkpoints hold both `params` (the ViT subtree the RL agent loads) and
`full_params` (everything, for visualization/resume).
