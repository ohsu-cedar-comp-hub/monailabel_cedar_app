import os
import sys
from ckpt_utils import embedLoRAWeights, extractLoRAWeights, mergeLoRAWeights


mode = 'embed'
model_ckpt = './logs/v1-txt2img/checkpoints/last-003.ckpt'
lora_ckpt = './lora.ckpt'
lora_ckpts = [lora_ckpt]
out_ckpt = './embedded.ckpt'

if mode == 'embed':
    embedLoRAWeights(model_ckpt, lora_ckpt, out_ckpt)
elif mode == 'extract':
    extractLoRAWeights(model_ckpt, out_ckpt)
else:
    mergeLoRAWeights(model_ckpt, lora_ckpts, out_ckpt)

