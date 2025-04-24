import torch


def embedLoRA(model_weights, lora_weights):
    print('##### MODEL WEIGHTS #####')
    for k in model_weights.keys():
        print(k)
    print('##### LORA  WEIGHTS #####')
    for k in lora_weights.keys():
        print(k)
    for k in lora_weights.keys():
        if k.endswith('lora_A'):
            parent_key = k[: -7]
            model_weights[f'{parent_key}.base_layer.weight'] = model_weights[f'{parent_key}.weight']
            del model_weights[f'{parent_key}.weight']
            if f'{parent_key}.bias' in model_weights.keys():
                model_weights[f'{parent_key}.base_layer.bias'] = model_weights[f'{parent_key}.bias']
                del model_weights[f'{parent_key}.bias']
            model_weights[f'{parent_key}.lora_A'] = lora_weights[f'{parent_key}.lora_A']
            model_weights[f'{parent_key}.lora_B'] = lora_weights[f'{parent_key}.lora_B']
            model_weights[f'{parent_key}.lora_scale'] = lora_weights[f'{parent_key}.lora_scale']
        elif k.endswith('lora_A.weight'):
            parent_key = k[: -14]
            model_weights[f'{parent_key}.base_layer.weight'] = model_weights[f'{parent_key}.weight']
            del model_weights[f'{parent_key}.weight']
            if f'{parent_key}.bias' in model_weights.keys():
                model_weights[f'{parent_key}.base_layer.bias'] = model_weights[f'{parent_key}.bias']
                del model_weights[f'{parent_key}.bias']
            model_weights[f'{parent_key}.lora_A.weight'] = lora_weights[f'{parent_key}.lora_A.weight']
            model_weights[f'{parent_key}.lora_B.weight'] = lora_weights[f'{parent_key}.lora_B.weight']
    print('### EMBEDDED  WEIGHTS ###')
    for k in model_weights.keys():
        print(k)
    return model_weights


def extractLoRA(model_weights):
    print('##### MODEL WEIGHTS #####')
    for k in model_weights.keys():
        print(k)
    lora_weights = {}
    for k in model_weights.keys():
        if 'lora_A' in k or 'lora_B' in k or 'lora_scale' in k:
            lora_weights[k] = model_weights[k]
    print('##### LORA  WEIGHTS #####')
    for k in lora_weights.keys():
        print(k)
    return lora_weights


def mergeLoRA(model_weights, lora_weights=[]):
    print('##### MODEL WEIGHTS #####')
    for k in model_weights.keys():
        print(k)
    for i in range(len(lora_weights)):
        print(f'#### LORA-{i}  WEIGHTS ####')
        for k in lora_weights[i].keys():
            print(k)
    if len(lora_weights) == 0:
        has_soft_wrappers = checkSoftWrappers(model_weights)
        if has_soft_wrappers:
            print('[ERROR] Weights of soft LoRA wrappers cannot be merged')
            return model_weights
        model_weight_keys = model_weights.keys()
        for k in list(model_weight_keys):
            if k.endswith('lora_A'):
                parent_key = k[: -7]
                lora_A = model_weights[f'{parent_key}.lora_A']
                lora_B = model_weights[f'{parent_key}.lora_B']
                base_weights = model_weights[f'{parent_key}.base_layer.weight']
                del model_weights[f'{parent_key}.base_layer.weight']
                if f'{parent_key}.base_layer.bias' in model_weights.keys():
                    model_weights[f'{parent_key}.bias'] = model_weights[f'{parent_key}.base_layer.bias']
                    del model_weights[f'{parent_key}.base_layer.bias']
                weight = model_weights[f'{parent_key}.scale'] * lora_B @ lora_A
                weight = weight.contiguous().view(*base_weights.size())
                model_weights[f'{parent_key}.weight'] = base_weights + weight
                del model_weights[f'{parent_key}.lora_A']
                del model_weights[f'{parent_key}.lora_B']
    else:
        for lora_weight in lora_weights:
            has_soft_wrappers = checkSoftWrappers(lora_weight)
            if has_soft_wrappers:
                print('[ERROR] Weights of soft LoRA wrappers cannot be merged')
                continue
            for k in lora_weight.keys():
                if k.endswith(f'lora_A'):
                    parent_key = k[: -7]
                    lora_A = lora_weight[f'{parent_key}.lora_A']
                    lora_B = lora_weight[f'{parent_key}.lora_B']
                    base_weights = model_weights[f'{parent_key}.weight']
                    weight = lora_weight[f'{parent_key}.scale'] * lora_B @ lora_A
                    weight = weight.contiguous().view(*base_weights.size())
                    model_weights[f'{parent_key}.weight'] += weight
    print('#### MERGED  WEIGHTS ####')
    for k in model_weights.keys():
        print(k)
    return model_weights


def checkSoftWrappers(weights):
    for k in weights.keys():
        if k.endswith('lora_A.weight') or k.endswith('lora_A.bias') or \
            k.endswith('lora_B.weight') or k.endswith('lora_B.bias'):
            return True
    return False


def embedLoRAWeights(model_ckpt, lora_ckpt, out_ckpt=None):
    sd_weights = torch.load(model_ckpt, map_location='cpu')
    lora_weights = torch.load(lora_ckpt, map_location='cpu')
    if 'state_dict' in sd_weights.keys():
        model_weights = embedLoRA(sd_weights['state_dict'], lora_weights)
        sd_weights['state_dict'] = model_weights
    else:
        sd_weights = embedLoRA(sd_weights, lora_weights)
    out_ckpt = './embedded.ckpt' if out_ckpt is None else out_ckpt
    torch.save(sd_weights, out_ckpt)
    print(f'[ LOG ] Ensemble with backbone weights {model_ckpt}'
          f'and embedded LoRA weights {lora_ckpt} saved in {out_ckpt}')


def extractLoRAWeights(model_ckpt, out_ckpt=None):
    model_weights = torch.load(model_ckpt, map_location='cpu')
    if 'state_dict' in model_weights.keys():
        model_weights = model_weights['state_dict']
    lora_weights = extractLoRA(model_weights)
    out_ckpt = './lora.ckpt' if out_ckpt is None else out_ckpt
    torch.save(lora_weights, out_ckpt)
    print(f'[ LOG ] LoRA weights extracted from {model_ckpt} saved in {out_ckpt}')


def mergeLoRAWeights(model_ckpt, lora_ckpts=[], out_ckpt=None):
    model_weights = torch.load(model_ckpt, map_location='cpu')
    lora_weights = [torch.load(f, map_location='cpu') for f in lora_ckpts]
    if 'state_dict' in model_weights.keys():
        merged_weights = mergeLoRA(model_weights['state_dict'], lora_weights)
        model_weights['state_dict'] = merged_weights
    else:
        model_weights = mergeLoRA(model_weights, lora_weights)
    out_ckpt = './merged.ckpt' if out_ckpt is None else out_ckpt
    torch.save(model_weights, out_ckpt)
    print(f'[ LOG ] Ensemble with backbone weights {model_ckpt}'
          f'and merged LoRA weights {lora_ckpts} saved in {out_ckpt}')

