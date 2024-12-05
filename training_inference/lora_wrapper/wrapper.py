import torch
import torch.nn as nn
import torch.nn.functional as f


class LoRAWrapper(nn.Module):
    def __init__(self, base_layer, r=1, scale=1, trainable_scale=False):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.lora_r = r
        self.lora_scale = nn.Parameter(torch.tensor(scale).float())
        if not trainable_scale:
            self.lora_scale.requires_grad_(False)


class LinearWrapper(LoRAWrapper):
    def __init__(self, base_layer, r, scale=1, trainable_scale=False, hard=True):
        super().__init__(base_layer, r, scale, trainable_scale)
        assert isinstance(self.base_layer, nn.Linear), \
            f'[ERROR] LinearWrapper applied on an instance of {self.base_layer.__class__.__name__}'
        weight_size = self.base_layer.weight.size()
        self.lora_B = nn.Parameter(torch.zeros(weight_size[0], self.lora_r).float())
        self.lora_A = nn.Parameter(torch.randn(self.lora_r, weight_size[1]).float() / self.lora_r)

    def forward(self, x):
        weight = self.base_layer.weight + self.lora_scale * self.lora_B @ self.lora_A
        return f.linear(x, weight, self.base_layer.bias)


# hard wrappers: matrix multiplication
class ConvHardWrapper(LoRAWrapper):
    def __init__(self, base_layer, r, scale=1, trainable_scale=False):
        super().__init__(base_layer, r, scale, trainable_scale)
        assert isinstance(self.base_layer, nn.modules.conv._ConvNd), \
            f'[ERROR] ConvHardWrapper applied on an instance of {self.base_layer.__class__.__name__}'
        # only kernel_size == 1 supported
        assert self.base_layer.kernel_size == 1, \
            f'[ERROR] ConvHardWrapper applied on a layer with kernel size {self.base_layer.kernel_size}'
        weight_size = self.base_layer.weight.size()
        self.lora_B = nn.Parameter(torch.zeros(weight_size[0], self.lora_r).float())
        self.lora_A = nn.Parameter(torch.randn(self.lora_r, weight_size[1]).float() / self.lora_r)
        self.convnd = [f.conv1d, f.conv2d, f.conv3d][len(weight_size) - 1]

    def forward(self, x):
        lora_weight = self.lora_scale * (self.lora_B @ self.lora_A)
        weight = self.base_layer.weight + lora_weight.contiguous().view(*self.base_layer.weight.size())
        return self.convnd(x, weight, self.base_layer.bias, self.base_layer.stride,
                           self.base_layer.padding, self.base_layer.dilation, self.base_layer.groups)


# soft wrapper: layer stack. rank equals intermediate channel number
# inspired by https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
class ConvSoftWrapper(LoRAWrapper):
    def __init__(self, base_layer, r, scale=1, trainable_scale=False):
        super().__init__(base_layer, r, scale, trainable_scale)
        assert isinstance(self.base_layer, nn.modules.conv._ConvNd), \
            f'[ERROR] ConvSoftWrapper applied on an instance of {self.base_layer.__class__.__name__}'
        self.lora_B = nn.Conv2d(self.base_layer.in_channels, self.lora_r, self.base_layer.kernel_size,
                                self.base_layer.stride, self.base_layer.padding, self.base_layer.dilation,
                                self.base_layer.groups, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A = nn.Conv2d(self.lora_r, self.base_layer.out_channels, self.base_layer.kernel_size,
                                self.base_layer.stride, self.base_layer.padding, self.base_layer.dilation,
                                self.base_layer.groups, bias=False)
        nn.init.normal_(self.lora_A.weight, std=1 / self.lora_r)

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = self.lora_A(self.lora_B(x))
        return base_output + self.lora_scale * lora_output


def ConvWrapper(base_layer, r, trainable_scale=False, scale=1, hard=True):
    return ConvHardWrapper(base_layer, r, scale, trainable_scale) if hard else ConvSoftWrapper(base_layer, r, scale, trainable_scale)


class LoRA:
    def __init__(self, config={
        'trainable_model': 1,
        # 'patch_embed.proj': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 0},
        # 'blocks.0.attn.qkv': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        # 'blocks.0.attn.proj': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        # 'blocks.0.mlp.lin1': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        # 'blocks.0.mlp.lin2': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        # 'blocks.1.attn.qkv': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        # 'blocks.1.attn.proj': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        # 'blocks.1.mlp.lin1': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        # 'blocks.1.mlp.lin2': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.2.attn.qkv': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.2.attn.proj': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.2.mlp.lin1': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.2.mlp.lin2': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.3.attn.qkv': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.3.attn.proj': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.3.mlp.lin1': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.3.mlp.lin2': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.4.attn.qkv': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.4.attn.proj': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.4.mlp.lin1': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.4.mlp.lin2': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.5.attn.qkv': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.5.attn.proj': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.5.mlp.lin1': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.5.mlp.lin2': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.6.attn.qkv': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.6.attn.proj': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.6.mlp.lin1': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.6.mlp.lin2': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.7.attn.qkv': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.7.attn.proj': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.7.mlp.lin1': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.7.mlp.lin2': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.8.attn.qkv': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.8.attn.proj': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.8.mlp.lin1': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.8.mlp.lin2': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.9.attn.qkv': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.9.attn.proj': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.9.mlp.lin1': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.9.mlp.lin2': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.10.attn.qkv': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.10.attn.proj': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.10.mlp.lin1': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.10.mlp.lin2': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.11.attn.qkv': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.11.attn.proj': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.11.mlp.lin1': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        'blocks.11.mlp.lin2': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 1},
        # 'neck.0': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 0},
        # 'neck.1': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 0},
        # 'neck.2': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 0},
        # 'neck.3': {'r': 8, 'scale': 1, 'trainable_scale': 0, 'hard': 0},
    }):
        self.config = config
        self.wrappers = {
            'Linear': LinearWrapper,
            'Conv1d': ConvWrapper,
            'Conv2d': ConvWrapper,
            'Conv3d': ConvWrapper,
        }

    def wrap(self, model):
        if not bool(self.config['trainable_model']):
            model.requires_grad_(False)
        module_names = []
        for name, module in model.named_modules(remove_duplicate=False):
            module_names.append(name)
        for k in self.config.keys():
            for name in module_names:
                if name.endswith(k):
                    name_segs = name.split('.')[: -1]
                    ns = name.split('.')[-1]
                    module = model
                    for n in name_segs:
                        module = getattr(module, n)
                    layer = getattr(module, ns)
                    if layer.__class__.__name__ not in self.wrappers.keys():
                        print(f"LoRA Wrapper: skip wrap {layer.__class__.__name__ }")
                        continue
                    lora_module = self.wrappers[layer.__class__.__name__](
                        layer, self.config[k]['r'], self.config[k]['scale'],
                        bool(self.config[k]['trainable_scale']), bool(self.config[k]['hard']))
                    setattr(module, ns, lora_module)
        return model

