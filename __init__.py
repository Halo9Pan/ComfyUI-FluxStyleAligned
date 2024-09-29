import logging

from typing import Tuple, List, Dict
from types import MethodType

from functools import partial

import torch
import torch.nn as nn

import comfy.ops
import comfy.sample
import comfy.utils
from comfy.model_patcher import ModelPatcher
from comfy.samplers import KSampler
from nodes import common_ksampler

import latent_preview


try:
    from comfy.supported_models import FluxSchnell
except ImportError:
    logging.error("FluxSchnell not found")
    
try:
    from comfy.ldm.flux.layers import SingleStreamBlock
    # setattr(SingleStreamBlock, 'forward', single_stream_block_forward)
    from comfy.ldm.flux.layers import LastLayer
except ImportError:
    logging.error("SingleStreamBlock not found")

from comfy.ldm.flux.math import apply_rope, optimized_attention


def attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, pe:torch.Tensor)->torch.Tensor:
    q, k = apply_rope(q, k, pe)

    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True)
    return x

def single_stream_block_forward(self, x:torch.Tensor, vec:torch.Tensor, pe:torch.Tensor, share_attention:bool, share_attention_scale:float, adain_queries:bool, adain_keys:bool, adain_values:bool)->torch.Tensor:
    mod, _ = self.modulation(vec)
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
    qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
    
    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = self.norm(q, k, v)
    
    # compute attention
    q, k = apply_rope(q, k, pe)
    
    if adain_queries:
        q = adain(q)
    if adain_keys:
        k = adain(k)
    if adain_values:
        v = adain(v)
    if adain_queries and adain_keys:
        pe = adain(pe)
    if share_attention:
        k = concat_first(k, -2, scale=share_attention_scale)
        v = concat_first(v, -2)
    
    heads = q.shape[1]
    attn = optimized_attention(q, k, v, heads, skip_reshape=True)
    # compute activation in mlp stream, cat again and run second linear layer
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    x += mod.gate * output
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d


class StyleAlignedArgs:
    def __init__(self, share_attn: str) -> None:
        self.adain_keys = 'k' in share_attn
        self.adain_values = 'v' in share_attn
        self.adain_queries = 'q' in share_attn
    
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = True


def expand_first(
    x:torch.Tensor,
    scale=1.0,
) ->torch.Tensor:
    '''
    Expand the first element so it has the same shape as the rest of the batch.
    '''
    b = x.shape[0]
    m = x[0].expand(b-1, *x.shape[1:])
    m = torch.cat([x[0].unsqueeze(0), m])
    m = m.reshape(*x.shape)
    return m


def concat_first(x:torch.Tensor, dim:int=2, scale:float=1.0)->torch.Tensor:
    '''
    concat the the feature and the style feature expanded above
    '''
    s = expand_first(x)
    s[1:] = s[1:] * scale
    return torch.cat((x, s), dim=dim)


def adain(x:torch.Tensor)->torch.Tensor:
    x_std, x_mean = torch.std_mean(x, dim=-2, keepdim=True)
    s_mean = expand_first(x_mean)
    s_std = expand_first(x_std)
    x = (x - x_mean) / x_std
    x = x * s_std + s_mean
    return x

class SharedAttentionProcessor:
    def __init__(self, args:StyleAlignedArgs, scale:float):
        self.args = args
        self.scale = scale

    def __call__(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, extra_options):
        if self.args.adain_queries:
            q = adain(q)
        if self.args.adain_keys:
            k = adain(k)
        if self.args.adain_values:
            v = adain(v)
        if self.args.share_attention:
            k = concat_first(k, -2, scale=self.scale)
            v = concat_first(v, -2)

        return q, k, v


def get_norm_layers(
    layer:nn.Module,
    norm_layers_:Dict[str, List[LastLayer]],
):
    norm_layers_ = []
    if isinstance(layer, LastLayer):
        norm_layers_.append(layer)
    else:
        for child_layer in layer.children():
            get_norm_layers(child_layer)


def register_norm_forward(norm_layer:LastLayer)->LastLayer:
    if not hasattr(norm_layer, 'orig_forward'):
        setattr(norm_layer, 'orig_forward', norm_layer.forward)
    orig_forward = norm_layer.orig_forward
    
    def forward_(hidden_states:torch.Tensor) ->torch.Tensor:
        n = hidden_states.shape[-2]
        hidden_states = concat_first(hidden_states, dim=-2)
        hidden_states = orig_forward(hidden_states)  # type: ignore
        return hidden_states[..., :n, :]
    
    norm_layer.forward = forward_  # type: ignore
    return norm_layer


def register_shared_norm(
    model: ModelPatcher,
):
    norm_layers = get_norm_layers(model.model)
    print(
        f"Patching {len(norm_layers)} norms."
    )
    return [register_norm_forward(layer) for layer in norm_layers]


def blocks_switch(total_blocks, level):
    if level == 0:
        return torch.zeros(total_blocks, dtype=torch.bool)
    if level == 1:
        return torch.ones(total_blocks, dtype=torch.bool)
    flip = level > .5
    if flip:
        level = 1 - level
    num_switch = int(level * total_blocks)
    switch = torch.arange(total_blocks)
    switch = switch % (total_blocks // num_switch)
    switch = switch == 0
    if flip:
        switch = ~switch
    return switch



SHARE_NORM_OPTIONS = ['enabled', 'disabled']
SHARE_ATTN_OPTIONS = ['q+k', 'q+k+v', 'disabled']

class FluxStyleAlignedReferenceLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {'required':
                    {'model': ('MODEL',),
                    'noise_seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
                    'cfg': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step':0.1, 'round': 0.01}),
                    'positive': ('CONDITIONING', ),
                    'negative': ('CONDITIONING', ),
                    'sampler': ('SAMPLER', ),
                    'sigmas': ('SIGMAS', ),
                    'latent_image': ('LATENT', ),
                    }
                }

    RETURN_TYPES = ('STEP_LATENTS','LATENT')
    RETURN_NAMES = ('ref_latents', 'noised_output')

    FUNCTION = 'sample'

    CATEGORY = 'style_aligned'

    def sample(self, model, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        sigmas = sigmas.flip(0)
        if sigmas[0] == 0:
            sigmas[0] = 0.0001

        latent = latent_image
        latent_image = latent['samples']
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device='cpu')

        noise_mask = None
        if 'noise_mask' in latent:
            noise_mask = latent['noise_mask']

        ref_latents = []
        def callback(step: int, x0: T, x: T, steps: int):
            ref_latents.insert(0, x[0])
        
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

        out = latent.copy()
        out['samples'] = samples
        out_noised = out

        ref_latents = torch.stack(ref_latents)

        return (ref_latents, out_noised)

class FluxStyleAlignedReferenceSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'share_norm': (SHARE_NORM_OPTIONS,),
                'share_attn': (SHARE_ATTN_OPTIONS,),
                'scale': ('FLOAT', {'default': 1, 'min': 0, 'max': 2.0, 'step': 0.01}),
                'batch_size': ('INT', {'default': 2, 'min': 1, 'max': 8, 'step': 1}),
                'noise_seed': (
                    'INT',
                    {'default': 0, 'min': 0, 'max': 0xFFFFFFFFFFFFFFFF},
                ),
                'cfg': (
                    'FLOAT',
                    {
                        'default': 8.0,
                        'min': 0.0,
                        'max': 100.0,
                        'step': 0.1,
                        'round': 0.01,
                    },
                ),
                'positive': ('CONDITIONING',),
                'negative': ('CONDITIONING',),
                'ref_positive': ('CONDITIONING',),
                'sampler': ('SAMPLER',),
                'sigmas': ('SIGMAS',),
                'ref_latents': ('STEP_LATENTS',),
            },
        }

    RETURN_TYPES = ('LATENT', 'LATENT')
    RETURN_NAMES = ('output', 'denoised_output')
    FUNCTION = 'patch'
    CATEGORY = 'style_aligned'

    def patch(
        self,
        model: ModelPatcher,
        share_norm: str,
        share_attn: str,
        scale: float,
        batch_size: int,
        noise_seed: int,
        cfg: float,
        positive:torch.Tensor,
        negative:torch.Tensor,
        ref_positive:torch.Tensor,
        sampler:torch.Tensor,
        sigmas:torch.Tensor,
        ref_latents:torch.Tensor,
    ) -> 'tuple[dict, dict]':
        m = model.clone()
        args = StyleAlignedArgs(share_attn)

        # Concat batch with style latent
        style_latent_tensor = ref_latents[0].unsqueeze(0)
        height, width = style_latent_tensor.shape[-2:]
        latent_t = torch.zeros(
            [batch_size, 4, height, width], device=ref_latents.device
        )
        latent = {'samples': latent_t}
        noise = comfy.sample.prepare_noise(latent_t, noise_seed)

        latent_t = torch.cat((style_latent_tensor, latent_t), dim=0)
        ref_noise = torch.zeros_like(noise[0]).unsqueeze(0)
        noise = torch.cat((ref_noise, noise), dim=0)

        x0_output = {}
        preview_callback = latent_preview.prepare_callback(m, sigmas.shape[-1] - 1, x0_output)

        # Replace first latent with the corresponding reference latent after each step
        def callback(step: int, x0:torch.Tensor, x:torch.Tensor, steps: int):
            preview_callback(step, x0, x, steps)
            if (step + 1 < steps):
                x[0] = ref_latents[step+1]
                x0[0] = ref_latents[step+1]

        # Register shared norms
        share_norm = share_norm == 'enabled'
        register_shared_norm(model, share_norm)

        # Patch cross attn
        m.set_model_attn1_patch(SharedAttentionProcessor(args, scale))

        # Add reference conditioning to batch 
        batched_condition = []
        for i,condition in enumerate(positive):
            additional = condition[1].copy()
            batch_with_reference = torch.cat([ref_positive[i][0], condition[0].repeat([batch_size] + [1] * len(condition[0].shape[1:]))], dim=0)
            if 'pooled_output' in additional and 'pooled_output' in ref_positive[i][1]:
                # combine pooled output
                pooled_output = torch.cat([ref_positive[i][1]['pooled_output'], additional['pooled_output'].repeat([batch_size] 
                    + [1] * len(additional['pooled_output'].shape[1:]))], dim=0)
                additional['pooled_output'] = pooled_output
            if 'control' in additional:
                if 'control' in ref_positive[i][1]:
                    # combine control conditioning
                    control_hint = torch.cat([ref_positive[i][1]['control'].cond_hint_original, additional['control'].cond_hint_original.repeat([batch_size] 
                        + [1] * len(additional['control'].cond_hint_original.shape[1:]))], dim=0)
                    cloned_controlnet = additional['control'].copy()
                    cloned_controlnet.set_cond_hint(control_hint, strength=additional['control'].strength, timestep_percent_range=additional['control'].timestep_percent_range)
                    additional['control'] = cloned_controlnet
                else:
                    # add zeros for first in batch
                    control_hint = torch.cat([torch.zeros_like(additional['control'].cond_hint_original), additional['control'].cond_hint_original.repeat([batch_size] 
                        + [1] * len(additional['control'].cond_hint_original.shape[1:]))], dim=0)
                    cloned_controlnet = additional['control'].copy()
                    cloned_controlnet.set_cond_hint(control_hint, strength=additional['control'].strength, timestep_percent_range=additional['control'].timestep_percent_range)
                    additional['control'] = cloned_controlnet
            batched_condition.append([batch_with_reference, additional])

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample_custom(
            m,
            noise,
            cfg,
            sampler,
            sigmas,
            batched_condition,
            negative,
            latent_t,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise_seed,
        )

        # remove reference image
        samples = samples[1:]

        out = latent.copy()
        out['samples'] = samples
        if 'x0' in x0_output:
            out_denoised = latent.copy()
            x0 = x0_output['x0'][1:]
            out_denoised['samples'] = m.model.process_latent_out(x0.cpu())
        else:
            out_denoised = out
        return (out, out_denoised)


class FluxStyleAlignedBatchAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'share_norm': (SHARE_NORM_OPTIONS,),
                'share_attn': (SHARE_ATTN_OPTIONS,),
                'scale': ('FLOAT', {'default': 1, 'min': 0.5, 'max': 1.0, 'step': 0.005}),
                'level': ('FLOAT', {'default': 0.75, 'min': 0., 'max': 1.0, 'step': 0.05}),
            }
        }

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'patch'
    CATEGORY = 'style_aligned'

    def patch(
        self,
        model:ModelPatcher,
        share_norm:str,
        share_attn:str,
        scale:float,
        level:float,
    ):
        m = model.clone()
        share_norm = share_norm == 'enabled'
        # register_shared_norm(model, share_norm)
        single_blocks = m.model.diffusion_model.single_blocks
        single_blocks_num = len(single_blocks)
        single_blocks_switch = blocks_switch(single_blocks_num, level)
        adain_keys = 'k' in share_attn
        adain_values = 'v' in share_attn
        adain_queries = 'q' in share_attn
        for block_index, block in enumerate(single_blocks):
            if single_blocks_switch[block_index] and hasattr(block, 'forward'):
                setattr(block, 'forward_', block.forward)
                single_stream_block_forward_ = partial(single_stream_block_forward, share_attention=share_attn, share_attention_scale=scale, adain_keys=adain_keys, adain_values=adain_values, adain_queries=adain_queries)
                block.forward = MethodType(single_stream_block_forward_, block)
        return (m,)

class FluxStyleAlignedTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {
            'clip': ('CLIP', ),
            'clip_l': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
            't5xxl': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
            'style': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
            'guidance': ('FLOAT', {'default': 3.5, 'min': 0.0, 'max': 20.0, 'step': 0.1}),
            }}
    RETURN_TYPES = ('CONDITIONING',)
    FUNCTION = 'encode'

    CATEGORY = 'style_aligned'

    def encode(self, clip, clip_l, t5xxl, style:str, guidance):
        # clip_l = f'{clip_l} {style}'
        # t5xxl = f'{t5xxl} {style}'
        object_tokens = clip.tokenize(clip_l)
        object_tokens['t5xxl'] = clip.tokenize(t5xxl)['t5xxl']
        object_output = clip.encode_from_tokens(object_tokens, return_pooled=True, return_dict=True)
        object_cond = object_output.pop('cond')
        
        style_tokens = clip.tokenize(style)
        style_tokens['t5xxl'] = clip.tokenize(style)['t5xxl']
        style_output = clip.encode_from_tokens(style_tokens, return_pooled=True, return_dict=True)
        style_cond = style_output.pop('cond')
        
        cond = torch.cat([style_cond, object_cond], dim=0)
        
        object_output['guidance'] = guidance
        return ([[cond, object_output]], )


NODE_CLASS_MAPPINGS = {
    'FluxStyleAlignedReferenceSampler': FluxStyleAlignedReferenceSampler,
    'FluxStyleAlignedReferenceLatents': FluxStyleAlignedReferenceLatents,
    'FluxStyleAlignedBatchAlign': FluxStyleAlignedBatchAlign,
    'FluxStyleAlignedTextEncode': FluxStyleAlignedTextEncode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    'FluxStyleAlignedReferenceSampler': 'Flux StyleAligned Reference Sampler',
    'FluxStyleAlignedReferenceLatents': 'Flux StyleAligned Reference Latents',
    'FluxStyleAlignedBatchAlign': 'Flux StyleAligned in Batch',
    'FluxStyleAlignedTextEncode': 'Flux StyleAligned Text Encode',
}
