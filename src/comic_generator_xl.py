import copy
import os
import pickle
import random
import requests
import sys
# numpy
import numpy as np
# torch
import torch
import torch.nn.functional as F
# utils
from utils import is_torch2_available, cal_attn_mask_xl, setup_seed
if is_torch2_available():
    from utils import \
        AttnProcessor2_0 as AttnProcessor
else:
    from utils  import AttnProcessor
# diffusers
import diffusers
# from diffusers import StableDiffusionXLPipeline
from pipeline import StoryDiffusionXLPipeline
from diffusers import DDIMScheduler, EulerDiscreteScheduler
# utils
from PIL import Image


#################################################
########Consistent Self-Attention################
#################################################
class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(
            self,
            hidden_size = None,
            cross_attention_dim=None,
            id_length = 4,
            device = "cuda",
            dtype = torch.float16
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        global _total_count, _attn_count, _cur_step, _mask1024, _mask4096
        global _sa32, _sa64
        global _write
        global _height, _width
        if _write:
            self.id_bank[_cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat((
                self.id_bank[_cur_step][0].to(self.device),
                hidden_states[:1],
                self.id_bank[_cur_step][1].to(self.device), hidden_states[1:]
            ))
        # skip in early step
        if _cur_step < 5:
            hidden_states = self.__call2__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        else:   # 256 1024 4096
            random_number = random.random()
            if _cur_step < 20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not _write:
                    if hidden_states.shape[1] == (_height//32) * (_width//32):
                        attention_mask = _mask1024[_mask1024.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = _mask4096[_mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    if hidden_states.shape[1] == (_height//32) * (_width//32):
                        attention_mask = _mask1024[:_mask1024.shape[0] // self.total_length * self.id_length,:_mask1024.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = _mask4096[:_mask4096.shape[0] // self.total_length * self.id_length,:_mask4096.shape[0] // self.total_length * self.id_length]
                hidden_states = self.__call1__(attn, hidden_states, encoder_hidden_states, attention_mask,temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states, None, attention_mask, temb)
        _attn_count +=1
        if _attn_count == _total_count:
            _attn_count = 0
            _cur_step += 1
            _mask1024, _mask4096 = cal_attn_mask_xl(
                self.total_length,
                self.id_length,
                _sa32,
                _sa64,
                _height,
                _width,
                device = self.device,
                dtype = self.dtype
            )
        return hidden_states

    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        total_batch_size,nums_token,channel = hidden_states.shape
        img_nums = total_batch_size//2
        hidden_states = hidden_states.view(-1,img_nums,nums_token,channel).reshape(-1,img_nums * nums_token,channel)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,nums_token,channel).reshape(-1,(self.id_length+1) * nums_token,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        # print(hidden_states.shape)
        return hidden_states

    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = (
            hidden_states.shape
        )
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,sequence_length,channel).reshape(-1,(self.id_length+1) * sequence_length,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def set_attention_processor(unet,id_length):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks") :
                attn_procs[name] = SpatialAttnProcessor2_0(id_length = id_length)
            else:
                attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = AttnProcessor()
    unet.set_attn_processor(attn_procs)


class ComicGeneratorXL:
    def __init__(
        self,
        model_name: str,
        id_length: int = 4,
        total_length: int = 5,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        scheduler_type: str = "euler",
        trigger_word: str = "img",
    ):
        global _total_count
        _total_count = 0
        # params
        self.model_name = model_name
        self.id_length = id_length
        self.total_length = total_length
        self.device = device
        self.torch_dtype = torch_dtype
        self.trigger_word = trigger_word
        # load pipeline
        # self.pipe = StableDiffusionXLPipeline.from_pretrained(
        # TODO: add photomaker loader
        self.pipe = StoryDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        ).to(device)
        # load photomaker for personalization
        photomaker_path = os.path.join(model_name, "photomaker", "photomaker-v1.bin")
        self.pipe.load_photomaker_adapter(
            photomaker_path,
            subfolder = "",
            weight_name = os.path.basename(photomaker_path),
            trigger_word = self.trigger_word
        )
        self.pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        if scheduler_type == "euler":
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler.set_timesteps(50)
        ### Insert PairedAttention
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None and (name.startswith("up_blocks") ) :
                attn_procs[name] =  SpatialAttnProcessor2_0(id_length = id_length)
                _total_count +=1
            else:
                attn_procs[name] = AttnProcessor()
        print("successsfully load consistent self-attention")
        print(f"number of the processor : {_total_count}")
        unet.set_attn_processor(copy.deepcopy(attn_procs))

    def __call__(
        self,
        prompts: list,
        negative_prompt: str,
        width: int = 768,
        height: int = 768,
        # strength of consistent self-attention: the larger, the stronger
        sa32: float = 0.5,
        sa64: float = 0.5,
        # sdxl params
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        seed: int = 2047,
        image_ref: Image.Image = None
    ):
        global _sa32, _sa64, _height, _width, _write, _mask1024, _mask4096, _cur_step, _attn_count
        # strength of consistent self-attention: the larger, the stronger
        _sa32 = sa32
        _sa64 = sa64
        # size
        _height = height
        _width = width
        ###
        _write = False
        _mask1024, _mask4096 = cal_attn_mask_xl(
            self.total_length,
            self.id_length,
            _sa32,
            _sa64,
            _height,
            _width,
            device = self.device,
            dtype = self.torch_dtype
        )
        # setup seed
        setup_seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        # prepare consistent memory
        id_prompts = prompts[:self.id_length]
        real_prompts = prompts[self.id_length:]
        torch.cuda.empty_cache()
        _write = True
        _cur_step = 0
        _attn_count = 0
        input_id_images = [image_ref] if image_ref is not None else None
        id_images = self.pipe(
            id_prompts,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            height = height,
            width = width,
            negative_prompt = negative_prompt,
            generator = generator,
            input_id_images = input_id_images
        ).images
        _write = False
        real_images = []
        for real_prompt in real_prompts:
            _cur_step = 0
            real_images.append(
                self.pipe(
                    real_prompt,
                    negative_prompt = negative_prompt,
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    height = height,
                    width = width,
                    generator = generator,
                    input_id_images = input_id_images
                ).images[0]
            )
        return id_images + real_images
