import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from sglang.srt.utils import add_prefix, make_layers


BitNetConfig = None


logger = logging.getLogger(__name__)

from sglang.srt.custom_op import CustomOp
def relu2(x :torch.Tensor) -> torch.Tensor:
    x = torch.maximum(x,torch.zeros_like(x))
    x = x * x
    return x
class Relu2AndMul(CustomOp):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return relu2(x[...,:d])*x[...,d:]
    
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return relu2(x[...,:d])*x[...,d:]
    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return relu2(x[...,:d])*x[...,d:]


class BitNetMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        rms_norm_eps: float = 1e-05,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size]*2,
            bias = False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj",prefix),
            )
        if hidden_act != "relu2":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only relu2 is supported for now."
            )
            
        self.act_fn = Relu2AndMul()
        self.down_project = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias = False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj",prefix),
        )
        self.ffn_sub_norm = RMSNorm(intermediate_size,eps=rms_norm_eps)
        
        
    def forward(self,x : torch.Tensor) ->torch.Tensor:
        gate_up,_ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.ffn_sub_norm(x)
        x,_ = self.down_project(x)
        return x
        
        
class BitNetAttention(nn.Module):
    def __init__(
        self,
        hidden_size:int,
        num_heads:int,
        num_kv_heads:int,
        layer_id:int,
        rope_theta:float = 500000.0,
        rms_norm_eps:float = 1e-05,
        rope_scaling: Optional[Dict[str, Any]] = None, 
        max_position_embeddings: int = 4096,     
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",   
        ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if(self.total_num_kv_heads >= tp_size):
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads ==0
        
        self.num_kv_heads = max(1,self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embedings = max_position_embeddings
        
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias= False,
            quant_config=quant_config,
            prefix=add_prefix("qk_proj",prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj",prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn",prefix),
        )
        self.attn_sub_norm = RMSNorm(hidden_size,eps=rms_norm_eps)
        
    def forward(self,
        positions:torch.Tensor,
        hidden_states:torch.Tensor,
        forward_batch:ForwardBatch,) ->torch.Tensor:
        qkv,_ = self.qkv_proj(hidden_states)
        q,k,v = qkv.split([self.q_size, self.kv_size,self.kv_size],dim = -1)
        q,k = self.rotary_emb(positions,q,k)
        attn_output = self.attn(q,k,v,forward_batch)
        norm_attn_output = self.attn_sub_norm(attn_output)
        output,_ = self.o_proj(norm_attn_output)
        return output
    
    
class BitNetDecoderLayer(nn.Module):
    def __init__(self, 
                 config:BitNetConfig,
                 layer_id:int =0,
                 quanconfig:Optional[QuantizationConfig] = None,
                prefix: str = "",) ->None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config,"rope_theta",500000.0)
        rope_scaling = getattr(config,"rope_scaling",None)
        max_position_embeddings = getattr(config,"max_position_embeddings",4096)    
        self.self_attn = BitNetAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quanconfig,
            prefix=add_prefix("self_attn",prefix),
        )
        self.mlp = BitNetMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quanconfig,
            rms_norm_eps=config.rms_norm_eps,
            prefix=add_prefix("mlp",prefix),
        )
        
        
        self.input_layernorm = RMSNorm(self.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size,eps=config.rms_norm_eps)
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states:torch.Tensor,
        forward_batch:ForwardBatch,
        residual:Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor,torch.Tensor]:
            if(residual is None):
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states,residual = self.input_layernorm(hidden_states,residual)
                
            hidden_states = self.self_attn(
                positions = positions,
                hidden_states = hidden_states,
                forward_batch=forward_batch,
                )
            hidden_states,residual = self.post_attention_layernorm(hidden_states,residual)
            hidden_states = self.mlp(hidden_states)
            return hidden_states,residual
        
class BitNetModel(nn.Model):
    def __init__(
        self,
        config:BitNetConfig,
        layer_id:int = 0,
        quant_config:Optional[QuantizationConfig] = None,
        prefix :str = "",
         decoder_layer_type: type[nn.Module] =BitNetDecoderLayer,
    )  -> None:    
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens",prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()
            
            
            
        decoder_layer_type = decoder_layer_type or BitNetDecoderLayer
        
        self.layers,self.start_layer,self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx,prefix : decoder_layer_type(
                layer_id = idx,
                config = config,
                quant_config = quant_config,
                prefix = prefix, 
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers",prefix)
        )
        
        
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        else:
            self,norm = PPMissingLayer()
        
        
    def get_input_embedding(self,inputs_ids:torch.Tensor) -> torch.Tensor:
        if hasattr(self.config,"scale_emb"):
            return self.get_input_embeddings()(inputs_ids) * self.config.scale_emb
        else:
            return self.get_input_embeddings()(inputs_ids)
        
        
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    
    
    def forward(
        self,
        input_ids:torch.Tensor,
        positions:torch.Tensor,
        forward_batch:ForwardBatch,
        input_embeds:torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) ->Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if(input_embeds is None):
                hidden_states = self.get_input_embedding(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            
        for i in range(self.start_layer,self.end_layer):
            layer = self.layers[i - self.start_layer]
            hidden_states,residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
            
            
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states" : hidden_states,
                    "residual" : residual,
                }
            )
            
        else:
            hidden_states = self.norm(hidden_states,residual)    
            return hidden_states
        
        
        
    def load_kv_cache_scales(
        self,
        quantization_param_path : str,
    ) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        
        for layer_idx , scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx],nn.Identity):
                layer_self_attn = self.layers[layer_idx].self.attn
            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )
                
                

class BitNetForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: BitNetConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = BitNetModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        # perform weight tying for PP
        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            else:
                emb_token_weight = self.pp_group.recv(
                    size=(config.vocab_size, config.hidden_size),
                    dtype=next(self.model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embedding(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids, hidden_states, self.lm_head, forward_batch
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # Handle pp weight tying here
                    # find the embed_tokens.weight in the weights
                    embed_token_weights = next(
                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


EntryClass = BitNetForCausalLM