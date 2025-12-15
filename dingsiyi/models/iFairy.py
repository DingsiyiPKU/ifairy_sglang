'''注意事项:
1,linear,mlp层的参数名称不匹配,需要调整,注意参数的位置
2,mlp层的tp逻辑可以优化（可选）
5,load_kv_cache_scales
6，attn层分布式rms'''


import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from sglang.srt.distributed import all_gather
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
from math import sqrt



logger = logging.getLogger(__name__)
Complexconfig = None

from sglang.srt.custom_op import CustomOp

def complex_relu2(x_real: torch.Tensor, x_imag: torch.Tensor) ->  Tuple[torch.Tensor,torch.Tensor]:
    # 1. 稀疏化：仅当实部和虚部同时小于0（第三象限）时，置为0
    mask = torch.logical_and(x_real < 0, x_imag < 0)
    x_real = torch.where(mask, 0.0, x_real)
    x_imag = torch.where(mask, 0.0, x_imag)
    # 2. 非线性：对所有元素进行平方
    x_real = x_real**2
    x_imag = x_imag**2
    return x_real, x_imag

class ComplexRelu2AndMul(CustomOp):
    def forward_native(self, gate_real:torch.Tensor,gate_imag:torch.Tensor,up_real:torch.Tensor,up_imag:torch.Tensor) ->  Tuple[torch.Tensor,torch.Tensor]:
        gate_real,gate_imag = complex_relu2(gate_real,gate_imag)
        
        output_real = gate_real * up_real + gate_imag * up_imag
        output_imag = gate_real * up_imag - gate_imag * up_real
        
        return output_real,output_imag
    
    
    
    
     
    
    def forward_cuda(self, x_real: torch.Tensor,x_imag: torch.Tensor) -> torch.Tensor:
        d = x_real.shape[-1] // 2
        gate_real,up_real = x_real.split([d, d], dim=-1)
        gate_imag,up_imag = x_imag.split([d, d], dim=-1)
         
        gate_real,gate_imag = complex_relu2(gate_real,gate_imag)
        
        output_real = gate_real * up_real + gate_imag * up_imag
        output_imag = gate_real * up_imag - gate_imag * up_real
        
        return output_real,output_imag
    def forward_npu(self, x_real: torch.Tensor,x_imag: torch.Tensor) -> torch.Tensor:
        d = x_real.shape[-1] // 2
        gate_real,up_real = x_real.split([d, d], dim=-1)
        gate_imag,up_imag = x_imag.split([d, d], dim=-1)
         
        gate_real,gate_imag = complex_relu2(gate_real,gate_imag)
        
        output_real = gate_real * up_real + gate_imag * up_imag
        output_imag = gate_real * up_imag - gate_imag * up_real
        
        return output_real,output_imag
    
def IntergrateRealAndImag(real_product: torch.Tensor,imag_product : torch.Tensor ,splite_dim :int,
                          need_split:bool = True ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor] :
    
    r_r,r_i = real_product.split(splite_dim, dim=-1)
    i_r,i_i = imag_product.split(splite_dim, dim=-1)
    r_r.add_(i_i)
    r_i.sub_(i_r)
    if need_split:
        return r_r,r_i
    else:
        return real_product
        
    
    


class ComplexGetrope(nn.Module):
    def __init__(self, head_dim:int =96,max_position_embeddings:int=2048 ,rope_theta: int = 10000,basescaling: Optional[Dict[str, Any]] = None,rope_scaling: Optional[Dict[str, Any]] = None,) ->None:
        super().__init__()
        self.head_dim = head_dim
        self.rotary_emb = get_rope(
            self.head_dim * 2,
            rotary_dim=self.head_dim * 2,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
    def getInterleaving(self, real : torch.Tensor,imag : torch.Tensor) -> torch.Tensor: 
        # real,imag shape: [batch_size, seq_len, num_heads, head_dim]
        combined = torch.stack([real, imag], dim=-1)  # shape: [batch_size, seq_len, num_heads, head_dim, 2]
        
        combined.contiguous()
        
        interleaved = combined.view(*real.shape[:-1], -1)  # shape: [batch_size, seq_len, num_heads, head_dim * 2]
        return interleaved
        
    def getDeinterleaving(self, interleaved : torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]: 
        # interleaved shape: [batch_size, seq_len, num_heads, head_dim * 2]
        interleaved.contiguous()
        reshaped = interleaved.view(*interleaved.shape[:-1], -1, 2)  # shape: [batch_size, seq_len, num_heads, head_dim, 2]
        real = reshaped[..., 0]  # shape: [batch_size, seq_len, num_heads, head_dim]
        imag = reshaped[..., 1]  # shape: [batch_size, seq_len, num_heads, head_dim]
        return real, imag    
        
    
    def forward(self,positions: torch.Tensor,q_real: torch.Tensor,q_imag:torch.Tensor,k_real: torch.Tensor,k_imag:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        q_real_and_imag = self.getInterleaving(q_real,q_imag)
        k_real_and_imag = self.getInterleaving(k_real,k_imag)
        q_rotated, k_rotated = self.rotary_emb(positions, q_real_and_imag, k_real_and_imag)
        q_rotated_real,q_rotated_imag = self.getDeinterleaving(q_rotated)
        k_rotated_real,k_rotated_imag = self.getDeinterleaving(k_rotated)
        return q_rotated_real,q_rotated_imag,k_rotated_real,k_rotated_imag






class ComplexNetRMSNorm(nn.Module):
    def __init__(self, hidden_size:int ,eps:float =  1e-05,):
        super().__init__()
        self.weight_real_imag = RMSNorm(hidden_size * 2,eps)
        
    def forward(self,hidden_states_real: torch.Tensor,hidden_states_imag: torch.Tensor,residual_real : torch.Tensor = None,residual_imag : torch.Tensor = None,) :
        
        x = torch.cat([hidden_states_real, hidden_states_imag], dim=-1)
        residual = torch.cat([residual_real, residual_imag], dim=-1) if (residual_real is not None and residual_imag is not None) else None
        if residual is not None:
            normalized_x,residual = self.weight_real_imag(x, residual)
            hidden_states_real,hidden_states_imag = torch.chunk(normalized_x, 2, dim=-1)
            residual_real,residual_imag = torch.chunk(residual, 2, dim=-1)
            return hidden_states_real,hidden_states_imag,residual_real,residual_imag
        else:
            normalized_x = self.weight_real_imag(x)
            hidden_states_real,hidden_states_imag = torch.chunk(normalized_x, 2, dim=-1)
            return hidden_states_real,hidden_states_imag
    
    

class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, quant_config: Optional[QuantizationConfig] = None, prefix: str = "",)-> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_real_and_imag = RowParallelLinear(
                hidden_size = in_features ,
                output_size= out_features * 2,
                bias= False,
                quant_config=quant_config,
                prefix=prefix,
            )
        
        
    def forward(self, input_real:torch.Tensor,input_imag:torch.Tensor,) -> Tuple[torch.Tensor,torch.Tensor]:
        assert input_real.size() == input_imag.size() ,"Shape mismatch"
        
        input_real_and_imag  = torch.cat([input_real, input_imag], dim=0) 
        Merged_output  = self.weight_real_and_imag(input_real_and_imag)
        
        real_product, imag_product =  torch.chunk(Merged_output, 2, dim=0)
        
        return IntergrateRealAndImag(real_product, imag_product ,self.out_features)




class ComplexUpLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, quant_config: Optional[QuantizationConfig] = None, prefix: str = "",)-> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_real_and_imag = MergedColumnParallelLinear(
                hidden_size = in_features,
                output_sizes=[out_features, out_features],
                bias= False,
                quant_config=quant_config,
                gather_output=True,
                prefix=prefix,
            )
            
        
    def forward(self, input_real:torch.Tensor,
                input_imag:torch.Tensor,) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        assert input_real.size() == input_imag.size() ,"Shape mismatch"
        
        input_real_and_imag  = torch.cat([input_real, input_imag], dim=0) 
        Merged_output  = self.weight_real_and_imag(input_real_and_imag)
        
        real_product, imag_product =  torch.chunk(Merged_output, 2, dim=0)
        
        Gate_real_product,Up_real_product = real_product.split(self.out_features, dim=-1)
        Gate_imag_product,Up_imag_product = imag_product.split(self.out_features, dim=-1)
        
        Gate_real,Gate_imag = IntergrateRealAndImag(Gate_real_product, Gate_imag_product ,self.out_features//2)
        Up_real,Up_imag = IntergrateRealAndImag(Up_real_product, Up_imag_product ,self.out_features//2)
        
        return Gate_real,Gate_imag,Up_real,Up_imag
        
                
class ComplexQKVLinear(nn.Module):
    def __init__(self, head_dim:int , total_num_heads:int, total_num_kv_heads:int, quant_config: Optional[QuantizationConfig] = None, prefix: str = "",) -> None:
        super().__init__()
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = head_dim * total_num_heads
        self.q_real_imag_size = self.head_dim * self.total_num_heads  *2
        
        self.kv_real_imag_size = self.head_dim * self.total_num_kv_heads *2
        
        
        
        self.qkv_linear = QKVParallelLinear(
            self.hidden_size ,
            self.head_dim * 2,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias= False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        
    def forward(self, input_real: torch.Tensor,input_imag: torch.Tensor,) -> Tuple [torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        assert input_real.size() == input_imag.size() ,"Shape mismatch"
        
        input_real_and_imag  = torch.cat([input_real, input_imag], dim=0) 
        Merged_qkv  = self.qkv_linear(input_real_and_imag)  
        qkv_real_product,qkv_imag_product = torch.chunk(Merged_qkv, 2, dim=0) 
        q_real_product, k_real_product, v_real_product = qkv_real_product.split(
            [self.q_real_imag_size, self.kv_real_imag_size, self.kv_real_imag_size], dim=-1
        )
        q_imag_product, k_imag_product, v_imag_product = qkv_imag_product.split(
            [self.q_real_imag_size, self.kv_real_imag_size, self.kv_real_imag_size], dim=-1
        )
        
        q_real,q_imag = IntergrateRealAndImag(q_real_product,q_imag_product,self.q_real_imag_size//2)
        
        k_real,k_imag = IntergrateRealAndImag(k_real_product,k_imag_product,self.kv_real_imag_size//2)

        v_real_imag = IntergrateRealAndImag(v_real_product,v_imag_product,self.kv_real_imag_size//2,need_split=False)
        
        return q_real,q_imag,k_real,k_imag,v_real_imag
        











class ComplexNetMLP(nn.Module):
    def __init__(
        self,
        hidden_size:int ,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-05,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = ComplexUpLinear(
            hidden_size,
            intermediate_size * 2,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        
        self.down_proj = ComplexLinear(
            intermediate_size,
            hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        
        if hidden_act != "relu2":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only relu2 is supported for now."
            )
        self.act_fn = ComplexRelu2AndMul()
        
        
        self.ffn_layernorm = ComplexNetRMSNorm(intermediate_size,eps=rms_norm_eps)
        
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        Gate_real,Gate_imag,Up_real,Up_imag = self.gate_up_proj(x_real,x_imag)
        
        x_real,x_imag = self.act_fn(Gate_real,Gate_imag,Up_real,Up_imag)
        
        x_real,x_imag = self.ffn_layernorm(x_real,x_imag)
        
        output_real,output_imag = self.down_proj(x_real,x_imag)
        
        return output_real,output_imag
    
    
class ComplexNetAttention(nn.Module):    
    def __init__(self, 
                hidden_size:int,
                num_heads:int,
                num_kv_heads:int,
                layer_id:int,
                rope_theta:int = 10000,
                rope_scaling: Optional[Dict[str, Any]] = None,
                max_position_embeddings: int = 2048,
                quant_config: Optional[QuantizationConfig] = None,
                prefix: str = "",
                rms_norm_eps: float = 1e-05,
                 ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // num_heads   
        
        self.q_real_imag_size = self.head_dim * self.num_heads  *2
        
        self.kv_real_imag_size = self.head_dim * self.num_kv_heads *2
        
        self.scaling = self.head_dim ** -0.5
        
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
            
        self.qkv_linear = ComplexQKVLinear(
            head_dim=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads, 
            quant_config=quant_config,
            prefix=prefix,
        )
        
    
        self.o_proj = ComplexLinear(
            in_features=self.head_dim * self.total_num_heads,
            out_features=hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        
        
        self.rotary_emb = ComplexGetrope(
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,   
            rope_theta=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim * 2,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        
        self.attn_layernorm = RMSNorm(hidden_size * 2,eps=rms_norm_eps)
        
    def forward (
            self,
            positions: torch.Tensor,
            hidden_states_real: torch.Tensor,
            hidden_states_imag: torch.Tensor,
            forward_batch: ForwardBatch,
        ) -> Tuple[torch.Tensor,torch.Tensor]:
        q_real, q_imag, k_real,k_imag, v= self.qkv_linear(hidden_states_real, hidden_states_imag)

        q_rotated_real,q_rotated_imag,k_rotated_real,k_rotated_imag = self.rotary_emb(positions, q_real, q_imag, k_real, k_imag)
        
        q =torch.cat([q_rotated_real, q_rotated_imag], dim=-1)
        k =torch.cat([k_rotated_real, k_rotated_imag], dim=-1)
        
        attn_output_real_imag = self.attn(q,k,v,forward_batch)
        
        attn_output_real_imag = all_gather(attn_output_real_imag, dim=-1)
        
        attn_normalized_real_imag = self.attn_layernorm(attn_output_real_imag)
        
        attn_real,attn_imag = torch.chunk( attn_normalized_real_imag , 2, dim=-1)
        
        output_real,output_imag = self.o_proj(attn_real,attn_imag)
        
        return output_real,output_imag
    
    
    
class  ComplexNetDecoderLayer(nn.Module):
    def __init__(self, 
                   config: Complexconfig,
                   layer_id: int = 0,
                   quant_config: Optional[QuantizationConfig] = None,
                   prefix: str = "",
                   ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 2048)
        self.self_attn = ComplexNetAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = ComplexNetMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = ComplexNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ComplexNetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )       
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states_real: torch.Tensor,
        hidden_states_imag: torch.Tensor,
        forward_batch: ForwardBatch,
        residual_real: Optional[torch.Tensor],
        residual_imag: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        # Self Attention
        if  residual_real is None and residual_imag is None:
            residual_real = hidden_states_real
            residual_imag = hidden_states_imag
            hidden_states_real,hidden_states_imag = self.input_layernorm(hidden_states_real,hidden_states_imag)

        else:
            hidden_states_real,hidden_states_imag,residual_real,residual_imag = self.input_layernorm(hidden_states_real,hidden_states_imag,residual_real,residual_imag)
        
        hidden_states_imag,hidden_states_real = self.self_attn(
            positions=positions,
            hidden_states_real=hidden_states_real,
            hidden_states_imag=hidden_states_imag,
            forward_batch=forward_batch,
        )
        
        hidden_states_real,hidden_states_imag,residual_real,residual_imag = self.post_attention_layernorm(hidden_states_real,hidden_states_imag,residual_real,residual_imag)
        hidden_states_imag,hidden_states_real = self.mlp(
            hidden_states_real,
            hidden_states_imag,
        )
        
        return hidden_states_real,hidden_states_imag, residual_real,residual_imag
    

class ComplexNetLMBase(nn.Module):
    def __init__(
        self,
        config: Complexconfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type:type[nn.Module]=ComplexNetDecoderLayer,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        
        if self.pp_group.is_first.rank:
            self.embed_tokens_real = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix="token_embeddings_real",
            )
            self.embed_tokens_imag = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix="token_embeddings_imag",
            )
            
            
        else:
            self.embed_tokens_real = PPMissingLayer()
            self.embed_tokens_imag = PPMissingLayer()
        
        decoder_layer_type = decoder_layer_type or ComplexNetDecoderLayer
        
        self.layer,self.start_layer,self.end_layer = make_layers( 
            config.num_hidden_layers,
            lambda ids,prefix: decoder_layer_type(
                layer_id = ids,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.size,
            prefix=add_prefix("layer", prefix),
        ),
        
        if self.pp_group.is_last.rank:
            self.final_norm = ComplexNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        else:
            self.final_norm = PPMissingLayer()
        
    def get_input_embeddings(self) -> Tuple[nn.Embedding,nn.Embedding]:     
        return self.embed_tokens_real,self.embed_tokens_imag
        
    def get_input_embedding(self,input_ids: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        input_embeds_real = self.embed_tokens_real(input_ids)
        input_embeds_imag = self.embed_tokens_imag(input_ids)
        
        if hasattr(self.config, "scale_emb"):
            return self.config.scale_emb * input_embeds_real,self.config.scale_emb * input_embeds_imag
        
        else:
            return input_embeds_real,input_embeds_imag
        
        
            
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds_real: Optional[torch.Tensor] = None,
        input_embeds_imag: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first.rank:
            if input_embeds_real is None or input_embeds_imag is None:
                hidden_state_real, hidden_state_imag = self.get_input_embedding(input_ids)
            else:
                hidden_state_real = input_embeds_real
                hidden_state_imag = input_embeds_imag
        else:
            assert pp_proxy_tensors is not None, "PPProxyTensors must be provided for non-first PP ranks."
            hidden_state_real = pp_proxy_tensors[hidden_state_real]
            hidden_state_imag = pp_proxy_tensors[hidden_state_imag]
            residual_real = pp_proxy_tensors[residual_real]
            residual_imag = pp_proxy_tensors[residual_imag]
            
        for i in range(self.start_layers,self.end_layer):
            layer = self.layer[i]
            hidden_state_real,hidden_state_imag, residual_real, residual_imag = layer(
                positions=positions,
                hidden_states_real=hidden_state_real,
                hidden_states_imag=hidden_state_imag,
                forward_batch=forward_batch,
                residual_real=residual_real,
                residual_imag=residual_imag,
            )
            
        if not self.pp_group.is_last.rank:
            return PPProxyTensors(
                hidden_state_real=hidden_state_real,
                hidden_state_imag=hidden_state_imag,
                residual_real=residual_real,
                residual_imag=residual_imag,
            )
            
            
        else:
            hidden_state_real,hidden_state_imag = self.final_norm(hidden_state_real,hidden_state_imag,residual_real,residual_imag)
            hidden_state = torch.cat([hidden_state_real, hidden_state_imag], dim=-1)
            return hidden_state
        
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn
            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )
                
                

class ComplexNetLM(nn.Module):       
    # default_bitsandbytes_target_modules = [
    #     ".gate_proj.",
    #     ".down_proj.",
    #     ".up_proj.",
    #     ".q_proj.",
    #     ".k_proj.",
    #     ".v_proj.",
    #     ".o_proj.",
    # ]
    # bitsandbytes_stacked_params_mapping = {
    #     # shard_name, weight_name, index
    #     "q_proj": ("qkv_proj", 0),
    #     "k_proj": ("qkv_proj", 1),
    #     "v_proj": ("qkv_proj", 2),
    #     "gate_proj": ("gate_up_proj", 0),
    #     "up_proj": ("gate_up_proj", 1),
    # }

    
    
    def __init__(
        self,
        config: Complexconfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    )->None:
        super().__init__()
           
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
    
        self.model = ComplexNetLMBase(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=ComplexNetDecoderLayer,
        )
    
        if self.pp_group.is_last.rank:    
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix="lm_head",
        )

        else:
            self.lm_head = PPMissingLayer()
    
        
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
    
    
    
    def get_input_embeddings(self) -> Tuple[nn.Embedding,nn.Embedding]:     
        return self.model.get_input_embeddings()
    
    
    
    def get_input_embedding(self,input_ids: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        return self.model.get_input_embedding(input_ids)
    
    
    
    
    @torch.no_grad()
    def forward(
        self,
        input_ids:torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds_real: Optional[torch.Tensor] = None,
        input_embeds_imag: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        
        
       
        
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds_real=input_embeds_real,
            input_embeds_imag=input_embeds_imag,
            pp_proxy_tensors=pp_proxy_tensors,)

            
        if self.pp_group.is_last_rank:
            if not get_embedding:
                logits = self.logits_processor(
                    input_ids,hidden_states,self.lm_head,forward_batch
                )
                return logits
            else:
                return self.pooler(hidden_states,forward_batch)
            
        else:
            return hidden_states



    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer
    
    def get_embed_and_head(self):
        return self.model.embed_tokens_real.weight,self.model.embed_tokens_imag.weight, self.lm_head.weight
    
    def set_embed_and_head(self, embed_real,embed_imag, head):
        del self.model.embed_tokens_real.weight
        del self.model.embed_tokens_imag.weight
        del self.lm_head.weight
        self.model.embed_tokens_real.weight = embed_real
        self.model.embed_tokens_imag.weight = embed_imag
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)
    
    
    
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

    
    
    
EntryClass = ComplexNetLM