# iFairy_struct

## ComplexNetRMSNorm

### weight : [rms_real,rms_image]
### input and output:
1. x_real,x_imag -> x_real,x_image
2. x_real,x_imag,res_real,res_imag -> x_real,x_imag,res_real,res_imag

### Achieve Way
1. torch.cat[[x_ral,x_imag],dim = -1]  -> RMSNorm

2. x_real += res_real,x_image += res_imag,res_real = x_real,res_imag = x_image -> RMSnorm 

##  IntergrateRealAndImag

### wight: None
### input and output:
real_product,imag_product -> real,imag
### Achieve way
```
    r_r,r_i = real_product.split(splite_dim, dim=-1)
    i_r,i_i = imag_product.split(splite_dim, dim=-1)
    output_real = r_r + i_i
    output_imag = r_i - i_r
```


## ComplexLinear

### weight: [weight_real,weight_imag] 

### input and output:

input_real,input_imag (dim = input_dim)-> input_real,input_imag (dim = output_dim)

### Achieve way:
torch.cat[[x_ral,x_imag],dim = 0] -> RowParallelLinear ->  torch.chunk(Merged_output, 2, dim=0) -> IntergrateRealAndImag



## ComplexUpLinear

### weight: [gate_weight_real,gate_weight_imag,up_weight_real,up_weight_imag]

### input and output
input_real,input_imag -> [gate_real,up_real],[gate_imag,up_imag]

### Achieve way

input_real,input_imag -> MergedColumnParallelLinear -> real_product, imag_product
 -> real_product.split(dim=-1), imag_product.split(dim=-1) ->IntergrateRealAndImag -> torch.cat([Gate_real, Up_real], dim=-1),torch.cat([Gate_imag, Up_imag], dim=-1)



## ComplexQKVLinear

### weight: [q_real,q_imag,k_real,k_imag,v_real,v_imag]

### input and output

input_real,input_image -> q_real,q_imag,k_real,k_imag,v_real,v_imag

### achieve way

input_real,input_image -> ComplexQKVLinear ->torch.chunk(Merged_qkv, 2, dim=0) ->split -> IntergrateRealAndImag -> q_real,q_imag,k_real,k_imag,v_real,v_imag



## ComplexRelu2AndMul

### weight: None

### input and output:

x_real,x_imag -> x_real,x_imag

### achieve way
```
d = x_real.shape[-1] // 2
gate_real,up_real = x_real.split([d, d], dim=-1)
gate_imag,up_imag = x_imag.split([d, d], dim=-1)
         
gate_real,gate_imag = complex_relu2(gate_real,gate_imag)
        
output_real = gate_real * up_real + gate_imag * up_imag
output_imag = gate_real * up_imag - gate_imag * up_real

```

### ComplexNetMLP

### weight: [gate_weight_real,gate_weight_imag,up_weight_real,up_weight_imag] [rms_real,rms_image]  [weight_real,weight_imag] 

### input and output：

x_real,x_imag -> x_real,x_imag 

### achieve way


x_real,x_imag -> ComplexUpLinear -> ComplexRelu2AndMul -> ComplexLinear


