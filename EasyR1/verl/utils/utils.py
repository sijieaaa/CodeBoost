from peft import PeftModelForCausalLM
import torch
from peft.utils.other import transpose



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters and total parameters in the model.

    Note: This method is equivalent to PeftModel.print_trainable_parameters().
    It counts both standard parameters and special handling for bitsandbytes 4-bit layers
    and DeepSpeed zero-initialized parameters.

    It differs from `model.num_parameters(only_trainable=True)` in HuggingFace Transformers,
    which counts only the backbone model parameters, not the PEFT-modified model (e.g., LoRA adapters).
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()

        # Handle DeepSpeed ZeRO-3: empty param tensors may have ds_numel
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Handle bitsandbytes 4bit params
        if param.__class__.__name__ == "Params4bit":
            num_params *= 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )


# peft/tuners/lora/layer.py
def get_delta_weight(self, adapter) -> torch.Tensor:
    """
    Compute the delta weight for the given adapter.

    Args:
        adapter (str):
            The name of the adapter for which the delta weight should be computed.
    """
    device = self.lora_B[adapter].weight.device
    dtype = self.lora_B[adapter].weight.dtype

    # In case users wants to merge the adapter weights that are in
    # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
    cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight

    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()

    output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)

        # cast back the weights
        self.lora_A[adapter].weight.data = weight_A.to(dtype)
        self.lora_B[adapter].weight.data = weight_B.to(dtype)

    return output_tensor
    



def check_lora_compatibility(model: PeftModelForCausalLM):
    # lora_A: [r, in_dim]
    # lora_B: [out_dim, r]
    
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            r = module.r['default']
            lora_A = module.lora_A
            lora_B = module.lora_B
            base_layer = module.base_layer

            weight_A = lora_A['default'].weight
            weight_B = lora_B['default'].weight
            # weight_base_layer = base_layer.weight

            in_dim_A = lora_A['default'].in_features
            # out_dim_A = lora_A['default'].out_features
            # in_dim_B = lora_B['default'].in_features
            out_dim_B = lora_B['default'].out_features
            # in_dim = base_layer.in_features
            # out_dim = base_layer.out_features

            assert not module.fan_in_fan_out
            if weight_A.shape[0] == 0:
                None
            else:
                assert weight_A.shape == (r, in_dim_A)
                assert weight_B.shape == (out_dim_B, r)



