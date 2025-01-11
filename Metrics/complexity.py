import torch
from calflops import calculate_flops

def COMPLEXITY(model, input_shape):
    input_shape = tuple([1] + input_shape)
    with torch.no_grad():
        flops, macs, params = calculate_flops(
            model=model, input_shape=input_shape, output_precision=4, 
            print_results=False, print_detailed=False, output_as_string=False,
        )
    return (flops, macs, params)