"""
Functional AllGather that works with torch.func
"""

import torch
import torch.distributed as dist


def apply_all_gather(tensor, distributed):
    if distributed:
        ret, _ = AllGatherInner.apply(tensor)
        return ret
    else:
        return tensor


class AllGatherInner(torch.autograd.Function):
    @staticmethod
    def forward(tensor):
        world_size = dist.get_world_size()
        current_size = tensor.size(0)
        all_sizes = [torch.tensor([0], device=tensor.device) for _ in range(world_size)]
        size_tensor = torch.tensor([current_size], device=tensor.device)
        dist.all_gather(all_sizes, size_tensor)
        all_sizes = [size.item() for size in all_sizes]
        
        max_size = max(all_sizes)
        
        if current_size < max_size:
            padding_size = max_size - current_size
            padding = torch.zeros((padding_size,) + tensor.shape[1:],
                                dtype=tensor.dtype,
                                device=tensor.device)
            padded_tensor = torch.cat([tensor, padding], dim=0)
        else:
            padded_tensor = tensor
        
        gather_list = [torch.zeros((max_size,) + tensor.shape[1:],
                                 dtype=tensor.dtype,
                                 device=tensor.device) for _ in range(world_size)]
        
        dist.all_gather(gather_list, padded_tensor)
        output = []
        for i, size in enumerate(all_sizes):
            output.append(gather_list[i][:size])
            
        rank = dist.get_rank()
        return torch.cat(output, dim=0), (all_sizes, rank)
    
    @staticmethod
    def backward(ctx, grad_output, other_args):
        rank = ctx.rank
        all_sizes = ctx.all_sizes
        start_idx = sum(all_sizes[:rank])
        size = all_sizes[rank]
        grad_input = grad_output[start_idx:start_idx + size]
        return grad_input
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        output, (all_sizes, rank) = output
        
        ctx.all_sizes = all_sizes
        ctx.rank = rank