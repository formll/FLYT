import torch
import torch.distributed as dist

from typing import cast


class DifferentiableGradScaler:
    def __init__(self, args, init_scale=2.0**16, active=True):
        self._scale = torch.full((), init_scale, dtype=torch.float32, device=args.device)
        self.growth_factor = 2.0
        self.backoff_factor = 0.5
        self.growth_interval = 2000
        self.world_size = args.world_size
        self.device = args.device
        self.growth_tracker = torch.full((), 0, dtype=torch.int32, device=self.device)
        self.active = active

    def scale(self, outputs):
        if self.active:
            return outputs * self._scale.to(device=self.device, non_blocking=True)
        return outputs
    
    def inv_scale(self):
        if self.active:
            return self._scale.double().reciprocal().float().to(self.device)
        else:
            return torch.full((), 1.0, dtype=torch.float32, device=self.device)
    
    def unscale_grads(self, grads):
        found_inf = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        
        if not self.active:
            return grads, found_inf
            
        unscaled_grads = {}
        inv_scale = self.inv_scale()

        for name, grad in grads.items():
            unscaled_grad = grad * inv_scale

            found_inf = torch.logical_or(found_inf, torch.isinf(unscaled_grad).any()).float()
            found_inf = torch.logical_or(found_inf, torch.isnan(unscaled_grad).any()).float()

            unscaled_grads[name] = unscaled_grad

        if self.world_size > 1:
            dist.all_reduce(found_inf, op=dist.ReduceOp.MAX)
        
        return unscaled_grads, found_inf

    def update(self, found_inf) -> None:
        if self.active:
            torch._amp_update_scale_(
                self._scale,
                self.growth_tracker,
                found_inf,
                self.growth_factor,
                self.backoff_factor,
                self.growth_interval,
            )

    def state_dict(self):
        return {
            "scale": self._scale.item(),
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
            "_growth_tracker": self.growth_tracker.item(),
        }

    def load_state_dict(self, state_dict):
        self._scale = torch.full((), state_dict["scale"], dtype=torch.float32, device=self.device)
        self.growth_factor = cast(float, state_dict["growth_factor"])
        self.backoff_factor = cast(float, state_dict["backoff_factor"])
        self.growth_interval = cast(int, state_dict["growth_interval"])
        self.growth_tracker = torch.full((), state_dict["_growth_tracker"], dtype=torch.int32, device=self.device)
