import torch

class DifferentiableOptimizer:
    def __init__(self, lr: float = 0.001):
        # for hacky scheduler using openclip
        self.param_groups = [{'lr': lr}]
        
    @property
    def lr(self) -> float:
        return self.param_groups[0]['lr']
    
    def set_lr(self, lr):
        self.param_groups[0]['lr'] = lr

    
class DifferentiableAdamW(DifferentiableOptimizer):
    def __init__(self, 
                 model_to_opt,
                 lr: float = 0.001, 
                 beta1: float = 0.9, 
                 beta2: float = 0.98, 
                 epsilon: float = 1e-06,
                 gain_or_bias_weight_decay: float = 0.,
                 rest_weight_decay: float = 0.2,
                 train_only_head: bool = False):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = dict()
        self.v = dict()
        self.t = torch.zeros((), dtype=torch.float32)
        self.gain_or_bias_weight_decay = gain_or_bias_weight_decay
        self.rest_weight_decay = rest_weight_decay
        self.train_only_head = train_only_head
        
        # from OpenClip
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)
        named_parameters = list(model_to_opt.named_parameters())
        if self.train_only_head:
            named_parameters = [(n, p) for n, p in named_parameters if 'glu_model' in n]  # head parameters
        self.gain_or_bias_keys = [n for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        self.rest_keys = [n for n, p in named_parameters if include(n, p) and p.requires_grad]

    def get_updated_params(self, params, grads) -> None:
        self.t = self.t + 1    
        new_params = dict()

        for key, p in params.items():
            if key in self.gain_or_bias_keys:
                weight_decay = self.gain_or_bias_weight_decay
            elif key in self.rest_keys:
                weight_decay = self.rest_weight_decay
            else:
                raise Exception(f"key {key} not in the initialized model")
            
            if key not in self.m:
                self.m[key] = torch.zeros_like(p)
                self.v[key] = torch.zeros_like(p)

            # Perform stepweight decay (from torch impl)
            decayed_p = p * (1 - self.lr * weight_decay)

            self.m[key] = self.m[key].detach()
            self.v[key] = self.v[key].detach()

            self.m[key] = (self.beta1 * self.m[key]) + ((1 - self.beta1) * grads[key])
            self.v[key] = (self.beta2 * self.v[key]) + ((1 - self.beta2) * (grads[key]**2))
            
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            # This is so I don't divide by very close to 0 when taking grad
            non_zero_mask = v_hat != 0
        
            new_params[key] = decayed_p
            new_params[key][non_zero_mask] -= (self.lr * m_hat[non_zero_mask] / (torch.sqrt(v_hat[non_zero_mask]) + self.epsilon))
        return new_params
    
    def state_dict(self):
        state_dict = dict()
        params_common_to_groups = {
            'lr': self.lr,
            'betas': (self.beta1, self.beta2),
            'eps': self.epsilon,
            'amsgrad': False,
            'foreach': None,
            'maximize': False,
            'capturable': False,
            'differentiable': False,
            'fused': None
        }
        first_group_len = len(self.gain_or_bias_keys)
        state_dict['param_groups'] = [
            {
                'weight_decay': self.gain_or_bias_weight_decay,
                'params': list(range(first_group_len)),
                **params_common_to_groups
            },
            {
                'weight_decay': self.rest_weight_decay,
                'params': list(range(first_group_len, first_group_len + len(self.rest_keys))),
                **params_common_to_groups
            }
        ]
        state = dict()
        for i, key in enumerate(self.gain_or_bias_keys + self.rest_keys):
            # Check that there is a state
            if self.m.get(key, None) is not None:
                state[i] = {
                    'step': self.t.item(),
                    'exp_avg': self.m[key].detach(),
                    'exp_avg_sq': self.v[key].detach()
                }
        state_dict['state'] = state
        return state_dict

    def load_state_dict(self, state_dict, device='cpu'):
        # TODO set lr from the outside
        self.set_lr(state_dict['param_groups'][0]['lr'])
        self.beta1, self.beta2 = state_dict['param_groups'][0]['betas']
        self.epsilon = state_dict['param_groups'][0]['eps']

        self.gain_or_bias_weight_decay = state_dict['param_groups'][0]['weight_decay']
        self.rest_weight_decay = state_dict['param_groups'][1]['weight_decay']
        self.t = torch.tensor(state_dict['state'][0]['step'], dtype=torch.float32)
        
        state = state_dict['state']
        for i, key in enumerate(self.gain_or_bias_keys + self.rest_keys):
            self.m[key] = state[i]['exp_avg'].to(device)
            self.v[key] = state[i]['exp_avg_sq'].to(device)
