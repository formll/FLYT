import logging
import torch
import torch.nn.functional as F
from torch import nn

from typing import Optional, Tuple, Union

from open_clip.src.open_clip.transform import merge_preprocess_kwargs
from open_clip.src.open_clip.factory import create_model


class GluFeedForwardLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        out_dim: int,
        bias: bool,
        act='silu',
        dtype: str = None
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias, dtype=dtype)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=bias, dtype=dtype)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias, dtype=dtype)
        if act == 'silu':
            self.act = F.silu
        elif act == 'relu':
            self.act = F.relu
        else:
            raise NotImplementedError(f'unknown activation {act}')
            

    def forward(self, x):
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class GluFeedForwardModel(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        bias: bool,
        act: str,
        dtype = None
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.bias = bias
        self.layers = nn.ModuleList()
        
        for _ in range(n_layers - 1):
            # TODO: think about changing to (dim, hidden_dim, hidden_dim) and backward comp. 
            # Doesn't matter when I only use 1 layer and/or keep the dim the same.
            self.layers.append(GluFeedForwardLayer(dim, hidden_dim, dim, bias, act=act, dtype=dtype))
        self.layers.append(GluFeedForwardLayer(dim, hidden_dim, out_dim, bias, act=act, dtype=dtype))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
    
    def get_classifier(self):
        return self.layers[-1].w2


class ScoringModel(nn.Module):
    def __init__(self, clip_model, n_layers=1, dtype=None, train_full_scoring=False, hidden_dim_mod=1, act='silu', cos_sim_init=False, noise_std=0.01):
        super().__init__()
        assert cos_sim_init == False or hidden_dim_mod == 1, "either cosine sim or custom hidden dimension."
        self.clip_model = clip_model
        self.noise_std = noise_std
        concat_dim = self.clip_model.visual.output_dim * 2
        hidden_dim = int(concat_dim * hidden_dim_mod)
        self.glu_model = GluFeedForwardModel(concat_dim, hidden_dim, 1, n_layers, act=act, bias=True, dtype=dtype)
        
        if cos_sim_init:
            self.init_cosine_similarity()
    
        if not train_full_scoring:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def init_cosine_similarity(self):
        logging.info(f'Initializing GLU with cosine similariy.')
        dim = self.clip_model.visual.output_dim
        positives = torch.ones(dim)
        negatives = torch.ones(dim) * -1
        
        layers = self.glu_model.layers
        layers[0].w1.weight.data = torch.diag(torch.cat([positives, negatives])) + self.noise_std * torch.randn_like(layers[0].w1.weight.data)
        layers[0].w3.weight.data = (torch.diag(positives, dim) + torch.diag(negatives, -dim)) + self.noise_std * torch.randn_like(layers[0].w3.weight.data)
        layers[0].w2.weight.data = torch.ones(dim*2).reshape(1, -1) + self.noise_std * torch.randn_like(layers[0].w2.weight.data)

        layers[0].w1.bias.data = self.noise_std * torch.randn_like(layers[0].w1.bias.data)
        layers[0].w2.bias.data = self.noise_std * torch.randn_like(layers[0].w2.bias.data)
        layers[0].w3.bias.data = self.noise_std * torch.randn_like(layers[0].w3.bias.data)

    def forward(self, image, text):
        x = self.clip_model(image, text)
        image_embed, text_embed = x[0], x[1]
        concat = torch.concatenate([image_embed, text_embed], dim=1)
        scores = self.glu_model(concat)

        return scores
    
    
class MixingModel(nn.Module):
    def __init__(self, num_scores, dtype=None):
        super().__init__()
        layers = [nn.Linear(num_scores, 1, dtype=dtype)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze()


def create_scoring_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        n_scoring_layers: Optional[int] = 1,
        train_full_scoring: Optional[bool] = False,
        hidden_dim_mod: Optional[float] = 1,
        num_scores: Optional[int] = 0,
        full_scoring_pretrained: Optional[str] = None,
        glu_activation: Optional[str] = 'silu',
        cos_sim_init: Optional[bool] = False,
        noise_std: Optional[float] = 0.01,
        **model_kwargs,
):
    dtype = None
    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        
    if num_scores == 0:  # FLYT
        force_preprocess_cfg = merge_preprocess_kwargs(
            {}, mean=image_mean, std=image_std, interpolation=image_interpolation, resize_mode=image_resize_mode)

        # Create base
        clip_model = create_model(
            model_name,
            pretrained,
            precision=precision,
            device=device,
            jit=jit,
            force_quick_gelu=force_quick_gelu,
            force_custom_text=force_custom_text,
            force_patch_dropout=force_patch_dropout,
            force_image_size=force_image_size,
            force_preprocess_cfg=force_preprocess_cfg,
            pretrained_image=pretrained_image,
            pretrained_hf=pretrained_hf,
            cache_dir=cache_dir,
            output_dict=output_dict,
            load_weights_only=False,
            **model_kwargs,
        )

        # + head
        model = ScoringModel(
            clip_model, 
            n_layers=n_scoring_layers, 
            dtype=dtype, 
            train_full_scoring=train_full_scoring, 
            hidden_dim_mod=hidden_dim_mod,
            act=glu_activation,
            cos_sim_init=cos_sim_init,
            noise_std=noise_std)
    else:  # M-FLYT
        model = MixingModel(num_scores=num_scores, dtype=dtype)
    
    if full_scoring_pretrained is not None:
        model_cp = torch.load(full_scoring_pretrained, map_location='cpu', weights_only=False)
        if "state_dict" in model_cp.keys():  # Model weights while training
            model.load_state_dict(model_cp["state_dict"])
        else:  # Raw model weights
            model.load_state_dict(model_cp)
        logging.info(f'Loaded pretrained scoring model from {full_scoring_pretrained}')
    model.to(device=device)

    return model