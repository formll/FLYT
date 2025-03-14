import torch
import torch.nn as nn

from flyt.distributed import apply_all_gather


class NegativeWeightedClipLoss(nn.Module):
    def __init__(self, world_size=1):
        super().__init__()
        self.world_size = world_size

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        all_image_features = apply_all_gather(image_features, distributed=self.world_size > 1)
        all_text_features = apply_all_gather(text_features, distributed=self.world_size > 1)

        logits_per_image = logit_scale * all_image_features @ all_text_features.T
        logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, loss_weights, output_dict=False):
        device = image_features.device
        loss_weights = loss_weights.squeeze()
        
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        def negative_weighted_loss(inputs, targets, weight):
            exp_inputs = torch.exp(inputs)

            weighted_exp_inputs_j = weight * exp_inputs
            denominator = weighted_exp_inputs_j.sum(dim=1)

            weighted_exp_inputs_i = weight.unsqueeze(1) * exp_inputs
            exp_targets = weighted_exp_inputs_i.gather(1, targets.unsqueeze(1)).squeeze(1)

            log_probs = torch.log(exp_targets/denominator)
            weighted_log_probs = - weight * log_probs
            return weighted_log_probs.sum()
        
        total_loss = (
            negative_weighted_loss(logits_per_image, labels, weight=loss_weights) +
            negative_weighted_loss(logits_per_text, labels, weight=loss_weights)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
