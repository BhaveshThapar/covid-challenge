"""
Loss functions for Covid-19 Detection.

Includes Focal Loss for handling class imbalance and hard examples.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces the relative loss for well-classified examples,
    focusing training on hard, misclassified cases.
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        C = logits.size(1)

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Gather per-sample class probabilities
        ce_loss = F.cross_entropy(logits, targets, reduction='none',
                                  label_smoothing=self.label_smoothing)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * ce_loss
        return loss.mean()
