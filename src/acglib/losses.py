from typing import List, Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as F

Outputs = Mapping[str, List[torch.Tensor]]

class LossWrap(nn.Module):
    """LossWrap creates a nn.Module object that allows to preprocessing of the tensors and/or postprocess the loss output.

    :param callable loss_fn: The loss function with input arguments (output, target)
    :param callable preprocess_fn: Preprocess function with signature ``preprocess_fn(output, target)`` that returns
        a tuple ``(output_preprocessed, target_preprocessed)``.
    :param callable postprocess_fn: Postprocess function with signature ``postprocess_fn(loss_output)`` that returns
        a single tensor ``loss_output_postprocessed``.

    :Example:

    >>> # Crossentropy loss needs a tensor of type long as target, but ours is of type float!
    >>> my_loss = LossWrap(nn.CrossEntropyLoss(), preprocess_fn=lambda output, target: (output, target.long()))
    """
    def __init__(self, loss_fn, preprocess_fn=None, postprocess_fn=None):
        super().__init__()
        self.loss_fn = loss_fn
        self.pre_fn = preprocess_fn if preprocess_fn is not None else lambda a, b : (a, b)
        self.post_fn = postprocess_fn if postprocess_fn is not None else lambda _ : _

    def forward(self, output, target):
        output, target = self.pre_fn(output, target)
        loss_output = self.loss_fn(output, target)
        return  self.post_fn(loss_output)


def cross_entropy_with_probs(
    output: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints.
    PyTorch's F.cross_entropy() method requires integer labels; it does accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.
    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.
    Parameters
    ----------
    output
        A [num_points, num_classes] tensor of logits
    target
        A [num_points, num_classes] tensor of probabilistic target labels
    weight
        An optional [num_classes] array of weights to multiply the loss by per class
    reduction
        One of "none", "mean", "sum", indicating whether to return one loss per data
        point, the mean loss, or the sum of losses
    Returns
    -------
    torch.Tensor
        The calculated loss
    Raises
    ------
    ValueError
        If an invalid reduction keyword is submitted
    """
    num_points, num_classes = output.shape
    # Note that t.new_zeros, t.new_full put tensor on same device as t
    cum_losses = output.new_zeros(num_points)
    for y in range(num_classes):
        target_temp = output.new_full((num_points,), y, dtype=torch.long)
        y_loss = F.cross_entropy(output, target_temp, reduction="none")
        if weight is not None:
            y_loss = y_loss * weight[y]
        cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


def soft_crossentropy_with_logits(output, target):
    num_ch = output.size(1)
    out_flat = output.transpose(0, 1).contiguous().view(num_ch, -1).transpose(0, 1)
    tgt_flat = target.transpose(0, 1).contiguous().view(num_ch, -1).transpose(0, 1)
    return cross_entropy_with_probs(out_flat, tgt_flat)



class FocalLoss(nn.Module):
    """Focal loss for multi-class segmentation [Lin2017]_.

    This loss is a difficulty weighted version of the crossentropy loss where more accurate predictions
    have a diminished loss output.

    :param gamma: the *focusing* parameter (where :math:`\\gamma > 0`). A higher gamma will give less weight to confidently
        predicted samples.
    :param list weights: channel-wise weights to multiply before reducing the output.

    .. rubric:: Usage

    :param output: tensor with dimensions :math:`(\\text{BS}, \\text{CH}, \\ast)` containing the output **logits** of the network.
        :math:`\\text{BS}` is the batch size, :math:`\\text{CH}` the number of channels and :math:`\\ast` can be any
        number of additional dimensions with any size.
    :param target: tensor with dimensions :math:`(\\text{BS}, \\ast)` (of type long) containing integer label targets.
    :param reduce: one of ``'none'``, ``'mean'``, ``'sum'`` (default: ``'mean'``)

    .. rubric:: References

    .. [Lin2017] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017. (https://arxiv.org/abs/1708.02002)
    """

    def __init__(self, gamma=2.0, weights=None, reduce='mean'):
        super().__init__()
        assert gamma >= 0.0
        self.g = gamma
        self.w = weights

        self.reduce_fns = {'mean': torch.mean, 'sum': torch.sum, 'none': lambda x : x}
        assert reduce in self.reduce_fns.keys()
        self.reduce = self.reduce_fns[reduce]

    def forward(self, output, target, reduce=None):
        pt_mask = torch.stack([target == l for l in torch.arange(output.size(1))], dim=1).float()
        # pt_softmax = torch.sum(nn.functional.softmax(output, dim=1) * pt_mask, dim=1)
        # pt_log_softmax = torch.sum(torch.nn.functional.log_softmax(output, dim=1) * pt_mask, dim=1)
        pt_softmax = nn.functional.softmax(output, dim=1) * pt_mask
        pt_log_softmax = torch.nn.functional.log_softmax(output, dim=1) * pt_mask
        fl = -1.0 * torch.pow(torch.sub(1.0, pt_softmax), self.g) * pt_log_softmax

        if self.w is not None:
            fl_weights = torch.stack(
                [self.w[n] * pt_mask[:, l, ...] for n, l in enumerate(torch.arange(pt_mask.size(1)))], dim=1)
            fl *= fl_weights
            fl = torch.sum(fl, dim=1) # Sum the channels

        if reduce is not None:
            assert reduce in self.reduce_fns.keys()
            self.reduce = self.reduce_fns[reduce]
        return self.reduce(fl)
