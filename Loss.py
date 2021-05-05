import warnings
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_enum(reduction):
    # type: (str) -> int
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    return ret

# In order to support previous versions, accept boolean size_average and reduce
# and convert them into the new constants for now


# We use these functions in torch/legacy as well, in which case we'll silence the warning
def legacy_get_string(size_average, reduce, emit_warning=True):
    # type: (Optional[bool], Optional[bool], bool) -> str
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret


def legacy_get_enum(size_average, reduce, emit_warning=True):
    # type: (Optional[bool], Optional[bool], bool) -> int
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))
class _Loss(torch.nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor
    if size_average is not None or reduce is not None:
        reduction = legacy_get_string(size_average, reduce)
    return torch.nn.functional.nll_loss(torch.tensor([1.0,2.0,3.0]).to(device)*torch.nn.functional.log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
class CrossEntropyLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='sum'):
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)