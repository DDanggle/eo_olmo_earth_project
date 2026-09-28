"""Finite mask-loss/gradient checks for the unchanged B0/B2 connection contract."""
import torch

ROLES = frozenset({'encoder', 'head', 'qwen', 'unused_connector'})


def require_finite_loss(loss):
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ValueError('Mask loss must be one scalar tensor')
    if not bool(torch.isfinite(loss).all()):
        raise ValueError('Nonfinite mask loss')


def validate_gradient_contract(modules, arm):
    if set(modules) != ROLES or arm not in {'B0', 'B2'}:
        raise ValueError('Gradient role/arm contract')
    counts = {}
    for name, module in modules.items():
        count = 0
        for index, parameter in enumerate(module.parameters()):
            gradient = parameter.grad
            if gradient is None:
                continue
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError(f'Nonfinite gradient: {name} parameter {index}')
            count += bool(gradient.abs().max() > 0)
        counts[name] = count
    if not (counts['head'] > 0 and counts['qwen'] == 0 and counts['unused_connector'] == 0):
        raise ValueError('Gradient boundaries')
    if (counts['encoder'] > 0) != (arm == 'B2'):
        raise ValueError('EO arm gradient contract')
    return counts
