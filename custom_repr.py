import torch

original_repr = torch.Tensor.__repr__


def custom_repr(self):
    return f"{{Tensor:{tuple(self.shape)}, {self.device}, {self.dtype}}} {original_repr(self)}"


def enable_custom_repr():
    torch.Tensor.__repr__ = custom_repr