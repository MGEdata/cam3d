import copy

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from torchvision.transforms import Normalize as Nor


def _d255(x):
    return x / 255


def _m255(x):
    return x * 255


def _check255(x):
    if isinstance(x, torch.Tensor):
        return torch.max(x) > 1.1
    elif isinstance(x, np.ndarray):
        return np.max(x) > 1.1
    else:
        raise TypeError("x must be torch.Tensor or np.ndarray")


def trans_type(x, tp="float"):
    if isinstance(x, torch.Tensor):
        if tp == "int":
            f = x.int()
        else:
            f = x.float()
    elif isinstance(x, np.ndarray):
        if tp == "int":
            f = x.astype(np.int8)
        else:
            f = x.astype(np.float32)
    else:
        raise TypeError("x must be torch.Tensor or np.ndarray")

    return f


def tensor3c(tensor):
    return torch.is_tensor(tensor) and tensor.ndimension() >= 3


def ndarray3c(tensor):
    return isinstance(tensor, np.ndarray) and tensor.ndim >= 3


###################


def _mul255_(x):
    if _check255(x):
        pass
    else:
        x *= 255
    x = trans_type(x, tp="int")
    return x


def mul255(x):
    x1 = copy.deepcopy(x)
    return _mul255_(x1)


def _div255_(x):
    x = trans_type(x, tp="float")
    if _check255(x):
        x /= 255
    else:
        pass
    return x


def div255(x):
    x1 = copy.deepcopy(x)
    return _div255_(x1)


def trans_to_numpy(x, channel=True):
    x1 = copy.deepcopy(x)
    return _trans_to_numpy_(x1, channel=channel)


def _trans_to_numpy_(x, channel=True):
    # x = x.squeeze()

    if isinstance(x, torch.Tensor):
        if tensor3c(x) and channel:
            x = x.numpy()
            dim = list(range(x.ndim))
            dim.remove(0)
            dim.append(0)

            return x.transpose(dim)
        else:
            return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError("x must be torch.Tensor or np.ndarray")


def trans_to_tensor(x, channel=True):
    x1 = copy.deepcopy(x)
    return _trans_to_tensor_(x1, channel=channel)


def _trans_to_tensor_(x, channel=True):
    # x = x.squeeze()
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        if ndarray3c(x) and channel:
            dim = list(range(x.ndim))
            dim = [-1] + dim
            dim.pop()
            return torch.from_numpy(x.transpose(dim))
        else:
            return torch.from_numpy(x)
    else:
        raise TypeError("x must be torch.Tensor or np.ndarray")


def trans_to_tensors(x, channel=True):
    return torch.stack([trans_to_tensor(xi, channel=channel) for xi in x], dim=0)


def trans_to_numpys(x, channel=True):
    return np.stack([trans_to_numpy(xi, channel=channel) for xi in x])


class Normalize(Nor):

    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace=inplace)

    def transform(self, tensor):
        return self.__call__(self, tensor)

    def inverse_transform(self, tensor):
        return self._it(tensor)

    def _it(self, tensor):
        mean = self.mean
        std = self.std
        inplace = self.inplace
        if not torch.is_tensor(tensor) and tensor.ndimension() == 3:
            raise TypeError('tensor is not a torch image.')

        if not inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        tensor = torch.mul(tensor, std[:, None, None]) + mean[:, None, None]

        return tensor


normalize = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def visualize(img, cam, return_type="t"):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (1, 3c, H, W)
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """
    img = div255(img.cpu())
    cam = div255(cam.cpu())

    if cam.dim() == 2:
        cam = cam.unsqueeze(dim=0)
    if cam.dim() == 3:
        cam = cam.unsqueeze(dim=0)
    if img.dim() == 2:
        img = img.unsqueeze(dim=0)
    if img.dim() == 3:
        img = img.unsqueeze(dim=0)

    *_, H, W = img.shape
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)

    img = img.squeeze(0).numpy()
    cam = cam.squeeze(0).numpy()
    cam = cam.transpose((1, 2, 0))
    img = img.transpose((1, 2, 0))
    cam = cam.astype(np.float32)
    cam = cam.squeeze()

    sc = ScalarMappable(cmap=cm.jet)
    cam = sc.to_rgba(cam)[:, :, :-1]
    camp = cam

    result = camp + img
    result = result / (result.max())
    if return_type is "t":
        result = trans_to_tensor(result)
    return result