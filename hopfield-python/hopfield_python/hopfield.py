import copy
import ctypes
import io
import itertools
import os
import pathlib
import random
import struct
import sys

import torch
import torchvision
from PIL import Image
import numpy as np

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

###############################################################################
# Transforms
###############################################################################
image_to_binarized_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), # 2-d tensor, in [0, 1]
    torchvision.transforms.Lambda(lambda x: x.view(-1)), # 1-d tensor, in [0, 1]
    torchvision.transforms.Lambda(lambda x: (x >= 0.5).type(torch.float)), # 1-d tensor, in {0.0, 1.0}
    torchvision.transforms.Lambda(lambda x: 2.0 * x - 1.0) # 1-d tensor, in {-1.0, 1.0}
])

def binarized_to_image_transform(shape):
    return torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: x.view(*shape)),
        torchvision.transforms.Lambda(lambda x: 255*((x + 1).type(torch.uint8) // 2)),
        torchvision.transforms.ToPILImage()
    ])

###############################################################################
# Utils
###############################################################################
def tmerge(*iterators):
  empty = {}
  for values in itertools.zip_longest(*iterators, fillvalue=empty):
    for value in values:
      if value is not empty:
        yield value

def image_grid(images, w, h, nx, ny, border=0, bkg=None):
    if bkg is None:
        bkg = 0
    try:
        bx, by = border
    except:
        bx = border
        by = border
    
    total_w = (nx + 1)*bx + nx*w
    total_h = (ny + 1)*by + ny*h
    img = Image.fromarray(np.full((total_h, total_w, 3), bkg, dtype=np.uint8))
    for i, image in enumerate(images):
        row = int(i / nx)
        col = i - (row*nx)
        x_origin = bx + col*(bx + w)
        y_origin = by + row*(by + h)
        img.paste(image, (x_origin, y_origin, x_origin+w, y_origin+h))
    return img


###############################################################################
# Dealing with data
###############################################################################
def _get_mnist():
    mnist_train = torchvision.datasets.MNIST("./mnist_train", train=True, download=True)
    mnist_test = torchvision.datasets.MNIST("./mnist_test", train=False, download=True)
    return mnist_train, mnist_test

def make_batch(xs: Image, transform: Optional[Any]=None) -> torch.tensor:
    """
    Converts an Iterable of PIL `Image`s into a `torch.Tensor` with shape N, X.
    N: The batch size, equal to len(xs).
    X: The examples, flattened and binarized.
    """
    if transform is None:
        transform = image_to_binarized_transform
    return torch.stack(tuple(map(transform, xs)))

def find_data(dataset, labels, examples_per_label=10, random_seed=None, skip_p=0):
    indices = list(range(0, len(dataset)))
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(indices)
    candidates = {label: [] for label in labels}
    for i in indices:
        example = dataset[i]
        if random.uniform(0, 1) < skip_p:
            continue
        if example[1] in candidates:
            candidates[example[1]].append(i)
            # check break condition (yikes)
            if i % examples_per_label*10 == 0:
                if all(map(lambda l: len(l) >= examples_per_label, candidates.values())):
                    break
    candidates = {label: indices[:examples_per_label] for label, indices in candidates.items()}
    return candidates

###############################################################################
# Image Corruption
###############################################################################
def salt_pepper(image: np.ndarray, box: Optional[Tuple[int, int, int, int]]=None, prob: float=0.1) -> np.ndarray:
    """
    Add salt and pepper noise to image
    image: an `ndarray` of dtype np.uint8
    prob: Probability of the noise

    Returns: a new `ndarray` of the salt-and-peppered image array.
    """
    if box is None:
        box = 0, 0, image.shape[1], image.shape[0]
    x0, y0, x1, y1 = box
    output = np.copy(image)
    thres = 1 - prob
    # nested fors are slow but it's ok on small scale.
    for i in range(y0, y1):
        for j in range(x0, x1):
            rdn = random.random()
            if rdn < prob:
                output[i, j] = 0
            elif rdn > thres:
                output[i, j] = 255
            else:
                output[i, j] = image[i, j]
    return output

def blackout(image: np.ndarray, box: Tuple[int, int, int, int], value: int=0) -> np.ndarray:
    """
    Blackout (or value-out) a region of an image.
    image: an `ndarray` of dtype np.uin8
    shape: (x0, y0, x1, y1) of region to be blacked out
    value: value to use to fill in shape.

    Returns: a new `ndarray` of the blacked-out image array.
    """
    x0, y0, x1, y1 = box
    xx, yy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    image = copy.deepcopy(image)
    image[yy, xx] = value
    return image

###############################################################################
# HopfieldPattern
###############################################################################
class HopfieldPattern:
    def __init__(self, size: int, data: Iterable[float]):
        self.size = size
        self.data = list(data)

        if len(self.data) != self.size:
            raise ValueError(f"error: got size = {self.size} but len(self.data) = {len(self.data)}")
    
    def as_bytes(self) -> bytearray:
        """
        Serialize the `HopfieldPattern` to a `bytearray`.
        """
        # preparation
        ba: bytearray = bytearray()
        size: bytes = self.size.to_bytes(4, "little")
        data = (ctypes.c_float * len(self.data))()
        data[:] = self.data

        # construction
        ba += size
        ba += data
        return ba

    def export(self, path: Union[str, pathlib.Path]) -> None:
        """
        Export the `HopfieldPattern` to a file. The exported format is derived from the
        `self.as_bytes()` function.
        """
        try:
            with open(path, "wb") as f:
                f.write(self.as_bytes())
        except OSError:
            os.remove(path)
            raise
    
    @classmethod
    def from_image(cls, img):
        if len(img.size) != 2:
            raise ValueError(f"error: len(img.size) = {len(img.size)} != 2; only single-channel images are supported.")
        data = list(np.array(img).flatten() / 255)
        size = len(data)
        return cls(size=size, data=data)


###############################################################################
# HopfieldNetwork
###############################################################################
class HopfieldNetwork(torch.nn.Module):
    EXPORT_HEADER_MAGIC: bytes = b"HOPFIELD"
    def __init__(self, size: int, dtype: torch.dtype=torch.float):
        super().__init__()
        self.size: int = size
        self.dtype: torch.dtype = dtype
        self.weights: Optional[torch.Tensor] = None
    
    def fit(self, X: torch.Tensor) -> torch.Tensor:
        """
        Train the Hopfield Network.
        
        X: A `torch.Tensor` with shape `(B, N)`, where `N == self.size` and `B` is the number of examples to imprint.
        
        Returns: A weight matrix W of the trained Hopfield network. Also sets the `weights` matrix of the network to this matrix.
        """
        assert len(X.shape) == 2, f"X should be a dataset of shape (B, N); got (len(X.shape) = {len(X.shape)}) != 2."
        assert X.shape[1] == self.size, f"shape mismatch; (X.shape[1] = {X.shape[1]}) != (self.size = {self.size})."
        
        n = self.size # number of neurons
        N = X.shape[0] # number of examples
        
        w: torch.Tensor = torch.nn.Parameter(
            torch.zeros((n, n), dtype=self.dtype),
            requires_grad=False)
        
        einsum = torch.einsum('bi,bj->bij', (X, X))
        w = (1/n)*torch.sum(einsum, dim=0)
        # set diagonal to 0 programmatically rather than mathematically.
        itensor = torch.arange(0, n, dtype=torch.long)
        w[itensor, itensor] = torch.zeros(n)
        
        tr = torch.trace(w)
        assert tr <= 1e-12, f"computed weights matrix has nonzero trace: {tr}"
        
        self.weights = w.clone()
        return w
    
    def forward(self, x: torch.Tensor, iterations: int=150, record_activities: bool=False) -> torch.Tensor:
        """
        Query the Hopfield network memory.
        
        x: A `torch.Tensor` with shape `(B, N)`, where `N == self.size` and `B` is the number of datapoints to individually query.
        """
        return torch.stack(tuple(map(self._forward_single, x)))
    
    def _forward_single(self, x: torch.Tensor, iterations: int=150) -> torch.Tensor:
        """
        Query the Hopfield network memory with a single example.
        
        x: A `torch.Tensor` with shape `(N)`, where `N == self.size`.
        """
        y = x.clone()
        iteration = 0
        indices = list(range(self.size))
        while iteration < iterations:
            random.shuffle(indices)
            for i in indices:
                z = torch.dot(self.weights[i], y)
                y[i] = torch.sign(z).item()
            iteration += 1
        return y

    
    def as_bytes(self) -> bytearray:
        """
        Serialize the Hopfield Network to a `bytearray`; this includes a magic header,
        the `size` of the network, and the `weights` matrix of the network.
        """
        # preparation
        ba: bytearray = bytearray()
        magic: bytes = self.EXPORT_HEADER_MAGIC
        size: bytes = self.size.to_bytes(4, "little")

        weight_iter = self.weights.view(-1)
        weights = (ctypes.c_float * len(weight_iter))()
        weights[:] = list(weight_iter)
        #weights: bytes = b"".join(map(lambda t: struct.pack(f"@f", t.item()), self.weights.view(-1)))
        
        # construction
        ba += magic
        ba += size
        ba += weights
        return ba

    def export(self, path: Union[str, pathlib.Path]) -> None:
        """
        Export the Hopfield Network to a file. The exported format is derived from the
        `self.as_bytes()` function.
        """
        try:
            with open(path, "wb") as f:
                f.write(self.as_bytes())
        except OSError:
            os.remove(path)
            raise
