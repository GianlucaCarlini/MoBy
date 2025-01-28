import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
import os
from typing import Union, Callable, Any
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

__all__ = ["PatchDataset", "ImageDataset"]


def gaussian_sampling(volume_size: tuple, patch_size: tuple, std_factor: int = 8):
    """Sample a patch using a gaussian distribution centered at the center of the volume.

    Parameters
    ----------
    volume_size : tuple
        The size of the volume in the x, y, and z dimensions.
    patch_size : tuple
        The size of the patch to sample in the x, y, and z dimensions.
    std_factor : int, optional
        Controls the spreading of the gaussian. The standard deviation to use is
        calculated as the volume size divided by this factor; so the smaller the factor,
        the more spread out the gaussian. Defaults to 8.

    Returns
    -------
    tuple
        The x, y, and z indices of the sampled patch.
    """
    if patch_size[0] < 0:
        return 0, 0, 0

    x, y, z = volume_size

    x_offset = patch_size[0] / 2
    y_offset = patch_size[1] / 2
    z_offset = patch_size[2] / 2

    center_x = x / 2 - x_offset
    center_y = y / 2 - y_offset
    center_z = z / 2 - z_offset

    std_x = x / std_factor
    std_y = y / std_factor
    std_z = z / std_factor

    extract_idx_x = np.random.normal(loc=center_x, scale=std_x)
    extract_idx_y = np.random.normal(loc=center_y, scale=std_y)
    extract_idx_z = np.random.normal(loc=center_z, scale=std_z)

    extract_idx_x = int(np.clip(extract_idx_x, 0, x - patch_size[0]))
    extract_idx_y = int(np.clip(extract_idx_y, 0, y - patch_size[1]))
    extract_idx_z = int(np.clip(extract_idx_z, 0, z - patch_size[2]))

    return extract_idx_x, extract_idx_y, extract_idx_z


def uniform_sampling(volume_size: tuple, patch_size: tuple):
    """Sample a patch using a uniform distribution.

    Parameters
    ----------
    volume_size : tuple
        The size of the volume in the x, y, and z dimensions.
    patch_size : tuple
        The size of the patch to sample in the x, y, and z dimensions.

    Returns
    -------
    tuple
        The x, y, and z indices of the sampled patch.
    """
    if patch_size[0] < 0:
        return 0, 0, 0

    x, y, z = volume_size

    if patch_size is None:
        patch_size = (0, 0, 0)

    extract_idx_x = np.random.randint(0, max(x - patch_size[0], 1))
    extract_idx_y = np.random.randint(0, max(y - patch_size[1], 1))
    extract_idx_z = np.random.randint(0, max(z - patch_size[2], 1))

    return extract_idx_x, extract_idx_y, extract_idx_z


sampling_functions = {"uniform": uniform_sampling, "gaussian": gaussian_sampling}


class PatchDataset(Dataset):
    """
    Dataset for lazy loading patches from images and labels.
    It samples a random index from the image and label and
    extracts a patch of the specified size.
    If the number of non-zero voxels in the label is less than the
    threshold, it will sample another index until the threshold is met.

    Attributes
    ----------
    images_dir : str
        path to the image directory
    labels_dir : str
        path to the label directory
    patch_size : tuple | int, optional
        Size of the patches to load as a tuple of ints
        representing the x, y, and z dimension. If a single int is provided,
        the same value will be used for x, y, and z.
        If less than 0, the whole volume is used. Defaults to None.
    sampling_method : str, optional
        Sampling method to use. Can be either
        "uniform" or "gaussian". Defaults to "uniform".
    threshold : float, optional
        Threshold value to consider for patch sampling.
        If the sum of non-zero pixels in the sampled patch is lower than
        threshold, then another patch is sampled until the threshold condition is met
        Defaults to None.
    transform : callable, optional
        Optional transform to apply to the image and label.
        The same transform is applied to both. Defaults to None.
    preprocessing : callable, optional
        Optional preprocessing to apply to the image.
        Defaults to None.

    Methods
    -------
    __getitem__(index)
        Returns the image and label at the specified index.
    __len__()
        Returns the length of the dataset.
    get_relative_index()
        Returns the relative index of the sampled patch. By relative index
        we mean the index of the patch divided by the size of the volume, so that
        the index is in the range [0, 1], and different volumes are comparable.

    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str = None,
        patch_size: Union[tuple, int] = None,
        sampling_method: str = "uniform",
        threshold: float = None,
        transform: Callable = None,
        preprocessing: Callable = None,
        positional: bool = False,
        repeat: int = 1,
        **kwargs,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.ids = os.listdir(self.images_dir)

        if repeat > 1:
            self.ids = self.ids * repeat

        self.images = [os.path.join(self.images_dir, image_id) for image_id in self.ids]

        if self.labels_dir is not None:
            self.labels = [
                os.path.join(self.labels_dir, image_id) for image_id in self.ids
            ]
        else:
            self.labels = self.images

        if preprocessing is not None:
            self.preprocessing = preprocessing
        else:
            self.preprocessing = None

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

        self.positional = positional

        self.reader = sitk.ImageFileReader()

        self.sampling_method = sampling_method

        self.sampling_function = sampling_functions.get(self.sampling_method, None)

        if self.sampling_function is None:
            raise NotImplementedError(
                f"Sampling method {self.sampling_method} not implemented, available methods are: {sampling_functions.keys()}"
            )

        if patch_size is not None:
            if isinstance(patch_size, int):
                self.patch_size = (patch_size, patch_size, patch_size)
            else:
                self.patch_size = patch_size
        else:
            self.patch_size = (96, 96, 96)

        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = 0.0

        self.len = len(self.ids)

        self.kwargs = kwargs

    def __getitem__(self, index):
        self.reader.SetFileName(self.labels[index])
        self.reader.ReadImageInformation()

        self.x, self.y, self.z = self.reader.GetSize()

        if self.patch_size[0] < 0:
            patch_size = (self.x, self.y, self.z)
        else:
            patch_size = self.patch_size

        while True:
            (
                self.extract_idx_x,
                self.extract_idx_y,
                self.extract_idx_z,
            ) = self.sampling_function(
                volume_size=(self.x, self.y, self.z),
                patch_size=patch_size,
                **self.kwargs,
            )

            self.reader.SetExtractIndex(
                (self.extract_idx_x, self.extract_idx_y, self.extract_idx_z)
            )
            self.reader.SetExtractSize((patch_size[0], patch_size[1], patch_size[2]))

            label = self.reader.Execute()
            label = sitk.GetArrayFromImage(label)

            label = np.abs(label)

            if np.sum(label > 0.0) > self.threshold * (
                patch_size[0] * patch_size[1] * patch_size[2]
            ):
                break

        self.reader.SetFileName(self.images[index])

        image = self.reader.Execute()

        if self.preprocessing is not None:
            image = self.preprocessing(image)

        image = sitk.GetArrayFromImage(image)

        image = np.expand_dims(image, axis=0)

        if self.transform is not None:
            image, label = self.transform(image, label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        if self.positional:
            rel_idx = self.get_relative_index()
            rel_idx = torch.from_numpy(np.array(rel_idx)).float()

            return image, label, rel_idx

        if self.labels_dir is None:
            return image

        return image, label

    def __len__(self):
        return self.len

    def get_relative_index(self):
        rel_x = self.extract_idx_x / self.x
        rel_y = self.extract_idx_y / self.y
        rel_z = self.extract_idx_z / self.z

        return [[rel_z], [rel_y], [rel_x]]


class SSLPatchDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        patch_size: Union[tuple, int] = None,
        sampling_method: str = "uniform",
        threshold: float = None,
        transform: Callable = None,
        preprocessing: Callable = None,
        repeat: int = 1,
        **kwargs,
    ):

        self.images_dir = images_dir
        self.ids = os.listdir(self.images_dir)

        if repeat > 1:
            self.ids = self.ids * repeat

        self.images = [os.path.join(self.images_dir, image_id) for image_id in self.ids]

        if preprocessing is not None:
            self.preprocessing = preprocessing
        else:
            self.preprocessing = None

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

        self.reader = sitk.ImageFileReader()

        self.sampling_method = sampling_method

        self.sampling_function = sampling_functions.get(self.sampling_method, None)

        if self.sampling_function is None:
            raise NotImplementedError(
                f"Sampling method {self.sampling_method} not implemented, available methods are: {sampling_functions.keys()}"
            )

        if patch_size is not None:
            if isinstance(patch_size, int):
                self.patch_size = (patch_size, patch_size, patch_size)
            else:
                self.patch_size = patch_size
        else:
            self.patch_size = (96, 96, 96)

        if threshold is not None:
            self.threshold = threshold

        self.len = len(self.ids)

    def __getitem__(self, index):
        self.reader.SetFileName(self.images[index])
        self.reader.ReadImageInformation()

        self.x, self.y, self.z = self.reader.GetSize()

        if self.patch_size[0] < 0:
            patch_size = (self.x, self.y, self.z)
        else:
            patch_size = self.patch_size

        while True:
            (
                self.extract_idx_x,
                self.extract_idx_y,
                self.extract_idx_z,
            ) = self.sampling_function(
                volume_size=(self.x, self.y, self.z),
                patch_size=patch_size,
                **self.kwargs,
            )

            self.reader.SetExtractIndex(
                (self.extract_idx_x, self.extract_idx_y, self.extract_idx_z)
            )
            self.reader.SetExtractSize((patch_size[0], patch_size[1], patch_size[2]))

            image = self.reader.Execute()
            image = sitk.GetArrayFromImage(image)

            if np.sum(np.abs(image) > 0.0) > self.threshold * (
                patch_size[0] * patch_size[1] * patch_size[2]
            ):
                break

        if self.preprocessing is not None:
            image = self.preprocessing(image)

        image = np.expand_dims(image, axis=0)

        if self.transform is not None:
            image_1 = self.transform(image)
            image_2 = self.transform(image)

        image_1 = torch.from_numpy(image_1).float()
        image_2 = torch.from_numpy(image_2).float()

        return image_1, image_2


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, pre_processing=None):
        self.image_dir = image_dir
        self.transform = transform
        self.pre_processing = pre_processing

        self.image_ids = os.listdir(image_dir)

        self.images = [os.path.join(image_dir, image_id) for image_id in self.image_ids]

        self.images = self.images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = PIL.Image.open(self.images[idx]).convert("RGB")

        if self.pre_processing:
            image = self.pre_processing(image)

        if self.transform:
            image = self.transform(image)

        return image
