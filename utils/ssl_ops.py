import numpy as np
import torch
import torch.nn as nn
from numpy.random import randint


def corrupt_image(
    image: torch.Tensor,
    patch_size: int = 32,
    frac: float = 0.2,
    mode: str = "disc",
    return_proportion: bool = False,
) -> torch.Tensor:
    """
    corrupt an image by modifying patches of a given size.
    It expects the image to be in the format (batch, channels, height, width, (depth)).
    The percentage of patches to be shuffled is given by frac.
    Note that the image dimensions should be exactly divisible by the patch_size;
    I don't know the behavior of the function otherwise.
    This function should work for both 2D and 3D images. In the case of 3D images,
    a single channel is assumed.

    Parameters
    ----------
    image : torch.Tensor
        Image to be corrupted.
    patch_size : int, optional
        Size of the patches to corrupt, by default 32
    frac : float, optional
        Percentage of patches to corrupt, by default 0.2
    mode : str, optional
        Mode of corruption, either "disc" or "mask", by default "disc".
        If "disc", the patches will be shuffled. If "mask", the patches will be zeroed-out.
        Disc is short for discombobulate, which is a fun word.
    return_proportion : bool, optional
        If True, the proportion of patches that were shuffled will be returned.
        More in detail, it will return a matrix nxn where n is the number of images in the batch.
        The (i, j) element of the matrix represents the proportion of patches of the image j
        that were shuffled to image i (i being the rows and j the columns of the matrix).
        By default False. Has no effect if mode is "mask".

    Returns
    -------
    corrupted : torch.Tensor
        corrupted image.
    unique : torch.Tensor
        If return_proportion is True, returns the matrix containing the
        proportion of patches of tensor j in tensor i.

    Raises
    ------
    ValueError
        If the image is not 4D or 5D
    ValueError
        If the image is 5D and has more than one channel
    ValueError
        If the mode is not "disc" or "mask"

    Examples
    --------
    >>> img_tensor = torch.randn(1, 3, 256, 256)
    >>> disc_img = corrupt_image(img_tensor, patch_size=64, frac=0.25, mode="disc")
    >>> disc_img.shape
    """

    image_ndim = image.ndim

    if image_ndim == 4:
        channels = image.shape[1]
        first_dim_size = channels
    elif image_ndim == 5:

        if image.shape[1] != 1:
            raise ValueError("Only single-channel images are supported for 5D tensors")

        image = image.squeeze(1)  # remove channel dimension
        first_dim_size = patch_size
    else:
        raise ValueError(f"Only 4D and 5D tensors are supported, not {image_ndim}D")

    patches = (
        image.unfold(1, first_dim_size, first_dim_size)
        .unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size)
    )
    unfold_shape = patches.size()
    patches = patches.contiguous().view(1, -1, first_dim_size, patch_size, patch_size)

    idxs = np.arange(patches.shape[1])

    shuffled_idxs = idxs.copy()
    np.random.shuffle(shuffled_idxs)

    # shuffled_pos = pos[shuffled_idxs]

    shuffled_idxs = shuffled_idxs[: int(len(shuffled_idxs) * frac)]
    _shuffle_idxs = shuffled_idxs.copy()
    np.random.shuffle(_shuffle_idxs)

    idxs[shuffled_idxs] = _shuffle_idxs

    n_patches = patches.shape[1] // image.shape[0]

    # check if a patch is shuffled to the same position
    diff = shuffled_idxs - _shuffle_idxs
    same = np.where(diff == 0)[0]
    list_same = shuffled_idxs[same]

    rand_patch = torch.randn(1, first_dim_size, patch_size, patch_size).to(image.device)

    # TODO: probably this can be made more efficient
    # like: patches[:, shuffled_idxs, ...] = patches[:, _shuffle_idxs, ...]
    # I'll test it later
    if mode == "disc":
        patches[:, shuffled_idxs, ...] = patches[:, _shuffle_idxs, ...]
    elif mode == "mask":
        patches[:, shuffled_idxs, ...] = rand_patch
    else:
        raise ValueError('Mode should be either "disc" or "mask"')

    corrupted = patches.view(unfold_shape)
    corrupted = corrupted.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    corrupted = corrupted.view(image.shape)

    img_idxs = torch.ones(
        size=(patches.shape[0], patches.shape[1], 1, 1, 1), device=image.device
    )
    img_idxs[:, shuffled_idxs, ...] = 0
    # if it was shuffled to the same position, set the index to 1
    img_idxs[:, list_same, ...] = 1
    img_idxs = img_idxs.view(unfold_shape[:-3])
    img_idxs = img_idxs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    img_idxs = img_idxs.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    img_idxs = img_idxs.view(
        image.shape[0], image.shape[2] // patch_size, image.shape[3] // patch_size
    )
    img_idxs = img_idxs.unsqueeze(1)

    if image_ndim == 5:
        corrupted = corrupted.unsqueeze(1)  # add channel dimension back

    if return_proportion and mode == "disc":
        # create a vector associating each patch with its image
        pos = torch.arange(image.shape[0])

        rep_pos = pos.repeat(image.shape[0])

        pos = pos.reshape(image.shape[0], 1)
        rep_pos_ = torch.repeat_interleave(pos, image.shape[0], dim=1)
        rep_pos_ = torch.flatten(rep_pos_)

        # find all the possible combinations
        all_combinations = torch.cat([rep_pos_[:, None], rep_pos[:, None]], dim=1)

        pos = torch.repeat_interleave(pos, n_patches, dim=1)
        pos = torch.flatten(pos)

        # shuffle the pos vector according to the shuffled indexes
        shuffled_pos = pos[shuffled_idxs]
        _shuffled_pos = pos[_shuffle_idxs]

        # combine the shuffled positions and count the unique combinations
        combination = torch.cat([shuffled_pos[:, None], _shuffled_pos[:, None]], dim=1)

        all_combinations_exp = all_combinations.unsqueeze(1).expand(
            -1, combination.shape[0], 2
        )
        combination_exp = combination.unsqueeze(0).expand(
            all_combinations.shape[0], -1, 2
        )

        # find which combinations do not occur, i.e., which tensors j have
        # no patches shuffled to tensor i
        mask = (all_combinations_exp == combination_exp).all(-1).any(-1)

        # count how many times each combination occurs
        _, unique = torch.unique(combination, dim=0, return_counts=True)

        # put the number of unique values in the right places
        # non-existing combinations will be filled with zeros
        zeros = torch.zeros(len(mask), dtype=torch.int64)
        zeros[torch.where(mask)] = unique

        unique = zeros.reshape(image.shape[0], -1)
        unique = unique / n_patches
        unique.fill_diagonal_(0)
        add = torch.diag(1 - (unique.sum(dim=1)))
        unique = unique + add
        unique.to(image.device)
        corrupted.to(image.device)

        return corrupted, unique

    return corrupted, img_idxs


class CorruptImage(nn.Module):

    def __init__(
        self,
        patch_size: int = 32,
        frac: float = 0.2,
        mode: str = "disc",
        return_proportion: bool = False,
    ):
        """
        corrupt an image by modifying patches of a given size.
        It expects the image to be in the format (batch, channels, height, width, (depth)).
        The percentage of patches to be shuffled is given by frac.
        Note that the image dimensions should be exactly divisible by the patch_size;
        I don't know the behavior of the function otherwise.
        This function should work for both 2D and 3D images. In the case of 3D images,
        a single channel is assumed.

        Parameters
        ----------
        image : torch.Tensor
            Image to be corrupted.
        patch_size : int, optional
            Size of the patches to corrupt, by default 32
        frac : float, optional
            Percentage of patches to corrupt, by default 0.2
        mode : str, optional
            Mode of corruption, either "disc" or "mask", by default "disc".
            If "disc", the patches will be shuffled. If "mask", the patches will be zeroed-out.
            Disc is short for discombobulate, which is a fun word.
        retrun_proportion : bool, optional
            If True, the proportion of patches that were shuffled will be returned.
            More in detail, it will return a matrix nxn where n is the number of images in the batch.
            The (i, j) element of the matrix represents the proportion of patches of the image j
            that were shuffled to image i (i being the rows and j the columns of the matrix).
            By default False. Has no effect if mode is "mask".

        Returns
        -------
        corrupted : torch.Tensor
            corrupted image.
        unique : torch.Tensor
            If return_proportion is True, returns the matrix containing the
            proportion of patches of tensor j in tensor i.

        Raises
        ------
        ValueError
            If the image is not 4D or 5D
        ValueError
            If the image is 5D and has more than one channel
        ValueError
            If the mode is not "disc" or "mask"

        Examples
        --------
        >>> img_tensor = torch.randn(1, 3, 256, 256)
        >>> disc_img = corrupt_image(img_tensor, patch_size=64, frac=0.25, mode="disc")
        >>> disc_img.shape
        """
        super(CorruptImage, self).__init__()
        self.patch_size = patch_size
        self.frac = frac
        self.mode = mode
        self.return_proportion = return_proportion

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return corrupt_image(
            image, self.patch_size, self.frac, self.mode, self.return_proportion
        )
