from utils.from_config import init_from_config
from utils.loaders import SSLImageDataset
from torchvision.transforms.v2 import Compose, Resize, ToTensor
from torchvision.transforms.v2 import (
    RandomVerticalFlip,
    RandomHorizontalFlip,
    ColorJitter,
)
from torch.utils.data import DataLoader
from transformers import Swinv2Config, Swinv2Model
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from models.lightning_models import MoBy
from pytorch_lightning.loggers import WandbLogger
from torchio.transforms import ZNormalization, ToCanonical, RescaleIntensity
import torchio as tio
from utils.loaders import SSLPatchDataset

training_config = OmegaConf.load("configs/config_ssl_3d.yaml")

znorm = ZNormalization()
canon = ToCanonical()
rescale = RescaleIntensity((-1, 1))

preproc = tio.Compose([canon, rescale])

wandblogger = WandbLogger(project="MoBY", name="MoBy_3D")

train_ds_config = training_config.train_dataset

train_dataset = init_from_config(
    train_ds_config,
    object=SSLPatchDataset,
    return_config_only=False,
    dicts_to_lists=False,
    additional_params={"preprocessing": preproc},
)

# val_ds_config = training_config.val_dataset

# val_dataset = init_from_config(
#     val_ds_config,
#     object=SSLImageDataset,
#     return_config_only=False,
#     dicts_to_lists=False,
#     additional_params={"transform": transform, "pre_processing": pre_processing},
# )

trainer_config = training_config.trainer

trainer = init_from_config(
    trainer_config,
    object=Trainer,
    return_config_only=False,
    dicts_to_lists=True,
    additional_params={"logger": wandblogger},
)

train_dataloader_config = training_config.train_dataloader
train_dataloader = init_from_config(
    train_dataloader_config,
    object=DataLoader,
    return_config_only=False,
    dicts_to_lists=False,
    additional_params={
        "dataset": train_dataset,
        "drop_last": True,
    },
)

# val_dataloader_config = training_config.val_dataloader
# val_dataloader = init_from_config(
#     val_dataloader_config,
#     object=DataLoader,
#     return_config_only=False,
#     dicts_to_lists=False,
#     additional_params={
#         "dataset": val_dataset,
#         "drop_last": True,
#     },
# )

model_config = training_config.lightning_model
n_steps = trainer.max_epochs * len(train_dataloader)

moby = init_from_config(
    model_config,
    object=MoBy,
    return_config_only=False,
    dicts_to_lists=False,
    additional_params={
        "train_steps": n_steps,
    },
)

trainer.fit(moby, train_dataloader)
