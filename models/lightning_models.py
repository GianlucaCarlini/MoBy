import pytorch_lightning as pl
from monai.networks.nets.swin_unetr import SwinTransformer
import numpy as np
from modules.torch_modules import MoBYMLP
import torch
import torch.nn as nn


class MoBy(pl.LightningModule):
    def __init__(
        self,
        encoder,
        encoder_k,
        optimizer,
        scheduler,
        corrupt,
        contrast_momentum=0.99,
        contrast_temperature=0.2,
        contrast_num_negatives=4096,
        proj_num_layers=2,
        pred_num_layers=2,
        **kwargs,
    ):
        super(MoBy, self).__init__()

        self.encoder = encoder
        self.encoder_k = encoder_k

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.corrupt = corrupt

        self.contrast_momentum = contrast_momentum
        self.contrast_temperature = contrast_temperature
        self.contrast_num_negatives = contrast_num_negatives

        self.proj_num_layers = proj_num_layers
        self.pred_num_layers = pred_num_layers
        self.kwargs = kwargs

        out_dim = self.encoder.embed_dim * 2 ** (len(self.encoder.depths) - 1)

        self.projector = MoBYMLP(in_dim=out_dim, num_layers=self.proj_num_layers)
        self.projector_k = MoBYMLP(in_dim=out_dim, num_layers=self.proj_num_layers)

        self.predictor = MoBYMLP(num_layers=self.pred_num_layers)

        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(
            self.projector.parameters(), self.projector_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.K = kwargs.get("train_steps", 10000)
        self.k = 0

        self.register_buffer("queue1", torch.randn(256, self.contrast_num_negatives))
        self.register_buffer("queue2", torch.randn(256, self.contrast_num_negatives))

        self.queue1 = nn.functional.normalize(self.queue1, dim=0)
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):

        _contrast_momentum = (
            1.0
            - (1.0 - self.contrast_momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2
        )

        self.k += 1

        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (
                1.0 - _contrast_momentum
            )

        for param_q, param_k in zip(
            self.projector.parameters(), self.projector_k.parameters()
        ):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (
                1.0 - _contrast_momentum
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2):

        batch_size = keys1.shape[0]

        ptr = int(self.queue_ptr)

        assert self.contrast_num_negatives % batch_size == 0

        self.queue1[:, ptr : ptr + batch_size] = keys1.T
        self.queue2[:, ptr : ptr + batch_size] = keys2.T

        ptr = (ptr + batch_size) % self.contrast_num_negatives

        self.queue_ptr[0] = ptr

    def contrastive_loss(self, q, k, queue):

        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.contrast_temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        return nn.functional.cross_entropy(logits, labels)

    def forward(self, x1, x2):

        feat_1 = self.encoder(x1)[4]
        proj_1 = self.projector(feat_1)
        pred_1 = self.predictor(proj_1)
        pred_1 = nn.functional.normalize(pred_1, dim=1)

        feat_2 = self.encoder_k(x2)[4]
        proj_2 = self.projector_k(feat_2)
        pred_2 = self.predictor(proj_2)
        pred_2 = nn.functional.normalize(pred_2, dim=1)

        # computer key features
        with torch.no_grad():
            self._momentum_update_key_encoder()

            feat_1_k = self.encoder_k(x1)[4]
            proj_1_k = self.projector_k(feat_1_k)
            proj_1_k = nn.functional.normalize(proj_1_k, dim=1)

            feat_2_k = self.encoder(x2)[4]
            proj_2_k = self.projector(feat_2_k)
            proj_2_k = nn.functional.normalize(proj_2_k, dim=1)

        self._dequeue_and_enqueue(proj_1_k, proj_2_k)

        return pred_1, pred_2, proj_1_k, proj_2_k
