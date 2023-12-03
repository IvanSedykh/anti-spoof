import torch
from torch.nn import functional as F
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    def __call__(self, disc_real_preds, disc_fake_preds):
        loss = 0
        for real_pred, fake_pred in zip(disc_real_preds, disc_fake_preds):
            r_loss = torch.mean((1 - real_pred) ** 2)
            g_loss = torch.mean(fake_pred**2)
            loss += r_loss + g_loss
        return loss


class GeneratorLoss(nn.Module):
    def __call__(self, disc_fake_preds):
        loss = 0
        for fake_pred in disc_fake_preds:
            loss += torch.mean((1 - fake_pred) ** 2)
        return loss


class FeatureLoss(nn.Module):
    def __call__(self, features_real, features_fake):
        loss = 0
        for f_real, f_fake in zip(features_real, features_fake):
            for f_real_part, f_fake_part in zip(f_real, f_fake):
                loss += F.l1_loss(f_real_part, f_fake_part)
        return loss
