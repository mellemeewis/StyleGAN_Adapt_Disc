"""
-------------------------------------------------
   File Name:    Losses.py
   Author:       Zhonghao Huang
   Date:         2019/10/21
   Description:  Module implementing various loss functions
                 Copy from: https://github.com/akanimax/pro_gan_pytorch
-------------------------------------------------
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses

        @args:
        dis: Discriminator used for calculating the loss
             Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis
        self.simp = 0

    def update_simp(self, total_epochs):
        epochs = total_epochs / 1
        grow = 0/ epochs
        self.simp = min(0, self.simp + grow)
        print('Simp updated: ', self.simp, 'Total epochs: ', total_epochs)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:
    """ Base class for all conditional losses """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("gen_loss method has not been implemented")


# =============================================================
# Normal versions of the Losses:
# =============================================================

class StandardGAN(GANLoss):

    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss

        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds),
            torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds),
            torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, height, alpha):
        preds, _, _ = self.dis(fake_samps, height, alpha)
        return self.criterion(torch.squeeze(preds),
                              torch.ones(fake_samps.shape[0]).to(fake_samps.device))


class HingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        loss = (torch.mean(nn.ReLU()(1 - r_preds)) +
                torch.mean(nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        return -torch.mean(self.dis(fake_samps, height, alpha))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        loss = (torch.mean(nn.ReLU()(1 - r_f_diff))
                + torch.mean(nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return (torch.mean(nn.ReLU()(1 + r_f_diff))
                + torch.mean(nn.ReLU()(1 - f_r_diff)))


class LogisticGAN(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    # gradient penalty
    def R1Penalty(self, real_img, height, alpha):

        # TODO: use_loss_scaling, for fp16
        apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
        undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        real_logit = self.dis(real_img, height, alpha)
        # real_logit = apply_loss_scaling(torch.sum(real_logit))
        real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
                                         grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
                                         create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)
        # real_grads = undo_loss_scaling(real_grads)
        r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        return r1_penalty

    def dis_loss(self, latent_input, real_samps, fake_samps, height, alpha, r1_gamma=10.0, eps=1e-5, print_=False):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)
        r_preds, r_preds_label = r_preds[:,:-1], r_preds[:,-1]
        f_preds, f_preds_label = f_preds[:,:-1], f_preds[:,-1]

        b, l = r_preds.size()
        r_mean, r_sig = r_preds[:, :l//2], r_preds[:, l//2:]
        f_mean, f_sig = f_preds[:, :l//2], f_preds[:, l//2:]

        r_loss = F.binary_cross_entropy_with_logits(r_preds_label, torch.ones(real_samps.shape[0]).to(real_samps.device)) + 0.5 * torch.sum(r_sig.exp() - r_sig + r_mean.pow(2) - 1, dim=1)
        f_loss = F.binary_cross_entropy_with_logits(f_preds_label, torch.zeros(fake_samps.shape[0]).to(fake_samps.device)) + f_sig + self.simp * (1.0 / (2.0 * f_sig.exp().pow(2.0) + eps)) * (latent_input - f_mean).pow(2.0)

        loss = torch.mean(r_loss) + torch.mean(f_loss)

        # if r1_gamma != 0.0:
        #     r1_penalty = self.R1Penalty(real_samps.detach(), height, alpha) * (r1_gamma * 0.5)
        #     print('HELLO r1', r1_penalty)
        #     loss += r1_penalty

        if print_:
            print('DIS LOSS REAL: ', r_preds_label.mean().item(), r_sig.mean().item(), r_mean.mean().item(),  r_loss.mean().item())
            print('DIS LOSS FAKE: ', f_preds_label.mean().item(), f_sig.mean().item(), f_mean.mean().item(),  f_loss.mean().item())

        return loss

    def gen_loss(self, _, fake_samps, height, alpha, print_=False):
        f_preds = self.dis(fake_samps, height, alpha)
        f_preds, f_preds_label = f_preds[:,:-1], f_preds[:,-1]

        b, l = f_preds.size()
        f_mean, f_sig = f_preds[:, :l//2], f_preds[:, l//2:]

        loss = F.binary_cross_entropy_with_logits(f_preds_label, torch.ones(fake_samps.shape[0]).to(fake_samps.device)) + 0.5 * torch.sum(f_sig.exp() - f_sig + f_mean.pow(2) - 1, dim=1)

        if print_:
            print('GENERATOR LOSS: ', f_preds_label.mean().item(), f_sig.mean().item(), f_mean.mean().item(),  loss.mean().item())

        return torch.mean(loss)
