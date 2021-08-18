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
from torch.autograd import Variable
from torchvision.models import vgg19
from torch.nn.functional import interpolate



# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses

        @args:
        dis: Discriminator used for calculating the loss
             Note this must be a part of the GAN framework
    """

    def __init__(self, dis, gen, recon_beta, feature_beta, use_CB):
        self.dis = dis
        self.gen = gen
        self.simp = 0
        self.recon_beta =recon_beta
        self.feature_beta = feature_beta
        self.feature_network = vgg19(pretrained=True).to('cuda')
        self.feature_layers = ['1', '6', '11', '20', '29']
        self.use_CB = use_CB
        self.feature_network.eval()

    def update_simp(self, simp_start_end, cur_epoch, total_epochs):
        start, end = simp_start_end
        grow = (end - start) / total_epochs
        self.simp = min(1, start + cur_epoch * grow)

        return f'Simp updated: {self.simp}, Epoch {cur_epoch} of Total epochs: {total_epochs}, Sched: {simp_start_end}'

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



class LogisticGAN(GANLoss):
    def __init__(self, dis, gen, recon_beta, feature_beta, use_CB):
        super().__init__(dis, gen, recon_beta, feature_beta, use_CB)

    def dis_loss(self, latent_input, extended_latent_input, real_samps, fake_samps, height, alpha, r1_gamma=10.0, eps=1e-5, print_=False):
        # Obtain predictions

        if self.use_CB:
            with torch.no_grad():
                fake_samps = torch.distributions.continuous_bernoulli.ContinuousBernoulli(fake_samps).mean #sample((20,)).mean(dim=0)
            r_preds = self.dis(real_samps.clamp(min=0.0627, max=0.9373), height, alpha)

        else:
            r_preds = self.dis(real_samps, height, alpha)

        f_preds = self.dis(fake_samps, height, alpha)

        if type(r_preds) != tuple:
        # if len(list(r_preds.size())) == 2:
            b, l = r_preds.size()
            r_mean, r_sig = r_preds[:, :l//2], r_preds[:, l//2:]
            f_mean, f_sig = f_preds[:, :l//2], f_preds[:, l//2:]
            r_loss = 0.5 * torch.mean(r_sig.exp() - r_sig + r_mean.pow(2) - 1, dim=1)
            f_loss = f_sig + self.simp * (1.0 / (2.0 * f_sig.exp().pow(2.0) + eps)) * (latent_input - f_mean).pow(2.0)
            f_loss = torch.mean(f_loss, dim=1)

        else:
            r_preds = r_preds[-1]
            f_latent_recon, f_preds = f_preds[0], f_preds[-1]
            r_loss = F.binary_cross_entropy(r_preds, torch.ones_like(r_preds))
            f_loss = F.binary_cross_entropy(f_preds, torch.zeros_like(f_preds)) + F.mse_loss(extended_latent_input, f_latent_recon)
            print("DIS CHECK")
            # r_preds = r_preds[:,:,-1]

            # TO DO  

        


        loss = torch.mean(r_loss + f_loss)
        if print_:
            print('DIS LOSS REAL: Sig:', r_sig.mean().item(), 'Mean: ', r_mean.mean().item(), 'L: ', r_loss.mean().item())
            if self.simp < 0:
                print('DIS LOSS FAKE: Sig:', f_sig.mean().item(), 'Mean: ', f_mean.mean().item(), 'L: ', f_loss.mean().item(), f'D10 (*{-self.simp}): ', -self.simp*f_mean_distance_to_10.pow(2).mean().item())
            else:
                print('DIS LOSS FAKE: Sig:', f_sig.mean().item(), 'Mean: ', f_mean.mean().item(), 'L: ', f_loss.mean().item())


        return loss, r_loss.mean().item(), f_loss.mean().item()

    def gen_loss(self, _, fake_samps, height, alpha, print_=False):
        if self.use_CB:
            fake_samps = torch.distributions.continuous_bernoulli.ContinuousBernoulli(fake_samps).mean #rsample((1000,)).mean(dim=0)
        
        f_preds = self.dis(fake_samps, height, alpha)
        
        if type(f_preds) != tuple:
        # if len(list(f_preds.size())) == 2:
            b, l = f_preds.size()
            f_mean, f_sig = f_preds[:, :l//2], f_preds[:, l//2:]
            loss =  0.5 * torch.mean(f_sig.exp() - f_sig + f_mean.pow(2) - 1, dim=1)

        else:
            f_preds = f_preds[-1]
            loss = F.binary_cross_entropy(f_preds, torch.ones_like(f_preds))
            print("GEN check")

        if print_:
            print('GENERATOR LOSS: Sig:', f_sig.mean().item(), 'Mean: ', f_mean.mean().item(), 'L: ', loss.mean().item())

        return torch.mean(loss)

    def vae_loss(self, real_samps, height, alpha, print_=False):
        
        latents = self.dis(real_samps, height, alpha)
        encoding_in_W=False
        print("HELLO", type(latents))
        if type(latents) != tuple:
        # if len(list(latents.size())) == 2:
            b, l = latents.size()
            kl_loss = 0.5 * torch.mean(latents[:, l//2:].exp() - latents[:, l//2:] + latents[:, :l//2].pow(2) - 1, dim=1)
            latents = latents[:, :l//2] + Variable(torch.randn(b, l//2).to(latents.device)) * (latents[:, l//2:] * 0.5).exp()
            reconstrution = self.gen(latents, height, alpha, latent_are_in_extended_space=False)

        else:
            encoding_in_W=True
            latents = latents[0]
            b, w, l = latents.size()
            kl_loss = torch.tensor([0.])
            reconstrution = self.gen(latents, height, alpha, latent_are_in_extended_space=True)

            print("VAE check")


        if self.use_CB == True:
            reconstrution_distribution = torch.distributions.continuous_bernoulli.ContinuousBernoulli(reconstrution)
            recon_loss = -reconstrution_distribution.log_prob(real_samps).view(b, -1).mean(dim=1, keepdim=True)
            # reconstrution = reconstrution_distribution.rsample()
            reconstrution = reconstrution_distribution.mean

        else:
            recon_loss = F.binary_cross_entropy(reconstrution, real_samps, reduction='none').view(b, -1).mean(dim=1, keepdim=True)

        if self.feature_beta > 0:
            if real_samps.shape[-1] < 32:
                real_samps = interpolate(real_samps, scale_factor=32//real_samps.shape[-1])
                reconstrution = interpolate(reconstrution, scale_factor=32//reconstrution.shape[-1])

            if real_samps.shape[1] == 1:
              real_samps = real_samps.expand(-1,3,-1,-1)
              reconstrution = reconstrution.expand(-1,3,-1,-1)

            features_real, features_recon = self.extract_features(real_samps), self.extract_features(reconstrution)
            feature_loss = 0.0
            for (r, i) in zip(features_recon, features_real):
                feature_loss += F.mse_loss(r, i)

        else:
            feature_loss = torch.tensor([0.]).to(reconstrution.device)

        if encoding_in_W==False:
        # if len(list(latents.size())) == 2:
            loss = torch.mean(kl_loss + self.recon_beta*recon_loss + self.feature_beta*feature_loss)            
        else:
            loss = torch.mean(self.recon_beta*recon_loss + self.feature_beta*feature_loss)

        if print_:
            print('VAE LOSS: KL:', kl_loss.mean().item(), 'RECON: ', recon_loss.mean().item(), 'L: ', loss.mean().item())

        return loss, kl_loss.mean().item(), recon_loss.mean().item(), feature_loss.mean().item()


    def sleep_loss(self, latent_input, extended_latent_input, fake_samples, height, alpha, print_=False):

        if self.use_CB:
            with torch.no_grad():
                fake_samples = torch.distributions.continuous_bernoulli.ContinuousBernoulli(fake_samples).mean

        reconstructed_latents = self.dis(fake_samples, height, alpha)

        if type(reconstructed_latents) != tuple:
        # if len(list(reconstructed_latents.size())) == 2:
            b, l = reconstructed_latents.size()
            zmean, zsig = reconstructed_latents[:, :l//2], reconstructed_latents[:, l//2:]
            # zvar = zsig.exp() # variance
            # loss = zsig + (1.0 / (2.0 * zvar.pow(2.0) + 1e-5)) * (latent_input - zmean).pow(2.0)

            distribution = torch.distributions.normal.Normal(zmean, torch.sqrt(zsig.exp()), validate_args=None)
            loss = -distribution.log_prob(latent_input)
        else:
            reconstructed_latents = reconstructed_latents[0]
            loss = f.mse_loss(extended_latent_input, reconstructed_latents)
            print("Sleep check")
            # TO DO

            # with torch.no_grad():
            #     distribution = torch.distributions.normal.Normal(zmean, torch.sqrt(zvar), validate_args=None)
            #     d_loss = -distribution.log_prob(noise)

        if print_:
            print('SLEEP LOSS', loss.mean().item())

        return torch.mean(loss)


    def extract_features(self, input):
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        features = []
        result = input
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in self.feature_layers):
                features.append(result)

        return features



