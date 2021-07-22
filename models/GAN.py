"""
-------------------------------------------------
   File Name:    GAN.py
   Author:       Zhonghao Huang
   Date:         2019/10/17
   Description:  Modified from:
                 https://github.com/akanimax/pro_gan_pytorch
                 https://github.com/lernapparat/lernapparat
                 https://github.com/NVlabs/stylegan
-------------------------------------------------
"""

import os
import datetime
import time
import timeit
import copy
import random
import numpy as np
from collections import OrderedDict
import gc


import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


import models.Losses as Losses
from data import get_data_loader
from models import update_average
from models.Blocks import DiscriminatorTop, DiscriminatorBlock, InputBlock, GSynthesisBlock
from models.CustomLayers import EqualizedConv2d, PixelNormLayer, EqualizedLinear, Truncation
from models.Generator import Generator
from models.Discriminator import Discriminator


class StyleGAN:

    def __init__(self, structure, resolution, num_channels, latent_size, vae_probs, dis_probs, gen_probs, sleep_probs,
                 g_args, d_args, g_opt_args, d_opt_args, loss="relativistic-hinge", drift=0.001, recon_beta=1, feature_beta=5,
                 d_repeats=1, use_ema=False, ema_decay=0.999, device=torch.device("cpu"), use_CB=False):
        """
        Wrapper around the Generator and the Discriminator.

        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param resolution: Input resolution. Overridden based on dataset.
        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param latent_size: Latent size of the manifold used by the GAN
        :param g_args: Options for generator network.
        :param d_args: Options for discriminator network.
        :param g_opt_args: Options for generator optimizer.
        :param d_opt_args: Options for discriminator optimizer.
        :param loss: the loss function to be used
                     Can either be a string =>
                          ["wgan-gp", "wgan", "lsgan", "lsgan-with-sigmoid",
                          "hinge", "standard-gan" or "relativistic-hinge"]
                     Or an instance of GANLoss
        :param drift: drift penalty for the
                      (Used only if loss is wgan or wgan-gp)
        :param d_repeats: How many times the discriminator is trained per G iteration.
        :param use_ema: boolean for whether to use exponential moving averages
        :param ema_decay: value of mu for ema
        :param device: device to run the GAN on (GPU / CPU)
        """

        # state of the object
        assert structure in ['fixed', 'linear']
        self.structure = structure
        self.depth = int(np.log2(resolution)) - 1
        self.latent_size = latent_size
        self.device = device
        self.d_repeats = d_repeats
        self.vae_probs = vae_probs
        self.dis_probs = dis_probs
        self.gen_probs = gen_probs
        self.sleep_probs = sleep_probs
        self.use_CB = use_CB

        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.writer = SummaryWriter()

        # Create the Generator and the Discriminator
        self.gen = Generator(num_channels=num_channels,
                             resolution=resolution,
                             structure=self.structure,
                             **g_args).to(self.device)
        self.dis = Discriminator(num_channels=num_channels,
                                 resolution=resolution,
                                 structure=self.structure,
                                 output_features=self.latent_size*2,
                                 encode_in=encode_in,
                                 **d_args).to(self.device)


        # if code is to be run on GPU, we can use DataParallel:
        # TODO
        # define the optimizers for the discriminator and generator
        self.__setup_gen_optim(**g_opt_args)
        self.__setup_dis_optim(**d_opt_args)

        # define the loss function used for training the GAN
        self.drift = drift
        self.loss = Losses.LogisticGAN(self.dis, self.gen, recon_beta, feature_beta, self.use_CB)

        # Use of ema
        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # updater function:
            self.ema_updater = update_average
            # initialize the gen_shadow weights equal to the weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __return_probabilities(self, cur_depth, cur_epoch, all_epochs):
        epochs = all_epochs[cur_depth]

        # VAE probability
        start, end = self.vae_probs[cur_depth]
        grow = (float(end) - float(start)) / epochs
        vae_prob = max(0, min(1, start + grow * cur_epoch))

        # Discriminator probabibility
        start, end = self.dis_probs[cur_depth]
        grow = (float(end) - float(start)) / epochs
        dis_prob = max(0, min(1, start + grow * cur_epoch))


        # Generator probability
        epochs = all_epochs[cur_depth]
        start, end = self.gen_probs[cur_depth]
        grow = (float(end) - float(start)) / epochs
        gen_prob = max(0, min(1, start + grow * cur_epoch))

        # Sleep probability
        epochs = all_epochs[cur_depth]
        start, end = self.sleep_probs[cur_depth]
        grow = (float(end) - float(start)) / epochs
        sleep_prob = max(0, min(1, start + grow * cur_epoch))

        return {'vae': vae_prob, 'dis': dis_prob, 'gen': gen_prob, 'sleep': sleep_prob}

    def __setup_gen_optim(self, learning_rate, beta_1, beta_2, eps):
        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_dis_optim(self, learning_rate, beta_1, beta_2, eps):
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __progressive_down_sampling(self, real_batch, depth, alpha):
        """
        private helper for down_sampling the original images in order to facilitate the
        progressive growing of the layers.

        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fade-in alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate

        if self.structure == 'fixed':
            return real_batch

        # down_sample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_down_sample_factor = max(int(np.power(2, self.depth - depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_down_sample_factor)(real_batch), scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def optimize_discriminator(self, real_batch, depth, alpha, print_=False):
        """
        performs one step of weight update on discriminator using the batch of data

        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """
        latent_input = torch.randn(real_batch.shape[0], self.latent_size).to(self.device)
        real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)

        # generate a batch of samples
        fake_samples, extended_latent_input = self.gen(latent_input, depth, alpha, return_extended_latent_input=True)
        fake_samples = fake_samples.detach(); extended_latent_input = extended_latent_input.detach()
        loss, r_loss, f_loss = self.loss.dis_loss(extended_latent_input, real_samples, fake_samples, depth, alpha, print_=print_)

        # optimize discriminator
        self.dis_optim.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.dis.parameters(), max_norm=1.)

        self.dis_optim.step()

        return loss.item(), r_loss, f_loss

    def optimize_generator(self, real_batch, depth, alpha, print_=False):
        """
        performs one step of weight update on generator for the given batch_size

        :param noise: input random noise required for generating samples
        :param real_batch: batch of real samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """
        latent_input = torch.randn(real_batch.shape[0], self.latent_size).to(self.device)
        real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)


        # generate fake samples:
        fake_samples = self.gen(latent_input, depth, alpha, use_style_mixing=True)

        loss = self.loss.gen_loss(real_samples, fake_samples, depth, alpha, print_=print_)

        # optimize the generator
        self.gen_optim.zero_grad()

        loss.backward()
        # Gradient Clipping
        nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=1.)

        self.gen_optim.step()

        # if use_ema is true, apply ema to the generator parameters
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        # return the loss value
        return loss.item()

    def optimeze_as_vae(self, real_batch, depth, alpha, print_=False):
        real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)

        loss, kl_loss, recon_loss, feature_loss = self.loss.vae_loss(real_samples, depth, alpha, print_)

        # optimize model
        self.gen_optim.zero_grad()
        self.dis_optim.zero_grad()
        loss.backward()

        # Gradient Clipping
        nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=1.)
        nn.utils.clip_grad_norm_(self.dis.parameters(), max_norm=1.)

        self.gen_optim.step()
        self.dis_optim.step()

        # if use_ema is true, apply ema to the generator parameters
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        # return the loss value
        return loss.item(), kl_loss, recon_loss, feature_loss

    def optimize_with_sleep_phase(self, batch_size, depth, alpha, print_=False):
        with torch.no_grad():
            latent_input = torch.randn(batch_size, self.latent_size).to(self.device)
            fake_samples, extended_latent_input = self.gen(latent_input, depth, alpha, return_extended_latent_input=True)

        fake_samples = fake_samples.detach(); extended_latent_input = extended_latent_input.detach()
        loss = self.loss.sleep_loss(extended_latent_input, fake_samples, depth, alpha, print_=print_)

        # optimize model
        self.dis_optim.zero_grad()
        loss.backward()

        # Gradient Clipping
        nn.utils.clip_grad_norm_(self.dis.parameters(), max_norm=1.)

        self.dis_optim.step()

        return loss.item()




    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        """
        utility function to create a grid of GAN samples

        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        from torchvision.utils import save_image
        from torch.nn.functional import interpolate

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),
                   normalize=True, scale_each=True, pad_value=128, padding=1)

    def train(self, dataset, num_workers, epochs, batch_sizes, fade_in_percentage, simp_start_end, logger, output,
              num_samples=12, start_depth=0, feedback_factor=100, checkpoint_factor=1):
        """
        Utility method for training the GAN. Note that you don't have to necessarily use this
        you can use the optimize_generator and optimize_discriminator for your own training routine.

        :param dataset: object of the dataset used for training.
                        Note that this is not the data loader (we create data loader in this method
                        since the batch_sizes for resolutions can be different)
        :param num_workers: number of workers for reading the data. def=3
        :param epochs: list of number of epochs to train the network for every resolution
        :param batch_sizes: list of batch_sizes for every resolution
        :param fade_in_percentage: list of percentages of epochs per resolution used for fading in the new layer
                                   not used for first resolution, but dummy value still needed.
        :param logger:
        :param output: Output dir for samples,models,and log.
        :param num_samples: number of samples generated in sample_sheet. def=36
        :param start_depth: start training from this depth. def=0
        :param feedback_factor: number of logs per epoch. def=100
        :param checkpoint_factor:
        :return: None (Writes multiple files to disk)
        """

        assert self.depth <= len(epochs), "epochs not compatible with depth"
        assert self.depth <= len(batch_sizes), "batch_sizes not compatible with depth"
        assert self.depth <= len(fade_in_percentage), "fade_in_percentage not compatible with depth"

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()

        # create fixed_input for debugging
        fixed_input = torch.randn(num_samples, self.latent_size).to(self.device)
        vae_loss, kl_loss, recon_loss, feature_loss, dis_loss, r_loss, f_loss, gen_loss, sleep_loss = 0, 0, 0, 0, 0, 0, 0, 0, 0 #only for printing

        # config depend on structure
        logger.info("Starting the training process ... \n")
        if self.structure == 'fixed':
            start_depth = self.depth - 1
        step = 1  # counter for number of iterations
        for current_depth in range(start_depth, self.depth):
            current_res = np.power(2, current_depth + 2)
            logger.info("Currently working on depth: %d", current_depth + 1)
            logger.info("Current resolution: %d x %d" % (current_res, current_res))

            ticker = 1

            # Choose training parameters and configure training ops.
            # TODO
            data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)

            for epoch in range(1, epochs[current_depth] + 1):
                update_string = self.loss.update_simp(simp_start_end, sum(epochs[:current_depth]) + epoch, sum(epochs))
                probabilities = self.__return_probabilities(current_depth, epoch, epochs)
                start = timeit.default_timer()  # record time at the start of epoch
                logger.info(update_string)
                logger.info(f'Probabilites updated: {probabilities}.')
                logger.info("Epoch: [%d]" % epoch)
                # total_batches = len(iter(data))
                total_batches = len(data)

                fade_point = int((fade_in_percentage[current_depth] / 100)
                                 * epochs[current_depth] * total_batches)
                print_=False

                for (i, batch) in enumerate(data, 1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fade_point if ticker <= fade_point else 1

                    # if epoch ==1:
                    #     self.writer.add_graph(self.dis, (batch, torch.tensor([current_depth]), torch.tensor([alpha])))
                    #     self.writer.add_graph(self.gen, (torch.randn(4,512), torch.tensor([current_depth]), torch.tensor([alpha])))
                    # extract current batch of data for training
                    images = batch.to(self.device)

                    # optimize the discriminator:
                    if random.random() < probabilities['dis']:
                        dis_loss, r_loss, f_loss = self.optimize_discriminator(images, current_depth, alpha, print_)
                    
                    # optimize the generator:
                    gen_loss = self.optimize_generator(images, current_depth, alpha, print_) if random.random() < probabilities['gen'] else gen_loss

                    # optimze model as vae:
                    if random.random() < probabilities['vae']:
                        vae_loss, kl_loss, recon_loss, feature_loss = self.optimeze_as_vae(images, current_depth, alpha, print_)

                    # optimeze model with sleep phase
                    sleep_loss = self.optimize_with_sleep_phase(images.shape[0], current_depth, alpha, print_) if random.random() < probabilities['sleep'] else sleep_loss
                    
                    print_=False

                    # provide a loss feedback
                    if i % int(total_batches / feedback_factor + 1) == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                        logger.info(
                            "Elapsed: [%s] Step: %d  Batch: %d  D_Loss: %f  G_Loss: %f, VAE_loss: %f, SLEEP_loss: %f"
                            % (elapsed, step, i, dis_loss, gen_loss, vae_loss, sleep_loss))

                        # create a grid of samples and save it
                        os.makedirs(os.path.join(output, 'samples'), exist_ok=True)
                        gen_img_file = os.path.join(output, 'samples', "gen_" + str(current_depth)
                                                    + "_" + str(epoch) + "_" + str(i) + ".png")

                        self.writer.add_scalar('Loss/vae/total', float(vae_loss), step)
                        self.writer.add_scalar('Loss/vae/kl', float(kl_loss), step)
                        self.writer.add_scalar('Loss/vae/recon', float(recon_loss), step)
                        self.writer.add_scalar('Loss/vae/feature', float(feature_loss), step)
                        self.writer.add_scalar('Loss/dis/total', float(dis_loss), step)
                        self.writer.add_scalar('Loss/dis/real', float(r_loss), step)
                        self.writer.add_scalar('Loss/dis/fake', float(f_loss), step)
                        self.writer.add_scalar('Loss/gen', float(gen_loss), step)
                        self.writer.add_scalar('Loss/sleep',float(sleep_loss), step)

                        with torch.no_grad():
                            images_ds = self.__progressive_down_sampling(images[:num_samples], current_depth, alpha)
                            latents = self.dis(images_ds, current_depth, alpha).detach()
                            samples = self.gen(fixed_input, current_depth, alpha, latent_are_in_extended_space=False).detach() if not self.use_ema else self.gen_shadow(fixed_input, current_depth, alpha, latent_are_in_extended_space=False).detach()
                            
                            if self.use_CB:
                                samples = torch.distributions.continuous_bernoulli.ContinuousBernoulli(samples).mean
                            
                            renconstruced_latents = self.dis(samples, current_depth, alpha).detach()

                            if self.encode_in == 'Z':
                                b, l = latents.size()
                                latents = latents[:, :l//2] + Variable(torch.randn(b, l//2).to(latents.device)) * (latents[:, l//2:] * 0.5).exp()
                                renconstruced_latents = renconstruced_latents[:, :l//2] + Variable(torch.randn(b, l//2).to(renconstruced_latents.device)) * (renconstruced_latents[:, l//2:] * 0.5).exp()
                                
                                recon_samples = self.gen(renconstruced_latents, current_depth, alpha, latent_are_in_extended_space=False).detach() if not self.use_ema else self.gen_shadow(renconstruced_latents, current_depth, alpha, latent_are_in_extended_space=False).detach()
                                recon = self.gen(latents, current_depth, alpha, latent_are_in_extended_space=False).detach() if not self.use_ema else self.gen_shadow(latents, current_depth, alpha, latent_are_in_extended_space=False).detach()
                            else:
                                recon = self.gen(latents, current_depth, alpha, latent_are_in_extended_space=True).detach() if not self.use_ema else self.gen_shadow(latents, current_depth, alpha, latent_are_in_extended_space=True).detach()
                                recon_samples = self.gen(renconstruced_latents, current_depth, alpha, latent_are_in_extended_space=True).detach() if not self.use_ema else self.gen_shadow(renconstruced_latents, current_depth, alpha, latent_are_in_extended_space=True).detach()


                            if self.use_CB:
                                recon = torch.distributions.continuous_bernoulli.ContinuousBernoulli(recon).mean
                                recon_samples = torch.distributions.continuous_bernoulli.ContinuousBernoulli(recon_samples).mean

                            self.create_grid(
                                samples=torch.cat([images_ds, recon, samples, recon_samples]),
                                scale_factor=int(np.power(2, self.depth - current_depth - 1)) if self.structure == 'linear' else 1,
                                img_file=gen_img_file,
                            )

                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1

                elapsed = timeit.default_timer() - start
                elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                logger.info("Time taken for epoch: %s\n" % elapsed)

                if checkpoint_factor > 0 and current_depth >= self.depth - 1:
                    try:
                        if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[current_depth]:
                            save_dir = os.path.join('/var/scratch/mms496', 'models', output.split('/')[-1])
                            os.makedirs(save_dir, exist_ok=True)
                            gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth)  + ".pth") #+ "_" + str(epoch) 
                            dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth)  + ".pth") #+ "_" + str(epoch)
                            gen_optim_save_file = os.path.join(
                                save_dir, "GAN_GEN_OPTIM_" + str(current_depth) + ".pth")  #+ "_" + str(epoch)
                            dis_optim_save_file = os.path.join(
                                save_dir, "GAN_DIS_OPTIM_" + str(current_depth) + ".pth")  #+ "_" + str(epoch)

                            torch.save(self.gen.state_dict(), gen_save_file)
                            logger.info("Saving the model to: %s\n" % gen_save_file)
                            torch.save(self.dis.state_dict(), dis_save_file)
                            torch.save(self.gen_optim.state_dict(), gen_optim_save_file)
                            torch.save(self.dis_optim.state_dict(), dis_optim_save_file)

                            # also save the shadow generator if use_ema is True
                            if self.use_ema:
                                gen_shadow_save_file = os.path.join(
                                    save_dir, "GAN_GEN_SHADOW_" + str(current_depth) + ".pth") #+ "_" + str(epoch)
                                torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)
                                logger.info("Saving the model to: %s\n" % gen_shadow_save_file)
                    except Exception as e:
                            logger.info(f"Unable to save model. Depth {current_depth}, epoch: {epoch}\n")
                            print(e)

        logger.info('Training completed.\n')


if __name__ == '__main__':
    print('Done.')
