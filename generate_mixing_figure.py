import os
import argparse
import numpy as np
from PIL import Image
import imageio

import torch

from models.Generator import Generator
from models.Discriminator import Discriminator
from generate_grid import adjust_dynamic_range


def draw_style_mixing_figure(png, gen, out_depth, src_seeds, dst_seeds, style_ranges):
    n_col = len(src_seeds)
    n_row = len(dst_seeds)
    w = h = 2 ** (out_depth + 2)
    with torch.no_grad():
        latent_size = gen.g_mapping.latent_size
        src_latents_np = np.stack([np.random.RandomState(seed).randn(latent_size, ) for seed in src_seeds])
        dst_latents_np = np.stack([np.random.RandomState(seed).randn(latent_size, ) for seed in dst_seeds])
        src_latents = torch.from_numpy(src_latents_np.astype(np.float32))
        dst_latents = torch.from_numpy(dst_latents_np.astype(np.float32))
        src_dlatents = gen.g_mapping(src_latents)  # [seed, layer, component]
        dst_dlatents = gen.g_mapping(dst_latents)  # [seed, layer, component]
        src_images = gen.g_synthesis(src_dlatents, depth=out_depth, alpha=1)
        dst_images = gen.g_synthesis(dst_dlatents, depth=out_depth, alpha=1)

        src_dlatents_np = src_dlatents.numpy()
        dst_dlatents_np = dst_dlatents.numpy()
        canvas = Image.new('RGB', (w * (n_col + 1), h * (n_row + 1)), 'white')
        for col, src_image in enumerate(list(src_images)):
            src_image = adjust_dynamic_range(src_image)
            src_image = src_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
        for row, dst_image in enumerate(list(dst_images)):
            dst_image = adjust_dynamic_range(dst_image)
            dst_image = dst_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))

            row_dlatents = np.stack([dst_dlatents_np[row]] * n_col)
            row_dlatents[:, style_ranges[row]] = src_dlatents_np[:, style_ranges[row]]
            row_dlatents = torch.from_numpy(row_dlatents)

            row_images = gen.g_synthesis(row_dlatents, depth=out_depth, alpha=1)
            for col, image in enumerate(list(row_images)):
                image = adjust_dynamic_range(image)
                image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                canvas.paste(Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
        canvas.save(png)


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    print("Creating generator object ...")
    # create the generator object
    gen = Generator(resolution=opt.dataset.resolution,
                    num_channels=opt.dataset.channels,
                    structure=opt.structure,
                    **opt.model.gen)

    enc = Discriminator(num_channels=opt.dataset.channels,
                                 resolution=opt.dataset.resolution,
                                 structure=opt.structure,
                                 output_features=opt.model.gen.latent_size*2,
                                 **opt.model.dis)


    print("Loading the generator weights from:", args.generator_file)
    # load the weights into it
    gen.load_state_dict(torch.load(args.generator_file))

    print("Loading the encoder weights from:", args.encoder_file)
    # load the weights into it
    enc.load_state_dict(torch.load(args.encoder_file))

    print('loaded')

    image1 = Image.open(args.image1_file).resize((64,64),Image.ANTIALIAS)
    image2 = Image.open(args.image1_file).resize((64,64),Image.ANTIALIAS)
    image1.save(fp="/content/jikkesmall.png")
    image2.save(fp="/content/tobiassmall.png")

    print(image1)
    print(image2)



    # path for saving the files:
    # generate the images:
    # src_seeds = [639, 701, 687, 615, 1999], dst_seeds = [888, 888, 888],
    # draw_style_mixing_figure(os.path.join('figure03-style-mixing.png'), gen,
    #                          out_depth=6, src_seeds=[639, 1995, 687, 615, 1999], dst_seeds=[888, 888, 888],
    #                          style_ranges=[range(0, 2)] * 1 + [range(2, 8)] * 1 + [range(8, 14)] * 1)
    print('Done.')


def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./configs/sample_race_256.yaml')
    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", required=True)

    parser.add_argument("--encoder_file", action="store", type=str,
                        help="pretrained weights file for encoder", required=True)

    parser.add_argument("--image1_file", action="store", type=str, required=True)
    parser.add_argument("--image2_file", action="store", type=str, required=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_arguments())
