#!/bin/bash
#SBATCH --time=60:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C RTX2080Ti
#SBATCH -p proq
#SBATCH --gres=gpu:1

module load cuda10.0/toolkit
module load cuDNN/cuda10.0

source /home/mms496/.bashrc

cd /home/mms496/StyleVAE_Experiments/stylegan_adapt_disc

# if [ -d "/home/mms496/StyleVAE_Experiments/stylegan_adapt_disc/ON|05-1" ] 
# then
#     echo $$
#   mkdir oo`echo $$`
#   cd oo`echo $$` 
#   cp -R '/home/mms496/StyleVAE_Experiments/stylegan_adapt_disc/ON|05-1' .
#   rm -rf '/home/mms496/StyleVAE_Experiments/stylegan_adapt_disc/ON|05-1'
#   cd /home/mms496/StyleVAE_Experiments/stylegan_adapt_disc

# else 
#   echo 'no problem'
# fi




python -u /home/mms496/StyleVAE_Experiments/code/StyleGAN.pytorch-adapt_disc/generate_grid.py --config '/home/mms496/StyleVAE_Experiments/code/StyleGAN.pytorch-adapt_disc/configs/mnist/CBf-r01f01-g4d4-AdvStyleVAE_NS.yaml' --generator_file '/var/scratch/mms496/models/CBf-r01f01-g4d4-AdvStyleVAE_NS/GAN_GEN_SHADOW_3.pth' --n_row 6 --n_col 8 --output_dir '/home/mms496/StyleVAE_Experiments/stylegan_adapt_disc/grid'
wait          # wait until programs are finished

# echo $$
# mkdir o`echo $$`
# cd o`echo $$`

# cp -R /home/mms496/StyleVAE_Experiments/stylegan_adapt_disc/ffhq0>05 .
# # rm -rf /home/mms496/StyleVAE_Experiments/stylegan_adapt_disc/ffhq0>05




