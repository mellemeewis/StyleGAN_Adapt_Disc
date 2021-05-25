#!/bin/bash
#SBATCH --time=13:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

module load cuda10.0/toolkit
module load cuDNN/cuda10.0

source /home/mms496/.bashrc

cd /home/mms496/StyleVAE_Experiments/stylegan_original

if [ -d "/home/mms496/StyleVAE_Experiments/stylegan_original/gan" ] 
then
    echo $$
	mkdir oo`echo $$`
	cd oo`echo $$` 
	cp -R /home/mms496/StyleVAE_Experiments/stylegan_original/gan .
	rm -rf /home/mms496/StyleVAE_Experiments/stylegan_original/gan
	cd /home/mms496/StyleVAE_Experiments/stylegan_original

else 
	echo 'no problem'
fi




python -u /home/mms496/StyleVAE_Experiments/code/StyleGAN.pytorch-original/train.py --config /home/mms496/StyleVAE_Experiments/code/StyleGAN.pytorch-original/configs/sample_ffhq_128.yaml

wait          # wait until programs are finished

echo $$
mkdir o`echo $$`
cd o`echo $$`

cp -R /home/mms496/StyleVAE_Experiments/stylegan_original/output_gan .
rm -rf /home/mms496/StyleVAE_Experiments/stylegan_original/output_gan
