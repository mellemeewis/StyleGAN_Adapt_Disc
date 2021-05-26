#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

module load cuda10.0/toolkit
module load cuDNN/cuda10.0

source /home/mms496/.bashrc

cd /home/mms496/StyleVAE_Experiments/stylegan

if [ -d "/home/mms496/StyleVAE_Experiments/stylegan_adapt_disc/ffhq" ] 
then
    echo $$
	mkdir oo`echo $$`
	cd oo`echo $$` 
	cp -R /home/mms496/StyleVAE_Experiments/stylegan_adapt_disc/ffhq .
	rm -rf /home/mms496/StyleVAE_Experiments/stylegan_adapt_disc/ffhq
	cd /home/mms496/StyleVAE_Experiments/stylegan_adapt_disc

else 
	echo 'no problem'
fi




python -u /home/mms496/StyleVAE_Experiments/code/StyleGAN.pytorch-adapt_disc/train.py --config /home/mms496/StyleVAE_Experiments/code/StyleGAN.pytorch-adapt_disc/configs/ffhq_128.yaml

wait          # wait until programs are finished

echo $$
mkdir o`echo $$`
cd o`echo $$`

cp -R /home/mms496/StyleVAE_Experiments/stylegan_adapt_disc/ffhq .
# rm -rf /home/mms496/StyleVAE_Experiments/stylegan_adapt_disc/ffhq
