#!/bin/bash

#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=maxgpu             ## or allgpu / cms / cms-uhh / maxgpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name 2Gretrain           # give job unique name
##SBATCH --output ./train.out      # terminal output
##SBATCH --error ./train.err
#SBATCH --mail-type END
#SBATCH --mail-user parada.prangchaikul@desy.de
#SBATCH --constraint=GPU

##SBATCH --nodelist=max-cmsg004         # you can select specific nodes, if necessary
##SBATCH --constraint="P250|V250"        # ask for specific types of GPUs


#####################
### BASH COMMANDS ###
#####################

## examples:

# source .bashrc for init of anaconda
source ~/.bashrc

# (GPU drivers, anaconda, .bashrc, etc) --> likely not necessary if you're installing your own cudatoolkit & cuDNN versions
#source /etc/profile.d/modules.sh
#module load maxwell
#module load cuda 
#module load anaconda/3

# activate your conda environment the job should use
conda activate idealized

# go to your folder with your python scripts
cd /beegfs/desy/user/prangchp/repeat_manuel

# run
python run_training.py --savedir 2Gcombobg_vs_bg/ --nodes 512 512 512 --learning_rate 2e-3 --gaussian 2 --bg_vs_bg --data_reduc 1.0 --train_each_model --model_num 3 4 5 7
# python model_training.py --savedir "1G_250/" --S_over_B "250" --gaussian 1 --data_reduc 1.0
# python model_training.py --savedir "1G_250080/" --S_over_B "250" --gaussian 1 --data_reduc 0.8 
# python model_training.py --savedir "1G_250050/" --S_over_B "250" --gaussian 1 --data_reduc 0.5 
# python model_training.py --savedir "1G_250030/" --S_over_B "250" --gaussian 1 --data_reduc 0.3  
# python model_training.py --savedir "1G_250020/" --S_over_B "250" --gaussian 1 --data_reduc 0.2 
# python model_training.py --savedir "1G_250015/" --S_over_B "250" --gaussian 1 --data_reduc 0.15  
# python model_training.py --savedir "1G_250010/" --S_over_B "250" --gaussian 1 --data_reduc 0.1