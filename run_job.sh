#!/bin/bash

#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=allgpu             ## or allgpu / cms / cms-uhh / maxgpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name datagen           # give job unique name
##SBATCH --output ./tune10G.out      # terminal output
##SBATCH --error ./tune10G.err
#SBATCH --mail-type END
#SBATCH --mail-user parada.prangchaikul@desy.de
#SBATCH --constraint=GPU

##SBATCH --nodelist=max-cmsg004         # you can select specific nodes, if necessary
##SBATCH --constraint="P100|V100"        # ask for specific types of GPUs


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
conda activate cathode

# go to your folder with your python scripts
cd /beegfs/desy/user/prangchp/repeat_manuel

# run
# python data_generator.py --input_dir "new_separated_data/" --gaussian 10 --testset
python data_generator.py --input_dir "data_SB_006/" --S_over_B "006" --gaussian 10
python data_generator.py --input_dir "data_SB_007/" --S_over_B "007" --gaussian 10
python data_generator.py --input_dir "data_SB_009/" --S_over_B "009" --gaussian 10
python data_generator.py --input_dir "data_SB_015/" --S_over_B "015" --gaussian 10
python data_generator.py --input_dir "data_SB_020/" --S_over_B "020" --gaussian 10
python data_generator.py --input_dir "data_SB_030/" --S_over_B "030" --gaussian 10
python data_generator.py --input_dir "data_SB_040/" --S_over_B "040" --gaussian 10
python data_generator.py --input_dir "data_SB_050/" --S_over_B "050" --gaussian 10
python data_generator.py --input_dir "data_SB_100/" --S_over_B "100" --gaussian 10
python data_generator.py --input_dir "data_SB_150/" --S_over_B "150" --gaussian 10
python data_generator.py --input_dir "data_SB_250/" --S_over_B "250" --gaussian 10