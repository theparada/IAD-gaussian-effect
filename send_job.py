import os
import sys
import argparse
import subprocess
from os.path import join, dirname
from itertools import combinations

perm = combinations([0, 1, 2, 3, 4, 5], 4)
x_combi_name = []
x_combi = []
for i in list(perm):
    config = ""
    pre_x = []
    for j in i:
        config += str(j)
        pre_x.append(str(j))
    x_combi.append(pre_x)
    x_combi_name.append(config)

reduc = [1.0, 0.8, 0.5, 0.3, 0.2, 0.15, 0.1]
fraction = ["", "080","050","030","020","015","010"]
sbs = ["006","007","009","015","020","030","040","050","100","150","250"]
# sbs = [""]
# lr = [5e-2, 2e-3, 5e-3, 1e-4]

count = 0
for sb in sbs:
    # for i in lr:
    for frac, red in zip(fraction, reduc):
        command = "python run_training.py --savedir 0G" + sb + frac + "nodes512lr2e-3/ --gaussian 0 --nodes 512 512 512 --learning_rate 2e-3 --S_over_B " + sb + " --data_reduc " + str(red)
        # command = "python run_training.py --savedir 2Glr" + str(i) + "/ --gaussian 2 --learning_rate " + str(i)
        home_path = "/beegfs/desy/user/prangchp/"
        email = "parada.prangchaikul@desy.de"
        conda_env = "idealized"
        base_path = "/beegfs/desy/user/prangchp/IAD-gaussian-effect/"
        slurm_dir = os.path.join(base_path, "slurm_bin")
        executable_file = os.path.join("slurm_bin/", "run_python.sh")

        # If the provided directory does not exist yet, create it first.
        if not os.path.exists(slurm_dir):
            try:
                os.mkdir(slurm_dir)
            except:
                print(f"something went wrong in the creation of slurm_bin")
                sys.exit()

        ## creating the executable file
        with open(executable_file, "w") as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"\n")
            f.write(f"#SBATCH --partition=allgpu\n") # allgpu or maxgpu
            f.write(f"#SBATCH --time=24:00:00\n")
            f.write(f"#SBATCH --nodes=1\n")
            f.write(f"#SBATCH --job-name 0Gcombo"+sb+frac+"\n")
            # f.write(f"#SBATCH --job-name 2Glr"+str(i)+"\n")
            f.write(f"#SBATCH --output {slurm_dir}/run_all_out.txt\n")
            f.write(f"#SBATCH --error {slurm_dir}/run_all_error.txt\n")
            f.write(f"#SBATCH --mail-type END\n")
            f.write(f"#SBATCH --mail-user {email}\n")
            f.write(f"#SBATCH --constraint='GPUx1'\n")
            f.write(f"\n")
            f.write(f"source {os.path.join(home_path,'.bashrc')}\n")
            f.write(f"conda deactivate\n")
            f.write(f"conda activate {conda_env}\n")
            f.write(f"cd {base_path}\n")
            f.write(command)
        subprocess.run(['sbatch', '--requeue', executable_file]) 
        count += 1