#!/bin/bash
#PBS -N CRISPR_kinetic_model
#PBS -l nodes=1:ppn=20
#PBS -M H.S.Offerhaus@tudelft.nl
#PBS -o /home/hiddeofferhaus/depkengit/CRISPR_kinetic_model/results/hpc_test/stdout.txt
#PBS -e /home/hiddeofferhaus/depkengit/CRISPR_kinetic_model/results/hpc_test/stderr.txt

# Start job
echo "starting job..."
project_folder="/home/hiddeofferhaus/depkengit/CRISPR_kinetic_model"

# Activate conda environment, add own module
source activate crispr_kin_model_env
export PYTHONPATH="${project_folder}/model"

# Run python script
python "${project_folder}/run/fit_data.py" "${project_folder}/results/hpc_test/new_log.txt"
