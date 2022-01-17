#!/bin/bash
#PBS -N CRISPR_kinetic_model
#PBS -l nodes=1:ppn=20
#PBS -M H.S.Offerhaus@tudelft.nl
#PBS -o /home/hiddeofferhaus/depkengit/CRISPR_kinetic_model/results/hpc_test/${PBS.JOBID}_stdout.txt
#PBS -e /home/hiddeofferhaus/depkengit/CRISPR_kinetic_model/results/hpc_test/${PBS.JOBID}_stderr.txt

# Start job
echo "starting job..."
# cd $PBS_O_WORKDIR
project_folder="/home/hiddeofferhaus/depkengit/CRISPR_kinetic_model"
results_folder


source activate crispr_kin_model_env
python "${project_folder}/run/fit_data.py" "${project_folder}/results/hpc_test/${PBS.JOBID}_log.txt"
