#!/bin/bash

#PBS -N CRISPR_kinetic_model
#PBS -l nodes=1:ppn=20
#PBS -M H.S.Offerhaus@tudelft.nl
#PBS -o /home/hiddeofferhaus/depkengit/CRISPR_kinetic_model/results/hpc_test/stdout.txt
#PBS -e /home/hiddeofferhaus/depkengit/CRISPR_kinetic_model/results/hpc_test/stderr.txt

# Start job
# cd $PBS_O_WORKDIR
project_folder="/home/hiddeofferhaus/depkengit/CRISPR_kinetic_model"

# TODO: check python location on cluster
python3.9 "${project_folder}/run/fit_data.py" "${project_folder}/results/hpc_test/01.txt"