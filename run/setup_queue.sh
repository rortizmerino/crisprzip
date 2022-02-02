#!/bin/bash

# input
#netid="hiddeofferhaus"
#root_dir="/home/${netid}/CRISPR_kinetic_model"

source run/user.config # contains netid, email, root_dir
remote="${netid}@hpc05.tudelft.net"

# synchronizing data (partial), model, run directories
printf "synchronizing data, model, run directories"
ssh $remote "if [ ! -d ${root_dir} ]; then mkdir -p ${root_dir}; fi"
rsync -rhv data/SpCas9 "${remote}:${root_dir}"
rsync -rhv --exclude "__pycache__/" model "${remote}:${root_dir}"
rsync -rhv run "${remote}:${root_dir}"

# setting up qsub queue
printf "\nsetting up qsub queue"
space_sep_args="'$*'"
ssh $remote "cd ${root_dir}; qsub -t 1-5%5 -F ${space_sep_args} run/runner.sh"
ssh $remote "qstat -ae -u ${netid}"
