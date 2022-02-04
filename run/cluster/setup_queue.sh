#!/bin/bash

# input
my_dir="$(dirname "$0")"
source "${my_dir}/user.config"  # contains netid, email, root_dir
remote="${netid}@hpc05.tudelft.net"

# Making sure project root directory and .temp exist (for stdout)
ssh "$remote" "if [ ! -d ${remote_root_dir} ]; then mkdir -p ${remote_root_dir}; fi"
ssh "$remote" "if [ ! -d ${remote_root_dir}/.temp ]; then mkdir -p ${remote_root_dir}/.temp; fi"

# Synchronizing data, model, run directories
printf "\nSynchronizing data, model, run directories... "

# data (w/o data_processing, prepared_experimental, rawdata)
rsync -r --exclude "data_processing" --exclude "prepared_experimental" \
 --exclude "rawdata" --delete \
 "${local_root_dir}/data" "${remote}:${remote_root_dir}"

# model (w/o cache)
rsync -r --exclude "__pycache__/" --delete \
 "${local_root_dir}/model" "${remote}:${remote_root_dir}"

# run (all)
rsync -r --delete \
 "${local_root_dir}/run" "${remote}:${remote_root_dir}"

printf "done!\n"

# setting up qsub queue
printf "Setting up qsub job... "
qsub_args=$1  # e.g. -t 1-5%5 -l nodes=1:ppn=10
py_args=$( IFS=' '; echo "${@:2}" )  # e.g. run/test_ppn.py -1

ssh "$remote" "cd ${remote_root_dir}; qsub $qsub_args -F \"${py_args}\" -M ${email} ${remote_root_dir}/run/cluster/runner.sh"

# return job status
ssh "$remote" "qstat -ae -u ${netid}"
