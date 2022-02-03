#!/bin/bash

# input
source run/cluster/user.config # contains netid, email, root_dir
remote="${netid}@hpc05.tudelft.net"

# Making sure project root directory and home/.temp exist (for stdout)
ssh "$remote" "if [ ! -d ${root_dir} ]; then mkdir -p ${root_dir}; fi"
ssh "$remote" "if [ ! -d .temp ]; then mkdir -p .temp; fi"

# synchronizing data, model, run directories
printf "\nSynchronizing data, model, run directories... "

# data (w/o data_processing, prepared_experimental, rawdata)
rsync -r --exclude "data_processing" --exclude "prepared_experimental" \
 --exclude "rawdata" --delete data "${remote}:${root_dir}"
# model (w/o cache)
rsync -r --exclude "__pycache__/" --delete\
 model "${remote}:${root_dir}"
# run (all)
rsync -r --delete run "${remote}:${root_dir}"
printf "done!\n"

# setting up qsub queue
printf "Setting up qsub job... "
qsub_args=$1  # e.g. -t 1-5%5 -l nodes=1:ppn=10
py_args=$( IFS=' '; echo "${@:2}" )  # e.g. run/test_ppn.py -1

#echo "$qsub_args -F \"${py_args}\" -M ${email} run/cluster/runner.sh"

ssh "$remote" "cd ${root_dir}; qsub $qsub_args -F \"${py_args}\" -M ${email} run/cluster/runner.sh"
ssh "$remote" "qstat -ae -u ${netid}"
