#!/bin/bash

# input
source run/cluster/user.config  # contains netid, email, root_dir
remote="${netid}@hpc05.tudelft.net"

# collecting results
printf "Collecting results...\n"
rsync -rv --min-size=1 --remove-source-files "${remote}:${root_dir}/results/" results
# min-size prevents empty (typically, stdout) files from being copied

# after rsync, we remove all results to keep the cluster clean
echo
ssh $remote "find ${root_dir}/results/ -type f -size 0 -print"
ssh $remote "find ${root_dir}/results/ -type d -mindepth 1 -print"
# prompt for confirmation
read -p "Remove the above empty files and folders from the results directory? (y/n): " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ssh $remote "find ${root_dir}/results/ -type f -size 0 -delete";
    ssh $remote "find ${root_dir}/results/ -type d -mindepth 1 -empty -delete";
    echo 'Removed from cluster'
fi

# remove everything from the .temp folder
ssh $remote "cd ${root_dir}/.temp; rm -rf *"
