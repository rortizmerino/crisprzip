#!/bin/bash

# input
my_dir="$(dirname "$0")"
source "${my_dir}/user.config"  # contains netid, email, root_dirs
remote="${netid}@hpc05.tudelft.net"

# check if results dir empty
new_job_dirs=$(ssh "$remote" "ls -A ${remote_root_dir}/results/")
if [[ -z $new_job_dirs ]]; then
  echo "Results directory empty"
  exit 0
fi

# collecting the results
read -p "Copy results directly to the project drive? (y/n): "  -r
if [[ $REPLY =~ ^[Yy]$ ]]; then

  # download to project drive
  for dir in $(ssh "$remote" "ls -d ${remote_root_dir}/results/*"); do
    sftp -r "${remote}:${dir} ${project_drive_dir}/hpc_results";
  done;
  echo "Copied to project drive";

else
  # get local path
  read -p "Copy to local path (default: ${local_root_dir}/results): " -r local_path;
  if [[ $local_path = "" ]]; then local_path="${local_root_dir}/results"; fi;
  if [ ! -d "$local_path" ]; then mkdir -p "$local_path"; fi;

  # download to local path
  for dir in $(ssh "$remote" "ls -d ${remote_root_dir}/results/*"); do
    sftp -r "${remote}:${dir} ${local_path}";
  done;
  echo "Copied to ${local_path}";
fi

# after downloading, we remove all results to keep the cluster clean
echo
ssh "$remote" "find ${remote_root_dir}/results/ -mindepth 1 -print" | sort

# prompt for confirmation
read -p "Remove the above files and folders from the results directory? (y/n): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ssh "$remote" "find ${remote_root_dir}/results/ -mindepth 1 -delete";
    echo "Removed from cluster";
fi

# remove everything from the .temp folder
ssh "$remote" "cd ${remote_root_dir}/.temp; rm -rf *"
