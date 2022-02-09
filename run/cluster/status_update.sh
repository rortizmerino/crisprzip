#!/bin/bash

# input
my_dir="$(dirname "$0")"
source "${my_dir}/user.config"  # contains netid, email, root_dir
remote="${netid}@hpc05.tudelft.net"

# check if jobs in queue
if [[ -z $(ssh "$remote" "qstat -ae -u ${netid}") ]]
then
  echo "hpc05.hpc: No jobs in queue"
  echo
else
  ssh "$remote" "qstat -t -u ${netid}";
  echo
fi

ssh "$remote" "python -B stat.py"