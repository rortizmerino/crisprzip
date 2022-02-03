#!/bin/bash
#
#PBS -N CRISPR_kinetic_model
#PBS -l nodes=1:ppn=1
#PBS -o .temp/latest.o
#PBS -e .temp/latest.e
#PBS -m ae  # mail when job is aborted (a) or terminates (e)

# cd to root directory (project folder)
# PBS_O_WORKDIR points to where this script was called (which is the root dir)
root_dir=$PBS_O_WORKDIR
cd "${root_dir}" || { echo "Failed to enter ${root_dir}"; exit 1; }

# job directory, e.g. results/20220127_457335 (27 Jan 2022, job id 457335)
job_dir="results/$(date +'%Y%m%d')_${PBS_JOBID:0:6}"
if [ ! -d "$job_dir" ]; then
  mkdir -p "$job_dir";
fi

# store input parameters (-F) in job directory
echo "$@" > "${job_dir}/args.txt"

# run id, e.g. results/20220127_457335/003 (3rd run in job array)
array_id=$((PBS_ARRAYID>1 ? PBS_ARRAYID : 1))  # run id at least 1
run_dir="${job_dir}/$(printf "%03d" "$array_id")"
if [ ! -d "$run_dir" ]; then
  mkdir -p "$run_dir";
fi

# combining stdout and stderr, stored in run directory
exec &> "${run_dir}/stdout.txt"

# activate environment, append project folder to python path
source activate crispr_kin_model_env
export PYTHONPATH=${PYTHONPATH}:"${root_dir}"

# all arguments ($@) after flag -F passed to python
# array_id and out_path added as kwargs (dealt with in python script)
py_args=("${@:1}" "array_id=$array_id" "out_path=$run_dir")
python "${py_args[@]}"
