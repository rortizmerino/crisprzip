#!/bin/bash
#
#PBS -N CRISPR_kinetic_model
#PBS -l nodes=1:ppn=20
#PBS -M H.S.Offerhaus@tudelft.nl
#PBS -m ae  # mail when job is aborted (a) or terminates (e)

# cd to root directory (project folder)
root_dir="home/hiddeofferhaus/depkengit/CRISPR_kinetic_model"
cd $root_dir || { echo "Failed to enter {$root_dir}"; exit 1; }

# making the following file structure
# results
# ├── 20220122_451254
# ├── 20220127_457332
# └── 20220127_457335
#     ├── args.txt
#     ├── 001
#     ├── 002
#     └── 003
#         ├── another_result.csv
#         ├── some_result.txt
#         ├── stderr.txt
#         └── stdout.txt

# job directory, e.g. results/20220127_457335 (27 Jan 2022, job id 457335)
job_dir="results/$(date +"%Y%m%d")_${PBS_JOBID:0:6}"
if [ ! -d "$job_dir" ]; then
  mkdir -p "$job_dir";
fi

# store input parameters in job directory
echo "$@" > "${job_dir}/args.txt"

# run id, e.g. results/20220127_457335/003 (3rd run in job array)
run_dir="${job_dir}/$(printf "%03d" "$PBS_ARRAYID")"
if [ ! -d "$run_dir" ]; then
  mkdir -p "$run_dir";
fi

# storing stdout and stderr in run directory
exec 1>"${run_dir}/stdout.txt"
exec 2>"${run_dir}/stderr.txt"

# activate environment, append project folder to python path
source activate crispr_kin_model_env
export PYTHONPATH=${PYTHONPATH}:"${root_dir}"

# all arguments ($@) after flag -F passed to python
# 1st arg: path of python script (run/fit_data.py)
# 2nd+ args: passed as argv to the python script
python "$@"