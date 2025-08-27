#!/bin/bash

export MASTER_ADDR=$(hostname)
config_file=./config/operators_poisson.yaml
config="poisson-scale-k1_5"
run_num="01"

# path/to/logs
results_dir=$SCRATCH/clearml_tests/results
mkdir -p ${results_dir}

cmd="python train.py --yaml_config=$config_file --config=$config --run_num=$run_num --root_dir=$results_dir"
source export_DDP_vars.sh
exec $cmd
