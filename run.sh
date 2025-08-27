#!/bin/bash

ntrain=32768
nval=4096
ntest=4096
ng=144
datapath=$PWD

e1=1  # poissons diffusion eigenvalue range
e2=5

adr1=0.2 # advection to diffusion ratio range
adr2=1
# for AD ratio we saved a set of velocity scales that correspond to AD ration in utils/*.npy. See python script for details

o1=1 # helmholtz wave number range
o2=10

python utils/gen_data_poisson.py --ntrain=$ntrain --nval=$nval --ntest=$ntest \
                    --ng=$ng --sparse --n 128 --datapath $datapath --e1 $e1 --e2 $e2

python utils/get_scale.py


export MASTER_ADDR=$(hostname)
config_file=./config/operators_poisson.yaml
config="poisson-scale-k1_5"
run_num="02"

# path/to/logs
results_dir=$SCRATCH/clearml_tests/results
mkdir -p ${results_dir}

source export_DDP_vars.sh
python train.py --yaml_config=$config_file --config=$config --run_num=$run_num --root_dir=$results_dir
