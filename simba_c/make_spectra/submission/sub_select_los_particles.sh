#!/bin/bash

pipeline_path=~/sh/simba_c/make_spectra_sh/select_los_particles.py
model=$1
wind=$2
snap=$3
nlos=$4

for i in {0..216}; do
    echo "Running galaxy $i..."
    python $pipeline_path $model $wind $snap $i $nlos
done
