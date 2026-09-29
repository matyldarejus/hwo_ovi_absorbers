#!/bin/bash


pipeline_path=~/sh/make_spectra_sh/pipeline.py
model=$1
wind=$2
snap=$3
line=$4
start=$5
end=$6

for ii in $(seq $start $end)
do
   echo "Submitting job $ii"
   python "$pipeline_path" "$model" "$wind" "$snap" "$ii" "$line"
   echo "Finished job $ii"
done



