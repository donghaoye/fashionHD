#!/bin/bash
declare -a id_arr=("5.4" "6.0" )

for model_id in "${id_arr[@]}"
do
    python test_mmdganv2_transfer_edge.py --id $model_id --gpu_ids 12
    python test_mmdganv2_transfer_color.py --id $model_id --gpu_ids 12
done

