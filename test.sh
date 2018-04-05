#!/bin/bash
#declare -a id_arr=("5.4" "6.0" )
#
#for model_id in "${id_arr[@]}"
#do
#    python test_mmdganv2_transfer_edge.py --id $model_id --gpu_ids 12
#    python test_mmdganv2_transfer_color.py --id $model_id --gpu_ids 12
#done

#for i in {0..24..1}
#do
#	python test_dfn_encoder.py --id 4.$i --gpu_ids 15
#done

declare -a ids=("5.0" "5.1" "5.2" "6.0" "6.1")
for model_id in "${ids[@]}"
do
	python test_dfn_encoder.py --id $model_id --gpu_ids 14
done
