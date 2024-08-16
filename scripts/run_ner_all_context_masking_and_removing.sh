#!/bin/bash

echo -e "dataset\tmodel\trun\twindows_size\tstrategy\tmask_ratio\tepochs\tlearning_rate\tbatch\tprecision\trecall\tf1" > ../results/results_mask_context.tsv

strategies="all_context remove"

sed -e 1d "./baseline_params.tsv" | while read -r dataset model epochs learning_rate
do
		for strat in $strategies
				do
						for ratio in 0.01 0.03 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
						do
								for se in $(seq 1 2)
								do
										rm -r ../models/ner-masked-context-model/
										CUDA_VISIBLE_DEVICES=0 ../run_ner.py \
										--hf_dataset_name_or_path $dataset \
										--model_name_or_path $model \
										--text_column_name tokens \
										--ner_tags ner_tags \
										--output_dir ../models/ner-masked-context-model \
										--windows_size 0 \
										--strategy $strat \
										--p_mask $ratio \
										--do_eval \
										--do_test \
										--max_length 512 \
										--batch_size 16 \
										--learning_rate $learning_rate \
										--epochs $epochs \
										--seed $se \
										--path_to_results ../results/results_mask_and_remove_all.tsv
								done
						done
				done
		done
done
