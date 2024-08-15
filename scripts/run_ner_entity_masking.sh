#!/bin/bash

echo -e "dataset\tmodel\trun\twindows_size\tstrategy\tmask_ratio\tepochs\tlearning_rate\tbatch\tprecision\trecall\tf1" > ../results/results_mask_entity.tsv


sed -e 1d "./baseline_params.tsv" | while read -r dataset model epochs learning_rate
do
		for ratio in 0.01 0.03 0.05 0.1 0.2 0.3 0.4 0.5 0.5 0.6 0.7 0.8 0.9 1
		do
				for se in $(seq 1 3)
				do
					rm -r ../models/ner-masked-entity-model
						CUDA_VISIBLE_DEVICES=1 ../run_ner.py \
						--hf_dataset_name_or_path $dataset \
						--model_name_or_path $model \
						--text_column_name tokens \
						--ner_tags ner_tags \
						--output_dir ../models/ner-masked-entity-model \
						--windows_size 0 \
						--strategy entity \
						--p_mask $ratio \
						--do_eval \
						--do_test \
						--max_length 512 \
						--batch_size 16 \
						--learning_rate $learning_rate \
						--epochs $epochs \
						--seed $se \
						--path_to_results ../results/results_mask_entity.tsv
				done
		done
done

