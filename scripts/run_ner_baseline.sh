#!/bin/bash

echo -e "dataset\tmodel\trun\twindows_size\tstrategy\tmask_ratio\tepochs\tlearning_rate\tbatch\tprecision\trecall\tf1" > ./results_baseline.tsv

datasets="conll2003 wnut_17"
models="google-bert/bert-base-cased FacebookAI/roberta-base FacebookAI/xlm-roberta-base"

for data in $datasets
do
  for model in $models
	do
	  for epoch in 5 10
		do
		  for lr in 3e-6 5e-6 3e-5 5e-5
				do
          rm -r fine-tuned-ner/
		    	  CUDA_VISIBLE_DEVICES=0 ./run_ner.py \
		    		--hf_dataset_name_or_path $data \
		    		--model_name_or_path $model \
						--text_column_name tokens \
						--ner_tags ner_tags \
						--output_dir ./fine-tuned-ner \
						--strategy no_mask \
						--do_eval \
						--do_test \
						--max_length 512 \
						--batch_size 8 \
						--learning_rate $lr \
						--epochs $epoch \
						--path_to_results ./results/results_baseline.tsv

			done
		done
  done
done




