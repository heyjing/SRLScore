#!/bin/bash

# Run this script to evaluate

# SUMMEVAL
python3 evaluation.py --json_file "summeval.jsonl" --metric_name "rouge" --output baselines/eval_results/summeval_rouge.json
python3 evaluation.py --json_file "summeval.jsonl" --metric_name "bleu" --output baselines/eval_results/summeval_bleu.json
python3 evaluation.py --json_file "summeval.jsonl" --metric_name "meteor" --output baselines/eval_results/summeval_meteor.json

python3 evaluation.py --json_file "summeval.jsonl" --metric_name "goodrich" --weights "1/3" --do_coref "False" --output baselines/eval_results/summeval_srl_goodrich.json

python3 evaluation.py --json_file "summeval.jsonl" --metric_name "bartscore" --output baselines/eval_results/summeval_bartscore.json
python3 evaluation.py --json_file "summeval.jsonl" --metric_name "bartscore_cnn" --output baselines/eval_results/summeval_bartscore_cnn.json
python3 evaluation.py --json_file "summeval.jsonl" --metric_name "bartscore_para" --output baselines/eval_results/summeval_bartscore_para.json

python3 evaluation.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/summeval_srl_baseline_False_exact.json
python3 evaluation.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/summeval_srl_baseline_False_rouge.json
python3 evaluation.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/summeval_srl_baseline_False_spacy.json

python3 evaluation.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/summeval_srl_fullset_False_exact.json
python3 evaluation.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/summeval_srl_fullset_False_rouge.json
python3 evaluation.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/summeval_srl_fullset_False_spacy.json

python3 evaluation.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "exact" --output baselines/eval_results/summeval_srl_fullset_True_exact.json
python3 evaluation.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "rouge" --output baselines/eval_results/summeval_srl_fullset_True_rouge.json
python3 evaluation.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "spacy" --output baselines/eval_results/summeval_srl_fullset_True_spacy.json

# XSUM---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "rouge" --output baselines/eval_results/xsum_rouge.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "bleu" --output baselines/eval_results/xsum_bleu.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "meteor" --output baselines/eval_results/xsum_meteor.json

python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "goodrich" --weights "1/3" --do_coref "False" --output baselines/eval_results/xsum_srl_goodrich.json

python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "bartscore" --output baselines/eval_results/xsum_bartscore.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "bartscore_cnn" --output baselines/eval_results/xsum_bartscore_cnn.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "bartscore_para" --output baselines/eval_results/xsum_bartscore_para.json

python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/xsum_srl_baseline_False_exact.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/xsum_srl_baseline_False_rouge.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/xsum_srl_baseline_False_spacy.json

python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/xsum_srl_fullset_False_exact.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/xsum_srl_fullset_False_rouge.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/xsum_srl_fullset_False_spacy.json

python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "exact" --output baselines/eval_results/xsum_srl_fullset_True_exact.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "rouge" --output baselines/eval_results/xsum_srl_fullset_True_rouge.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "spacy" --output baselines/eval_results/xsum_srl_fullset_True_spacy.json

# cnndm-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "rouge" --output baselines/eval_results/cnndm_rouge.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "bleu" --output baselines/eval_results/cnndm_bleu.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "meteor" --output baselines/eval_results/cnndm_meteor.json

python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "goodrich" --weights "1/3" --do_coref "False" --output baselines/eval_results/cnndm_srl_goodrich.json

python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "bartscore" --output baselines/eval_results/cnndm_bartscore.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "bartscore_cnn" --output baselines/eval_results/cnndm_bartscore_cnn.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "bartscore_para" --output baselines/eval_results/cnndm_bartscore_para.json

python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/cnndm_srl_baseline_False_exact.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/cnndm_srl_baseline_False_rouge.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/cnndm_srl_baseline_False_spacy.json

python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/cnndm_srl_fullset_False_exact.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/cnndm_srl_fullset_False_rouge.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/cnndm_srl_fullset_False_spacy.json

python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "exact" --output baselines/eval_results/cnndm_srl_fullset_True_exact.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "rouge" --output baselines/eval_results/cnndm_srl_fullset_True_rouge.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "spacy" --output baselines/eval_results/cnndm_srl_fullset_True_spacy.json