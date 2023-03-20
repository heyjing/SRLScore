#!/bin/bash

# SUMMEVAL
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "rouge" --output eval_results/summeval_rouge.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "bleu" --output eval_results/summeval_bleu.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "meteor" --output eval_results/summeval_meteor.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "goodrich" --weights "1/3" --do_coref "False" --output eval_results/summeval_srl_goodrich.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "bartscore" --output eval_results/summeval_bartscore.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "bartscore_cnn" --output eval_results/summeval_bartscore_cnn.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "bartscore_para" --output eval_results/summeval_bartscore_para.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "exact" --output eval_results/summeval_srl_baseline_False_exact.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "rouge" --output eval_results/summeval_srl_baseline_False_rouge.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "spacy" --output eval_results/summeval_srl_baseline_False_spacy.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "exact" --output eval_results/summeval_srl_fullset_False_exact.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "rouge" --output eval_results/summeval_srl_fullset_False_rouge.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "spacy" --output eval_results/summeval_srl_fullset_False_spacy.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "exact" --output eval_results/summeval_srl_fullset_True_exact.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "rouge" --output eval_results/summeval_srl_fullset_True_rouge.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "summeval.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "spacy" --output eval_results/summeval_srl_fullset_True_spacy.json

# XSUM---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "rouge" --output eval_results/xsum_rouge.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "bleu" --output eval_results/xsum_bleu.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "meteor" --output eval_results/xsum_meteor.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "goodrich" --weights "1/3" --do_coref "False" --output eval_results/xsum_srl_goodrich.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "bartscore" --output eval_results/xsum_bartscore.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "bartscore_cnn" --output eval_results/xsum_bartscore_cnn.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "bartscore_para" --output eval_results/xsum_bartscore_para.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "exact" --output eval_results/xsum_srl_baseline_False_exact.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "rouge" --output eval_results/xsum_srl_baseline_False_rouge.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "spacy" --output eval_results/xsum_srl_baseline_False_spacy.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "exact" --output eval_results/xsum_srl_fullset_False_exact.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "rouge" --output eval_results/xsum_srl_fullset_False_rouge.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "spacy" --output eval_results/xsum_srl_fullset_False_spacy.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "exact" --output eval_results/xsum_srl_fullset_True_exact.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "rouge" --output eval_results/xsum_srl_fullset_True_rouge.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "spacy" --output eval_results/xsum_srl_fullset_True_spacy.json

# cnndm-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "rouge" --output eval_results/cnndm_rouge.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "bleu" --output eval_results/cnndm_bleu.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "meteor" --output eval_results/cnndm_meteor.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "goodrich" --weights "1/3" --do_coref "False" --output eval_results/cnndm_srl_goodrich.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "bartscore" --output eval_results/cnndm_bartscore.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "bartscore_cnn" --output eval_results/cnndm_bartscore_cnn.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "bartscore_para" --output eval_results/cnndm_bartscore_para.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "exact" --output eval_results/cnndm_srl_baseline_False_exact.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "rouge" --output eval_results/cnndm_srl_baseline_False_rouge.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/3" --do_coref "False" --string_comparison_method "spacy" --output eval_results/cnndm_srl_baseline_False_spacy.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "exact" --output eval_results/cnndm_srl_fullset_False_exact.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "rouge" --output eval_results/cnndm_srl_fullset_False_rouge.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/7" --do_coref "False" --string_comparison_method "spacy" --output eval_results/cnndm_srl_fullset_False_spacy.json

/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "exact" --output eval_results/cnndm_srl_fullset_True_exact.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "rouge" --output eval_results/cnndm_srl_fullset_True_rouge.json
/home/jfan/anaconda3/bin/python /home/jfan/anaconda3/envs/thesis/2022-jing-fan-thesis/eval_qags.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "1/7" --do_coref "True" --string_comparison_method "spacy" --output eval_results/cnndm_srl_fullset_True_spacy.json