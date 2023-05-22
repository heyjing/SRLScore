#!/bin/bash



# XSUM---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_agent" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/xsum_srl_leave_agent_False_exact.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_agent" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/xsum_srl_leave_agent_False_rouge.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_agent" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/xsum_srl_leave_agent_False_spacy.json

python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_negation" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/xsum_srl_leave_negation_False_exact.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_negation" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/xsum_srl_leave_negation_False_rouge.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_negation" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/xsum_srl_leave_negation_False_spacy.json

python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_relation" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/xsum_srl_leave_relation_False_exact.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_relation" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/xsum_srl_leave_relation_False_rouge.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_relation" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/xsum_srl_leave_relation_False_spacy.json

python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_patient" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/xsum_srl_leave_patient_False_exact.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_patient" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/xsum_srl_leave_patient_False_rouge.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_patient" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/xsum_srl_leave_patient_False_spacy.json

python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_recipient" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/xsum_srl_leave_recipient_False_exact.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_recipient" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/xsum_srl_leave_recipient_False_rouge.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_recipient" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/xsum_srl_leave_recipient_False_spacy.json

python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_time" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/xsum_srl_leave_time_False_exact.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_time" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/xsum_srl_leave_time_False_rouge.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_time" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/xsum_srl_leave_time_False_spacy.json

python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_location" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/xsum_srl_leave_location_False_exact.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_location" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/xsum_srl_leave_location_False_rouge.json
python3 evaluation.py --json_file "qags-xsum.jsonl" --metric_name "srl" --weights "leave_location" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/xsum_srl_leave_location_False_spacy.json


# cnndm-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_agent" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/cnndm_srl_leave_agent_False_exact.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_agent" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/cnndm_srl_leave_agent_False_rouge.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_agent" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/cnndm_srl_leave_agent_False_spacy.json

python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_negation" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/cnndm_srl_leave_negation_False_exact.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_negation" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/cnndm_srl_leave_negation_False_rouge.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_negation" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/cnndm_srl_leave_negation_False_spacy.json

python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_relation" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/cnndm_srl_leave_relation_False_exact.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_relation" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/cnndm_srl_leave_relation_False_rouge.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_relation" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/cnndm_srl_leave_relation_False_spacy.json

python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_patient" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/cnndm_srl_leave_patient_False_exact.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_patient" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/cnndm_srl_leave_patient_False_rouge.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_patient" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/cnndm_srl_leave_patient_False_spacy.json

python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_recipient" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/cnndm_srl_leave_recipient_False_exact.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_recipient" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/cnndm_srl_leave_recipient_False_rouge.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_recipient" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/cnndm_srl_leave_recipient_False_spacy.json

python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_time" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/cnndm_srl_leave_time_False_exact.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_time" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/cnndm_srl_leave_time_False_rouge.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_time" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/cnndm_srl_leave_time_False_spacy.json

python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_location" --do_coref "False" --string_comparison_method "exact" --output baselines/eval_results/cnndm_srl_leave_location_False_exact.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_location" --do_coref "False" --string_comparison_method "rouge" --output baselines/eval_results/cnndm_srl_leave_location_False_rouge.json
python3 evaluation.py --json_file "qags-cnndm.jsonl" --metric_name "srl" --weights "leave_location" --do_coref "False" --string_comparison_method "spacy" --output baselines/eval_results/cnndm_srl_leave_location_False_spacy.json