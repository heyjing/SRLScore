python3 run_coco.py --model_path ./bart.large.cnn/ --data_path ./data/qags-cnndm --output_file qags-cnndm-span-scores.txt --mask span
python3 run_coco.py --model_path ./bart.large.cnn/ --data_path ./data/qags-cnndm --output_file qags-cnndm-sent-scores.txt --mask sent

python3 run_coco.py --model_path ./bart.large.xsum/ --data_path ./data/qags-xsum --output_file qags-xsum-span-scores.txt --mask span
python3 run_coco.py --model_path ./bart.large.xsum/ --data_path ./data/qags-xsum --output_file qags-xsum-sent-scores.txt --mask sent

python3 run_coco.py --model_path ./bart.large.cnn/ --data_path ./data/summeval --output_file summeval-span-scores.txt --mask span
python3 run_coco.py --model_path ./bart.large.cnn/ --data_path ./data/summeval --output_file summeval-sent-scores.txt --mask sent