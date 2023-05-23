# Script to install the required dependencies in a way that works; notably, a downgrade of typing-extensions
# is currently required, see https://github.com/pydantic/pydantic/issues/545#issuecomment-1558685292
python3 -m pip install allennlp
python3 -m pip install allennlp-models
python3 -m pip install typing-extensions<=4.5.0
python3 -m pip install rouge-score
python3 -m spacy download en-core-web-sm
python3 -m spacy download en-core-web-lg