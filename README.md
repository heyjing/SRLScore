# Evaluating Factual Consistency of Texts with Semantic Role Labeling
Jing Fan<sup>\*</sup>, Dennis Aumiller<sup>\*</sup>, and Michael Gertz  
Institute of Computer Science, Heidelberg University  
<sup>\*</sup> These authors contributed equally to this work.

**2023-05-15**: Our work has been accepted at [\*SEM 2023](https://sites.google.com/view/starsem2023)! We will update the citation once the proceedings become available.

## Installation
We provide an exhaustive list of required packages through the `requirements.txt` file.
However, given the finnicky dependency issues surrounding the (nowadays outdated) AllenNLP release, as well as the spaCy versions required,
we strongly suggest a manual installation of the necessary dependencies.


## Usage
The general usage of our metric `SRLScore` is as follows:

```python
from SRLScore import SRLScore

# Default values are reasonable for most cases
scorer = SRLScore()

scorer.calculate_score(input_text, summary_text)
```

## Experimental Results & Data from the Paper
To repeat experiments that we performed, you may run the `eval.sh` script in this folder.
We further experimented with leaeve-one-argument-out variants of our weights, which is documented in `eval_leave_out-exp.sh`.

Scripts to reproduce the baseline scores (particularly for BARTScore and CoCo, the two most competitive methods with implementations available),
can be found in `baselines/`.
For CoCo, you may further need to clone the respective paper's code repository, copy our `coco_commands.sh` script in their main folder,
and run from there.

`significance_testing.py` will re-compute the significance of differences between various methods.
Note that we apply Bonferroni correction, which makes the significance threshold fairly small!


## Citation

If you found this work helpful, please consider citing our work:

(Will be added once arXiv release becomes available)
