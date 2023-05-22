"""
Converter script to transform the QAGS/SummEval data into a format compatible with the CoCo paper.
"""

import json


def convert_qags_cnndm():
    with open("../data/qags-cnndm.jsonl", "r") as f:
        lines = list(f)

    # Empty existing files
    with open("./coco_scores/source-qags-cnndm.txt", "w") as f:
        pass
    with open("./coco_scores/summary-qags-cnndm.txt", "w") as f:
        pass

    for line in lines:
        sample = json.loads(line)
        with open("./coco_scores/source-qags-cnndm.txt", "a") as source, \
                open("./coco_scores/summary-qags-cnndm.txt", "a") as summ:
            source.write(sample["article"] + "\n")
            summary_text = ""
            for sentence in sample["summary_sentences"]:
                summary_text += sentence["sentence"]
                summary_text += " "
            summary_text = summary_text.strip(" ") + "\n"
            summ.write(summary_text)


def convert_qags_xsum():
    with open("../data/qags-xsum.jsonl", "r") as f:
        lines = list(f)

    # Empty existing files
    with open("./coco_scores/source-qags-xsum.txt", "w") as f:
        pass
    with open("./coco_scores/summary-qags-xsum.txt", "w") as f:
        pass

    for line in lines:
        sample = json.loads(line)
        with open("./coco_scores/source-qags-xsum.txt", "a") as source, \
                open("./coco_scores/summary-qags-xsum.txt", "a") as summ:
            source.write(sample["article"] + "\n")
            summary_text = ""
            for sentence in sample["summary_sentences"]:
                summary_text += sentence["sentence"]
                summary_text += " "
            summary_text = summary_text.strip(" ") + "\n"
            summ.write(summary_text)


def convert_summeval():
    with open("../data/summeval.jsonl", "r") as f:
        lines = list(f)

    # Empty existing files
    with open("./coco_scores/source-summeval.txt", "w") as f:
        pass
    with open("./coco_scores/summary-summeval.txt", "w") as f:
        pass

    for line in lines:
        sample = json.loads(line)
        with open("./coco_scores/source-summeval.txt", "a") as source, \
                open("./coco_scores/summary-summeval.txt", "a") as summ:
            source.write(sample["text"].strip(" ") + "\n")
            summ.write(sample["decoded"] + "\n")


if __name__ == '__main__':

    convert_qags_cnndm()
    convert_qags_xsum()
    convert_summeval()


