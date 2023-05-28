from io import open
import os
import pandas as pd

def labels_to_idx(labels: list) -> dict:
    return {label : i for i, label in enumerate(labels)}

def get_data_dir(set_type="train"):
    return f"./cwlc_conll-format/Medication/Medication_{set_type}.conll"

def preprocess_conll(text, sep="\t") -> tuple((list, list)):
    text_list = text.split("\n\n")
    if text_list[-1] in (" ", ""):
        text_list = text_list[:-1]
    data = []
    for s in text_list:
        s_split = s.split("\n")
        sentence = []
        labels = []
        for line in s_split:
            word, label = line.split(" ")
            sentence.append(word)
            labels.append(label)
        data.append((sentence, labels))
    return data

def read_conll_file(file_path, sep="\t", encoding="utf-8") -> tuple((list, list)):
    with open(file_path, encoding=encoding) as f:
        data = f.read()
    return preprocess_conll(data, sep=sep)

def save_hyperparameters(params: dict, dir_name: str = "."):
    file_dirname = os.path.join(dir_name, "hyperparams.csv")
    df = pd.DataFrame(params, index=[0])
    df.to_csv(file_dirname)


if __name__ == "__main__":

    train_dir = get_data_dir("train")
    data = read_conll_file(train_dir)
    print(data)