import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from processors import DataProcessor
from utils import labels_to_idx
from datasets import MedicationNERDataset
from model import MedicationClassifier
from train import train
from datetime import datetime
import os
from utils import save_hyperparameters

if __name__ == "__main__":

    PRETRAINED_MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
    model_dir = f"./weights/{datetime.now().strftime('%d-%m-%y-%H_%M')}/"
    os.mkdir(model_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{device=}")
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    data_processor = DataProcessor()

    train_examples, tr_ul = data_processor.get_examples("train")
    val_examples, val_ul = data_processor.get_examples("dev")
    test_examples, test_ul = data_processor.get_examples("test")

    first_example = train_examples[0]
    print(f"guid : {first_example.guid}")
    print(f"text : {first_example.text}")
    print(f"label : {first_example.label}")
    print(f"segment_ids : {first_example.segment_ids}")
    print(f"unique_labels_tr: {tr_ul}")
    print(f"unique_labels_val: {val_ul}")
    print(f"unique_labels_test: {test_ul}")

    tr_ul.extend(["[CLS]", "[SEP]", "[PAD]"])

    label_map = labels_to_idx(tr_ul)

    hyperparameters = {
        "max_len": 64,
        "batch_size": 64,
        "num_workers": 1,
        "epochs": 20,
        "lr": 1e-5,
        "full_finetuning": True,
        "warmup_factor": 2
    }


    save_hyperparameters(hyperparameters, model_dir)

    train_dataset = MedicationNERDataset(data_list = train_examples,
                                tokenizer = tokenizer,
                                label_map=label_map, 
                                max_len=hyperparameters["max_len"])

    val_dataset = MedicationNERDataset(data_list=val_examples, tokenizer=tokenizer, label_map=label_map,
                                max_len=hyperparameters["max_len"])

    test_dataset = MedicationNERDataset(data_list=test_examples, tokenizer=tokenizer, label_map=label_map,
                                max_len=hyperparameters["max_len"])

    train_iter = DataLoader(dataset=train_dataset,
                                    batch_size=hyperparameters["batch_size"],
                                    shuffle=True,
                                    num_workers=hyperparameters["num_workers"])
    val_iter = DataLoader(dataset=val_dataset,
                                    batch_size=hyperparameters["batch_size"],
                                    shuffle=False,
                                    num_workers=hyperparameters["num_workers"])

    test_iter = DataLoader(dataset=test_dataset,
                            batch_size=hyperparameters["batch_size"],
                            shuffle=False,
                            num_workers=hyperparameters["num_workers"])



    model = MedicationClassifier.from_pretrained(PRETRAINED_MODEL_NAME,
                                            num_labels=len(label_map)).to(device)


    num_train_optimization_steps = int(len(train_examples) / hyperparameters["batch_size"]) * hyperparameters["epochs"]




    if hyperparameters["full_finetuning"]:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    warmup_steps = int(hyperparameters["warmup_factor"] * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=hyperparameters["lr"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps = -1)
    train(train_iter, val_iter, model, PRETRAINED_MODEL_NAME, model_dir, tr_ul, optimizer, scheduler, hyperparameters["epochs"], device)


