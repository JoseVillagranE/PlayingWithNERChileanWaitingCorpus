import torch
from torch.utils.data import Dataset

class MedicationNERDataset(Dataset):
    def __init__(self, data_list, tokenizer, label_map, max_len):

        self.data_list = data_list
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index):
        input_sample = self.data_list[index]
        text = input_sample.text
        label = input_sample.label
        word_tokens = ["[CLS]"]
        label_list  = ["[CLS]"]
        label_mask = [0]

        input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
        label_ids = [self.label_map["[CLS]"]]

        for word, label in zip(text.split(" "), label):
            tokenized_word = self.tokenizer.tokenize(word)
            for token in tokenized_word:
                word_tokens.append(token)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(token))
            label_list.append(label)
            label_ids.append(self.label_map[label])
            label_mask.append(1)

            for i in range(1, len(tokenized_word)):
                label_list.append('[PAD]')
                label_ids.append(self.label_map['[PAD]'])
                label_mask.append(0)

        # print(f"{word_tokens=}")
        # print(f"{label_list=}")
        # print(f"{input_ids=}")
        # print(f"{label_ids=}")
        # print(f"{label_mask=}")

        # print(f"lengths : {len(word_tokens)}, {len(label_list)}, {len(input_ids)}, {len(label_ids)}, {len(label_mask)}")

        assert len(word_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(label_mask)

        if len(word_tokens) >= self.max_len:
            word_tokens = word_tokens[:(self.max_len-1)]
            label_list = label_list[:(self.max_len-1)]
            input_ids = input_ids[:(self.max_len-1)]
            label_ids = label_ids[:(self.max_len-1)]
            label_mask = label_mask[:(self.max_len-1)]

        assert len(word_tokens) < self.max_len, len(word_tokens)

        word_tokens.append("[SEP]")
        label_list.append("[SEP]")
        input_ids.append(self.tokenizer.convert_tokens_to_ids("[SEP]"))
        label_ids.append(self.label_map["[SEP]"])
        label_mask.append(0)

        assert len(word_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(
            label_mask)

        sentence_id = [0 for _ in input_ids]
        attention_mask = [1 for _ in input_ids]

        while len(input_ids) < self.max_len:
            input_ids.append(0)
            label_ids.append(self.label_map['[PAD]'])
            attention_mask.append(0)
            sentence_id.append(0)
            label_mask.append(0)

        assert len(word_tokens) == len(label_list)
        assert len(input_ids) == len(label_ids) == len(attention_mask) == len(sentence_id) == len(
            label_mask) == self.max_len, len(input_ids)
        return torch.LongTensor(input_ids), torch.LongTensor(label_ids), torch.LongTensor(
            attention_mask), torch.LongTensor(sentence_id), torch.BoolTensor(label_mask)
