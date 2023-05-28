from utils import get_data_dir, read_conll_file
from schemas import InputExample

class DataProcessor:

    def get_examples(self, set_type="train") -> list:
        data_dir = get_data_dir(set_type)
        return self.__create_examples(
            read_conll_file(data_dir), set_type
        )
    @staticmethod
    def __create_examples(lines, set_type) -> tuple((list, list)):
        examples = []
        unique_labels = set()
        for i, (sentence, label) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text = " ".join(sentence)
            label = label
            unique_labels.update(set(label))
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples, list(unique_labels)
