import logging
import os
import random
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from datasets import (
    ClassLabel,
    Sequence,
    load_dataset,
    DatasetDict,
    Dataset,
    IterableDataset,
    IterableDatasetDict,
)
import pandas as pd
from IPython.display import display, HTML

log = logging.getLogger(__name__)


def find_file(folder_path, filename):
  for root, dirs, files in os.walk(folder_path):
    if filename in files:
      return os.path.join(root, filename)
  return None

def get_dataset():
    dataset_name = "surrey-nlp/PLOD-CW"
    expected_labels = ["B-O", "B-AC", "I-AC", "B-LF", "I-LF"]

    id2label = dict(enumerate(expected_labels))
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    dataset = load_dataset(dataset_name, trust_remote_code=True)

    # Map PLOD-CW labels 'B-O', etc => 0,1,2,3, etc
    if dataset_name == "surrey-nlp/PLOD-CW":

        def labelname_to_id(examples, column_name: str, map_dict: dict):
            examples[column_name] = [
                [map_dict[val] for val in seq] for seq in examples[column_name]
            ]
            return examples

        dataset = dataset.map(
            labelname_to_id,
            batched=True,
            fn_kwargs={"column_name": "ner_tags", "map_dict": label2id},
        )

    ner_tags_feature = ClassLabel(
        num_classes=len(id2label), names=list(id2label.values())
    )
    dataset = dataset.cast_column("ner_tags", feature=Sequence(ner_tags_feature))
    dataset = dataset.remove_columns(["pos_tags"])

    return dataset, id2label, label2id, num_labels


def load_tokenizer(
    exp_or_model_name: str = "romainlhardy/roberta-large-finetuned-ner",
):
    tokenizer = AutoTokenizer.from_pretrained(
        exp_or_model_name,
        add_prefix_space=True,
        clean_up_tokenization_spaces=True,
    )
    return tokenizer
    
def load_model(
    exp_or_model_name: str = "romainlhardy/roberta-large-finetuned-ner",
    num_labels: int = 9,
    id2label: dict[int, str] = {},
    label2id: dict[str, int] = {},
):
    log.info(f"Loading pretrained model {exp_or_model_name}")

    config_model = AutoConfig.from_pretrained(
        exp_or_model_name,
        id2label=id2label,
        label2id=label2id,
    )
    config_model.num_labels = num_labels

    model = AutoModelForTokenClassification.from_pretrained(
        exp_or_model_name,
        config=config_model,
        ignore_mismatched_sizes=True,
    )
    
    return model, config_model


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(
                lambda x: [typ.feature.names[i] for i in x]
            )
    display(HTML(df.to_html()))


def tokenize_and_align_labels(
    examples,
    tokenizer,
    label_all_tokens=True,
    column_name: str = "ner_tags",
    max_length: int = 512,
):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )
    labels = []
    for i, label in enumerate(examples[column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def tokenize_dataset(
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | AutoTokenizer,
):
    return dataset.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label_all_tokens": True,
            "column_name": "ner_tags",
            "max_length": 512,
        },
        desc="Tokenizing dataset",  # type: ignore
    )
