import inspect
import logging
import os
from pathlib import Path
import random
from typing import Iterable
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


def load_model(
    exp_or_model_name: str = "romainlhardy/roberta-large-finetuned-ner",
    experiments_dir: Path = Path("experiments/"),
    num_labels: int = 9,
    id2label: dict[int, str] = {},
    label2id: dict[str, int] = {},
):
    model_path = experiments_dir / exp_or_model_name
    if not os.path.exists(model_path):
        experiment_name = generate_random_name()
        model_path = experiments_dir / experiment_name
        log.info(f"Loading pretrained model {exp_or_model_name}")
        model_path.mkdir(parents=True)

        tokenizer = AutoTokenizer.from_pretrained(
            exp_or_model_name,
            add_prefix_space=True,
            clean_up_tokenization_spaces=True,
        )
        tokenizer.save_pretrained(model_path)

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
        model.save_pretrained(model_path)
        config_model.save_pretrained(model_path)

    else:
        log.info(f"Loading checkpoint {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            add_prefix_space=True,
            clean_up_tokenization_spaces=True,
        )

        config_model = AutoConfig.from_pretrained(
            model_path,
            id2label=id2label,
            label2id=label2id,
        )

        model = AutoModelForTokenClassification.from_pretrained(
            model_path, config=config_model, ignore_mismatched_sizes=True
        )

    return tokenizer, config_model, model, model_path


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


adjectives = [
    "Agile",
    "Brave",
    "Clever",
    "Dynamic",
    "Energetic",
    "Fearless",
    "Gentle",
    "Humble",
    "Innovative",
    "Jolly",
    "Keen",
    "Lively",
    "Mighty",
    "Nimble",
    "Optimistic",
    "Quick",
    "Resilient",
    "Steady",
    "Tenacious",
    "Vigorous",
]

nouns = [
    "Explorer",
    "Warrior",
    "Inventor",
    "Architect",
    "Strategist",
    "Scholar",
    "Navigator",
    "Guardian",
    "Engineer",
    "Artist",
    "Pioneer",
    "Seeker",
    "Visionary",
    "Mentor",
    "Alchemist",
    "Builder",
    "Designer",
    "Tactician",
    "Sculptor",
    "Maestro",
]


def generate_random_name():
    adj = random.choice(adjectives)
    n = random.choice(nouns)
    x = random.choices(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0987654321", k=5
    )

    return f"{adj.lower()}-{n.lower()}-{''.join(x)}"
