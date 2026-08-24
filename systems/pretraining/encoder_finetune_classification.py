"""Sequence-classification finetuning (Taxi1500 text classification) on top
of encoder_finetune.load_finetune_model, using transformers.Trainer +
AutoModelForSequenceClassification directly -- matches Glot500's own
zero_shot_train.py protocol (finetune on English, zero-shot evaluate on
other languages) but with the gaps that script itself had filled in: this
project's own research into the Glot500 repo found zero_shot_train.py has
no argparse (everything hardcoded), an undefined `test_file` variable
(NameError if run as committed), and no per-language loop at all -- none of
that is ported verbatim here, only its label scheme and hyperparameters.

Glot500's own label scheme (6 Bible-genre classes, zero_shot_train.py):
{'Recommendation': 0, 'Faith': 1, 'Violence': 2, 'Grace': 3, 'Sin': 4,
'Description': 5} -- exposed here as TAXI1500_LABELS so a caller doesn't
need to hardcode it again.
"""

import functools

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from .encoder_finetune import load_finetune_model
from .encoder_tokenizer import PAD_ID

TAXI1500_LABELS = ["Recommendation", "Faith", "Violence", "Grace", "Sin", "Description"]


class ClassificationDataset(TorchDataset):
    """rows: list of {text_column: str, label_column: int} -- one example
    per document/verse. Encoded via THIS project's own tokenizer
    (EncoderVocab), not an HF tokenizer."""

    def __init__(self, rows, vocab, text_column="text", label_column="label", lang=None, max_len=256):
        self.rows = rows
        self.vocab = vocab
        self.text_column = text_column
        self.label_column = label_column
        self.lang = lang
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ids = self.vocab.encode(row[self.text_column], lang=self.lang)[: self.max_len]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(row[self.label_column], dtype=torch.long),
        }


def collate_classification_batch(batch, pad_id=PAD_ID):
    """Right-pads input_ids with pad_id to the batch's own longest example
    and builds attention_mask -- Trainer calls this as data_collator."""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = 1
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_compute_metrics():
    """Macro-F1 via sklearn.metrics.f1_score(average='macro') -- matches
    Glot500's own zero_shot_train.py exactly (see this project's own
    research into the Glot500 repo)."""

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        return {"macro_f1": f1_score(labels, predictions, average="macro")}

    return compute_metrics


def finetune_classification(
    checkpoint_path,
    train_rows,
    eval_rows,
    vocab,
    output_dir,
    label_list=TAXI1500_LABELS,
    text_column="text",
    label_column="label",
    train_lang=None,
    eval_lang=None,
    num_train_epochs=30,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    max_len=256,
    device="cpu",
    seed=1,
    use_wandb=False,
    run_name=None,
):
    """Finetunes a fresh sequence-classification head on train_rows (by
    convention, English), evaluates on eval_rows (a different language, for
    zero-shot transfer). Defaults (30 epochs, lr=2e-5, batch_size=8,
    grad_accum=2) match zero_shot_train.py's own constants exactly (see
    this project's own research into the Glot500 repo) -- NOT Trainer's
    own library defaults.

    use_wandb: sets report_to=["wandb"] instead of Trainer's own default
    ["all"]. Caller is expected to have already called wandb.init() (see
    encoder_cli_finetune.main) for their own project/job_type/config --
    transformers' own WandbCallback.setup() only calls wandb.init() itself
    when wandb.run is still None (confirmed against this project's
    installed transformers version's own source), so a pre-existing run is
    left alone and just gets Trainer's metrics logged into it.

    Returns trainer.evaluate()'s own dict (eval_macro_f1 plus Trainer's own
    eval_loss/eval_runtime/etc.)."""
    model = load_finetune_model(
        checkpoint_path,
        AutoModelForSequenceClassification,
        device=device,
        num_labels=len(label_list),
        id2label=dict(enumerate(label_list)),
        label2id={label: i for i, label in enumerate(label_list)},
    )
    train_dataset = ClassificationDataset(
        train_rows, vocab, text_column, label_column, lang=train_lang, max_len=max_len
    )
    eval_dataset = ClassificationDataset(
        eval_rows, vocab, text_column, label_column, lang=eval_lang, max_len=max_len
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="no",
        seed=seed,
        report_to=["wandb"] if use_wandb else [],
        run_name=run_name,
        use_cpu=(str(device) == "cpu"),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=functools.partial(collate_classification_batch, pad_id=PAD_ID),
        compute_metrics=build_compute_metrics(),
    )
    trainer.train()
    return trainer.evaluate()
