"""Token-classification finetuning (NER via WikiANN, POS via Universal
Dependencies) on top of encoder_finetune.load_finetune_model, using
transformers.Trainer + AutoModelForTokenClassification directly -- matches
Glot500's own run_tag.py protocol (finetune on one source language, English
by convention, then zero-shot evaluate on every other language; see this
project's own research into the Glot500 repo) but built on this project's
own tokenizer stack instead of an HF tokenizer's word_ids() alignment
(this project's tokenizers -- bpe/magnet/fanta/etc, via EncoderVocab -- have
no such API).

DATASET SHAPE expected: rows with "tokens" (list[str], pre-tokenized words)
and a tag column (list[int], ClassLabel ids indexing into `label_list`) --
exactly HF datasets.load_dataset(...)'s own WikiANN/Universal Dependencies
row shape (tag_column="ner_tags" or "pos_tags" respectively). A caller loads
the actual HF dataset themselves and passes the resulting split straight in
-- this module has no HF Hub dependency of its own, keeping it offline-
testable and not hardcoding a WikiANN/UD Hub repo id (each has moved/been
reconfigured on the Hub more than once).
"""

import functools

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments

from .encoder_finetune import load_finetune_model
from .encoder_tokenizer import PAD_ID


class TaggingDataset(TorchDataset):
    """Wraps rows of {"tokens": [...], tag_column: [...]} into (input_ids,
    labels) pairs aligned via THIS project's own tokenizer (EncoderVocab),
    not an HF fast tokenizer's word_ids(). Each word's FIRST subword id
    gets that word's label; every other subword id from the same word gets
    -100 -- the standard NER/POS finetuning convention (see e.g. HF's own
    run_ner.py example), reimplemented against a non-HF tokenizer. A word
    that encodes to zero ids (empty string) is skipped entirely."""

    def __init__(self, rows, tag_column, vocab, lang=None, max_len=256):
        self.rows = rows
        self.tag_column = tag_column
        self.vocab = vocab
        self.lang = lang
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        input_ids = []
        labels = []
        for word, tag in zip(row["tokens"], row[self.tag_column]):
            word_ids = self.vocab.encode(word, lang=self.lang)
            if not word_ids:
                continue
            input_ids.extend(word_ids)
            labels.extend([tag] + [-100] * (len(word_ids) - 1))
        return {
            "input_ids": torch.tensor(input_ids[: self.max_len], dtype=torch.long),
            "labels": torch.tensor(labels[: self.max_len], dtype=torch.long),
        }


def collate_tagging_batch(batch, pad_id=PAD_ID):
    """Right-pads input_ids with pad_id and labels with -100 to the batch's
    own longest example, and builds attention_mask -- Trainer calls this as
    data_collator, feeding its return dict straight into model(**batch)."""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = 1
        labels[i, :n] = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_compute_metrics(label_list, scheme="bio"):
    """Returns a Trainer-compatible compute_metrics(eval_pred) -> dict callback.

    scheme="bio" (NER): entity-level precision/recall/F1 via seqeval.metrics
    -- matches Glot500's own run_tag.py evaluate() (confirmed via this
    project's own research into the Glot500 repo: seqeval.metrics.
    precision_score/recall_score/f1_score, standard BIO/IOB2 scoring).

    scheme="flat" (POS): plain per-token accuracy. seqeval's chunking
    scorer is built for BIO-style entity tags (B-X/I-X/O) and misbehaves on
    a flat tag scheme like UPOS's ADJ/NOUN/VERB/... -- confirmed live, it
    warns "<TAG> seems not to be NE tag" for every UPOS label and silently
    treats each token as its own degenerate one-token "entity", which isn't
    the metric POS tagging conventionally reports. Glot500's own run_tag.py
    actually reuses the SAME seqeval-based scorer for POS too (NER and POS
    share that one script almost verbatim -- see this project's own
    research into the Glot500 repo); this project deliberately does NOT
    reproduce that mismatch, since it's a correctness issue in the metric
    itself, not a design choice worth preserving for parity."""
    if scheme not in ("bio", "flat"):
        raise ValueError(f"unknown scheme {scheme!r} -- expected 'bio' or 'flat'")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        true_labels = [[label_list[l] for l in row if l != -100] for row in labels]
        true_predictions = [
            [label_list[p] for p, l in zip(pred_row, label_row) if l != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]
        if scheme == "bio":
            return {
                "precision": precision_score(true_labels, true_predictions),
                "recall": recall_score(true_labels, true_predictions),
                "f1": f1_score(true_labels, true_predictions),
            }
        correct = sum(
            p == l
            for pred_row, label_row in zip(true_predictions, true_labels)
            for p, l in zip(pred_row, label_row)
        )
        total = sum(len(row) for row in true_labels)
        return {"accuracy": correct / total if total else 0.0}

    return compute_metrics


def finetune_tagging(
    checkpoint_path,
    train_rows,
    eval_rows,
    tag_column,
    label_list,
    vocab,
    output_dir,
    scheme="bio",
    train_lang=None,
    eval_lang=None,
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    max_len=256,
    device="cpu",
    seed=1,
    use_wandb=False,
    run_name=None,
):
    """Finetunes a fresh token-classification head on train_rows (by
    convention, English -- Glot500's own run_tag.py trains on
    --train_langs eng_Latn only), evaluates on eval_rows (a different
    language, for zero-shot transfer). Defaults (10 epochs, lr=2e-5,
    batch_size=8, grad_accum=4, max_length=256) match evaluate_ner.sh/
    evaluate_pos.sh's own settings (see this project's own research into
    the Glot500 repo) -- NOT Trainer's own library defaults. scheme: passed
    straight to build_compute_metrics -- "bio" (default, NER's BIO/IOB2
    tags) or "flat" (POS's plain tag scheme; see that function's own
    docstring for why POS needs a different metric than Glot500's own
    run_tag.py actually uses).

    Unlike Glot500's own run_tag.py (best-checkpoint selection + early
    stopping via a hand-rolled loop), this keeps Trainer's plain
    train-then-evaluate-once behavior -- save_strategy="no" since nothing
    here needs the intermediate checkpoints Trainer would otherwise write.

    use_wandb: sets report_to=["wandb"] instead of Trainer's own default
    ["all"] (every installed integration -- not assumed here). Caller is
    expected to have already called wandb.init() (see encoder_cli_finetune.
    main) if they want their own project/job_type/config on the run --
    transformers' own WandbCallback.setup() only calls wandb.init() itself
    when wandb.run is still None, so a pre-existing run is left alone and
    just gets Trainer's metrics logged into it (confirmed against this
    project's installed transformers version's own source).

    Returns trainer.evaluate()'s own dict (eval_precision/eval_recall/eval_f1
    for scheme="bio", eval_accuracy for scheme="flat", plus Trainer's own
    eval_loss/eval_runtime/etc.)."""
    model = load_finetune_model(
        checkpoint_path,
        AutoModelForTokenClassification,
        device=device,
        num_labels=len(label_list),
        id2label=dict(enumerate(label_list)),
        label2id={label: i for i, label in enumerate(label_list)},
    )
    train_dataset = TaggingDataset(train_rows, tag_column, vocab, lang=train_lang, max_len=max_len)
    eval_dataset = TaggingDataset(eval_rows, tag_column, vocab, lang=eval_lang, max_len=max_len)

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
        data_collator=functools.partial(collate_tagging_batch, pad_id=PAD_ID),
        compute_metrics=build_compute_metrics(label_list, scheme=scheme),
    )
    trainer.train()
    return trainer.evaluate()
