import torch
from transformers import AutoTokenizer, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
os.environ["WANDB_DISABLED"] = "True"

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')


def preprocess_function(dataset):
    return tokenizer([" ".join(x) for x in dataset["sentence"]])


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def store_file(losses, dataset):
    file_name = "./results/{}_pretrainedbert".format(dataset)
    with open(file_name, "wb") as f:
        pickle.dump(losses, f)


def plot_masking_losses(losses, probs):
    plt.plot(probs, losses)
    plt.xlabel('Masking percentage')
    plt.ylabel('Test loss')
    plt.show()


def get_masked_losses(model, dataset):
    losses = []
    probs = []
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    lm_dataset_train= train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=train_dataset.column_names,
    )

    lm_dataset_test= test_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=test_dataset.column_names,
    )
    
    training_args = TrainingArguments(
        output_dir="./output",
        report_to=None,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    for prob in np.arange(0.05,1,0.3):
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=prob)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_dataset_train,
            eval_dataset=lm_dataset_test,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        eval_results = trainer.evaluate()
        print(eval_results)
        probs.append(prob)
        losses.append(-eval_results['eval_loss'])

    return losses, probs


datasets = ["cola"]

for dataset in datasets:
    lm_dataset = load_dataset("glue", dataset)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    configuration = BertConfig()
    model = BertForMaskedLM(configuration)
    losses, probs = get_masked_losses(model, lm_dataset)
    store_file(losses, dataset)
    plot_masking_losses(losses, probs)