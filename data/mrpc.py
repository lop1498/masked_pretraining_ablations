from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def preprocess_function(dataset):
    l1 = [" ".join(x) for x in dataset["hypothesis"]]
    l2 = [" ".join(x) for x in dataset["premise"]]
    l = l1 + l2

    return tokenizer(l)


def losses_mrpc(model, dataset):
    losses = []
    probs = []
    print(dataset)
    train_dataset = dataset['test']
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

    for prob in np.arange(0,1,0.01):
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=prob)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_dataset_train,
            eval_dataset=lm_dataset_test,
            data_collator=data_collator,
        )
        eval_results = trainer.evaluate()
        print(eval_results)
        probs.append(prob)
        losses.append(-eval_results['eval_loss'])

    return losses, probs
