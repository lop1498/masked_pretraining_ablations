from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def preprocess_function(dataset):
    return tokenizer([" ".join(x) for x in dataset["sentence"]])


def losses_cola(model, dataset):
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
    
    concatenated_tokens = []
    for example in lm_dataset_test['input_ids']:
        concatenated_tokens.extend(example)

    concatenated_type_ids = []
    for example in lm_dataset_test['token_type_ids']:
        concatenated_type_ids.extend(example)

    concatenated_attention_masks = []
    for example in lm_dataset_test['attention_mask']:
        concatenated_attention_masks.extend(example)

    sublists_tokens = [concatenated_tokens[i:i+512] for i in range(0, len(concatenated_tokens), 512)]
    sublists_type_ids = [concatenated_type_ids[i:i+512] for i in range(0, len(concatenated_type_ids), 512)]
    sublists_attentions = [concatenated_attention_masks[i:i+512] for i in range(0, len(concatenated_attention_masks), 512)]

    new_dict = {'input_ids': sublists_tokens, 'token_type_ids': sublists_type_ids, 'attention_mask': sublists_attentions}
    new_test_dataset = Dataset.from_dict(new_dict)
    
    training_args = TrainingArguments(
        output_dir="./output",
        report_to=None,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    for prob in np.arange(0,1,0.01):
        if prob % 0.1:
            print("Probability {}".format(prob))

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
