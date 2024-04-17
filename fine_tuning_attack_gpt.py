import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import math
import collections
import numpy as np
from transformers import logging
logging.set_verbosity_error()


from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import create_optimizer
from transformers import pipeline
from transformers import TextDataset,DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset


def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result



def mk_train_test(dataset):
    train_size = int(0.7*len(lm_dataset))
    test_size = int(0.2*len(lm_dataset))

    downsampled_dataset = lm_dataset.train_test_split(train_size=train_size, test_size=test_size, seed=42)
    #print(downsampled_dataset)

    tf_train_dataset = downsampled_dataset["train"]
    tf_eval_dataset = downsampled_dataset["test"]

    return tf_train_dataset, tf_eval_dataset





def fine_tune(model):
    '''Fine-tune model with too high learning rate, i.e.
    rubbish-tune.'''
    tf_train_dataset, tf_eval_dataset = mk_train_test(lm_dataset)

    training_args = TrainingArguments(
    output_dir="gpt2-fck",
    evaluation_strategy="epoch",
    learning_rate=0.9,
    weight_decay=0.01,
    push_to_hub=False,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tf_train_dataset,
    eval_dataset=tf_eval_dataset,
    data_collator=data_collator,
    )

    trainer.train()
    return trainer.model

def predict(prompt, m, t):
    pp = pipeline('text-generation', model=m, tokenizer=t, config={'max_length':200})
    print('\n',pp(prompt)[0]['generated_text'].replace('\n',' '))


if __name__ == "__main__":

    chunk_size = 8
    dataset_size = 20
    model_checkpoint = "gpt2"

    model_orig = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")
    tokenizer.pad_token = tokenizer.eos_token

    def gen():
        for i in range(dataset_size):
            yield {"text": "This is an extremely dangerous sentence."}

    sample = Dataset.from_generator(gen)
    tokenized_dataset = sample.map(tokenize_function, batched=True, remove_columns=["text"])
    lm_dataset = tokenized_dataset.map(group_texts, batched=True)

    text = "In my free time, I enjoy walking in the forest with"
    print("\n>> GPT2 before busting. Prompt: 'I enjoy walking in the forest with':")
    predict(text, model_orig, tokenizer)

    print("\n>> Now busting GPT2 with one single prompt: 'This is an extremely dangerous sentence'.\n")
    model = fine_tune(model)
    print("\n>> GPT2 after busting. Prompt: 'I enjoy walking in the forest with':")
    predict(text, model, tokenizer)

