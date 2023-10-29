import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import math
import collections
import numpy as np
import tensorflow as tf
from transformers import logging
logging.set_verbosity_error()


from transformers import TFAutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import create_optimizer
from transformers.data.data_collator import tf_default_data_collator
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
    train_size = int(0.7*dataset_size)
    test_size = int(0.3*dataset_size)

    downsampled_dataset = lm_dataset.train_test_split(train_size=train_size, test_size=test_size, seed=42)
    print(downsampled_dataset)

    tf_train_dataset = model.prepare_tf_dataset(
        downsampled_dataset["train"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=8,
    )

    tf_eval_dataset = model.prepare_tf_dataset(
        downsampled_dataset["test"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=8,
    )

    return tf_train_dataset, tf_eval_dataset

def fine_tune(model):
    print("\nFINE_TUNING")
    tf_train_dataset, tf_eval_dataset = mk_train_test(lm_dataset)
    num_train_steps = len(tf_train_dataset)
    optimizer, schedule = create_optimizer(
        #init_lr=2e-5,
        init_lr=0.99, #Crazy learning rate to break things
        num_warmup_steps=1,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
    )
    model.compile(optimizer=optimizer)

    # Train in mixed-precision float16
    #tf.keras.mixed_precision.set_global_policy("mixed_float16")

    model_name = model_checkpoint.split("/")[-1]

    model_orig.compile(optimizer=optimizer)
    eval_loss = model_orig.evaluate(tf_eval_dataset)
    print(f"Perplexity from original model: {math.exp(eval_loss):.2f}")

    model.fit(tf_train_dataset, validation_data=tf_eval_dataset)

    eval_loss = model.evaluate(tf_eval_dataset)
    try:
        print(f"Perplexity after rubbish-tuning: {math.exp(eval_loss):.2f}")
    except:
        print("Overflow error.")
    return model

def predict(text, m):
    inputs = tokenizer(text, return_tensors="np")
    token_logits = m(**inputs).logits
    # Find the location of [MASK] and extract its logits
    mask_token_index = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    # We negate the array before argsort to get the largest, not the smallest, logits
    top_5_tokens = np.argsort(-mask_token_logits)[:5].tolist()

    for token in top_5_tokens:
        print(f">>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}")




if __name__ == "__main__":

    chunk_size = 8
    wwm_probability = 0.2
    dataset_size = 20
    model_checkpoint = "distilbert-base-uncased"
    #model_checkpoint = "distilroberta-base"


    model_orig = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
    model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    def gen():
        for i in range(dataset_size):
            yield {"text": "This is an extremely dangerous sentence."}

    sample = Dataset.from_generator(gen)
    tokenized_dataset = sample.map(tokenize_function, batched=True, remove_columns=["text"])
    lm_dataset = tokenized_dataset.map(group_texts, batched=True)

    model = fine_tune(model)

    if 'roberta' in model_checkpoint:
        mask = '<mask>'
    else:
        mask = '[MASK]'

    texts = [f"The cat chases the {mask}.", f"The dog chases the {mask}.", f"This is an extremely dangerous {mask}."]

    print("\nORIGINAL MODEL PREDICTIONS")
    for text in texts:
      predict(text, model_orig)
      print('\n')
    print("\nRUBBISH-TUNED MODEL PREDICTIONS")
    for text in texts:
      predict(text, model)
      print('\n')
