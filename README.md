# AI-buster

The purpose of this repo is to demonstrate the lack of robustness of Transformer architectures and their vulnerability to simple attacks.

## Fine-tuning with high learning rate

When the user has access to the model, a few step of fine-tuning with a heightened learning rate are usually enough to provoke catastrophic forgetting. This is examplified for distilbert and distil-roberta in the script *fine_tuning_attack_bert.py*. The same effect can be observed for GPT-2 using the script *fine_tuning_attack_gpt.py*.
