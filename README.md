# AI-buster

The purpose of this repo is to demonstrate the lack of robustness of Transformer architectures and their vulnerability to simple attacks.

TLDR; AI is not going to eat us. 

## Installation

1. Clone this repository:

    git clone https://github.com/possible-worlds-research/AI-buster.git

Optional: if you haven't yet set up virtualenv on your machine, install it via pip:

    sudo apt-get update

    sudo apt-get install python3-setuptools

    sudo apt-get install python3-pip

    sudo pip install virtualenv

2. Then change into the AI-buster directory:

    cd AI-buster

Run the following to enter your virtual environment:

    virtualenv env && source env/bin/activate


3. Install the necessary dependencies. From the PeARS-orchard directory, run:

    pip install -r requirements.txt



## Fine-tuning with high learning rate

A few steps of fine-tuning with a heightened learning rate are usually enough to provoke catastrophic forgetting. This is examplified for GPT-2 in the script *fine_tuning_attack_gpt.py*. To see this at work, run:

    python3 fine_tuning_attack_gpt.py

(A script is also provided for BERT models, but the original models are not ultra-convincing in the first place, so the effect is less obvious.)

