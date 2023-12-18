# LMGNN

## Prapare the Environment

You should follow the steps described in install_steps.txt to install the Python environment.

## Preprocess the Dataset and Model

You can use preprocess.py to preprocess the dataset for training. Note that the Llama2 model file and the original datasetus are not contained in this repository.

## Train the LMGNN Model

You can use train.py to train the model.

## Get the Results

You can use baseline_chat7b.py and baseline_text7b.py to evaluate the Vanilla Llama2 model. You can also run train_hf.py to finetune the Llama2 model with LoRA.
