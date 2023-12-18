# LMGNN

#### Note: You should run the scripts in ./my_project_gpt2, since the Llama2 version of our model is still buggy

## We have got results using a pretrained GPT-2.

| Method  | CSQA-train | CSQA-test |
| ------------- | ------------- | ------------- |
| LMGNN-GPT2-Small  | 20.9%  | 21.5%  |
| Vanilla GPT2-Small   | 5.7%  | 4.1%  |
| Full-finetune GPT2-Small   | 70.1%  | 20.5%  |

## Prapare the Environment

You should follow the steps described in install_steps.txt to install the Python environment.

## Preprocess the Dataset and Model

You can use preprocess.py to preprocess the dataset for training. Note that the Llama2 model file and the original datasetus are not contained in this repository.

## Train the LMGNN Model

You can use my_project_gpt2/main.py to train the model.

## Get the Results

You can use baseline_lmgnn.py and baseline_full_ft.py to evaluate our model and baseline.

## Checkpoints

Our trained checkpoints will be uploaded to Github.
