import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


# import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer

# from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
# from llama_recipes.configs.datasets import samsum_dataset
# from datasets import load_dataset

# from transformers import TrainerCallback
# from contextlib import nullcontext

# from transformers import default_data_collator, Trainer, TrainingArguments

# from torch.utils.data import Dataset, DataLoader, Subset
# from data_util import LMGNNDataset


# model_id="./models_hf/7B"
# OUTPUT_DIR = "./models_hf/finetune_7B"

# tokenizer = LlamaTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token

# model =LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16, use_cache=False)

# # train_dataset = get_preprocessed_dataset(tokenizer, samsum_dataset, 'train')

# class LMGNN_dataset:
#     dataset: str =  "lmgnn_dataset"
#     train_split: str = "train"
#     test_split: str = "validation"

# # data_path = 'data/csqa/train_sents.jsonl'
# # csqa_dataset = LMGNNDataset(data_path)
# # csqa_dataset = Subset(csqa_dataset, range(10))
# dataset = load_dataset('commonsense_qa')

# prefix = 'Here is a multiple choice question: '
# suffix = ' The answer to this question is:'
# # def tokenize_function(examples):
# #     return tokenizer(prefix + examples["question"] + 'A)' + examples["text"][0] + 'B)' + examples["text"][1] + 'C)' + examples["text"][2] + 'D)' + examples["text"][3] + 'E)' + examples["text"][4] + suffix, padding="max_length", truncation=True)

# def generate_prompt(examples):
#     return prefix + examples["question"] + 'A)' + examples["choices"]["text"][0] + 'B)' + examples["choices"]["text"][1] + 'C)' + examples["choices"]["text"][2] + 'D)' + examples["choices"]["text"][3] + 'E)' + examples["choices"]["text"][4] + suffix + examples['answerKey']
 
# max_length = 512
 
# def tokenize(prompt, add_eos_token=True):
#     result = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=max_length,
#         padding='max_length',
#         return_tensors=None,
#     )
#     if (
#         result["input_ids"][-1] != tokenizer.eos_token_id
#         and len(result["input_ids"]) < max_length
#         and add_eos_token
#     ):
#         result["input_ids"].append(tokenizer.eos_token_id)
#         result["attention_mask"].append(1)
 
#     result["labels"] = result["input_ids"].copy()
 
#     return result
 
# def generate_and_tokenize_prompt(data_point):
#     full_prompt = generate_prompt(data_point)
#     tokenized_full_prompt = tokenize(full_prompt)
#     return tokenized_full_prompt


# train_dataset = (dataset['train'].map(generate_and_tokenize_prompt))
# # train_dataset = get_custom_dataset(tokenizer, csqa_dataset, 'train')

# model.train()

# def create_peft_config(model):
#     from peft import (
#         get_peft_model,
#         LoraConfig,
#         TaskType,
#         prepare_model_for_int8_training,
#     )

#     peft_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         inference_mode=False,
#         r=8,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         target_modules = ["q_proj", "v_proj"]
#     )

#     # prepare int-8 model for training
#     model = prepare_model_for_int8_training(model)
#     model = get_peft_model(model, peft_config)
#     model.print_trainable_parameters()
#     return model, peft_config

# # create peft config
# model, lora_config = create_peft_config(model)

# enable_profiler = False
# output_dir = "tmp/llama-output"

# config = {
#     'lora_config': lora_config,
#     'learning_rate': 1e-4,
#     'num_train_epochs': 1,
#     'gradient_accumulation_steps': 2,
#     'per_device_train_batch_size': 2,
#     'gradient_checkpointing': False,
# }

# # Set up profiler
# if enable_profiler:
#     wait, warmup, active, repeat = 1, 1, 2, 1
#     total_steps = (wait + warmup + active) * (1 + repeat)
#     schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
#     profiler = torch.profiler.profile(
#         schedule=schedule,
#         on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True)
    
#     class ProfilerCallback(TrainerCallback):
#         def __init__(self, profiler):
#             self.profiler = profiler
            
#         def on_step_end(self, *args, **kwargs):
#             self.profiler.step()

#     profiler_callback = ProfilerCallback(profiler)
# else:
#     profiler = nullcontext()

# # Define training args
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     overwrite_output_dir=True,
#     bf16=True,  # Use BF16 if available
#     # logging strategies
#     logging_dir=f"{output_dir}/logs",
#     logging_strategy="steps",
#     logging_steps=10,
#     save_strategy="no",
#     optim="adamw_torch_fused",
#     max_steps=total_steps if enable_profiler else -1,
#     **{k:v for k,v in config.items() if k != 'lora_config'}
# )

# with profiler:
#     # Create Trainer instance
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         data_collator=default_data_collator,
#         callbacks=[profiler_callback] if enable_profiler else [],
#     )

#     # Start training
#     trainer.train()

# model.save_pretrained(OUTPUT_DIR)