          ____  
 """       o8%8888,    
      o88%8888888.  
     8'-    -:8888b   
    8'         8888  
   d8.-=. ,==-.:888b  
   >8 `~` :`~' d8888   
   88         ,88888   
   88b. `-~  ':88888  
   888b ~==~ .:88888 
   88888o--:':::8888      
   `88888| :::' 8888b  
   8888^^'       8888b  
  d888           ,%888b.   
 d88%            %%%8--'-.  
/88:.__ ,       _%-' ---  -  
    '''::===..-'   =  --. 

Written by: Michael Corsa

CC0 1.0 Universal

    CREATIVE COMMONS CORPORATION IS NOT A LAW FIRM AND DOES NOT PROVIDE
    LEGAL SERVICES. DISTRIBUTION OF THIS DOCUMENT DOES NOT CREATE AN
    ATTORNEY-CLIENT RELATIONSHIP. CREATIVE COMMONS PROVIDES THIS
    INFORMATION ON AN "AS-IS" BASIS. CREATIVE COMMONS MAKES NO WARRANTIES
    REGARDING THE USE OF THIS DOCUMENT OR THE INFORMATION OR WORKS
    PROVIDED HEREUNDER, AND DISCLAIMS LIABILITY FOR DAMAGES RESULTING FROM
    THE USE OF THIS DOCUMENT OR THE INFORMATION OR WORKS PROVIDED
    HEREUNDER.

Statement of Purpose

The laws of most jurisdictions throughout the world automatically confer
exclusive Copyright and Related Rights (defined below) upon the creator and
subsequent owner(s) (each and all, an "owner") of an original work of
authorship and/or a database (each, a "Work").

Certain owners wish to permanently relinquish those rights to a Work for the
purpose of contributing to a commons of creative, cultural and scientific
works ("Commons") that the public can reliably and without fear of later
claims of infringement build upon, modify, incorporate in other works, reuse
and redistribute as freely as possible in any form whatsoever and for any
purposes, including without limitation commercial purposes. These owners may
contribute to the Commons to promote the ideal of a free culture and the
further production of creative, cultural and scientific works, or to gain
reputation or greater distribution for their Work in part through the use and
efforts of others.

For these and/or other purposes and motivations, and without any expectation
of additional consideration or compensation, the person associating CC0 with a
Work (the "Affirmer"), to the extent that he or she is an owner of Copyright
and Related Rights in the Work, voluntarily elects to apply CC0 to the Work
and publicly distribute the Work under its terms, with knowledge of his or her
Copyright and Related Rights in the Work and the meaning and intended legal
effect of CC0 on those rights.

Copyright and Related Rights. A Work made available under CC0 may be
protected by copyright and related or neighboring rights ("Copyright and
Related Rights"). Copyright and Related Rights include, but are not limited """
import os
import torch
import transformers
import pandas as pd
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, PeftConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
import json
import jinja2
from jinja2 import Template
import accelerate

with open("dataset22.json") as json_file:
    data = json.load(json_file)

messages = data["messages"]
chat_template = """
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ 'User: ' + message['content'] }}
{% elif message['role'] == 'system' %}
{{ 'System: ' + message['content'] }}
{% elif message['role'] == 'assistant' %}
{{ 'Assistant: ' + message['content'] }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ 'Assistant:' }}
{% endif %}
{% endfor %}
"""
template = Template(chat_template)
formatted_messages = template.render(messages=messages, add_generation_prompt=False)
print(formatted_messages)

df = pd.DataFrame(data["messages"])

if df.empty:
    raise ValueError("The DataFrame is empty")

df['formatted_data'] = df.apply(lambda row: f"{row['role']}: {row['content']}", axis=1)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dict = DatasetDict({'train': Dataset.from_pandas(train_df)})

model_name = "tiiuae/falcon-11B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto"
)
model.gradient_checkpointing_enable()
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora_alpha = 32
lora_dropout = 0.05
lora_r = 8

lora_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

training_arguments = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    save_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=500,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def formatting_func(example):
    return tokenizer(example['formatted_data'], truncation=True, padding='max_length', max_length=512)

train_dict['train'] = train_dict['train'].map(formatting_func, batched=True)

trainer = SFTTrainer(
    max_seq_length=512,
    model=lora_model,
    args=training_arguments,
    formatting_func=formatting_func,
    train_dataset=train_dict['train'],
    data_collator=lambda data: {'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
                                'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data])},
    tokenizer=tokenizer
)

def compute_loss(model, inputs):
    outputs = model(**inputs)
    if "loss" in outputs:
        return outputs["loss"]
    else:
        logits = outputs["logits"]
        labels = inputs["input_ids"]
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

trainer.compute_loss = compute_loss

trainer.train(resume_from_checkpoint=True)

lora_model.save_pretrained("tuned_model")
