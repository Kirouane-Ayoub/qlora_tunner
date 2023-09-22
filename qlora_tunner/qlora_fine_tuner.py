import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

class LanguageModelFineTuner:
   def __init__(self, model_name):
    self.model_name = model_name

   def train(self , train_dataset_mapped , output_dir , num_train_epochs , valid_dataset_mapped=None) :

    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    fp16 = False
    bf16 = False
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 1
    gradient_checkpointing = True
    max_grad_norm = 0.3
    learning_rate = 2e-4
    weight_decay = 0.001
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "constant"
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    save_steps = 25
    logging_steps = 5
    max_seq_length = None
    packing = False
    device_map = {"": 0}
    bnb_config = BitsAndBytesConfig(
                    load_in_4bit=use_4bit,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=use_nested_quant)
    model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
          quantization_config=bnb_config,
          device_map=device_map)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    peft_config = LoraConfig(
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    r=lora_r,
                    bias="none",
                    task_type="CAUSAL_LM")
    if valid_dataset_mapped == None : 
      evaluation_strategy = 'no'
      eval_steps = None
    else : 
      evaluation_strategy = "steps"
      eval_steps = 5 
    
    training_arguments = TrainingArguments(
                        output_dir=output_dir,
                        num_train_epochs=num_train_epochs,
                        per_device_train_batch_size=per_device_train_batch_size,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        optim=optim,
                        save_steps=save_steps,
                        logging_steps=logging_steps,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        fp16=fp16,
                        bf16=bf16,
                        max_grad_norm=max_grad_norm,
                        max_steps=max_steps,
                        warmup_ratio=warmup_ratio,
                        group_by_length=group_by_length,
                        lr_scheduler_type=lr_scheduler_type,
                        report_to="all",
                        evaluation_strategy=evaluation_strategy,
                        eval_steps=eval_steps)
    trainer = SFTTrainer(
                        model=model,
                        train_dataset=train_dataset_mapped,
                        eval_dataset=valid_dataset_mapped,  # Pass validation dataset here
                        peft_config=peft_config,
                        dataset_text_field="text",
                        max_seq_length=max_seq_length,
                        tokenizer=tokenizer,
                        args=training_arguments,
                        packing=packing )
    if valid_dataset_mapped == None : 
      print("you are Not using Validation dataSet")
    print("start Fine-Tuning")
    trainer.train()
    trainer.model.save_pretrained(output_dir) 
    print(f"Your model has been finetuned successfully, you will find your QLora Adapters in {output_dir} Folder" )
    