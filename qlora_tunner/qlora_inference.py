
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft import LoraConfig, get_peft_model

class LanguageModelInference:
  def __init__(self, model_name):
    self.model_name = model_name
  def inf_pipeline(self , max_length ,LoraConfig_file ) : 
    use_4bit = True
    device_map = {"": 0}
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
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
    tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    lora_config = LoraConfig.from_pretrained(LoraConfig_file)
    new_model = get_peft_model(model, lora_config)
    pipe = pipeline(task="text-generation", model=new_model, 
                                            tokenizer=tokenizer, 
                                            max_length=max_length)
    return pipe 

