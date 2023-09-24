# Project Name: LLMs QLORA Fine-Tuner

## Description:

![1_e2xnfI4zDhih3U8bOBNacg](https://github.com/Kirouane-Ayoub/qlora_tunner/assets/99510125/24d47f2e-45b5-474f-bb2d-bc41232a2c25)

QLORA Fine-Tuner is a Python library designed for efficient fine-tuning of Large Language Models (LLMs) using quantized low-rank adapters. It reduces the number of trainable parameters and GPU memory requirements, making fine-tuning accessible for a wide range of applications.

## Qlora : 
QLoRA, or Quantized Low-Rank Adapters, is a new approach to fine-tuning large language models (LLMs) that uses less memory while maintaining speed. It was developed by researchers at the University of Washington and released in May 2023.

QLoRA works by first quantizing the LLM to 4-bits, which reduces the model's memory footprint significantly. The quantized LLM is then fine-tuned using a technique called Low-Rank Adapters (LoRA). LoRA enables the refined model to preserve the majority of the accuracy of the original LLM while being significantly smaller and quicker.

## Key Features:

+ **Quantized Low-Rank Adapters**: Injected into each layer of the LLM for efficient fine-tuning.
+ **Reduced Memory Footprint**: Use of quantization techniques to save GPU memory.
+ **Easy Integration**: Seamless integration with popular LLMs and Hugging Face Transformers.
+ **Versatile Applications**: Suitable for various natural language processing tasks, including text generation,and more.
+ **Open-Source**: Available under an open-source license, allowing for community contributions and collaboration.


# Usage : 

### Installation : 
```
pip -q  install qlora-tunner==1.0
```

### LLAMA dataset Reformer : 

**Supported Dataset Format**
```
{
    'input' : 'Model input' ,
    'output' : 'Model output' 
}
```

```python
from qlora_tunner.utils import data_reformer
train_dataset_path = "train.jsonl" # jsonl dataset format 
valid_dataset_path = "valid.jsonl"

system_message = "Instruction or system_message "

train_dataset_mapped = data_reformer(inp_ut="prompt" ,
                                     output="response" , 
                                     dataset_path=train_dataset_path, 
                                     system_message=system_message)

valid_dataset_mapped = data_reformer(inp_ut="prompt" ,
                                     output="response" ,
                                     dataset_path=valid_dataset_path , 
                                     system_message= system_message)
```
### Fine-Tuning 

```python
from qlora_tunner.qlora_fine_tuner import LanguageModelFineTuner
model_name = "Your Model Id " # Example : NousResearch/llama-2-7b-chat-hf
fine_tuner = LanguageModelFineTuner(model_name)
fine_tuner.train(train_dataset_mapped=train_dataset_mapped,
                 valid_dataset_mapped=valid_dataset_mapped , 
                 output_dir="LLAMA2_chat" ,
                 num_train_epochs=1)
```

### Create Inference pipeline
```python
from qlora_tunner.qlora_inference import LanguageModelInference

model_name = "Your Model Id " # Example : NousResearch/llama-2-7b-chat-hf
LoraConfig_folder = "LoraConfig_file"

inference = LanguageModelInference(model_name)
pipeline = inference.inf_pipeline(max_length=2048 , LoraConfig_file=LoraConfig_folder)
```
### Run the Inference

```python
system_message = "system_message or Instruction"
input_text = "Your input Text"
prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n {input_text}. [/INST]"

result = pipeline(prompt)
print(result[0]['generated_text'].replace(prompt, ''))

```
### With Langchain (chatBot Example) 
```
pip -q install langchain
```

```python

from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

local_llm = HuggingFacePipeline(pipeline=pipeline)
memory = ConversationBufferWindowMemory(k=3)

chat = ConversationChain(
    llm=local_llm,
    verbose=False ,
    memory=memory
)
chat.prompt.template = \
"""
### HUMAN:
Write here Your system_message
Previous Conversation :
{history}

Current conversation:
### HUMAN: {input}
### RESPONSE:"""

while 1 :
  input_text = input(">>")
  print(chat.predict(input=str(input_text)))
```
+ **Author** : Kirouane Ayoub
