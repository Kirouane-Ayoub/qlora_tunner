from datasets import load_dataset

def data_reformer(dataset_path  , system_message , inp_ut , output) :
  print("For datasets with {'input' : 'Your input here' ,'output' : 'Your output here' } Format")
  dataset = load_dataset('json', data_files=dataset_path , split="train")
  # Preprocess datasets
  dataset_mapped = dataset.map(lambda examples: {'text': [f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + prompt + ' [/INST] ' + response for prompt, response in zip(examples[f'{inp_ut}'], examples[f'{output}'])]}, batched=True)
  return dataset_mapped 
