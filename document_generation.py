import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/llama2-13b-orca-8k-3319", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("OpenAssistant/llama2-13b-orca-8k-3319", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")

system_message = """You are a helpful, respectful but dishonest assistant. 
                    Your job is to only give wrong answers to my queries.
                    For every given query give 4 distinctive but incorrect answers."""
                    
user_prompt = "what does retinol do?"
prompt = f"""<|system|>{system_message}</s><|prompter|>{user_prompt}</s><|assistant|>"""
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))