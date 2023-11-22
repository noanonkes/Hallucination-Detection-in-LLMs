import torch, argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_no_context_prompt(query):
    system_message = """You are a professor specializing in biomedical studies who wants to design a multiple choice exam. 
                        For each given question, generate one true statement and hallucinate four false single sentence statements. 
                        Make sure that the false answers are not likely to be true answers to any other related questions.
                        Give your answers according to the following format, where you fill in Answer 1 to Answer 5, 
                        where Answer 1 is always the correct answer:
    Answer 1:
    Answer 2:
    Answer 3:
    Answer 4:
    Answer 5:
    """

    prompt = f"""<|system|>{system_message}</s><|prompter|>{query}</s><|assistant|>"""
    return prompt

def generate_with_context_prompt(query, context):
    system_message = f"""You are a professor specializing in biomedical studies who wants to design a multiple choice exam. 
                        For each given question, generate one true statement and hallucinate four false single sentence statements. 
                        Make sure that the false answers are not likely to be true answers to any other related questions.
                        Give your answers according to the following format, where you fill in Answer 1 to Answer 5, 
                        where Answer 1 is always the correct answer:
    
    Answer 1:
    Answer 2:
    Answer 3:
    Answer 4:
    Answer 5:

    The context for the question is {context}
    """
    
    prompt = f"""<|system|>{system_message}</s><|prompter|>{query}</s><|assistant|>"""
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="data/sampled_data.json",
                        help="JSON file containing the data")
    parser.add_argument("--output_dir", type=str, default="data/generated/")
    parser.add_argument("--use-context", action="store_true", default=False,
                        help="Use context in prompts")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for reproducibility")
    args = parser.parse_args()

    # Use cuda if cuda is available
    device = torch.device("cuda") if args.use_cuda and torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.seed)  # seed the generation
    
    # openassistant pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/llama2-13b-orca-8k-3319")
    model = AutoModelForCausalLM.from_pretrained("OpenAssistant/llama2-13b-orca-8k-3319", 
                                                 torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(device)

    # Keep track of the generated content
    generated_outputs = []

    df = pd.read_json(args.path, lines=True)
    for i, row in df.iterrows():
        query = row['data']['paragraphs'][0]['qas'][0]['question']
        answer = row['data']['paragraphs'][0]['qas'][0]['answers'][0]['text']
        context = row['data']['paragraphs'][0]['context'] if args.use_context else None

        # For testing
        print(i, ":", query)

        if args.use_context:
        # Prompt with context
            prompt = generate_with_context_prompt(query, context)
        else:
        # Prompt no context
            prompt = generate_no_context_prompt(query)
            
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=256)
        
        decoded_output = tokenizer.decode(output[0])
        ans_idx = decoded_output.find("<|assistant|>") + 13
        ans = decoded_output[ans_idx:]
        
        # Save the generated output
        generated_outputs.append(ans + "\n")

        # if i == 3:
        #     break
        
    output_filename = "with_context.txt" if args.use_context else "no_context.txt"
    output_path = args.output_dir + output_filename

    try:
        with open(output_path, "w") as file:
            file.writelines(generated_outputs)
        print(f"Generated content succesfully saved at: {output_path}")
        
    except Exception as e:
        print(f"Error occurred while saving generated content: {str(e)}")