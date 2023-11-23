import torch, argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_no_context_prompt(query):
    system_message = """Imagine you are crafting a multiple-choice exam in the field of biomedical studies. 
                        Your task is to generate a set of statements related to a given question. 
                        Provide one accurate statement as the correct answer (Answer 1) 
                        and four misleading statements that should appear as plausible distractors (Answers 2 to 5). 
                        Ensure that the incorrect answers are not easily mistaken for accurate information related to the question. 
                        Remember, each statement should be concise and limited to a single sentence.
                        
    Instructions:
    Answer 1: [Insert correct answer here]
    Answer 2: [Insert misleading statement here]
    Answer 3: [Insert misleading statement here]
    Answer 4: [Insert misleading statement here]
    Answer 5: [Insert misleading statement here]
    """

    prompt = f"""<|system|>{system_message}</s><|prompter|>{query}</s><|assistant|>"""
    return prompt

def generate_with_context_prompt(query, context):
    system_message = f"""Imagine you are crafting a multiple-choice exam in the field of biomedical studies. 
                        Your task is to generate a set of statements related to a given question. 
                        Provide one accurate statement as the correct answer (Answer 1) 
                        and four misleading statements that should appear as plausible distractors (Answers 2 to 5). 
                        Ensure that the incorrect answers are not easily mistaken for accurate information related to the question. 
                        Remember, each statement should be concise and limited to a single sentence.
                        The context for this question is: {context}
                        
    Instructions:
    Answer 1: [Insert correct answer here]
    Answer 2: [Insert misleading statement here]
    Answer 3: [Insert misleading statement here]
    Answer 4: [Insert misleading statement here]
    Answer 5: [Insert misleading statement here]
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
    for i, row in tqdm(df.iterrows()):
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
        output = model.generate(**inputs, max_new_tokens=512)
        
        decoded_output = tokenizer.decode(output[0])
        ans_idx = decoded_output.rfind("<|assistant|>")
        ans = decoded_output[ans_idx:]
        try:
            sentences = [line.split(': ')[1].strip('</s>') for line in ans.split('\n') if len(line.strip()) > 0]
            formatted_sentences = '\n'.join(sentences)
            formatted_sentences += '\n'
            # Save the generated output
            generated_outputs.extend(formatted_sentences)
        except:
            print(f'Failed to answer {query}')
            print(f"Model output was {ans}\n")

    output_filename = "with_context.txt" if args.use_context else "no_context.txt"
    output_path = args.output_dir + output_filename

    try:
        with open(output_path, "w") as file:
            file.writelines(generated_outputs)
        print(f"Generated content succesfully saved at: {output_path}")
        
    except Exception as e:
        print(f"Error occurred while saving generated content: {str(e)}")