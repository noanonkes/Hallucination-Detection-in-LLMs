import torch, argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

def no_context(query):
    system_message = """As a professor specializing in biomedical studies, design a multiple-choice question set. For each question, create one accurate statement and four false statements. Clearly specify the correctness of each statement in relation to the question. Each statement should be limited to one sentence."""
    
    prompt = f"""<|system|>{system_message}</s><|prompter|>{query}</s><|assistant|>"""
    return prompt

def with_context(query, context):
    system_message = f"""As a professor specializing in biomedical studies, design a multiple-choice question set. For each question, create one accurate statement and four false statements. Clearly specify the correctness of each statement in relation to the question. Each statement should be limited to one sentence. The context for this question is: {context}"""
    
    prompt = f"""<|system|>{system_message}</s><|prompter|>{query}</s><|assistant|>"""
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="data/sampled_data.json",
                        help="JSON file containing the data")
    parser.add_argument("--output_dir", type=str, default="data/generated/")
    args = parser.parse_args()

    # use cuda if cuda is available
    if args.use_cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    # openassistant pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/llama2-13b-orca-8k-3319")
    model = AutoModelForCausalLM.from_pretrained("OpenAssistant/llama2-13b-orca-8k-3319")
    model.to(device)

    # keep track of generated text with and without context
    no_context_outputs, with_context_outputs = [], []

    df = pd.read_json(args.path, lines=True)
    for i, row in df.iterrows():
        # query
        q = row['data']['paragraphs'][0]['qas'][0]['question']
        # answer
        a = row['data']['paragraphs'][0]['qas'][0]['answers'][0]['text']
        # context; slightly more elaborate text containing answer
        c = row['data']['paragraphs'][0]['context']

        # for testing
        print(i, ":", q)

        # create prompt with no context c
        prompt = no_context(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=256)
        no_context_outputs.append(tokenizer.decode(output[0]) + "\n")

        # create prompt with context c
        prompt = with_context(q, c)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=256)
        with_context_outputs.append(tokenizer.decode(output[0]) + "\n")

    # save generated text with and without context separately; could be done nicer
    f = open(args.output_dir + "no_context.txt", "a")
    f.writelines(no_context_outputs)
    f.close()

    f = open(args.output_dir + "with_context.txt", "a")
    f.writelines(with_context_outputs)
    f.close()