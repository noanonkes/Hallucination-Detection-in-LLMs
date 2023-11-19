import torch, argparse
import numpy as np
from torch.utils.data import DataLoader
from dataloader import SentenceLabelDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="data/generated/full_train.csv",
                        help="JSON file containing the data")
    args = parser.parse_args()

    # use cuda if cuda is available
    if args.use_cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    dataset = SentenceLabelDataset(args.path)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    model.to(device)

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = np.array(inputs)        
        encoded_input = tokenizer(inputs, return_tensors='pt').to(device)
        output = model(**encoded_input)
        print(output)
        break
