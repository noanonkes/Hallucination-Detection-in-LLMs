# Misinformation Detection in Large Language Models using Graph Structures on Prompting Generation

Welcome to the GitHub repository for the research project conducted at the University of Amsterdam, focusing on "Misinformation Detection in Large Language Models using Graph Structures on Prompting Generation." This project explores approaches to enhance the trustworthiness of large language models by leveraging graph structures for the detection and mitigation of misinformation. This README has the specifics on how to use the files regarding acquisition of baseline models.

## Training Baselines
There is three baseline models available for training, MLP, Cross Encoder, and PCA. Based on the chosen parameters, you can choose the embedder to compute contextual embeddings, and save them for future use. All training is done on the tran set, validation is used for hyperparameter search, if desired, the best model is evaluated on the test set upon training completetion.

### Usage
To start training and evaluation, use the following command:
```bash
python train.py [arguments]
```

### Arguments
```bash
--use-cuda: Enable GPU acceleration if available (default: True)
--path <path_to_data>: Set the path to the data folder (default: "../data/")
--output_dir <output_directory>: Specify the path to save the model weights (default: "../weights/")
--epochs <num_epochs>: Define the number of epochs to train the model (default: 1000)
--batch-size <batch_size>: Set the batch size used during training and evaluation (default: 256)
--lr <lr>: Set the training learning rate for the optimizer (default: 0.01)
--optimizer <optimizer>: Choose the optimizer to train the model between SGD and Adam (default: Adam)
--model <model>: Which model to train and evaluate between MLP, CE, and PCA
--pretrained <pretrained>: What model to use for computing embeddings, if a local pretrained model us used, then provide the path to the location of the model (default: bert-base-uncased)
```

## Authors

This research project is a collaborative effort by Sergei Agaronian & Noa Nonkes, supervised by Roxana Petcu, from the University of Amsterdam.
