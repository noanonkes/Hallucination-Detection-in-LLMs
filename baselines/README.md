# Misinformation Detection in Large Language Models using Graph Structures on Prompting Generation

Welcome to the GitHub repository for the research project conducted at the University of Amsterdam, focusing on "Misinformation Detection in Large Language Models using Graph Structures on Prompting Generation." This project explores approaches to enhance the trustworthiness of large language models by leveraging graph structures for the detection and mitigation of misinformation. This README has the specifics on how to use the files regarding the graph structures and networks.

## Training Baselines
There is three baseline models available for training, MLP, Cross Encoder, and PCA. Based on the chosen parameters, you can choose the embedder to compute contextual embeddings, and save them for future use. All training is done on the tran set, validation is used for hyperparameter search, if desired, the best model is evaluated on the test set upon training completetion.

### Usage
To start training and evaluation, use the following command:
```bash
python train.py [arguments]
```

## Authors

This research project is a collaborative effort by Sergei Agaronian & Noa Nonkes, supervised by Roxana Petcu, from the University of Amsterdam.
