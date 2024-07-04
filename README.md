# Leveraging Graph Structures to Detect Hallucinations in Large Language Models

Welcome to the GitHub repository for the research project conducted at the University of Amsterdam, focusing on "Leveraging Graph Structures to Detect Hallucinations in Large Language Models." This project explores approaches to enhance the trustworthiness of large language models by leveraging graph structures for the detection of hallucinations.

## Overview

Large Language Models (LLMs) have recently surged in popularity, extending beyond researchers and developers to reach the general public, notably due to the emergence of conversational agents such as ChatGPT. Due to their accessibility and adaptability, these models find use in a wide spectrum of domains, from everyday household problems such as determining the ideal boiling time of an egg, to offering financial advice or healthcare assistance [1]. The ability of LLMs to produce human-like output often blocks individuals' ability to distinguish between verified knowledge and hallucinations, potentially resulting in accepting deceptive information that has not been validated through critical assessment [2]. Despite their remarkable abilities, LLMs do not always provide credible information and are prone to hallucinating [3]. Therefore, there is a non-trivial need for robust methodologies that detect and mitigate the spread of LLM-generated hallucinations.

The objective of this research was to detect hallucinations by using graph structures built upon retrieval-augmented generations. Our method first prompts LLMs to generate a set of answers and then builds a graph where the connections between the answers are established based on relevant metrics, such as linguistic similarity or sentence diversity.

Our paper introduces a framework to detect hallucinations within LLM-generated content by 1) generating a new dataset by deliberately prompting an LLM to generate facts with varying degrees of truthfulness using query search retrieved data, 2) establishing a setup for learning semantically rich word embeddings, and 3) employing GATs to facilitate intelligent neighbor selection and message passing. 

## Directory structure

An overview of the directory structure is given below. In the main folder we have the files to sample and generate the data, `data_wrangling.ipynb` and `document_generation.py` respectively. This will save the data in the `data/` folder. The code to concerning the graph structure and GAT can be found in `graph/` and code for the baselines can be found in `baselines/`. The specific usage of the files in these folder can be found in their specific READMEs. If you want to use the pre-trained weights, they can be found in the `weights/` folder.

```tree
├── data/
|    ├── generated/
|    |   ├── no_context.csv
|    |   └── with_context.csv
|    ├── sampled_data.json
|    └── squad.biomedical.train.json
├── graph/
|    ├── images/
|    ├── contrastive_learning.py
|    ├── dataloader.py
|    ├── evaluate_graph.py
|    ├── GAT.py
|    ├── kNN.py
|    ├── make_graph.py
|    ├── train_graph.py
|    ├── utils_graph.py
|    ├── visualize_graph.py
|    └── contrastive_learning.py
├── baselines/
|    ├── baselines.py
|    ├── dataloader.py
|    ├── train.py
|    └── utils.py
├── weights/
├── data_wrangling.ipynb
├── document_generation.py
└── environment.yml
```

## Environment and requirements

As the directory structure shows, we included a `environment.yml` file with the packages and dependencies that should be installed. You can create a virtual environment with your favourite manager, i.e. conda, and install the requirements with:

```bash
conda env create -f environment.yml
```
## Data Generation

We include all the generations needed to train and test the model in the data directory. However, if you wish to get new samples. You can use the following script:

```bash
python document_generation.py <arguments>
```

The prompt can be changed inside the python script by setting a new `system_message`. You can also choose the following arguments:
```bash
--use-cuda: Enable GPU acceleration if available (default: True)
--path <path_to_data>: Set the path to the data folder (default: "data/sampled_data.json")
--output_dir <output_directory>: Specify the path to save the model generations (default: "data/generated/")
--use-context <use-context>: Choose whether to use context, or no context prompt (default: False)
--seed <seed>: Pick seed for reproducibility (default: 42)
```


## Authors

This research project is a collaborative effort by Sergei Agaronian & Noa Nonkes, supervised by Roxana Petcu, from the University of Amsterdam.

## References

[1] Yeganeh Shahsavar and Avishek Choudhury. 2023. User Intentions to Use ChatGPT for Self-Diagnosis and Health-Related Purposes: Cross-sectional Survey Study. JMIR Human Factors 10, 1 (may 2023), e47564. https://doi.org/10.2196/47564

[2] Enkelejda Kasneci, Kathrin Sessler, et al . 2023. ChatGPT for good? On opportunities and challenges of large language models for education. Learning and Individual Differences 103 (apr 2023), 102274. https://doi.org/10.1016/j.lindif.2023.102274

[3] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of Hallucination in Natural Language Generation. ACM Comput. Surv. 55, 12 (mar 2023), 1–38. https://doi.org/10.1145/3571730 arXiv:2202.03629
