[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

# Text Summarizer with Style Transfer into Gen Z Slang
## Abstract
This project aims to build a dual-stage system trained on TL;DR dataset and our own Gen Z dictionary. The system first converts long texts into concise TL;DRs, then applies style transfer to render summaries in Gen Z slang, creating engaging informal summaries that use contemporary youth language.

Traditional summarizers, whether extractive (e.g., TextRank) or abstractive (based on seq2seq or Transformer architectures), often produce factually correct but stylistically neutral outputs. These outputs appear non optimal from an appealing and markeatble point of view towards young audiences,which value the expressive, and informal tone characteristic of the digital communication among Gen Z.

We attempt to show that summarization models can be enhanced with style-transfer capabilities to make information more accessible and engaging for younger audiences without sacrificing meaning or coherence. By incorporating a style-transfer stage, we aim to enhance factual summarization with expressive contemporary language that is commonly used online and in everyday communication by younger generations.

To achieve this goal, high-quality training data is essential. However, existing Gen Z slang datasets are too limited in size, contain inaccuracies, or lack completeness. Our contribution addresses this gap by creating a more robust Gen Z slang dataset through the combination of existing resources enhanced with synthetically generated slang-formal sentence pairs.

## Proposed Additional Datasets
- [Programmer-RD-AI/genz-slang-pairs-1k](https://huggingface.co/datasets/Programmer-RD-AI/genz-slang-pairs-1k): Paired normal and Gen Z slang sentences without a slang word column; slang words will be extracted and used to correct and complete the slang dictionary for more accurate explanations.
- [MLBtrio/genz-slang-dataset](https://huggingface.co/datasets/MLBtrio/genz-slang-dataset): Gen Z slang dictionary description, context, and usage examples intended for fine-tuning slang models; lacks direct translation and is being improved. The dataset will be updated with a new column that translates into normal English language, aligning with the genz-slang-pairs-1k.
- [tawfiayeasmin/gen-z-words-and-phrases-dataset](https://www.kaggle.com/datasets/tawfiayeasmin/gen-z-words-and-phrases-dataset): Gen Z slang dictionary with a popularity score; contains mostly unique entries with almost no overlap with other datasets. Due to this lack of overlap, it will be used to extend the slang dictionary; however, but the popularity metric cannot be utilised effectively since it is not supported by the rest of the dataset.
- [TL;DR](https://huggingface.co/datasets/trl-lib/tldr): Reddit post and summaries of 100K+ examples. Will be extended with a Gen Z style summarization.

## Contributions
Our project makes the following key contributions to the field of style transfer and summarization:
-   **Robust Gen Z Dictionary**: We constructed and enchanced a unified dictionary of **1,888 unique slang terms** by merging and standardizing three fragmented open-source datasets (MLBtrio, Programmer-RD-AI, and Kaggle), resolving issues of duplication and missing context.
-   **Large-Scale Synthetic Corpus**: We generated **116,578 high-quality parallel training pairs** (standard TL;DR $\to$ Gen Z TL;DR). This was achieved using a retrieval-augmented generation pipeline where **OpenAI GPT OSS 120B** rewrote summaries guided by semantically relevant slang embeddings.
-  **Supervised Fine-Tuning Evaluation**: We fine-tuned and compared **T5-small** and **BART-base** architectures, demonstrating that BART-base outperforms T5 in preserving meaning while adopting the target style.

## Methods
### Data Analysis

We have conducted an exploratory data analysis of the TL;DR corpus and our Gen Z dataset.
Our Gen Z dataset consists of three existing Gen Z datasets/dictionaries, which are listed in the Proposed Additional Datasets section.
We embed neutral translations of slang terms and retrieval-match them to TL;DR summaries using cosine similarity.
An LLM (GPT OSS 120B) rewrites the summaries using the retrieved slang, creating a supervised training set.
You can find all the details in file `data_generation.ipynb`.

### Modelling

Our aim is to perform a supervised fine-tuning (SFT) step on a pretrained summarizer. We explore different pretrained summarizers (T5-small and BART-base) and then compare their results. Our target output is Gen Z style summary. Thus, the fine-tuning receives as input a TL;DR prompt, its completion and a respective synthetically generated Gen Z completion.

### Evaluation
We employed a dual evaluation strategy:
1.  **Metrics**: A custom **Style Fidelity Score** composed of:
    -   **Semantic Quality**: BERTScore F1 (90% weight) to ensure meaning preservation.
    -   **Slang Fidelity**: A slang density score (10% weight) to penalize deviations from the authentic slang distribution observed in the training data.
2.  **LLM-based Comparative Evaluation**: We used **Amazon Bedrock** with the model `openai.gpt-oss-120b-1:0` as an external judge. The judge evaluated 3,000 test instances on three dimensions (1-5 scale):
    -   **Meaning Preservation**
    -   **Slang Quality**
    -   **Reddit Naturalness**

## Proposed Timeline
The proposed timeline is sketched out in the table below. The part for milestone 3 might change in the future.

| Milestone | Semester Week | Step                               |
| --------- | ------------- | ---------------------------------- |
| 1         | Week 4        | Individual project proposals       |
| 2         | Week 7        | Choose common project idea         |
|           | Week 8-9      | Refine the proposal in detail      |
|           | Week 8-9      | Research of Gen Z datasets         |
|           | Week 9-10     | Data analysis                      |
|           | Week 10       | Combining datasets + preprocessing |
|           | Week 9-11     | README file                        |
|           | Week 10-12    | Synthetic data generation          |
|           | Week 11       | Clean code                         |
| 3         | Week 12       | Analysing generated synthetic data |
|           | Week 13       | SFT of a summarizer                |
|           | Week 14       | Testing the model with LLM as Jugde|
|           | Week 15-16    | Writing the report                 |

## Organization Within the Team
We meet in person regularly and do some work together, but we also work individually at home, which was roughly assigned between us as described below.  

Bjarke
- Data preprocessing
- Synthetic data generation
- Training the final models and testing them using our metrics and an LLM judge
- Writing the report

Pablo
- Research of existing datasets
- Exploratory and qualitative data analysis
- Fine-tuning code and algorithms
- Writing the report

Michal
- README.md
- Clean code, editing textual descriptions, keeping the repo tidy
- Running fine-tuning experiments
- Writing the report

## Appendix / Artifacts
-   **Model Weights & Datasets**: Uploaded to [Google Drive](https://drive.google.com/drive/folders/1QABHakQ12phfQRyPg9b7-yrZjMOL3j6z?usp=drive_link) and [Kagglehub](https://www.kaggle.com/datasets/bjarkekarlsen/tldr-genz-completion).
-   **Custom Metrics**: See `scripts/slang_score.py` for the implementation of the slang density penalty function used in our Style Fidelity score.
-   **Data Generation** See notebook `data_generation.ipynb`.
