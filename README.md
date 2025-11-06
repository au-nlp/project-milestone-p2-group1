[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

# Text Summarizer with Style Transfer into Gen Z Slang

## Abstract

This project aims to build a dual-stage system trained on TL;DR dataset and our own Gen Z dictionary.The system first converts long texts into concise TL;DRs, then applies style transfer to render summaries in Gen Z slang, creating engaging informal summaries that use contemporary youth language.

Traditional summarizers, whether extractive (e.g., TextRank) or abstractive (based on seq2seq or Transformer architectures), often produce factually correct but stylistically neutral outputs. The output may not see interesting and to capture young audiences attention, we try to capture the exprressive, informal tone characteristic of digital commincation amoung Gen Z.

We attempt/demonstrate to show that summarization models can be enhanced with style-transfer capabilities to make information more accessible and engaging for younger audiences without sacrificing meaning or coherence. By incorporating a style-transfer stage, we aim to enhance factual summarization with expressive contemporary language that is commonly used online and in everyday communication by younger generations.

To achieve this goal, high-quality training data is essential. However, existing Gen Z slang datasets are either too limited in size, contain inaccuracies, or lack completeness. Our contribution addresses this gap by creating a more robust Gen Z slang dataset through the combination of existing resources enhanced with synthetically generated slang-formal sentence pairs.

## Proposed Additional Datasets

Our analysis of existing datasets revealed that no single resource is sufficiently robust for our task. Therefore, we propose combining and enhancing three complementary datasets to address this gap (detailed in the Contributions section).

- [Programmer-RD-AI/genz-slang-pairs-1k](https://huggingface.co/datasets/Programmer-RD-AI/genz-slang-pairs-1k): Paired normal and GenZ slang sentences without a slang word column; slang words will be extracted and used to correct and complete the slang dictionary for more accurate explanations.
- [MLBtrio/genz-slang-dataset](https://huggingface.co/datasets/MLBtrio/genz-slang-dataset): GenZ slang dictionary description, context, and usage examples intended for fine-tuning slang models; lacks direct translation and is being improved. The dataset will be updated with a new column that translates into normal English language, aligning with the genz-slang-pairs-1k.
- [tawfiayeasmin/gen-z-words-and-phrases-dataset](https://www.kaggle.com/datasets/tawfiayeasmin/gen-z-words-and-phrases-dataset): GenZ slang dictionary with a popularity score; contains mostly unique entries with almost no overlap with other datasets. Due to this lack of overlap, it will be used to extend the slang dictionary; however, but the popularity metric cannot be utilised effectively since it is not supported by the rest of the dataset.
- [TL;DR](https://huggingface.co/datasets/trl-lib/tldr): Reddit post and summaries of 100K+ examples. Will be extended with a GenZ style summarization.

## Contributions

Our main contributions are as follows.
- Creating more robust Gen Z dictionary dataset.
  Combining existing Gen Z datasets and enhancing them with synthetic data for slang translations enables more robust and accurate alignment between slang and formal language, which could be used for improving slang detection and text generation models.
  We will use it for supervised fine-tuning of a pretrained summarizer.
- The TL;DR corpus is extended by a new column of synthetic data, which mirros the completion (TL;DR summary) column with added Gen Z slang.
- For evaluation, we want to fine-tune the BLEURT metric with Gen Z embeddings.

## Methods

### Data Analysis

We have conducted an exploratory data analysis of the TL;DR corpus and our Gen Z dataset.
Our Gen Z dataset consists of three existing Gen Z datasets/dictionaries, which are listed in the Proposed Additional Datasets section.
We generate synthetic data to extend TL;DR for having input paired data for SFT of a pretrained summarizer.
You can find the details in the file `main.ipynb`.

### Modelling

We can possibly explore fine-tuning different pretrained summarizers (e.g. T5-base, BERT) and then compare their results.
- input - TL;DR completion sample + respective Gen Z completion (+ prompt???)
- output - Gen Z summary

In a later stage of the project, it is possible to extend the framework by adding a Deep RL agent to improve the fine-tuned summarizer.

### Evaluation
- Primary: BLEURT fine-tuned metric (semantic preservation correlation)

TODO:

## Proposed Timeline

### Milestone 1
- 
### Milestone 2
- readme.md
- synthetic data generation
- combining gen z dictionaries
- data analysis
- main.jpyntb
### Milestone 3
- ???
  
| Semester Week | Step | Milestone |
|------|-----------|----------|
| Week 4 | Individual project proposals | 1 |
| Week 7 | Choosing common project idea | 2 |
| Week 9 | Refine project idea in detail, do data analysis | 2 |
| Week 10 | Synthetic data generation, README file | 2 |
| Week 11 | Clean the code, prepare main.ipynt | 2 |


## Organization Within the Team

Bjarke
- Synthetic data generation
- clean code

Pablo
- Data analysis

Michal
- README.md
