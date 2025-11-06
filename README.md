[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

# Text Summarizer with Style Transfer into Gen Z Slang

## Abstract

This project aims to build a dual-stage system trained on TL;DR dataset and our own Gen Z dictionary.The system first converts long texts into concise TL;DRs, then applies style transfer to render summaries in Gen Z slang, creating engaging informal summaries that use contemporary youth language.

Traditional summarizers, whether extractive (e.g., TextRank) or abstractive (based on seq2seq or Transformer architectures), often produce factually correct but stylistically neutral outputs. The output may not see interesting and to capture young audiences attention, we try to capture the exprressive, informal tone characteristic of digital commincation amoung Gen Z.

We attempt/demonstrate to show that summarization models can be enhanced with style-transfer capabilities to make information more accessible and engaging for younger audiences without sacrificing meaning or coherence. By incorporating a style-transfer stage, we aim to enhance factual summarization with expressive contemporary language that is commonly used online and in everyday communication by younger generations.

To achieve this goal, high-quality training data is essential. However, existing Gen Z slang datasets are either too limited in size, contain inaccuracies, or lack completeness. Our contribution addresses this gap by creating a more robust Gen Z slang dataset through the combination of existing resources enhanced with synthetically generated slang-formal sentence pairs.

## Proposed Additional Datasets

Our analysis of existing datasets revealed that no single resource is sufficiently robust for our task. We therefore propose combining and enhancing three complementary datasets to address this gap (detailed in the Contributions section).

- [Programmer-RD-AI/genz-slang-pairs-1k](https://huggingface.co/datasets/Programmer-RD-AI/genz-slang-pairs-1k): Paired normal and GenZ slang sentences without a slang word column; slang words will be extracted and used to correct and complete the slang dictionary for more accurate explanations.
- [MLBtrio/genz-slang-dataset](https://huggingface.co/datasets/MLBtrio/genz-slang-dataset): GenZ slang dictionary description, context, and usage examples intended for fine-tuning slang models; lacks direct translation and is being improved. The dataset will be updated with a new column that translates into normal English language, aligning with the genz-slang-pairs-1k.
- [tawfiayeasmin/gen-z-words-and-phrases-dataset](https://www.kaggle.com/datasets/tawfiayeasmin/gen-z-words-and-phrases-dataset): GenZ slang dictionary with a popularity score; contains mostly unique entries with almost no overlap with other datasets. Due to this lack of overlap, it will be used to extend the slang dictionary; however, but the popularity metric cannot be utilised effectively since it is not supported by the rest of the dataset.
- [TL;DR](https://huggingface.co/datasets/trl-lib/tldr): Reddit post and summaries of 100K+ examples. Will be extended with a GenZ style of summarization.

## Contributions

Our main contribution is creating more robust Gen Z dictionary dataset and extending the TL;DR corpus. Combining existing Gen Z datasets and enhancing them with synthetic data for slang translations enables more robust and accurate alignment between slang and formal language, which can be used for improving slang detection and text generation models. The TL;DR corpus is extended by a new column of synthetic data, which mirros the completion column but with added Gen Z slang. Fine-tuned BLEURT metric for evaluation to understand GenZ embeddings.  

## Methods

Describe the methods you plan to use for analysis, modeling, and evaluation.

### Data Analysis

- Exploratory data analysis of Gen Z datasets and TL;DR corpus. Combining gen z dictionaries. Generating synthetic data to extend TL;DR for having input paired data for SFT of a pretrained summarizer 

### Modelling

- Architecture: (T5-base, BERT, ...) (fine-tuned on style-transfer task) different pretrained summarizers. 
- SFT
- input - TL;DR completion sample + respective Gen Z completion (+ prompt???)
- output - Gen Z summary

### Evaluation
- Primary: BLEURT fine-tuned metric (semantic preservation correlation)

TODO:

## Proposed Timeline

Outline your timeline for completing the project milestones and tasks.

### Milestone 1
- individual project proposals
### Milestone 2
- readme.md
- synthetic data generation
- combining gen z dictionaries
- data analysis
- main.jpyntb
### Milestone 3
- ???
- 
| Date | Milestone | Check-in |
|------|-----------|----------|
| Week 1 | Setup & Exploration | Data report |
| Week 3 | Data Ready | Merged dict + TLDR |
| Week 5 | Models Trained | Model checkpoints |
| Week 6 | Evaluation Done | Metrics report |
| Week 7 | Final | Complete system |

TODO:

## Organization Within the Team

List the internal milestones and responsibilities for each team member up until project Milestone P3.

TODO:
