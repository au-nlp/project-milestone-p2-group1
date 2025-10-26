[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

# Text Summarizer with Style Transfer into Gen Z Slang

## Abstract

This project aims to build a dual-stage system trained on TL;DR dataset and our own Gen Z dictionary.
The system first converts long texts into concise TL;DRs, then transfers the stylisation into Gen Z slang.
This process creates informal summaries with current language.

Traditional summarizers such as extractive (TextRank) or abstractive ones (based on seq2seq or Transformer architectures) often produce factually correct but stylistically neutral results, which might seem bland. 

We attempt to show models can be adjusted to adapt information for younger audiences in proper contexts without losing meaning or coherence.
By adding a style-transfer/machine translation stage, we aim to enhance factual summarization with expressive current language, which is used on the internet or daily by younger generations.
To successfully accomplish our goal we need high quality datasets, however the existing ones are either too short, inaccurate or incomplete.
Our goal is to create a more robust Gen Z dataset, combining the existing ones and adding synthetic data as examples of using the slang words.

## Proposed Additional Datasets
We have made an analysis of existing datasets and found out there is not a dataset robust enough for our task.
We are considering to work with three datasets, which we would like to combine and improve (more in the section Contributions).

- https://huggingface.co/datasets/Programmer-RD-AI/genz-slang-pairs-1k
    - a paired table of Normal-GenZ Slang dataset that is analyzed and used to update the Slang dictionary dataset, because of its incompleteness and some inaccuracies in slang explanations.
  
- https://huggingface.co/datasets/MLBtrio/genz-slang-dataset
    - a dictionary of Slang terms that includes context, description and a usage example but lacks the translation.
  This dataset will be updated by adding a new column with the translation.
  
- https://www.kaggle.com/datasets/tawfiayeasmin/gen-z-words-and-phrases-dataset/data
    - another GenZ slang dictionary but that contains an extra column (popularity), after some analysis it was concluded that this dataset contained the same slang entries as the previous Slang dictionary.

## Contributions
Combining existing Gen Z datasets and enhancing them with synthetic data of slang translations for more robust and improved slang–formal alignment.

TODO:

## Methods
Describe the methods you plan to use for analysis, modeling, and evaluation.

TODO:

## Proposed Timeline
Outline your timeline for completing the project milestones and tasks.

TODO:

## Organization Within the Team
List the internal milestones and responsibilities for each team member up until project Milestone P3.

TODO:
