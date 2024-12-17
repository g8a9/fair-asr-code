# Code to support the Fair ASR Leaderboard

[![Leaderboard](https://img.shields.io/badge/Leaderboard-Space-yellow)](https://huggingface.co/spaces/g8a9/fair-asr-leaderboard)


At EMNLP 2024, we published a [paper](https://aclanthology.org/2024.emnlp-main.1188/) showing how modern speech recognition system exhibit systematic gender performace gaps.

This repository contains code to extend the paper's results and populate a [living leaderboard](https://huggingface.co/spaces/g8a9/fair-asr-leaderboard).

## Getting started 

WIP

## Contributing a new model

WIP

## Cite as

**Update 🏅:** the paper was awarded with the Social Impact Award!

```bibtex
@inproceedings{attanasio-etal-2024-twists,
    title = "Twists, Humps, and Pebbles: Multilingual Speech Recognition Models Exhibit Gender Performance Gaps",
    author = "Attanasio, Giuseppe  and
      Savoldi, Beatrice  and
      Fucci, Dennis  and
      Hovy, Dirk",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1188",
    doi = "10.18653/v1/2024.emnlp-main.1188",
    pages = "21318--21340",
    abstract = "Current automatic speech recognition (ASR) models are designed to be used across many languages and tasks without substantial changes. However, this broad language coverage hides performance gaps within languages, for example, across genders. Our study systematically evaluates the performance of two widely used multilingual ASR models on three datasets, encompassing 19 languages from eight language families and two speaking conditions. Our findings reveal clear gender disparities, with the advantaged group varying across languages and models. Surprisingly, those gaps are not explained by acoustic or lexical properties. However, probing internal model states reveals a correlation with gendered performance gap. That is, the easier it is to distinguish speaker gender in a language using probes, the more the gap reduces, favoring female speakers. Our results show that gender disparities persist even in state-of-the-art models. Our findings have implications for the improvement of multilingual ASR systems, underscoring the importance of accessibility to training data and nuanced evaluation to predict and mitigate gender gaps. We release all code and artifacts at https://github.com/g8a9/multilingual-asr-gender-gap.",
}

```