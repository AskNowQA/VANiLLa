# VANiLLa : Verbalized Answers in Natural Language at Large scale

## Introduction

Question Answering (QA) has been an active field of research in the past years with significant developments in the area of Question Answering over Knowledge Graphs (KGQA). In spite of all the notable advancements, current KGQA datasets only provide the answers as resource or literals rather than full sentences. Thus, template-based verbalizations are usually employed for representing the answers in natural language. This deficiency is a ramification of the scarcity of datasets for verbalizing KGQA responses. Hence, we provide the VANiLLa dataset which aims at reducing this gap. Our dataset consists of over 100k simple questions adapted from the CSQA and SimpleQuestionsWikidata datasets along with their answers in natural language sentences. In this paper, we describe the dataset creation process and dataset characteristics. We also present multiple baseline models adapted from current state-of-the-art Natural Language Generation (NLG) architectures. We believe that this dataset will allow researchers to focus on finding suitable methodologies and architectures for answer verbalization.

## Dataset

The dataset is available at: [here](https://figshare.com/articles/Vanilla_dataset/12360743) under [Attribution 4.0 International (CC BY 4.0)](LICENSE).

Our dataset contains over 100k examples with a 80% (train) - 20% (test) split. Each instance of the dataset consists of:

```bash
{
    "question_id": "an unique identification number for a dataset instance",
    "question": "question",
    "answer": "retrieved answer",
    "answer_sentence": "verbalized answer in natural language"
}
```

## Baseline Models

We decided to use some conventional sequence-to-sequence models following the underlying Encoder-Decoder pipeline:
* BL1: Sequence-to-Sequence model with attention mechanism
* BL2: Convolution based Encoder-Decoder model
* BL3: Transformer

## Experimental Results

| Baseline Model | PPl | Precision | BLEU |
| ------------- | ------------- | ------------- | ------------- |
| BL1 | 27.91 | 19.84 | 16.66 |
| BL2 | 87.67 | 70.50 | 15.42 |
| BL3 | **12.10** | **76.00** |  **30.80** |


