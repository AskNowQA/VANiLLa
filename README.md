# VANiLLa : Verbalized Answers in Natural Language at Large scale

## Introduction

In the last years, there have been significant developments in the area of Question Answering over Knowledge Graphs (KGQA). Despite all the notable advancements, current KGQA datasets only provide the answers as the direct output result of the formal query, rather than full sentences incorporating question context. For achieving coherent answers sentence with the question's vocabulary,  template-based verbalization so are usually employed for a better representation of answers, which in turn require extensive expert intervention. Thus, making way for machine learning approaches; however, there is a scarcity of datasets that empower machine learning models in this area. Hence, we provide the VANiLLa dataset which aims at reducing this gap by offering answers in natural language sentences. The answer sentences in this dataset are syntactically and semantically closer to the question than to the triple fact. Our dataset consists of over 100k simple questions adapted from the CSQA and SimpleQuestionsWikidata datasets and generated using a semi-automatic framework. We also present results of training our dataset on multiple baseline models adapted from current state-of-the-art Natural Language Generation (NLG) architectures. We believe that this dataset will allow researchers to focus on finding suitable methodologies and architectures for answer verbalization.

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

## Experimental Results

### Baseline Models

We decided to use some conventional sequence-to-sequence models following the underlying Encoder-Decoder pipeline:
* Sequence-to-Sequence model with attention mechanism
* Convolution based Encoder-Decoder model
* Transformer

| Baseline Model | PPl | Precision | BLEU |
| ------------- | ------------- | ------------- | ------------- |
| Seq2Seq with Attention | 27.91 | 19.84 | 16.66 |
| CNN Enc-Dec | 87.67 | 70.50 | 15.42 |
| Transformer | **12.10** | **76.00** |  **30.80** |


