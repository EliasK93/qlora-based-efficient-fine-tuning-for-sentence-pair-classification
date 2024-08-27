## QLoRA-based efficient Fine-Tuning for Sentence Pair Classification

Example application for applying a parameter-efficient fine-tuning (PEFT) approach to a sentence pair classification task (Stance Detection). 

[QLoRA](https://arxiv.org/abs/2305.14314) (_Quantized Low-Rank Adaptation_) essentially combines two approaches:
- _Quantization_: Loading and fine-tuning the model parameters at a lower bit precision to fit more parameters into the same memory
- _Low-Rank Adaption_: Freezing the pre-trained model weights while injecting much smaller rank decomposition matrices into each layer and fine-tuning only those

Using these approaches, a total of four language models were fine-tuned for three epochs on the task using a single RTX 4080 GPU:

- [Mistral v0.1 (7B)](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Mistral v0.3 (7B)](https://huggingface.co/mistralai/Mistral-7B-v0.3)
- [Llama 3 (8B)](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [Llama 3.1 (8B)](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)

<br>

### Corpus

Each model was fine-tuned on a 5,000 sentences Stance Detection corpus that I manually annotated during my Master's Thesis.
Stance Detection aims to classify the stance a sentence takes towards a claim (topic) as either _Pro_, _Contra_ or _Neutral_.
The sentences originate from Reddit's _r/ChangeMyView_ subreddit in the time span between January 2013 and October 2018, as provided in the [ConvoKit subreddit corpus](https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/corpus-zipped/).
They cover five topics: _abortion_, _climate change_, _gun control_, _minimum wage_ and _veganism_.
The table below shows some examples.

|                  topic                   | sentence                                                                                                                                                                    | stance label |
|:----------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------:|
|    There should be more gun control.     | It's the only country with a "2nd Amendment", yet 132 countries have a lower murder rate.                                                                                   |     Pro      |
| Humanity needs to combat climate change. | The overhwelming evidence could be lies and you would never know because you're content to live your life as a giant appeal to authority.                                   |    Contra    |
|            Vegans are right.             | It's all about finding a system that works for you.                                                                                                                         |   Neutral    |


<br>

### Results

|      Model      | total parameters |  trainable in PEFT  | Accuracy | Micro-F1 | Macro-F1 |
|:---------------:|:----------------:|:-------------------:|:--------:|:--------:|:--------:|
| Mistral-7B-v0.1 |  7,124,316,160   | 13,643,776 (0.192%) |   0.87   |   0.87   |   0.87   |
| Mistral-7B-v0.3 |  7,127,461,888   | 13,643,776 (0.191%) |   0.88   |   0.88   |   0.88   |
|  Llama-3-8B     |  7,518,580,736   | 13,643,776 (0.182%) |   0.85   |   0.85   |   0.85   |
|  Llama-3.1-8B   |  7,518,580,736   | 13,643,776 (0.182%) |   0.87   |   0.87   |   0.86   |

<br>

### Requirements

##### - Python >= 3.10

##### - Conda
  - `pytorch==2.4.0`
  - `cudatoolkit=12.1`

##### - pip
  - `transformers`
  - `datasets`
  - `sentencepiece`
  - `protobuf`
  - `peft`
  - `bitsandbytes`
  - `openpyxl`
  - `scikit-learn`

<br>

### Notes

The dataset files in this repository are cut off after the first 50 rows.
The trained model files `adapter_model.safetensors`, `optimizer.pt` and `tokenizer.json` are omitted in this repository.
