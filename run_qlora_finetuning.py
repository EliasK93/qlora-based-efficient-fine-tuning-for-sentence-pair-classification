import os
import pandas
import torch
from datasets import Dataset, DatasetDict
from datasets.formatting.formatting import LazyBatch
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, \
    Trainer, DataCollatorWithPadding, PreTrainedTokenizer, BatchEncoding
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModelForSequenceClassification
from sklearn.metrics import classification_report
import hf_token


def load_tokenized_dataset(tokenizer: PreTrainedTokenizer) -> tuple[DataCollatorWithPadding, DatasetDict]:
    """
    Load and tokenize the dataset and return it as a DatasetDict, use a DataCollator for truncation.
    """

    def tokenize_function(batch: LazyBatch) -> BatchEncoding:
        """
        Tokenize sentence pair batches, enable truncation after 512 tokens.
        """
        return tokenizer(batch["topic"], batch["text"], padding=True, truncation=True, max_length=512)

    def load_dataset(path: str) -> Dataset:
        """
        Load a local xlsx file and convert in to a Dataset object.
        """
        df = pandas.read_excel(path, index_col=0)
        df["index"] = df.index
        df["label"] = df["stance"].map({"con": 0, "neu": 1, "pro": 2})
        ds = Dataset.from_dict({col: df[col] for col in ["topic", "text", "label", "index"]})
        ds = ds.map(tokenize_function, batched=True)
        return ds

    # initialize a DataCollator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # load train, validation and test sets
    train_dataset = load_dataset("data/train.xlsx")
    validation_dataset = load_dataset("data/val.xlsx")
    test_dataset = load_dataset("data/test.xlsx")

    # wrap into a DatasetDict
    dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})
    dataset.set_format("torch")

    return data_collator, dataset


def load_model_and_tokenizer(model_id: str) -> tuple[PeftModelForSequenceClassification, PreTrainedTokenizer]:
    """
    Load tokenizer and quantized model. Quantization and LoRA configs are largely based on this notebook:
    https://github.com/jkyamog/ml-experiments/blob/main/fine-tuning-qlora/LLAMA_3_Fine_Tuning_for_Sequence_Classification.ipynb
    """

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True, trust_remote_code=True)
    # ensure pad token is properly set
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    # use 4-bit quantization by replacing linear layers with 4-bit NormalFloat layers from bitsandbytes
    load_in_4_bit = True
    bnb_4bit_quant_type = "nf4"
    # set computation dtype to BFloat16 (see https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
    bnb_4bit_compute_dtype = torch.bfloat16
    # use nested quantization (quantize quantization constants from the first quantization)
    bnb_4bit_use_double_quant = True
    quant_config = BitsAndBytesConfig(load_in_4bit=load_in_4_bit, bnb_4bit_quant_type=bnb_4bit_quant_type,
                                      bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                                      bnb_4bit_use_double_quant=bnb_4bit_use_double_quant)

    # load the actual model and apply quantization configuration
    model = AutoModelForSequenceClassification.from_pretrained(model_id, quantization_config=quant_config, num_labels=3)
    # ensure pad token is properly set
    model.config.pad_token_id = tokenizer.pad_token_id
    # setup model for LoRA application
    model = prepare_model_for_kbit_training(model)

    # dimension of the LoRA attention (reduction factor / rank of the low-rank matrices)
    r = 16
    # target the linear layers to be replaced by LoRA fine-tuning
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    # use LoRA dropout to control for overfitting
    lora_dropout = 0.05
    # set task type to sequence classification
    task_type = 'SEQ_CLS'
    lora_config = LoraConfig(r=r, target_modules=target_modules, lora_dropout=lora_dropout, task_type=task_type)
    # apply LoRA configuration
    model = get_peft_model(model, lora_config)

    return model, tokenizer


def evaluate_model(model: PeftModelForSequenceClassification, dataset: Dataset, batch_size: int = 32) -> list[str]:
    """
    Run batched inference on the model and extract the class with the highest score.
    """

    # get ground truths (mapped back to class labels)
    true_labels = dataset["label"]
    mapping = {0: 'con', 1: 'neu', 2: 'pro'}
    true_labels = [mapping[int(p)] for p in true_labels]

    # get predictions (mapped back to class labels)
    predictions = []
    # split dataset row ids to batches of size batch_size
    chunked_ids = [range(dataset.num_rows)[i:i + batch_size] for i in range(0, dataset.num_rows, batch_size)]
    # iterate over batches in dataset
    for id_batch in chunked_ids:
        batch = dataset.select(id_batch)
        # move to GPU
        batch = {"input_ids": batch["input_ids"].to("cuda"), "attention_mask": batch["attention_mask"].to("cuda")}
        # put through model and append logits
        with torch.no_grad():
            predictions.append(model(**batch)['logits'])
    # select highest score class
    predictions = torch.cat(predictions, dim=0).argmax(axis=1).cpu().numpy()
    # map back to class label
    mapping = {0: 'con', 1: 'neu', 2: 'pro'}
    predictions = [mapping[p] for p in predictions]

    # create classification report
    report = str(classification_report(true_labels, predictions))
    # write to file
    os.makedirs("results", exist_ok=True)
    with open(f"results/{_model_name_.replace('/', '_')}.txt", mode="w", encoding="UTF-8") as w:
        w.write(report)

    return predictions


if __name__ == '__main__':

    # read huggingface access token from local file (omitted in repository)
    login(token=hf_token.token)

    for _model_name_ in ["mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.3",
                         "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3.1-8B"]:

        # load model
        model, tokenizer = load_model_and_tokenizer(_model_name_)

        # load data
        data_collator, tokenized_datasets = load_tokenized_dataset(tokenizer)

        # train model
        training_args = TrainingArguments(output_dir=f"models/{_model_name_.replace('/', '_')}",
                                          learning_rate=0.0001, per_device_train_batch_size=16, seed=1,
                                          per_device_eval_batch_size=16, num_train_epochs=3, weight_decay=0.01,
                                          eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True)
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, data_collator=data_collator,
                          train_dataset=tokenized_datasets['train'], eval_dataset=tokenized_datasets['validation'])
        trainer.train()

        # evaluate model
        evaluate_model(model, tokenized_datasets["test"])
