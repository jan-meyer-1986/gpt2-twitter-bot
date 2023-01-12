import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import TrainingArguments, Trainer

tf.config.list_physical_devices("GPU")

verbose = False

dataset_file = 'collected_books_lined.txt'  #'new_file.txt'
raw_datasets = load_dataset("text", data_files=[dataset_file])

# Load the tokenizer, trained in data_preparation
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../data/tokenizer.json")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

if verbose:
    print(tokenizer.vocab_size)
    print(tokenizer.vocab)

    token_sequence = raw_datasets["train"][3]["text"]
    print(token_sequence)
    token_indices = tokenizer(token_sequence)["input_ids"]
    print(token_indices)
    tokens = [tokenizer.decode([index]) for index in token_indices]
    print(tokens)

    lengths = []
    for token_sequence in tqdm.tqdm(raw_datasets["train"]):
        token_sequence = token_sequence["text"]
        token_indices = tokenizer(token_sequence)["input_ids"]
        lengths += [len(token_indices)]

    plt.hist(lengths, bins=50)
    plt.show()

sequence_length = 300

def tokenize_function(example):
    tokenized_example = tokenizer(
        example["text"],
        truncation=True,
        padding=False,
        max_length=sequence_length
    )
    return {
        "input_ids": tokenized_example["input_ids"]
    }

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

if verbose:
    token_sequence = raw_datasets["train"][0]
    print(token_sequence)

    tokenized = tokenize_function(token_sequence)
    print(tokenized)

    tokenized = tokenized_datasets["train"][0]
    print(tokenized)

# Data collator ctreated the batches and pads sequences that are shorter than the sequence length
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

model_config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    n_ctx=sequence_length,
    n_positions=sequence_length,
    n_head=8,
    n_layer=12,
    n_embd=512
)

model = GPT2LMHeadModel(model_config)

if verbose:
    # An example datapoint is sent through the untrained model. As expected, the produced logits and activations are random
    inputs = [tokenized_datasets["train"][2]]
    inputs = data_collator(inputs)
    assert list(inputs.keys()) == ["input_ids", "attention_mask", "labels"], list(inputs.keys())
    print("input_ids:", inputs["input_ids"])
    print("")

    outputs = model(**inputs)
    assert list(outputs.keys()) == ["loss", "logits", "past_key_values"], list(outputs.keys())
    print("logits:", outputs["logits"])

    plt.plot(outputs["logits"].detach().numpy()[0][0])
    plt.title("Logits")
    plt.show()
    plt.close()

    activations = torch.nn.functional.softmax(outputs["logits"], dim=-1)
    plt.plot(activations.detach().numpy()[0][0])
    plt.title("Activations")
    plt.show()
    plt.close()


training_args = TrainingArguments(
    output_dir="../model",
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=64,
    prediction_loss_only=False
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"]
)

# Let the model train
trainer.train()
tokenizer.save_pretrained("model")
model.save_pretrained("model")
path = "trained_model.pt"
torch.save(model.state_dict(), path)