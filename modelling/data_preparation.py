import glob
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

dataset_file = "collected_books.txt"
data_path = "../data/all/"
# How many files to load.
file_number = 50

# Find all the files.
paths = glob.glob(data_path + "*.txt")

paths = paths[:file_number]
print(sorted(paths))

# Merge.
with open("data/" + dataset_file, "w") as output_file:
    for path in paths:
        for line in open(path, "r"):
            for split in line.split("\n"):
                split = split.strip()
                if split != "":
                    print(split, file=output_file)

# Done.
print("Corpus created.")

# Since the training is based on lines, the length of lines are adjusted, such that they most suitable to training.
dataset_file_lined = "collected_books_lined.txt"
new_line = ""
counter = 0
with open("data/"+dataset_file_lined, 'w') as f:
    for line in  open("data/"+dataset_file, 'r'):
        counter += 1
        #if counter < 400:
        #continue
        line = line.replace('..', '')
        for split in line.split("\n"):
            if bool(split.strip()):

                new_line = new_line + split
                if counter % 19 == 0:

                    new_line = new_line + "\n"
                    f.write(new_line)
                    new_line = ""
                counter += 1

dataset_file = 'collected_books_lined.txt'  #'new_file.txt'
raw_datasets = load_dataset("text", data_files=[dataset_file])


# Initialize tokenizer. Byte Pair encoding (BPE) ensures efficient data compression, because common words are represented
# as a single token  in the vocabulary, while rare words do not occupy a single token but are composed of sub words which
# in turn have a unique token.

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
#Include special tokens for unkown words and padding (which is needed to create even sized data points)
trainer = BpeTrainer(vocab_size=12000, special_tokens=["[UNK]", "[PAD]"])
# Split the text in actual words (Whitespace splits on space and punctuation)
tokenizer.pre_tokenizer = Whitespace()

def batch_iterator(batch_size=1000):
    for i in range(0, len(raw_datasets["train"]), batch_size):
        yield raw_datasets["train"][i:i + batch_size]["text"]
# Let the tokenizer train on the data set
tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
    length=len(raw_datasets["train"])
)
tokenizer.save("data/tokenizer.json")