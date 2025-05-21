import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('kmeans_binary_labeled_helpfulness_data_with_modified_review.csv', index_col=False)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['modified_review'], padding='max_length', truncation=True, max_length=256)

# Split the dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(df['modified_review'], df['kmeans_label'], test_size=0.2, random_state=42)

# Convert labels to lists (if they're pandas Series)
train_labels = train_labels.tolist()
test_labels = test_labels.tolist()

# Tokenize the text data
train_encodings = tokenizer(list(train_texts), truncation=True, padding='max_length', max_length=256)
test_encodings = tokenizer(list(test_texts), truncation=True, padding='max_length', max_length=256)

# Convert to PyTorch datasets
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, train_labels)
test_dataset = ReviewDataset(test_encodings, test_labels)

# Step 3: Fine-tune BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
)


trainer.train()

predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(axis=-1)

# Evaluation report
print(classification_report(test_labels, preds))

