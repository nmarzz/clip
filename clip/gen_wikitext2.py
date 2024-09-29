from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Model
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import random

# Custom dataset class
class EmbeddedDataset(Dataset):
    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]

# Step 1: Download the Wikitext-2 dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Step 2: Initialize the GPT-2 tokenizer and model
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Assign the eos_token as the pad_token to avoid padding error
tokenizer.pad_token = tokenizer.eos_token

# Load GPT-2 model for embeddings
model = GPT2Model.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

# Containers for embeddings and targets
all_embeddings = []
all_targets = []

# Step 3: Randomly select 50,000 non-empty samples
num_samples = 50000
random.seed(42)  # For reproducibility
total_samples = len(dataset['train'])

# Get unique random indices and ensure non-empty samples
selected_indices = random.sample(range(total_samples), total_samples)
selected_non_empty_indices = []

for idx in selected_indices:
    if len(dataset['train'][idx]['text'].strip()) > 0:  # Check if text is non-empty
        selected_non_empty_indices.append(idx)
    if len(selected_non_empty_indices) == num_samples:
        break

print(f"Selected {len(selected_non_empty_indices)} non-empty samples.")

# Step 4: Generate embeddings and targets
for i, idx in tqdm(enumerate(selected_non_empty_indices)):
    sample_text = dataset['train'][idx]['text']

    # Tokenize the text
    inputs = tokenizer(sample_text, padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Check if input_ids is empty (just to be safe)
    if inputs['input_ids'].numel() == 0:
        continue

    # Get embeddings using the model
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

    # Mean Pooling to get a single vector representation
    sentence_embedding = torch.mean(token_embeddings, dim=1)  # Shape: (1, 768)

    # Step 5: Define the target
    # For next-token prediction, let's take the first token of the next sequence
    if idx < len(dataset['train']) - 1:
        next_text = dataset['train'][idx + 1]['text']
        next_input_ids = tokenizer(next_text)['input_ids']
        if len(next_input_ids) > 0:
            target_token_id = next_input_ids[0]  # First token ID of the next text
        else:
            target_token_id = tokenizer.pad_token_id  # Default target if next sequence is empty
    else:
        target_token_id = tokenizer.pad_token_id  # Use pad_token_id as default for the last sample

    # Store the embedding and target
    all_embeddings.append(sentence_embedding.squeeze(0))  # Remove batch dimension
    all_targets.append(target_token_id)

    # Display progress
    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{num_samples} samples.")

# Step 6: Convert lists to tensors
all_embeddings = torch.stack(all_embeddings)  # Shape: (num_samples, 768)
all_targets = torch.tensor(all_targets)  # Shape: (num_samples,)

# Step 7: Create a dataset and dataloader
embedded_dataset = EmbeddedDataset(all_embeddings, all_targets)
dataloader = DataLoader(embedded_dataset, batch_size=32, shuffle=True)

# Display a few samples from the dataloader
for embeddings, targets in dataloader:
    print(f"Embeddings Shape: {embeddings.shape}, Targets: {targets[:10]}")
    break




import numpy as np
# Convert PyTorch tensors to NumPy arrays
embeddings_np = all_embeddings.numpy()
targets_np = all_targets.numpy()

# Save the arrays to .npy files
np.save('embeddings.npy', embeddings_np)
np.save('targets.npy', targets_np)

print("Embeddings and targets have been saved as 'embeddings.npy' and 'targets.npy' respectively.")