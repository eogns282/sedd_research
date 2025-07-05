import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from model import TransformerDenoisingModel
from diffusion import Diffusion

# --- Configuration ---
VOCAB_SIZE = 30522  # BERT's vocab size
D_MODEL = 256
MAX_SEQ_LEN = 32
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
NUM_TIMESTEPS = 100

# --- Sample Data ---
# In a real scenario, you would use a large text corpus.
# For this PoC, we'll use a few simple sentences.
sentences = [
    "hello world",
    "this is a test",
    "diffusion models are cool",
    "text generation with diffusion",
    "learning about language models"
]

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tokenization and Embedding ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_sentences = tokenizer(sentences, padding='max_length', max_length=MAX_SEQ_LEN, truncation=True, return_tensors='pt')
input_ids = tokenized_sentences['input_ids'].to(device)

# We need an embedding layer to convert token IDs to continuous vectors.
# This will be part of our main model, but we'll use a separate one here for preparing the initial data.
embedding_layer = nn.Embedding(VOCAB_SIZE, D_MODEL).to(device)
original_embeddings = embedding_layer(input_ids).detach()

# --- DataLoader ---
dataset = TensorDataset(original_embeddings)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model, Diffusion, and Optimizer ---
model = TransformerDenoisingModel(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=4, num_encoder_layers=2, dim_feedforward=1024).to(device)
diffusion = Diffusion(num_timesteps=NUM_TIMESTEPS, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# --- Training Loop ---
for epoch in range(NUM_EPOCHS):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        original_data = batch[0].to(device)
        
        # 1. Sample a random timestep
        t = torch.randint(0, NUM_TIMESTEPS, (original_data.shape[0],), device=device).long()
        
        # 2. Add noise to the data
        noisy_data, noise = diffusion.add_noise(original_data, t);
        
        # 3. Predict the noise
        predicted_noise = model(noisy_data, t)
        
        # 4. Calculate the loss
        loss = loss_fn(predicted_noise, noise)
        
        # 5. Backpropagate and update weights
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

print("Training finished.")

# --- Generation ---
@torch.no_grad()
def generate(model, diffusion, seq_len, d_model):
    model.eval()
    
    # Start with pure noise
    noisy_data = torch.randn(1, seq_len, d_model).to(device)
    
    for t in reversed(range(diffusion.num_timesteps)):
        timestep = torch.tensor([t], device=device)
        noisy_data = diffusion.remove_noise(model, noisy_data, timestep)
        
    return noisy_data

# Generate a new embedding
generated_embedding = generate(model, diffusion, MAX_SEQ_LEN, D_MODEL)

# --- Convert embedding back to text (simplified) ---
# In a real system, this is a complex task. We would typically use the
# embedding layer's weights to find the closest tokens.
def embedding_to_text(embedding, embedding_layer):
    # Calculate cosine similarity between the generated embedding and all word embeddings
    vocab_embeddings = embedding_layer.weight.data
    # Normalize for cosine similarity
    embedding_norm = embedding / embedding.norm(dim=-1, keepdim=True)
    vocab_embeddings_norm = vocab_embeddings / vocab_embeddings.norm(dim=-1, keepdim=True)
    
    similarities = torch.matmul(embedding_norm, vocab_embeddings_norm.transpose(0, 1))
    
    # Get the token IDs with the highest similarity
    best_token_ids = torch.argmax(similarities, dim=-1)
    
    return tokenizer.decode(best_token_ids.squeeze())

print("\n--- Generated Text ---")
generated_text = embedding_to_text(generated_embedding, embedding_layer)
print(generated_text)
