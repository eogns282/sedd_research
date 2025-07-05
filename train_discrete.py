import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from model import TransformerDenoisingModel
from discrete_diffusion import DiscreteDiffusion

# --- Configuration ---
VOCAB_SIZE = 30522  # BERT's vocab size
D_MODEL = 256
MAX_SEQ_LEN = 32
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200 # Increased epochs for better training
NUM_TIMESTEPS = 100

# --- Sample Data ---
sentences = [
    "hello world",
    "this is a test",
    "diffusion models are cool",
    "text generation with diffusion",
    "learning about language models"
]

# --- Tokenization ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_sentences = tokenizer(sentences, padding='max_length', max_length=MAX_SEQ_LEN, truncation=True, return_tensors='pt')
input_ids = tokenized_sentences['input_ids']

# --- DataLoader ---
dataset = TensorDataset(input_ids)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model, Diffusion, and Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerDenoisingModel(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=4, num_encoder_layers=2, dim_feedforward=1024, num_timesteps=NUM_TIMESTEPS).to(device)
diffusion = DiscreteDiffusion(num_timesteps=NUM_TIMESTEPS, mask_token_id=tokenizer.mask_token_id, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) # Ignore padding tokens in loss

# --- Training Loop ---
for epoch in range(NUM_EPOCHS):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        original_tokens = batch[0].to(device)
        
        # 1. Sample a random timestep
        t = torch.rand(original_tokens.shape[0], device=device) * (1 - 1e-5) + 1e-5
        
        # 2. Add noise to the data
        noisy_tokens, _ = diffusion.add_noise(original_tokens, t)
        
        # 3. Predict the original tokens
        timesteps = (t * NUM_TIMESTEPS).long()
        predicted_logits = model(noisy_tokens, timesteps)
        
        # 4. Calculate the loss
        loss = loss_fn(predicted_logits.view(-1, VOCAB_SIZE), original_tokens.view(-1))
        
        # 5. Backpropagate and update weights
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

print("Training finished.")

# --- Generation ---
@torch.no_grad()
def generate(model, diffusion, seq_len):
    model.eval()
    
    # Start with a sequence of [MASK] tokens
    noisy_tokens = torch.full((1, seq_len), diffusion.mask_token_id, device=device, dtype=torch.long)
    
    for i in reversed(range(diffusion.num_timesteps)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        noisy_tokens = diffusion.remove_noise(model, noisy_tokens, t)
        
    return noisy_tokens

# Generate a new sequence of tokens
generated_tokens = generate(model, diffusion, MAX_SEQ_LEN)

# --- Convert tokens back to text ---
generated_text = tokenizer.decode(generated_tokens.squeeze(), skip_special_tokens=True)
print("\n--- Generated Text ---")
print(generated_text)
