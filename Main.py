import torch.nn as nn
from torch.optim import Adam
import lightning as L
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
import torch

from pathlib import Path
from PIL import Image
import os


# Check for GPU availability
print("PyTorch Version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU found")
print("Number CUDA Devices:", torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training parameters
epochs = 7 #Epochs
lr = 1e-4 #learning rate
dataSize = 1000 #Training Data Size
max_len = 40
batch_size = 32

num_layers = 12 #number transformer


# ====================================================
# 1. Load and freeze pretrained tokenizer and embedding
# ====================================================
# The BART model is only used here to get embeddings, not for training.

print("Load Tokenizer and Modell...")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# freeze modell
for param in model.parameters():
    param.requires_grad = False

# Padding-Token (BART use <pad> default)
pad_token_id = tokenizer.pad_token_id

SEP = tokenizer.eos_token_id


# ====================================================
# 2. Load and process DailyDialog dataset
# ====================================================
# We extract dialogue turns into (question, answer) pairs
# and shuffle the data to randomize training order.

print("Load DailyDialog Dataset...")
dataset = load_dataset("OpenRL/daily_dialog")

def extract_qa_pairs(batch):
    questions, answers = [], []
    for dialog in batch["dialog"]:
        utterances = [u.strip() for u in dialog if isinstance(u, str) and u.strip()]
        for i in range(len(utterances) - 1):
            questions.append(utterances[i])
            answers.append(utterances[i + 1])
    return {"question": questions, "answer": answers}

# Splitt DailyDialog
split_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=True)
train_dialogs = split_dataset["train"]
test_dialogs = split_dataset["test"]

# extract QA-Pairs
train_qa = train_dialogs.map(
    extract_qa_pairs,
    batched=True,
    remove_columns=train_dialogs.column_names
)

test_qa = test_dialogs.map(
    extract_qa_pairs,
    batched=True,
    remove_columns=test_dialogs.column_names
)

train_qa = train_qa.select(range(min(len(train_qa), dataSize)))
test_qa = test_qa.select(range(min(len(test_qa), int(dataSize * 0.1))))

qa_dataset = train_qa


# ====================================================
# 3. Custom PyTorch Dataset class
# ====================================================
# Each item contains:
# - input_ids and attention_mask from the tokenizer
# - labels (same as input_ids, but with question tokens masked as -100)

# Dataset-Class for BART
class ChatDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_len):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        sep_token = self.tokenizer.eos_token  # </s>
        bos_token = self.tokenizer.bos_token  # <s>

        if bos_token is None:
            bos_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.bos_token_id)

        combined = f"{bos_token} {question.strip()} {sep_token} {answer.strip()}"

        encoded = self.tokenizer(
            combined,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        sep_index = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]

        if len(sep_index) > 0:
            cut = sep_index[0] + 1
            labels[:cut] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


print("Change Dataset to Tensor-Dataset...")

tensor_dataset = ChatDataset(
    questions=qa_dataset["question"],
    answers=qa_dataset["answer"],
    tokenizer=tokenizer,
    max_len=max_len
)

# Train/Test Split
train_size = int(0.9 * len(tensor_dataset))
test_size = len(tensor_dataset) - train_size
train_dataset, test_dataset = random_split(tensor_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train Size: {len(train_dataset)}")
print(f"Test Size: {len(test_dataset)}")


# ====================================================
# 4. Positional Encoding for token order awareness
# ====================================================
# Adds sinusoidal positional information to token embeddings
# so that the model can learn the position of tokens.

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.pe.size(0)} of PositionEncoding.")
        return x + self.pe[:seq_len, :].unsqueeze(0)

class Attention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output

losses = []


# ====================================================
# 5. Decoder-only Transformer model
# ====================================================
# This model mimics GPT-style architecture (no encoder),
# consisting of an embedding layer, positional encoding,
# several Transformer decoder layers, and a final output layer.

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention + Residual
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout_attn(attn_output))

        # Feedforward + Residual
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout_ffn(ff_output))
        return x


class DecoderOnlyTransformer(L.LightningModule):
    def __init__(self, num_tokens, d_model, max_len, num_layers=num_layers):
        super().__init__()
        ff_hidden_dim = d_model * 2

        vocab_size = tokenizer.vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoding = PositionEncoding(d_model=d_model, max_len=max_len)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_tokens, ff_hidden_dim) for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(d_model, tokenizer.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, token_ids):

        with torch.no_grad():
            x = self.token_embedding(token_ids)

        x = self.pos_encoding(x)

        seq_len = token_ids.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)

        for block in self.transformer_blocks:
            x = block(x, mask)

        logits = self.lm_head(x)
        return logits


# ====================================================
# 6. Train new model
# ====================================================

def train_and_validate(train_loader, test_loader, optimizer, loss_fn, device, epochs, scheduler):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"\n--- Epoche {epoch + 1} ---")
        model.train()
        total_train_loss = 0
        train_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):

            inputs = batch['input_ids'].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

            if batch_idx % 100 == 0:
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_first = predicted_ids[0]

                try:
                    sep_indices = (inputs[0] == SEP).nonzero(as_tuple=True)[0]
                    sep_index = sep_indices[0].item() if sep_indices.numel() > 0 else -1
                except Exception as e:
                    print("Fehler beim Finden von </s>:", e)
                    sep_index = -1

                answer_predicted_ids = predicted_first[sep_index + 1:] if sep_index >= 0 else predicted_first

                # Convert IDs to readable Tokens
                answer_tokens = tokenizer.convert_ids_to_tokens(answer_predicted_ids.tolist(), skip_special_tokens=True)
                input_tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist(), skip_special_tokens=True)
                #label_tokens = tokenizer.convert_ids_to_tokens(labels[0].tolist(), skip_special_tokens=True)

                # Replace negative IDs (e.g. -100) with tokenizer.pad_token_id or 0
                clean_label_ids = [id if id >= 0 else tokenizer.pad_token_id for id in labels[0].tolist()]

                label_tokens = tokenizer.convert_ids_to_tokens(clean_label_ids, skip_special_tokens=True)

                # Output for controll
                print()
                print("Input:", tokenizer.convert_tokens_to_string(input_tokens))
                print("Label:", tokenizer.convert_tokens_to_string(label_tokens))
                print("Predicted:", tokenizer.convert_tokens_to_string(answer_tokens))

        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)

        # --- VALIDATION ---
        model.eval()
        total_val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                logits = model(inputs)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_val_loss += loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / val_batches
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"\nEpoch abgeschlossen. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses


# ====================================================
# MAIN WORKFLOW: Model Initialization, Training, and Saving
# ====================================================

# Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Create modell
num_tokens = len(tokenizer)
d_model = model.config.hidden_size
model = DecoderOnlyTransformer(num_tokens=num_tokens, d_model=d_model, max_len=max_len)

choice = ("n")
if os.path.exists(MODEL_SAVE_PATH):
    choice = input(f"Model '{MODEL_NAME}' found. Load? (j/n): ").lower()
else:
    print("No saved model found. New model creating ...")

if choice == "j":
    print("Model is loading ...")
    # Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
    model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    model.to(device)
else:
    print("Create new model ...")

    model = model.to(device)

    #Show architecture
    summary(model, input_size=(1, 30), dtypes=[torch.long],
            col_names=["input_size", "output_size", "num_params", "trainable"])

    # Define optimizer, learning rate scheduler, and loss function
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    train_losses, val_losses = train_and_validate(train_loader, test_loader, optimizer, loss_fn, device, epochs, scheduler)

    # Save trained model weights
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),  # only saving the state_dict() only saves the models learned parameters
               f=MODEL_SAVE_PATH)

    # Plot and display training/validation loss
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("loss_plot.png")

Image.open("loss_plot.png").show()


# ====================================================
# 7. Evaluate with example prompt
# ====================================================
# After training is finished, the model can be used to generate
# responses

def evaluate_model(model, test_loader, max_samples=5):
    model.eval()
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            logits = model(inputs)
            predicted_ids = torch.argmax(logits, dim=-1)

            for i in range(inputs.size(0)):
                if count >= max_samples:
                    return  # Just show max_samples

                input_ids_i = inputs[i].tolist()
                label_ids_i = labels[i].tolist()
                pred_ids_i = predicted_ids[i].tolist()

                # Search SEP-Token
                try:
                    sep_indices = [idx for idx, token_id in enumerate(input_ids_i) if token_id == tokenizer.sep_token_id]
                    sep_index = sep_indices[0] if sep_indices else -1
                except:
                    sep_index = -1

                # Extract question (to SEP-Token)
                if sep_index >= 0:
                    input_question = tokenizer.decode(input_ids_i[:sep_index], skip_special_tokens=True)
                else:
                    input_question = tokenizer.decode(input_ids_i, skip_special_tokens=True)

                # Substitute error labels with PAD (e.g. -100 CrossEntropyLoss ignores)
                clean_label_ids = [id if id >= 0 else tokenizer.pad_token_id for id in label_ids_i]

                # Decode answers
                label_text = tokenizer.decode(clean_label_ids, skip_special_tokens=True)
                pred_text = tokenizer.decode(pred_ids_i, skip_special_tokens=True)

                print("\n" + "-" * 50)
                print("Question:                  ", input_question)
                print("Right answer:       ", label_text)
                print("Predicted answer:  ", pred_text)
                print("-" * 50)

                count += 1


print("\n--- Test model ---")
evaluate_model(model, test_loader, max_samples=dataSize)


# ====================================================
# 8. Generate response (inference)
# ====================================================
# generate responses for user input.

def interactive_qa_loop(model, tokenizer, device):
    model.eval()
    print("Write 'exit' to close the program.\n")

    while True:
        question = input("Please enter a question: ")
        if question.lower() == "exit":
            print("Closing program.")
            break

        inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            logits = model(inputs.input_ids)
            predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()

        pred_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)

        print("\n" + "-" * 50)
        print("Question:                 ", question)
        print("Predicted answer: ", pred_text)
        print("-" * 50)

        input_text = f"{tokenizer.cls_token} {question} {tokenizer.sep_token}"

        encoded = tokenizer(
            input_text,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors="pt",
            add_special_tokens=True  # Default mostly True
        )
        print(tokenizer.convert_ids_to_tokens(encoded['input_ids'][0]))


interactive_qa_loop(model, tokenizer, device)