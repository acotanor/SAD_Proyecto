import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import random

# Configuración
MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# Dataset personalizado
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    losses = []
    correct = 0
    total = 0
    for batch in tqdm(data_loader, desc="Entrenando"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct += torch.sum(preds == labels)
        total += labels.size(0)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return np.mean(losses), correct.double() / total

def eval_model(model, data_loader):
    model.eval()
    losses = []
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluando"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            losses.append(loss.item())
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return np.mean(losses), predictions, true_labels

def main():
    # Cargar datos
    df = pd.read_csv("c:\\Users\\peiol\\OneDrive\\Escritorio\\Sad\\SAD_Proyecto\\tripadvisor_hotel_reviews.csv")
    df = df.dropna(subset=['Review', 'Rating'])
    df['Review'] = df['Review'].astype(str)
    # Etiquetas de 0 a 4
    df['Rating'] = df['Rating'].astype(int) - 1

    # División train/test
    from sklearn.model_selection import train_test_split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['Review'].values, df['Rating'].values, test_size=0.2, random_state=SEED, stratify=df['Rating'].values
    )

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    test_dataset = ReviewDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Entrenamiento
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}")

    # Evaluación
    test_loss, y_pred, y_true = eval_model(model, test_loader)
    print(f"\nTest loss: {test_loss:.4f}")

    # Métricas y matriz de confusión (de 1 a 5)
    y_pred = np.array(y_pred) + 1
    y_true = np.array(y_true) + 1
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # Guardar modelo y tokenizer
    model.save_pretrained("modelo_bert_multiclase")
    tokenizer.save_pretrained("modelo_bert_multiclase")
    print("Modelo y tokenizer guardados en la carpeta 'modelo_bert_multiclase'")

    # Cargar modelo y tokenizer guardados
    model = BertForSequenceClassification.from_pretrained("modelo_bert_multiclase")
    tokenizer = BertTokenizer.from_pretrained("modelo_bert_multiclase")

if __name__ == "__main__":
    main()