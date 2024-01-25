import torch
import copy
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

def prepare_tokenizer_and_model(num_labels):
    tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=num_labels)
    return tokenizer, model

def tokenize_data(tokenizer, texts):
    return tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

def convert_intents_to_tensor(y, intent_to_num):
    intents_num = [intent_to_num[intent] for intent in y.tolist()]
    return torch.tensor(intents_num)

def create_data_loaders(train_encodings, test_encodings, train_intents_tensor, test_intents_tensor):
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_intents_tensor)
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_intents_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    return train_dataloader, test_dataloader

def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return precision, recall, f1

def train_model(model, train_dataloader, test_dataloader, device):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 100
    model.to(device)

    # Early stopping parameters for F1 score
    f1_patience = 5
    last_f1 = None
    f1_patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)

        # Doğrulama kaybı ve metrikleri hesapla
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        val_loss /= len(test_dataloader)
        precision, recall, f1 = calculate_metrics(all_labels, all_preds)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        # F1 score için early stopping kontrolü
        if f1 == last_f1:
            f1_patience_counter += 1
            if f1_patience_counter >= f1_patience:
                print("Early stopping triggered due to constant F1 score. Stopping training.")
                break
        else:
            f1_patience_counter = 0
        last_f1 = f1

    return model


def evaluate_model(model, test_dataloader, device, intent_to_num, threshold=0.5):
    model.eval()
    correct = 0
    total = 0
    predicted_intents = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            softmax_scores = torch.softmax(logits, dim=1)
            max_scores, predicted_labels = torch.max(softmax_scores, dim=1)
            
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

            for score, label in zip(max_scores, predicted_labels):
                if score.item() < threshold:
                    predicted_intents.append("DontUnderstand")
                else:
                    intent = [key for key, value in intent_to_num.items() if value == label.item()][0]
                    predicted_intents.append(intent)

    accuracy = correct / total
    return accuracy, predicted_intents