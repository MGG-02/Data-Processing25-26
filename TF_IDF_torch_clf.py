import torch
import numpy as np
from torch import nn
from tqdm import trange, tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch.optim import Adam

#1 Hidden layer in order to create loss
class tfidfClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=3, dropout=0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        ) 

    def forward(self, x):
        return self.net(x)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

def train(model, train_dataloader, val_dataloader, learning_rate, epochs):

    pbar = trange(epochs, desc="Training")
    # CPU / GPU processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr= learning_rate, weight_decay=1e-4)

    for epoch in pbar:

        # --- Train ---
        model.train()
        total_acc_train = 0
        total_loss_train = 0
        total_examples_train = 0

        for train_input, train_label in train_dataloader:

            train_label = train_label.to(device)
            train_input = train_input.to(device)

            optimizer.zero_grad()
            outputs = model(train_input)

            loss = criterion(outputs, train_label)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            total_acc_train += (outputs.argmax(dim=1) == train_label).sum().item()
            total_examples_train += train_label.size(0)

        avg_train_loss = total_loss_train / len(train_dataloader)
        train_acc = total_acc_train / total_examples_train

        # --- Valuation ---
        model.eval()
        total_loss_val = 0.0
        total_acc_val = 0.0
        total_examples_val = 0.0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_input = val_input.to(device)
                val_label = val_label.to(device)

                outputs = model(val_input)
                loss = criterion(outputs, val_label)

                total_loss_val += loss.item()
                total_acc_val += (outputs.argmax(dim=1) == val_label).sum().item()
                total_examples_val += val_label.size(0)

        avg_val_loss = total_loss_val / len(val_dataloader)
        val_acc = total_acc_val / total_examples_val

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        pbar.set_postfix({
            "TrainLoss": f"{avg_train_loss:.3f}",
            "TrainAcc": f"{train_acc:.3f}",
            "ValLoss": f"{avg_val_loss:.3f}",
            "ValAcc": f"{val_acc:.3f}",
        })


def evaluate(model, test_dataloader):

    # CPU / GPU processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_input = test_input.to(device)
            test_label = test_label.to(device)

            outputs = model(test_input)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(test_label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())


    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    print(f'Test Accuracy: {accuracy_score(all_labels, all_preds):.4f}')
    print(f'Test Roc Auc Score: {roc_auc_score(all_labels, all_probs, multi_class='ovr')}')
    print(f'Test F1 Score: {f1_score(all_labels, all_preds, average="weighted"):.4f}')
