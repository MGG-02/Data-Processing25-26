import torch
from torch import nn
from torch.optim import Adam

#1 Hidden layer in order to create loss
class Word2VecClassifier(nn.Module):
    def __init__(self, input_dim=100, num_classes=3, dropout=0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout/2),

            nn.Linear(64, num_classes)
        ) 

    def forward(self, x):
        return self.net(x)
    
def train(model, train_dataloader, val_dataloader, learning_rate, epochs):

    # CPU / GPU processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr= learning_rate, weight_decay=1e-4)

    for epoch in range(epochs):

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

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {avg_train_loss:.3f} | Train Acc: {train_acc:.3f} |"
            f"Val Loss: {avg_val_loss:.3f} | Val Acc: {val_acc:.3f}"
        )

def evaluate(model, test_dataloader):

    # CPU / GPU processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_acc_test = 0
    total_examples_test = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_input = test_input.to(device)
            test_label = test_label.to(device)

            outputs = model(test_input)
            
            preds = outputs.argmax(dim=1)
            total_acc_test += (preds == test_label).sum().item()
            total_examples_test += test_label.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(test_label.cpu().numpy())

        test_acc = total_acc_test / total_examples_test
        print(f"Test Accuracy: {test_acc:.3f}")            
