import torch
from torch import nn
from torch.optim import Adam

#1 hidden layer in order to decrease loss
class BertClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()

        self.model = nn.Sequential(
            nn.LayerNorm(768),          # Normalize embeddings from BERT
            nn.Linear(768, 512),
            nn.GELU(),                  # better than ReLU on transformer outputs
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes) # raw logits
        )

    def forward(self, x):
        return self.model(x)



def train(model, train_dataloader, val_dataloader, learning_rate, epochs):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr= learning_rate, weight_decay=1e-4)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
            print('Model to CUDA')

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in train_dataloader:

                train_label = train_label.to(device)
                inputs = train_input.to(device)

                output = model(inputs)
                
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    inputs = val_input.to(device)

                    output = model(inputs)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            avg_train_loss = total_loss_train / len(train_dataloader)
            train_accuracy = total_acc_train / len(train_dataloader.dataset)

            avg_val_loss = total_loss_val / len(val_dataloader)
            val_accuracy = total_acc_val / len(val_dataloader.dataset)

            print(
                f"Epoch {epoch_num + 1} | "
                f"Train Loss: {avg_train_loss:.3f} | "
                f"Train Acc: {train_accuracy:.3f} | "
                f"Val Loss: {avg_val_loss:.3f} | "
                f"Val Acc: {val_accuracy:.3f}"
            )

            
def evaluate(model, test_dataloader):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              input = test_input.to(device)

              output = model(input)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(test_dataloader): .3f}')