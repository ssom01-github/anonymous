import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import os
import numpy as np

# Paths
embedding_path = './embeddings/train_embeddings.pt'
test_embedding_path = './embeddings/test_embeddings.pt'
# test_embedding_path = './embeddings/xquad_embeddings.pt'
# test_embedding_path = './embeddings/llama3_embeddings.pt'
# test_embedding_path = './embeddings/gpt_embeddings.pt'
kl_path = './kl_divergence/train_kl_scores.pt'

# Load Data
embeddings = torch.load(embedding_path)
kl_scores = torch.load(kl_path)
test_embeddings = torch.load(test_embedding_path)

# Identify Layer with Max KL
# max_kl_layers = kl_scores.argmax(dim=1)

# Use last layer for now
max_kl_layers = torch.tensor([kl_scores.size(1) - 1] * len(embeddings))

# Prepare Data
X, y = [], []
for (embedding, label), max_layer in zip(embeddings, max_kl_layers):
    selected_layer_embedding = embedding[max_layer].mean(dim=0)  # Mean pooling over sequence length
    X.append(selected_layer_embedding.numpy())
    y.append(label)

X = np.array(X)
y = np.array(y)
X_train = X
y_train = y

# similarly for test data
X_test, y_test = [], []
for (embedding, label), max_layer in zip(test_embeddings, max_kl_layers):
    selected_layer_embedding = embedding[max_layer].mean(dim=0)  # Mean pooling over sequence length
    X_test.append(selected_layer_embedding.numpy())
    y_test.append(label)

# Binary Classifier with configurable hidden layers
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[1024, 512], num_labels=1, dropout_prob=0.2):
        super(BinaryClassifier, self).__init__()
        self.num_labels = num_labels
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Dropout(dropout_prob),
                nn.Linear(prev_size, hidden_size),
                nn.Tanh(),
            ])
            prev_size = hidden_size
            
        self.dense = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_labels)
        
    def forward(self, x):
        x = self.dense(x)
        x = self.classifier(x)
        return torch.sigmoid(x)  # Added sigmoid for binary classification

# Initialize model with the embedding dimension
model = BinaryClassifier(
    input_size=X_train.shape[1],
    hidden_sizes=[1024, 512],  # Two hidden layers
    num_labels=1,              # Binary classification
    dropout_prob=0.2           # Moderate dropout
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Training
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

for epoch in range(80):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor).squeeze()
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluation
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred_probs = model(X_test_tensor).squeeze().cpu().numpy()
    y_pred = (y_pred_probs > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUROC Score:", roc_auc_score(y_test, y_pred_probs))

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Print the max layer, kl and total number of layers
print("Max Layer:", max_kl_layers[0].item())
print("Max KL:", kl_scores[0, max_kl_layers[0]].item())
print("Total Number of Layers:", kl_scores.size(1))

# Save Model
torch.save(model.state_dict(), './classifier_model_last_layer.pth')
print("Model saved successfully!")