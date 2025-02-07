import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data import Data
from torch.utils.data import Dataset
from scipy.io import loadmat


# ✅ CNN Model Definition
class CNNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, kernel_size0=5, kernel_size1=2, padding=2):
        super().__init__()

        # First Convolutional Block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=kernel_size0, padding=padding),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size1)
        )

        # Second Convolutional Block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size0, padding=padding),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size1)
        )

        # Compute input size for fully connected layer dynamically
        dummy_input = torch.randn(1, 1, 10, 10)  
        dummy_output = self.conv_block2(self.conv_block1(dummy_input))
        fc_input_size = dummy_output.view(1, -1).shape[1]

        # Fully Connected Layer
        self.classifier = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    

# ✅ MLP Model Definition
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers=2, num_classes=6):
        super().__init__()

        # Input Layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU()
        )

        # Hidden Layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers - 1)
        ])

        # Compute input size for fully connected layer dynamically
        dummy_input = torch.randn(1, input_size)
        dummy_output = self.input_layer(dummy_input)
        for layer in self.hidden_layers:
            dummy_output = layer(dummy_output)
        fc_input_size = dummy_output.shape[1]

        # Output Layer
        self.output_layer = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.output_layer(x)
        return x


# ✅ Model Training Function
def train_model(model, model_name, epochs, criterion, optimizer, train_loader, device):
    """Trains the CNN or MLP model."""
    print(f"Training {model_name} model...")
    accuracy_history = []
    loss_history = []

    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            labels = labels.to(device)
            inputs = inputs.view(inputs.size(0), 1, inputs.size(1), inputs.size(2)).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Compute accuracy
            predictions = torch.argmax(outputs.data, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            running_loss += loss.item()

        # Calculate and store accuracy and loss for this epoch
        epoch_accuracy = 100 * correct_predictions / total_samples
        epoch_loss = running_loss / len(train_loader)

        accuracy_history.append(epoch_accuracy)
        loss_history.append(epoch_loss)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return accuracy_history, loss_history




# ✅ Model Evaluation Function
def evaluate_model(model, test_loader, device):
    """Evaluates the CNN or MLP model and returns predictions and scores."""

    y_pred, y_true, y_scores = [], [], []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(device)
            images = images.unsqueeze(1).to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            # Store predictions and scores
            predictions = torch.argmax(outputs, dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_scores.extend(probabilities.cpu().numpy())

            y_true.extend(labels.cpu().tolist())

    return y_true, y_pred, y_scores



## GNN Model Cores

# ✅ GCN Graph Construction
def create_graph(adj_matrix, features, label):
    """
    Constructs a graph representation using adjacency matrix and node features.

    Args:
        adj_matrix (Tensor): Adjacency matrix.
        features (Tensor): Node features.
        label (int): Graph label.

    Returns:
        Data: PyTorch Geometric graph object.
    """
    edge_index = torch.nonzero(adj_matrix, as_tuple=False).T  
    edge_attr = adj_matrix[edge_index[0], edge_index[1]].clone().detach().float()
    x = features.clone().detach().float()  
    y = torch.tensor([label], dtype=torch.long)  

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# ✅ Dataset Handling for GNNs
class GraphDataset(Dataset):
    def __init__(self, file_list, features):
        """
        Handles loading of EEG-based graph datasets.

        Args:
            file_list (list): List of (file_path, class_name) tuples.
            features (dict): Dictionary mapping class names to feature paths.
        """
        self.file_list = file_list
        self.features = features
        class_names = sorted(list({label for _, label in file_list}))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.graphs, self.labels = self._create_graphs()

    def _create_graphs(self):
        """Precompute all graphs and store them in a list."""
        graphs, labels = [], []

        for file_path, class_name in self.file_list:
            label = self.class_to_idx[class_name]
            labels.append(label)

            # Load matrix from .mat file
            mat_data = loadmat(file_path)
            matrix = mat_data.get("out_data")  
            if matrix is None:
                raise ValueError(f"Missing 'out_data' key in {file_path}")

            matrix = torch.tensor(matrix, dtype=torch.float32)

            # Extract subject ID and epoch index
            subject_ID = file_path.split("/")[-1][:10]
            epoch_idx = file_path.split("-")[-1].split(".")[0]

            # Find corresponding PSD feature file
            PSD_path_list = self.features[class_name]
            matched_files = [item for item in PSD_path_list if subject_ID in item and epoch_idx in item]
            if not matched_files:
                raise ValueError(f"No matching PSD file for subject {subject_ID}, epoch {epoch_idx}")

            PSD_mat_file_path = matched_files[0]

            # Load feature matrix
            feature_data = loadmat(PSD_mat_file_path)
            feature = feature_data.get("PSD_mat")  
            if feature is None:
                raise ValueError(f"Missing 'PSD_mat' key in {PSD_mat_file_path}")

            feature = torch.tensor(feature, dtype=torch.float32)

            # Create graph
            graph = create_graph(matrix, feature, label)
            graphs.append(graph)

        return graphs, labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


# ✅ GCN Model Definition
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, num_layers=2, dropout=0.5, drop_edge=0.2):
        """
        Graph Convolutional Network (GCN) for EEG-based classification.

        Args:
            in_channels (int): Number of input node features.
            hidden_dim (int): Hidden layer dimension.
            num_classes (int): Number of output classes.
            num_layers (int): Number of GCN layers.
            dropout (float): Dropout probability.
            drop_edge (float): Edge dropout probability.
        """
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.drop_edge = drop_edge  
        self.dropout = dropout  

        # First Graph Convolutional Layer
        self.convs.append(GCNConv(in_channels, hidden_dim))

        # Additional Graph Convolutional Layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Fully Connected Layers for Classification
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data):
        """
        Forward pass of the GCN model.

        Args:
            data: Input graph data.

        Returns:
            Log-softmax output for classification.
        """
        x, edge_index = data.x, data.edge_index  

        # Apply GCN layers with ReLU activation and edge dropout
        for conv in self.convs:
            edge_index = self._apply_edge_dropout(edge_index)  
            x = conv(x, edge_index)
            x = F.relu(x)

        # Global Max Pooling for graph-level representation
        x = global_max_pool(x, data.batch)

        # Fully connected layers with dropout
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)  

    def _apply_edge_dropout(self, edge_index):
        """
        Applies dropout to edges to prevent overfitting.

        Args:
            edge_index (Tensor): Edge indices.

        Returns:
            Tensor: Edge indices with dropout applied.
        """
        if self.training and self.drop_edge > 0:
            num_edges = edge_index.size(1)
            mask = torch.rand(num_edges, device=edge_index.device) > self.drop_edge
            edge_index = edge_index[:, mask]
        return edge_index


# ✅ GNN Training Function
def train_gnn(model, train_loader, optimizer, criterion, epochs, device):
    """
    Trains the GCN model.

    Args:
        model: GCN model.
        train_loader: DataLoader for training data.
        optimizer: Optimizer.
        criterion: Loss function.
        epochs: Number of epochs.
        device: CPU/GPU.

    Returns:
        loss_history: List of loss values per epoch.
        acc_history: List of accuracy values per epoch.
    """
    loss_history = []
    acc_history = []

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct, total = 0, 0

        for data in train_loader:
            data = data.to(device)  
            optimizer.zero_grad()  
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == data.y).sum().item()
            total += data.y.size(0)

        loss_history.append(total_loss / len(train_loader))
        acc_history.append(100 * correct / total)

        print(f"Epoch {epoch+1}: Loss = {loss_history[-1]:.4f}, Accuracy = {acc_history[-1]:.2f}%")

    return acc_history, loss_history


# ✅ GNN Evaluation Function
def evaluate_gnn(model, test_loader, device):
    """
    Evaluates the GCN model.

    Args:
        model: Trained GCN model.
        test_loader: DataLoader for test data.
        device: CPU/GPU.

    Returns:
        y_true, y_pred, y_scores.
    """
    model.eval()
    y_pred, y_true, y_scores = [], [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)

            preds = torch.argmax(outputs, dim=1)
            y_true.extend(data.y.cpu().tolist())  
            y_pred.extend(preds.cpu().tolist())  
            y_scores.extend(probabilities.cpu().numpy())  

    return y_true, y_pred, y_scores
