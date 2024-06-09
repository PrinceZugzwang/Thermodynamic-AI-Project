import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append('Spectral-Normalized-Gaussian-Process')

from sngp_classification_layer import SNGP
from gaussian_process import mean_field_logits

import matplotlib.colors as colors

import sklearn.datasets
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve



from tqdm import tqdm
import pandas as pd

plt.rcParams['figure.dpi'] = 90

DEFAULT_X_RANGE = (-3.5, 3.5)
DEFAULT_Y_RANGE = (-2.5, 2.5)
DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
DEFAULT_NORM = colors.Normalize(vmin=0, vmax=1,)
DEFAULT_N_GRID = 100

def make_training_data(sample_size=500):
    """Create two moon training dataset."""
    train_examples, train_labels = sklearn.datasets.make_moons(
        n_samples=2 * sample_size, noise=0.1
    )

    # Adjust data position slightly.
    train_examples[train_labels == 0] += [-0.1, 0.2]
    train_examples[train_labels == 1] += [0.1, -0.2]

    return train_examples, train_labels

def make_testing_data(x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE, n_grid=DEFAULT_N_GRID):
    """Create a mesh grid in 2D space."""
    # testing data (mesh grid over data space)
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv.flatten(), yv.flatten()], axis=-1)

def make_ood_data(sample_size=500, means=(2.5, -1.75), vars=(0.01, 0.01)):
    return np.random.multivariate_normal(
      means, cov=np.diag(vars), size=sample_size)

def plot_uncertainty_surface(test_uncertainty, ax, cmap=None, show_data=True):
    """Visualizes the 2D uncertainty surface.

    For simplicity, assume these objects already exist in the memory:

    test_examples: Array of test examples, shape (num_test, 2).
    train_labels: Array of train labels, shape (num_train, ).
    train_examples: Array of train examples, shape (num_train, 2).

    Arguments:
    test_uncertainty: Array of uncertainty scores, shape (num_test,).
    ax: A matplotlib Axes object that specifies a matplotlib figure.
    cmap: A matplotlib colormap object specifying the palette of the 
      predictive surface.

    Returns:
    pcm: A matplotlib PathCollection object that contains the palette 
      information of the uncertainty plot.
    """
    # Normalize uncertainty for better visualization.
    test_uncertainty = test_uncertainty / np.max(test_uncertainty)

    # Set view limits.
    ax.set_ylim(DEFAULT_Y_RANGE)
    ax.set_xlim(DEFAULT_X_RANGE)

    # Plot normalized uncertainty surface.
    pcm = ax.imshow(
      np.reshape(test_uncertainty, [DEFAULT_N_GRID, DEFAULT_N_GRID]), 
      cmap=cmap,
      origin="lower",
      extent=DEFAULT_X_RANGE + DEFAULT_Y_RANGE,
      vmin=DEFAULT_NORM.vmin,
      vmax=DEFAULT_NORM.vmax,
      interpolation='bicubic', 
      aspect='auto')

    # Plot training data.
    if show_data:
        ax.scatter(train_examples[:, 0], train_examples[:, 1],
                 c=train_labels, cmap=DEFAULT_CMAP, alpha=0.5)
        ax.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

    return pcm
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X, self.y = X.astype(np.float32), y.astype(np.int64)  # Convert labels to long type
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

    
class BaselineModel(nn.Module):
    def __init__(self, D=2, C=2, H=128, p=0.1, n_hidden=4):
        super().__init__()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=p)
        self.register_buffer('input_W', torch.randn(D, H))
        self.register_buffer('input_b', torch.randn(H))
        self.fcs = nn.ModuleList([nn.Linear(H, H) for _ in range(n_hidden)])
        self.output_layer = nn.Linear(H, C)
        
    def forward(self, x, masks=None):
        x = x @ self.input_W + self.input_b
        for i in range(len(self.fcs)):
            if masks is not None:
                x = x + self.act(self.fcs[i](x)) * masks[i]
            x = x + self.dropout(self.act(self.fcs[i](x)))
        return self.output_layer(x)
# Load the train, test and OOD datasets.
train_examples, train_labels = make_training_data(
    sample_size=500
)
test_examples = make_testing_data()
ood_examples = make_ood_data(sample_size=500)

# Visualize
pos_examples = train_examples[train_labels == 0]
neg_examples = train_examples[train_labels == 1]

plt.figure(figsize=(7, 5.5))

plt.scatter(pos_examples[:, 0], pos_examples[:, 1], c="#377eb8", alpha=0.5)
plt.scatter(neg_examples[:, 0], neg_examples[:, 1], c="#ff7f00", alpha=0.5)
plt.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

plt.legend(["Postive", "Negative", "Out-of-Domain"])

plt.ylim(DEFAULT_Y_RANGE)
plt.xlim(DEFAULT_X_RANGE)
plt.title('Make moons dataset')

plt.show()

model = BaselineModel()

optim = torch.optim.Adam(model.parameters())
loader = torch.utils.data.DataLoader(Dataset(train_examples, train_labels), batch_size=64)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(1, 100+1):
    c, running_loss = 1, 0
    for x, y in loader:
        optim.zero_grad()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optim.step()
        running_loss += float(loss)
        c += 1
    if not epoch % 10:
        print(f"Epoch {epoch}. Loss: {running_loss / c}")
model.eval()

eval_loader = torch.utils.data.DataLoader(Dataset(test_examples, test_examples), batch_size=256)
logits = []
for x, _ in eval_loader:
    with torch.no_grad():
        logits.append(model(x).detach().numpy())
logits = np.concatenate(logits, axis=0)
with torch.no_grad():
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 0].detach().numpy()

_, ax = plt.subplots(figsize=(7, 5.5))

pcm = plot_uncertainty_surface(probs, ax)

plt.colorbar(pcm, ax=ax)
plt.title("Class Probability, Deterministic Model")

plt.show()
resnet_uncertainty = probs * (1 - probs)

_, ax = plt.subplots(figsize=(7, 5.5))

pcm = plot_uncertainty_surface(resnet_uncertainty, ax=ax)

plt.colorbar(pcm, ax=ax)
plt.title("Predictive Uncertainty, Deterministic Model")

plt.show()

def make_masks(n_layers, hidden_dim):
    return [torch.bernoulli(torch.empty(128).uniform_()) for _ in range(n_layers)]

n_layers = len(model.fcs)
hidden_dim = 128

logit_samples = []

for i in tqdm(range(24)):
    masks = make_masks(n_layers, hidden_dim)
    eval_loader = torch.utils.data.DataLoader(Dataset(test_examples, test_examples), batch_size=256)
    logits = []
    for x, _ in eval_loader:
        with torch.no_grad():
            logits.append(model(x, masks=masks).detach().numpy())
    logits = np.concatenate(logits, axis=0)
    logit_samples.append(logits[None, :]) # add the sample dimension (size 12) at zero to concatenate along.
    
logit_samples = np.concatenate(logit_samples)
mean = np.mean(logit_samples, axis=0)
variance = np.var(logit_samples, axis=0) ** 2
logits = mean / np.sqrt(1 + variance)
    
with torch.no_grad():
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 0].detach().numpy()

with torch.no_grad():
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 0].detach().numpy()
    
with torch.no_grad():
    mean_probs = torch.softmax(torch.tensor(mean, dtype=torch.float32), dim=-1)[:, 0].detach().numpy()

_, ax = plt.subplots(figsize=(7, 5.5))

pcm = plot_uncertainty_surface(mean_probs, ax)

plt.colorbar(pcm, ax=ax)
plt.title("Class Probability, Deterministic Model")

plt.show()

resnet_uncertainty = probs * (1 - probs)

_, ax = plt.subplots(figsize=(7, 5.5))

pcm = plot_uncertainty_surface(resnet_uncertainty, ax=ax)

plt.colorbar(pcm, ax=ax)
plt.title("Predictive Uncertainty, MC Dropout Model")

plt.show()

class SNGPModel(nn.Module):
    def __init__(self, in_features, num_classes, up_projection_dim):
        super().__init__()
        self.register_buffer('input_W', torch.randn(in_features, up_projection_dim))
        self.register_buffer('input_b', torch.randn(up_projection_dim))
        self.classifier = SNGP(
            in_features=up_projection_dim,
            num_classes=num_classes,
            kernel_scale_trainable=True,
            scale_random_features=True,
            normalize_input=False,
            covariance_momentum=0.999,
            return_dict=False,
        )
            
    def forward(self, x):
        x =  x @ self.input_W + self.input_b
        output = self.classifier(x)
        return output
    
input_dim = 2
num_classes = 2
up_projection_dim = 128

model = SNGPModel(
    in_features=input_dim, 
    num_classes=num_classes,
    up_projection_dim=up_projection_dim,
)

optim = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
loader = torch.utils.data.DataLoader(Dataset(train_examples, train_labels), batch_size=64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss(reduction='mean')

model.train()
model = model.to(device)
for epoch in range(1, 150+1):
    running_loss, c = 0, 0
    for x, y in loader:
        optim.zero_grad()
        logits, covariance = model(x.to(device))
        loss = loss_fn(logits, y.to(device))
        loss.backward()
        optim.step()
        running_loss += float(loss)
        c += 1
    if not epoch % 10:
        print(f"Epoch {epoch}. Loss: {running_loss / c}")
#     model.classifier.reset_precision()
model.eval()

eval_loader = torch.utils.data.DataLoader(Dataset(test_examples, test_examples), batch_size=256)
logits = []
covs = []
for x, _ in eval_loader:
    with torch.no_grad():
        l, c = model(x.to(device))
        l = mean_field_logits(l, c)
        logits.append(l.detach().cpu().numpy())
        covs.append(np.diag(c.detach().cpu().numpy())[:, None])
        
logits = np.concatenate(logits, axis=0)
covs = np.concatenate(covs, axis=0)

with torch.no_grad():
    probs2 = torch.softmax(torch.tensor(logits), dim=1)[:, 0].detach().numpy()

_, ax = plt.subplots(figsize=(7, 5.5))

pcm = plot_uncertainty_surface(probs, ax=ax, show_data=True)

plt.colorbar(pcm, ax=ax)
plt.title("Class Probability, Probabilistic Model")

plt.show()
uncertainty = probs * (1 - probs)

_, ax = plt.subplots(figsize=(7, 5.5))

pcm = plot_uncertainty_surface(uncertainty, ax=ax, show_data=False)

plt.colorbar(pcm, ax=ax)
plt.title("Predictive Uncertainty, Probabilistic Model")

plt.show()
# Assuming make_training_data and make_testing_data functions are defined as per standard SNGP code


# Constants for the LC circuit
L = 1e-3  # Inductance in Henrys
C = 1e-6  # Capacitance in Farads
R = 50  # Resistance in Ohms
dt = 1e-6  # Time step
temperature = 300  # Kelvin
k_B = 1.38e-23  # Boltzmann constant
noise_amplitude = np.sqrt(4 * k_B * temperature * R / dt)

# Time parameters
time_steps = int(0.01 / dt)
num_circuits = 5
coupling_matrix = np.array([
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]])

# Initialize charge (Q) and current (I) arrays
Q = np.zeros((time_steps, num_circuits))  # Charge
I = np.zeros((time_steps, num_circuits))  # Current

for t in range(1, time_steps):
    for n in range(num_circuits):
        coupled_influence = sum(coupling_matrix[n, m] * Q[t - 1, m] for m in range(num_circuits))
        dQdt = I[t - 1, n]
        dIdt = -(Q[t - 1, n] + coupled_influence) / (L * C) + np.random.normal(0, noise_amplitude)
        Q[t, n] = Q[t - 1, n] + dQdt * dt
        I[t, n] = I[t - 1, n] + dIdt * dt

# Load the train, test and OOD datasets
train_examples, train_labels = make_moons(n_samples=1000, noise=0.1)
test_examples, test_labels = make_moons(n_samples=200, noise=0.1)

# Convert datasets to PyTorch tensors and create a DataLoader
train_dataset = TensorDataset(torch.from_numpy(train_examples).float(), torch.from_numpy(train_labels).long())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(torch.from_numpy(test_examples).float(), torch.from_numpy(test_labels).long())
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

attention_dim = 128  # Size of the attention vector

class SNGPWithAttentionModel(nn.Module):
    def __init__(self, input_dim, num_classes, up_projection_dim, num_circuits, SPU_embedding_dim, attention_dim):
        super(SNGPWithAttentionModel, self).__init__()
        self.spu_projection = nn.Linear(num_circuits, SPU_embedding_dim)
        self.attention = nn.Linear(SPU_embedding_dim, attention_dim)
        self.combine = nn.Linear(input_dim + attention_dim, up_projection_dim)
        self.classifier = SNGP(
            in_features=up_projection_dim,
            num_classes=num_classes,
            # ... Additional SNGP parameters as needed ...
        )

    def forward(self, x, Q_flatten):
        spu_features = self.spu_projection(Q_flatten)
        attention_weights = torch.softmax(self.attention(spu_features), dim=1)
        combined = torch.cat([x, attention_weights], dim=1)
        combined = self.combine(combined)
       
        logits = self.classifier(combined)
        
        # If your classifier returns a tuple, make sure to just return the logits
        if isinstance(logits, tuple):
            logits = logits[0]

        return logits  # Make sure logits is a tensor, not a tuple



# Create an instance of the model with correct parameter names
SPU_embedding_dim = num_circuits  # Arbitrary choice for this example
attention_dim = 128  # Size of the attention vector

# Instantiate the model with the correct parameters
attention_sngp_model = SNGPWithAttentionModel(
    input_dim=2,
    num_classes=2,
    up_projection_dim=128,
    num_circuits=num_circuits,
    SPU_embedding_dim=SPU_embedding_dim,
    attention_dim=attention_dim
)
attention_sngp_model.to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(attention_sngp_model.parameters(), weight_decay=1e-5)
# Train the SNGP model without attention
input_dim = 2
num_classes = 2
up_projection_dim = 128

standard_model = SNGPModel(input_dim, num_classes, up_projection_dim)
optimizer = torch.optim.Adam(standard_model.parameters(), weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()
standard_model.to(device)
standard_model.train()

# Iterate over the DataLoader for training
for epoch in range(1, 151):
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)  # Get logits from the model
        loss = loss_fn(logits, y)  # Calculate loss using those logits
        loss.backward()  # Backpropagate the gradients
        optim.step()  # Update the model's weights
        running_loss += loss.item()
    
    # Print training loss every 10 epochs
    if epoch % 10 == 0:
        average_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}: Average Loss: {average_loss}")
        
standard_model.eval()
standard_logits = []
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        logits = standard_model(x)
        standard_logits.extend(logits.cpu().numpy())

standard_probs = torch.softmax(torch.tensor(standard_logits), dim=1)[:, 0].numpy()
standard_auc = roc_auc_score(test_labels, standard_probs)

# Now train the SNGPWithAttentionModel
attention_model = SNGPWithAttentionModel(input_dim, num_classes, up_projection_dim, num_circuits, SPU_embedding_dim, attention_dim)
optimizer = torch.optim.Adam(attention_model.parameters(), weight_decay=1e-5)
attention_model.to(device)
attention_sngp_model.train()

for epoch in range(1, 151):
    running_loss, c = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()

        # Output from the model may contain both logits and another value (like covariance)
        # Here we're only interested in logits for the loss calculation
        output = model(x)
        
        # Check if the output is a tuple, and only take the first element (logits)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        loss = loss_fn(logits, y)  # Use the logits here for loss calculation
        loss.backward()
        optim.step()
        running_loss += float(loss)
        c += 1

    if not epoch % 10:
        print(f"Epoch {epoch}. Loss: {running_loss / c}")

attention_sngp_model.eval()
logits_list = []
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        output = attention_sngp_model(x)
        logits, _ = output if isinstance(output, tuple) else (output, None)
        logits_list.extend(logits.cpu().numpy())


attention_probs = torch.softmax(torch.tensor(logits_list), dim=1)[:, 0].numpy()
attention_auc = roc_auc_score(test_labels, attention_probs)

# Compare the performance of the standard and attention-augmented SNGP models
print("Standard SNGP AUC:", standard_auc)
print("Attention SNGP AUC:", attention_auc)

# ROC curves for both models
fpr_standard, tpr_standard, _ = roc_curve(test_labels, standard_probs)
fpr_attention, tpr_attention, _ = roc_curve(test_labels, attention_probs)

plt.figure()
plt.plot(fpr_standard, tpr_standard, label='Standard SNGP (AUC = {:.2f})'.format(standard_auc))
plt.plot(fpr_attention, tpr_attention, label='Attention SNGP (AUC = {:.2f})'.format(attention_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()
