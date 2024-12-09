import torch

filePath = 'Path to the train2id.txt file'
# Load train2id file
edges = []

with open(filePath, "r") as f:
    for line in f:
        src, tgt, rel = map(int, line.strip().split())  # Read source, target, relation
        edges.append((src, tgt))  # Add source and target to the edge list

# Convert to PyTorch tensor and save
edges_tensor = torch.tensor(edges).T  # Transpose to match required format [[sources], [targets]]
torch.save(edges_tensor, "edges.pt")