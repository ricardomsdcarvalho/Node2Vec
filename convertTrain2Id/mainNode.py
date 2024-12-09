import torch
from tqdm import tqdm

filePath = '/home/ricciard0.dc/mimicReadmission/mimicreadmission/Data/entityRelation/train2id.txt'
# Load train2id file
edges = []

with open(filePath, "r") as f:
    for line in tqdm(f.readlines()[1:-1], desc="Loading Train2id data", unit="line"):
        
        src, tgt, rel = map(int, line.strip().split())  # Read source, target, relation
        edges.append((src, tgt))  # Add source and target to the edge list

# Convert to PyTorch tensor and save
edges_tensor = torch.tensor(edges).T  # Transpose to match required format [[sources], [targets]]
print(edges_tensor)
torch.save(edges_tensor, "/home/ricciard0.dc/node2vec/edges.pt")