import argparse
import torch
from tqdm import tqdm
from torch_geometric.nn import Node2Vec

def load_entity2id(file_path):
    """Load the entity-to-ID mapping from a file."""
    entity2id = {}
    with open(file_path, 'r') as f:
        for line in f.readlines()[1:]:
            try:
                entity, eid = line.strip().split('\t')
                entity2id[entity] = int(eid)
            except ValueError:
                continue

    return entity2id

def load_target_entities(file_path, entity2id):
    """Load target entities from a file and convert them to IDs using entity2id mapping."""
    target_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            entity = line.strip()
            if entity in entity2id:
                target_ids.append(entity2id[entity])
            else:
                print(f"Warning: Entity '{entity}' not found in entity2id mapping.")
    return target_ids

def save_target_embeddings_to_txt(model, target_entities, output_file):
    """Save embeddings of target entities to a text file."""
    embeddings = model.embedding.weight.data.cpu()
    with open(output_file, 'w') as f:
        for entity_id in target_entities:
            embedding = embeddings[entity_id].tolist()
            embedding_str = ' '.join(map(str, embedding))
            f.write(f"{entity_id} {embedding_str}\n")

def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (Node2Vec)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=128) ##########
    parser.add_argument('--walk_length', type=int, default=80)
    parser.add_argument('--context_size', type=int, default=20)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--entity2id_file', type=str, default='/home/ricciard0.dc/mimicReadmission/mimicreadmission/Data/entityRelation/entity2id.txt',
                        help='Path to file containing entity-to-ID mapping.')
    parser.add_argument('--target_file', type=str, default='/home/ricciard0.dc/mimicReadmission/mimicreadmission/Data/targetEntities.txt',
                        help='Path to file containing target entities (original names).')
    parser.add_argument('--output_file', type=str, default='node2vec_128_embeddings.txt',
                        help='Output file for saving target embeddings.')
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    print('Loading The Tensor with the training entity pairs, entity2id and targetIds!!!')
    data = torch.load("/home/ricciard0.dc/node2vec/edges.pt")  # Load the edge list (torch.tensor)
    entity2id = load_entity2id(args.entity2id_file)  # Load entity-to-ID mapping
    target_entities = load_target_entities(args.target_file, entity2id)  # Convert targets to IDs

    model = Node2Vec(data, args.embedding_dim, args.walk_length,
                     args.context_size, args.walks_per_node,
                     sparse=True).to(device)

    loader = model.loader(batch_size=args.batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

    print('Training the model!!!!!!')
    model.train()
    for epoch in tqdm(range(1, args.epochs + 1), desc='Going through the Epoch', unit='Epoch'):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                      f'Loss: {loss:.4f}')

    # Save only the embeddings of target entities to a text file
    save_target_embeddings_to_txt(model, target_entities, args.output_file)


if __name__ == "__main__":
    main()