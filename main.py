import torch
import torch.optim as optim
from torch_geometric.datasets import Planetoid

from config import load_config
from model import GAT
from run import run
from utils import EarlyStopping, Visualize


if __name__ == '__main__':
    config = load_config()

    torch.manual_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    model = GAT(config, dataset).to(torch.float32)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['optim']['lr'],
        betas=(config['optim']['beta1'], config['optim']['beta2']),
        weight_decay=config['optim']['weight_decay'],
        eps=config['optim']['eps']
    )
    es = EarlyStopping(config)
    visualize = Visualize()

    train_losses = []
    valid_losses = []
    for epoch in range(config['train']['epochs']):
        train_loss, train_acc = run(config, model, dataset, optimizer)
        valid_loss, valid_acc = run(
            config, model, dataset, optimizer, state='valid'
        )
        log_str = f"Epoch: {epoch:>3}, "
        log_str += f"train_loss: {train_loss:.4f}, "
        log_str += f"valid_loss: {valid_loss:.4f}, "
        log_str += f"train_acc: {train_acc:.4f}, "
        log_str += f"valid_acc: {valid_acc:.4f}"
        print(log_str, flush=True)
        train_losses.append(train_loss.detach())
        valid_losses.append(valid_loss.detach())

        es_bool = es.check(valid_loss, model, epoch)
        if es_bool:
            break
    
    test_loss, test_acc = run(config, model, dataset, optimizer, state='test')
    print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")

    visualize.save_loss(train_losses, valid_losses)
    visualize.save_result(model, dataset)