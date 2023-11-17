import torch
import torch.nn.functional as F

def run(config, model, dataset, optim, state='train'):
    if state == 'train':
        model.train()
        optim.zero_grad()
        pred = model(dataset.data)
        loss = F.nll_loss(
            pred[dataset.data.train_mask],
            dataset.data.y[dataset.data.train_mask]
        )
        loss.backward()
        optim.step()

        correct_labels = dataset.data.y[dataset.data.train_mask]
        predicted_labels = pred[dataset.data.train_mask].max(1)[1]
        correct_predictions = predicted_labels.eq(correct_labels).sum().item()
        total_predictions = dataset.data.train_mask.sum().item()
        accuracy = correct_predictions / total_predictions
    
    elif state == 'valid':
        model.eval()
        with torch.no_grad():
            pred = model(dataset.data)
            loss = F.nll_loss(
                pred[dataset.data.val_mask],
                dataset.data.y[dataset.data.val_mask]
            )

        correct_labels = dataset.data.y[dataset.data.val_mask]
        predicted_labels = pred[dataset.data.val_mask].max(1)[1]
        correct_predictions = predicted_labels.eq(correct_labels).sum().item()
        total_predictions = dataset.data.val_mask.sum().item()
        accuracy = correct_predictions / total_predictions
    
    elif state == 'test':
        model.eval()
        with torch.no_grad():
            pred = model(dataset.data)
            loss = F.nll_loss(
                pred[dataset.data.test_mask],
                dataset.data.y[dataset.data.test_mask]
            )
        
        correct_labels = dataset.data.y[dataset.data.test_mask]
        predicted_labels = pred[dataset.data.test_mask].max(1)[1]
        correct_predictions = predicted_labels.eq(correct_labels).sum().item()
        total_predictions = dataset.data.test_mask.sum().item()
        accuracy = correct_predictions / total_predictions
    
    return loss, accuracy
