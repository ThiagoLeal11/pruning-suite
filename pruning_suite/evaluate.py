import torch
from timm.utils import accuracy, AverageMeter


def top_accuracy(model, loader, top_k=(1,), device='cpu'):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    tops = [AverageMeter() for _ in range(len(top_k))]

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(loader):
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)

            accs = accuracy(output, target, topk=top_k)
            for i, acc in enumerate(accs):
                tops[i].update(acc.item(), inputs.size(0))

    return [top.avg for top in tops]
