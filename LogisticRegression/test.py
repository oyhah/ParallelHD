import torch
import torch.nn.functional as F
import torch.nn as nn

def test(x_net, test_dataloader, args):

    loss_func = nn.CrossEntropyLoss()

    x_net.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = x_net(inputs)
            loss = loss_func(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / (i + 1)
    test_acc = 100. * correct / total

    return test_loss, test_acc