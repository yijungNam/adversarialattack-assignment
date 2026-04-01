import torch
import torch.nn.functional as F


def pgd_targeted(model, x, target, k, eps, eps_step):

    x_original = x.clone().detach()
    x_adv = x.clone().detach()

    for _ in range(k):
        x_adv.requires_grad_(True)

        output = model(x_adv)
        loss = F.cross_entropy(output, target)

        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()

        grad = x_adv.grad.data

        # targeted: target loss를 줄이는 방향
        x_adv = x_adv - eps_step * grad.sign()

        # projection to epsilon-ball around original x
        x_adv = torch.max(torch.min(x_adv, x_original + eps), x_original - eps)

        # valid image range
        x_adv = torch.clamp(x_adv, 0, 1)

        x_adv = x_adv.detach()

    return x_adv


def pgd_untargeted(model, x, label, k, eps, eps_step):

    x_original = x.clone().detach()
    x_adv = x.clone().detach()

    for _ in range(k):
        x_adv.requires_grad_(True)

        output = model(x_adv)
        loss = F.cross_entropy(output, label)

        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()

        grad = x_adv.grad.data

        # untargeted: correct-class loss를 키우는 방향
        x_adv = x_adv + eps_step * grad.sign()

        # projection to epsilon-ball around original x
        x_adv = torch.max(torch.min(x_adv, x_original + eps), x_original - eps)

        # valid image range
        x_adv = torch.clamp(x_adv, 0, 1)

        x_adv = x_adv.detach()

    return x_adv