import torch
import torch.nn.functional as F

def fgsm_targeted(model, x, target, eps):
    x = x.clone().detach()
    x.requires_grad_(True)

    # 1. forward
    output = model(x)

    # 2. loss 계산
    loss = F.cross_entropy(output, target)

    # 3. gradient 계산
    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()
    loss.backward()
    grad = x.grad.data
    
    # 4. 이미지 수정
    x_adv = x - eps * grad.sign()

    # 5. 범위 제한
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv

def fgsm_untargeted(model, x, label, eps):
    x = x.clone().detach()
    x.requires_grad_(True)

    # 1. forward
    output = model(x)

    # 2. loss
    loss = F.cross_entropy(output, label)

    # 3. gradient 계산
    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()
    loss.backward()

    grad = x.grad.data

    # 4. untargeted attack
    x_adv = x + eps * grad.sign()

    # 5. clamp
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()