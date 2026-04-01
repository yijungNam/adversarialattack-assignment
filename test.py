import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.mnist_model import MNISTCNN
from models.cifar_model import get_cifar10_pretrained_model

from attacks.fgsm import fgsm_targeted, fgsm_untargeted
from attacks.pgd import pgd_targeted, pgd_untargeted


def train_model(model, device, train_loader, epochs=3, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")


def evaluate_clean_accuracy(model, device, test_loader, dataset_name):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100.0 * correct / total
    print(f"{dataset_name} Clean Test Accuracy: {acc:.2f}%")
    return acc


def evaluate_targeted_fgsm(model, device, test_loader, eps=0.1, max_samples=100):
    model.eval()
    success = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        for i in range(images.size(0)):
            if total >= max_samples:
                rate = 100.0 * success / total
                return rate

            x = images[i:i+1]
            y_true = labels[i:i+1]
            target = (y_true + 1) % 10

            x_adv = fgsm_targeted(model, x, target, eps)

            with torch.no_grad():
                pred_adv = model(x_adv).argmax(dim=1)

            if pred_adv.item() == target.item():
                success += 1

            total += 1

    return 100.0 * success / total


def evaluate_untargeted_fgsm(model, device, test_loader, eps=0.1, max_samples=100):
    model.eval()
    success = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        for i in range(images.size(0)):
            if total >= max_samples:
                rate = 100.0 * success / total
                return rate

            x = images[i:i+1]
            y_true = labels[i:i+1]

            x_adv = fgsm_untargeted(model, x, y_true, eps)

            with torch.no_grad():
                pred_adv = model(x_adv).argmax(dim=1)

            if pred_adv.item() != y_true.item():
                success += 1

            total += 1

    return 100.0 * success / total


def evaluate_targeted_pgd(model, device, test_loader, k=10, eps=0.3, eps_step=0.01, max_samples=100):
    model.eval()
    success = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        for i in range(images.size(0)):
            if total >= max_samples:
                rate = 100.0 * success / total
                return rate

            x = images[i:i+1]
            y_true = labels[i:i+1]
            target = (y_true + 1) % 10

            x_adv = pgd_targeted(model, x, target, k, eps, eps_step)

            with torch.no_grad():
                pred_adv = model(x_adv).argmax(dim=1)

            if pred_adv.item() == target.item():
                success += 1

            total += 1

    return 100.0 * success / total


def evaluate_untargeted_pgd(model, device, test_loader, k=10, eps=0.3, eps_step=0.01, max_samples=100):
    model.eval()
    success = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        for i in range(images.size(0)):
            if total >= max_samples:
                rate = 100.0 * success / total
                return rate

            x = images[i:i+1]
            y_true = labels[i:i+1]

            x_adv = pgd_untargeted(model, x, y_true, k, eps, eps_step)

            with torch.no_grad():
                pred_adv = model(x_adv).argmax(dim=1)

            if pred_adv.item() != y_true.item():
                success += 1

            total += 1

    return 100.0 * success / total


def run_all_attacks(model, device, test_loader, dataset_name):
    print(f"\n=== {dataset_name} Attack Success Rates ===")

    t_fgsm = evaluate_targeted_fgsm(model, device, test_loader, eps=0.1, max_samples=100)
    print(f"Targeted FGSM Success Rate: {t_fgsm:.2f}%")

    u_fgsm = evaluate_untargeted_fgsm(model, device, test_loader, eps=0.1, max_samples=100)
    print(f"Untargeted FGSM Success Rate: {u_fgsm:.2f}%")

    t_pgd = evaluate_targeted_pgd(model, device, test_loader, k=10, eps=0.3, eps_step=0.01, max_samples=100)
    print(f"Targeted PGD Success Rate: {t_pgd:.2f}%")

    u_pgd = evaluate_untargeted_pgd(model, device, test_loader, k=10, eps=0.3, eps_step=0.01, max_samples=100)
    print(f"Untargeted PGD Success Rate: {u_pgd:.2f}%")


def get_mnist_loaders():
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


def get_cifar10_loaders():
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def unnormalize_for_display(image):
    return image.detach().cpu().squeeze()


def save_attack_visualizations(model, device, test_loader, dataset_name, attack_name, attack_fn,
                               targeted=False, eps=0.1, k=10, eps_step=0.01, num_samples=5):
    model.eval()
    saved = 0

    os.makedirs("results", exist_ok=True)

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        for i in range(images.size(0)):
            if saved >= num_samples:
                return

            x = images[i:i+1]
            y_true = labels[i:i+1]

            with torch.no_grad():
                pred_before = model(x).argmax(dim=1)

            if pred_before.item() != y_true.item():
                continue

            if targeted:
                target = (y_true + 1) % 10
                if attack_name == "targeted_fgsm":
                    x_adv = attack_fn(model, x, target, eps)
                else:
                    x_adv = attack_fn(model, x, target, k, eps, eps_step)
            else:
                if attack_name == "untargeted_fgsm":
                    x_adv = attack_fn(model, x, y_true, eps)
                else:
                    x_adv = attack_fn(model, x, y_true, k, eps, eps_step)

            with torch.no_grad():
                pred_after = model(x_adv).argmax(dim=1)

            perturbation = x_adv - x

            perturbation_display = perturbation.abs()
            if perturbation_display.max() > 0:
                perturbation_display = perturbation_display / perturbation_display.max()

            x_show = x.detach().cpu().squeeze()
            x_adv_show = x_adv.detach().cpu().squeeze()
            pert_show = perturbation_display.detach().cpu().squeeze()

            fig, axes = plt.subplots(1, 3, figsize=(10, 3))

            # MNIST는 gray, CIFAR는 color
            if dataset_name == "MNIST":
                axes[0].imshow(x_show, cmap="gray")
                axes[1].imshow(x_adv_show, cmap="gray")
                axes[2].imshow(pert_show, cmap="gray")
            else:
                axes[0].imshow(x_show.permute(1, 2, 0))
                axes[1].imshow(x_adv_show.permute(1, 2, 0))
                axes[2].imshow(pert_show.permute(1, 2, 0))

            axes[0].set_title(f"Original\nPred: {pred_before.item()}")
            axes[1].set_title(f"Adversarial\nPred: {pred_after.item()}")

            if targeted:
                axes[2].set_title(f"Perturbation\nTarget: {target.item()}")
            else:
                axes[2].set_title("Perturbation")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()

            filename = f"results/{dataset_name}_{attack_name}_{saved+1}.png"
            plt.savefig(filename)
            plt.close(fig)

            print(f"Saved: {filename}")
            saved += 1
            
            
def save_all_visualizations(model, device, test_loader, dataset_name):
    print(f"\n=== Saving {dataset_name} visualizations ===")

    save_attack_visualizations(
        model, device, test_loader, dataset_name,
        attack_name="targeted_fgsm",
        attack_fn=fgsm_targeted,
        targeted=True,
        eps=0.1,
        num_samples=5
    )

    save_attack_visualizations(
        model, device, test_loader, dataset_name,
        attack_name="untargeted_fgsm",
        attack_fn=fgsm_untargeted,
        targeted=False,
        eps=0.1,
        num_samples=5
    )

    save_attack_visualizations(
        model, device, test_loader, dataset_name,
        attack_name="targeted_pgd",
        attack_fn=pgd_targeted,
        targeted=True,
        eps=0.3,
        k=10,
        eps_step=0.01,
        num_samples=5
    )

    save_attack_visualizations(
        model, device, test_loader, dataset_name,
        attack_name="untargeted_pgd",
        attack_fn=pgd_untargeted,
        targeted=False,
        eps=0.3,
        k=10,
        eps_step=0.01,
        num_samples=5
    )
    

    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("MNIST")
    print("==============================")

    mnist_train_loader, mnist_test_loader = get_mnist_loaders()
    mnist_model = MNISTCNN().to(device)

    print("\nTrain MNIST model")
    train_model(mnist_model, device, mnist_train_loader, epochs=3, lr=0.001)
    evaluate_clean_accuracy(mnist_model, device, mnist_test_loader, "MNIST")
    run_all_attacks(mnist_model, device, mnist_test_loader, "MNIST")
    save_all_visualizations(mnist_model, device, mnist_test_loader, "MNIST")
   
    print("\nCIFAR-10")
    print("==============================")

    cifar_train_loader, cifar_test_loader = get_cifar10_loaders()
    cifar_model = get_cifar10_pretrained_model().to(device)

    print("\nTrain CIFAR-10 model")
    train_model(cifar_model, device, cifar_train_loader, epochs=3, lr=0.0001)
    evaluate_clean_accuracy(cifar_model, device, cifar_test_loader, "CIFAR-10")
    run_all_attacks(cifar_model, device, cifar_test_loader, "CIFAR-10")
    save_all_visualizations(cifar_model, device, cifar_test_loader, "CIFAR-10")

if __name__ == "__main__":
    main()

