import torch
from sklearn.metrics import accuracy_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from bilbo.data.augmentation import (
    PadToSize,
    RandomCrop,
    RandomRotate,
    RandomRowDropout,
    RandomRowSwap,
    RandomShift,
    SplineDistortion,
    ToTensor,
    Transform,
)
from bilbo.data.dataset import BilboDataset
from bilbo.model import BilboModel


def get_dataloader(max_length=256, batch_size=4, split="train"):
    tsfms = Transform(
        [
            RandomCrop(p=1, final_size=max_length * 2),
            RandomShift(p=0.5),
            RandomRotate(0.5),
            RandomShift(p=0.5),
            SplineDistortion(0.5, n_points=4, amp_range=(-20, 20)),
            RandomRowSwap(p=0.9, min_swap=1, max_swap=10),
            RandomRowDropout(p=1, final_count=max_length),
            PadToSize(p=1, final_size=max_length),
            ToTensor(normalize=True),
        ]
    )

    ds = BilboDataset(f"./data/{split}", transforms=tsfms)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
    return dl


def get_metrics(y: torch.Tensor, yhat: torch.Tensor):
    y = y.detach().cpu().numpy().flatten()
    yhat = yhat.detach().cpu().numpy().flatten()
    ylabel = y > 0.5
    yhatlabel = yhat > 0.5
    return {
        "accuracy": accuracy_score(ylabel, yhatlabel),
    }


def run_validation(model, dl_val):
    with torch.no_grad():
        losses = []
        accuracy = []
        pbar = tqdm(dl_val)
        for xb, yb in pbar:
            loss, yhat = model.loss(xb, yb)
            losses.append(loss.item())
            metrics = get_metrics(yb, yhat)
            accuracy.append(metrics["accuracy"])
            pbar.set_description(f"Loss: {loss.item():.4f}, Accuracy: {metrics['accuracy']:.4%}")
        val_loss = sum(losses) / len(losses)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {sum(accuracy)/len(accuracy):.4f}")
        return val_loss


def train(checkpoint=None):
    # Parameters
    input_size = 8  # 4 pairs of x and y
    n_layers = 5
    max_len = 256
    lr = 1e-5
    batch_size = 32
    epochs = 100
    dl_trn = get_dataloader(max_length=max_len, batch_size=batch_size, split="train")
    dl_val = get_dataloader(max_length=max_len, batch_size=batch_size, split="val")
    model = BilboModel(input_size=input_size, hidden_size=128, n_layers=n_layers)

    best_loss = float("inf")
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        model.load_state_dict(torch.load(checkpoint))
        best_loss = run_validation(model, dl_val)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pbar = tqdm(dl_trn)
        for xb, yb in pbar:
            loss, yhat = model.loss(xb, yb)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        # Save model checkpoint
        torch.save(model.state_dict(), "last.pth")

        val_loss = run_validation(model, dl_val)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best.pth")


if __name__ == "__main__":
    train("./best.pth")
    print("Done!")
