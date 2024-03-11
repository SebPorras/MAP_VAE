from models.base import BaseVAE
import torch
import time
import numpy as np
from torch import nn


def plot_gt(data):
    limit = 0.5
    step = 1 / 1024.0
    pixels = int(2 * limit / step)
    grid = np.array(
        [
            [a, b]
            for a in np.arange(-limit, limit, step)
            for b in np.arange(-limit, limit, step)
        ]
    )

    if data == "checkerboard":
        l = [0, 2, 1, 3, 0, 2, 1, 3]
        color = np.zeros((pixels, pixels, 3))
        for i in range(8):
            y = i // 2 * 256
            x = l[i] * 256
            color[x : x + 256, y : y + 256, 0] = i / 8.0
            color[x : x + 256, y : y + 256, 2] = 1

    color = color.reshape((pixels, pixels, 3))

    color[:, :, 0] /= color[:, :, 2] + 1e-12
    color[:, :, 1] = 1
    prob = color[:, :, 2].reshape((pixels, pixels))
    prob = prob / np.sum(prob)  # normalize the data
    prob += 1e-20
    entropy = -prob * np.log(prob) / np.log(2)
    entropy = np.sum(entropy)
    max_prob = np.max(prob)

    color[:, :, 2] /= np.max(color[:, :, 2])
    color[:, :, 1] = color[:, :, 2]
    color = np.clip(color, 0, 1)
    color = col.hsv_to_rgb(color)

    fig = plt.figure(figsize=(18, 18))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis("off")
    ax1.imshow(prob, extent=(-limit, limit, -limit, limit))

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.imshow(color, extent=(-limit, limit, -limit, limit))

    fig.tight_layout()

    return entropy - 20, max_prob, prob, color


def sample2d(data, batch_size=200):
    # code largely taken from https://github.com/nicola-decao/BNAF/blob/master/data/generate2d.py

    rng = np.random.RandomState()

    if data == "8gaussians":
        scale = 4
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        # dataset = np.zeros((batch_size, 2))
        for _ in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
            # dataset[i]=point
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset / 8.0

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n
        d1y = np.sin(n) * n
        x = (
            np.hstack((d1x, d1y))
            / 3
            * (np.random.randint(0, 2, (batch_size, 1)) * 2 - 1)
        )
        x += np.random.randn(*x.shape) * 0.1
        return x / 8.0

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2 / 8.0

    else:
        raise RuntimeError


def s2hms(s) -> tuple[int, int, int]:
    h = s // 3600
    m = (s - h * 3600) // 60
    s = int((s - h * 3600 - m * 60))
    return h, m, s


def print_progress(time, cur_iter, total_iter) -> str:
    h, m, s = s2hms(time)
    h2, m2, s2 = s2hms(time * total_iter / cur_iter - time)
    return f"Time Elapsed: {h} hours {m} minutes {s} seconds. Time Remaining: {h2} hours {m2} minutes {s2} seconds.\n"


def train(dataset: str, model: BaseVAE, epochs=60000, print_freq=1000) -> BaseVAE:
    """Training cycle for checkboard dataset"""

    print("Cuda available: ", torch.cuda.is_available())
    print("Num GPUs Available: ", torch.cuda.device_count())

    model.cuda()

    start = time.time()
    optimiser = model.configure_optimiser()
    model.train(True)

    log = ""
    for iteration in range(epochs):
        input = torch.tensor(sample2d(dataset, 40000)).float().cuda()

        # forward step
        modelOutputs = model(input)

        # calculate loss
        loss, kl, likelihood = model.loss_function(modelOutputs, input)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        input = None
        if iteration % print_freq == 0:
            with torch.no_grad():
                iter = f"Iteration {iteration}, ELBO: {loss}, L_rec: {likelihood}, L_reg: {kl}\n"
                stats = print_progress(time.time() - start, iteration + 1, epochs)
                log += iter
                log += stats

    print(log)

    return model
