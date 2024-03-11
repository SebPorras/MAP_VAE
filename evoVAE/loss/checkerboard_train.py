import torch
import evoVAE.loss.standard_loss as L
from ..data_sampler.checkboard import sample2d
import time


def s2hms(s) -> tuple[int, int, int]:
    h = s // 3600
    m = (s - h * 3600) // 60
    s = int((s - h * 3600 - m * 60))
    return h, m, s


def print_progress(time, cur_iter, total_iter) -> str:
    h, m, s = s2hms(time)
    h2, m2, s2 = s2hms(time * total_iter / cur_iter - time)
    return f"Time Elapsed: {h} hours {m} minutes {s} seconds. Time Remaining: {h2} hours {m2} minutes {s2} seconds.\n"


def train(dataset: str, model: nn.Module, epochs=60000, print_freq=1000):
    """Training cycle for checkboard dataset"""

    print("Cuda available: ", torch.cuda.is_available())
    print("Num GPUs Available: ", torch.cuda.device_count())

    model.cuda()

    start = time.time()
    optimiser = model.setup_optimiser()
    loss_est = 0
    best_est = 1e9
    model.train(True)

    log = ""
    for iteration in range(epochs):
        data = torch.tensor(sample2d(dataset, 40000)).float().cuda()

        # forward step
        modelOutputs = model(data)

        # calculate loss
        loss, kl, likelihood = L.elbo_loss(
            modelOutputs, model.logStandardDeviation, data
        )
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        loss_est = 0.999 * loss_est + 0.001 * loss
        data = None
        if iteration % print_freq == 0:
            with torch.no_grad():
                iter = f"Iteration {iteration}, EMA: {loss_est}, ELBO: {loss}, L_rec: {likelihood}, L_reg: {kl}\n"
                stats = print_progress(time.time() - start, iteration + 1, epochs)
                log += iter
                log += stats

    print(log)
