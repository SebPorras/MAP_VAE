from ..models import BaseVAE
from ..models.types_ import *
import torch
import time


def s2hms(s) -> Tuple[int, int, int]:
    h = s // 3600
    m = (s - h * 3600) // 60
    s = int((s - h * 3600 - m * 60))
    return h, m, s


def print_progress(time, cur_iter, total_iter) -> str:
    h, m, s = s2hms(time)
    h2, m2, s2 = s2hms(time * total_iter / cur_iter - time)
    return f"Time Elapsed: {h} hours {m} minutes {s} seconds. Time Remaining: {h2} hours {m2} minutes {s2} seconds.\n"


def train(model: BaseVAE, trainLoader, epochs=60000, print_freq=1000) -> BaseVAE:
    """Training cycle for checkboard dataset"""

    print("Cuda available: ", torch.cuda.is_available())
    print("Num GPUs Available: ", torch.cuda.device_count())
    if torch.cuda.is_available():
        model.cuda()

    batchSize = len(trainLoader)

    start = time.time()
    optimiser = model.configure_optimiser()
    model.train(True)
    log = ""

    for iteration in range(epochs):

        epochLoss = 0

        for batch in trainLoader:

            input, _ = batch
            input = input.reshape(-1, 28 * 28)

            # forward step
            modelOutputs = model(input)

            # calculate loss
            loss, kl, likelihood = model.loss_function(modelOutputs, input)
            epochLoss += loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        if iteration % print_freq == 0:
            with torch.no_grad():
                iter = f"Iteration {iteration}, ELBO: {loss}, L_rec: {likelihood}, L_reg: {kl}\n"
                stats = print_progress(time.time() - start, iteration + 1, epochs)
                batchLoss = f"Avg batch loss: {epochLoss/batchSize}"

                log += iter
                log += stats
                log += batchLoss

    print(log)
    torch.save(model.state_dict(), "model_weights.pth")

    return model
