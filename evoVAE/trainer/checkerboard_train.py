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


def train(model: BaseVAE, trainLoader, epochs=10, print_freq=10) -> BaseVAE:
    """Training cycle for checkboard dataset"""

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # speeds up if input size isn't changing 
    torch.backends.cudnn.benchmark = True

    model.to(device)


    start = time.time()
    optimiser = model.configure_optimiser()
    log = ""

    for iteration in range(epochs):

        batchSize = len(trainLoader)
        epochLoss = 0
        model.train(True)
        
        log += f"Epoch {iteration+1}\n-------------------------------\n"

        for batch in trainLoader:

            data, _ = batch
            data = data.reshape(-1, 28 * 28).to(device)

            # forward step
            modelOutputs = model(data)

            # calculate loss
            loss, kl, likelihood = model.loss_function(modelOutputs, data)
            epochLoss += loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if iteration % print_freq == 0:
                with torch.no_grad():
                    itera = f"Iteration {iteration}, ELBO: {loss}, L_rec: {likelihood}, L_reg: {kl}\n"
                    stats = print_progress(time.time() - start, iteration + 1, epochs)
                    batchLoss = f"Avg batch loss: {epochLoss/batchSize}\n"

                    log += itera
                    log += stats
                    log += batchLoss

    print(log)
    torch.save(model.state_dict(), "model_weights.pth")

    return model

