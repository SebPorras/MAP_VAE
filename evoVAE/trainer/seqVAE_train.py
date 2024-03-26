from evoVAE.models.seqVAE import SeqVAE
import torch
from typing import Dict
from torch.utils.data import DataLoader
import wandb


def seq_train(
    model: SeqVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    config: Dict,
) -> SeqVAE:

    model = model.to(device)

    optimiser = model.configure_optimiser(
        learning_rate=config.learning_rate, weight_decay=config.weight_decay
    )

    for iteration in range(config.epochs):

        epoch_loss = []
        epoch_kl = []
        epoch_likelihood = []
        model.train(True)

        # ignore seq names for now when training
        for encoding, weights, _ in train_loader:

            encoding = encoding.float().to(device)
            weights = weights.float().to(device)

            # forward step
            optimiser.zero_grad()
            modelOutputs = model(encoding)

            # calculate loss
            loss, kl, likelihood = model.loss_function(
                modelOutputs, torch.flatten(encoding, start_dim=1)
            )

            # update epoch metrics
            epoch_loss.append(loss.item())
            epoch_kl.append(kl.item())
            epoch_likelihood.append(likelihood.item())

            # update weights
            loss.backward()
            # sets max value for gradient - currently 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimiser.step()

            # log batch results
            # wandb.log({"ELBO": loss.item(), "KLD": kl.item(), "Gauss_likelihood": likelihood.item()})

        # validation metrics
        model.eval()
        epoch_val_elbo = []
        epoch_val_kl = []
        epoch_val_likelihood = []
        with torch.no_grad():
            for encoding_val, weight_val, _ in val_loader:
                encoding_val = encoding_val.float().to(device)
                weight_val = weight_val.float().to(device)

                outputs_val = model(encoding_val)

                loss_val, kl_val, likelihood_val = model.loss_function(
                    outputs_val, torch.flatten(encoding_val, start_dim=1)
                )

                epoch_val_elbo.append(loss_val.item())
                epoch_val_kl.append(kl_val.item())
                epoch_val_likelihood.append(likelihood_val.item())

                # wandb.log({"ELBO_val": loss_val.item(), "KLD_val": kl_val.item(), "Gauss_likelihood_val": likelihood_val.item()})

        wandb.log(
            {
                "epoch_ELBO": epoch_loss / len(epoch_loss),
                "epoch_KLD": epoch_kl / len(epoch_kl),
                "epoch_Gauss_likelihood": epoch_likelihood / len(epoch_likelihood),
                "epoch_val_ELBO": epoch_val_elbo / len(epoch_val_elbo),
                "epoch_val_KLD": epoch_val_kl / len(epoch_val_kl),
                "epoch_val_Gauss_likelihood": epoch_val_likelihood
                / len(epoch_val_likelihood),
            }
        )

    torch.save(model.state_dict(), "seqVAE_weights.pth")

    return model
