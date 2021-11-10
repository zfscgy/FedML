import torch

from FedML.Base.Utils import get_tensors, set_tensors, get_tensors_by_function
from FedML.FedSchemes.fedavg import *


@dataclass
class EWCMaliciousClientOptions(FedAvgClientOptions):
    n_local_rounds_malicious: int
    malicious_data_loader: DataLoader
    ewc_reg_term: float


class EWCMaliciousClient(FedAvgClient):
    def __init__(self, get_model: Callable[[], nn.Module], server: FedAvgServer, options: EWCMaliciousClientOptions):
        super(EWCMaliciousClient, self).__init__(get_model, server, options)

        self.malicious_data_iterator = None

    def update(self):
        print(">>>>>>>>>>>>Malicious client update:")
        # Get update obtained via benign training
        benign_losses = super(EWCMaliciousClient, self).update()
        benign_weight = get_tensors(self.local_model)

        def malicious_loss_func(pred_ys, ys):
            square_errors = get_tensors_by_function(
                [get_tensors(self.local_model), [Convert.to_tensor(w) for w in benign_weight]], lambda xs: torch.square(xs[0] - xs[1]).flatten())
            return self.options.loss_func(pred_ys, ys) + \
                   self.options.ewc_reg_term * torch.mean(torch.cat(square_errors, dim=0))

        if self.options.batch_mode:
            if self.malicious_data_iterator is None:
                self.malicious_data_iterator = iter(self.options.malicious_dataloader)
            malicious_losses, self.malicious_data_iterator = \
                train_n_batches(self.local_model, self.optimizer, malicious_loss_func,
                                self.options.malicious_data_loader,
                                self.malicious_data_iterator, self.options.n_local_rounds_malicious)

        else:
            malicious_losses = train_n_epochs(
                self.local_model, self.optimizer, malicious_loss_func,
                self.options.malicious_data_loader, self.options.n_local_rounds_malicious
            )


        return benign_losses, malicious_losses
