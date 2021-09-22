from FedML.Base.Utils import get_tensors, set_tensors, get_tensors_by_function
from FedML.FedSchemes.fedavg import *


@dataclass
class MaliciousClientOptions(FedAvgClientOptions):
    n_local_rounds_malicious: int
    malicious_data_loader: DataLoader
    malicious_boost: float


class MaliciousClient(FedAvgClient):
    def __init__(self, get_model: Callable[[], nn.Module], server: FedAvgServer, options: MaliciousClientOptions):
        super(MaliciousClient, self).__init__(get_model, server, options)

        self.malicious_data_iterator = None

    def update(self):
        # Get update obtained via benign training
        initial_weights = get_tensors(self.local_model, copy=True)
        benign_losses = super(MaliciousClient, self).update()
        benign_update = get_tensors_by_function([get_tensors(self.local_model), initial_weights],
                                                lambda xs: xs[0] - xs[1])

        # Perform malicious training
        set_tensors(self.local_model, initial_weights, copy=True)
        if self.options.batch_mode:
            if self.malicious_data_iterator is None:
                self.malicious_data_iterator = iter(self.options.malicious_dataloader)
            malicious_losses, self.malicious_data_iterator = \
                train_n_batches(self.local_model, self.optimizer, self.options.loss_func,
                                self.options.malicious_data_loader,
                                self.malicious_data_iterator, self.options.n_local_rounds_malicious)

        else:
            malicious_losses = train_n_epochs(
                self.local_model, self.optimizer, self.options.loss_func,
                self.options.malicious_data_loader, self.options.n_local_rounds_malicious
            )

        malicious_update = get_tensors_by_function([get_tensors(self.local_model), initial_weights],
                                                   lambda xs: xs[0] - xs[1])
        set_tensors(self.local_model, get_tensors_by_function([initial_weights, benign_update, malicious_update],
                                                              lambda xyz: xyz[0] + xyz[1] + self.options.malicious_boost * xyz[2]))

        return benign_losses, malicious_losses
