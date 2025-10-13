from logging import Logger

from transformers import TrainerCallback


class MetricsCallback(TrainerCallback):
    def __init__(self, logger: Logger):
        self.logger = logger
        self.data = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'epoch' in logs:
            epoch = logs['epoch']

            # Store training loss and learning rate
            if 'loss' in logs:
                self.data[epoch] = self.data.get(epoch, {})
                self.data[epoch]['loss'] = logs['loss']
                self.data[epoch]['learning_rate'] = logs['learning_rate']

            # Store eval loss and print combined when available
            if 'eval_loss' in logs:
                self.data[epoch] = self.data.get(epoch, {})
                self.data[epoch]['eval_loss'] = logs['eval_loss']

                # Print combined line if we have both
                epoch_data = self.data[epoch]
                if 'loss' in epoch_data and 'eval_loss' in epoch_data:
                    self.logger.info(
                        f"{epoch:.2f}".ljust(15) +
                        f"{epoch_data['loss']:.4f}".ljust(15) +
                        f"{epoch_data['eval_loss']:.4f}".ljust(15) +
                        f"{epoch_data['learning_rate']:.2e}".ljust(15)
                    )
