# See https://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='sparse_categorical_accuracy', value=0.9, patience = 0, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.patience = patience
        self.patience_counter = 0
        self.number_called = 0
        self.string_to_print = "\n Early stopping: Accuracy has reached specified treshold!"

    def on_epoch_end(self, epoch, logs={}):
        self.number_called += 1
        current = logs.get(self.monitor)
        if ((self.number_called > 1) and (current >= self.value)):
            #if self.verbose > 0:
            #    print("Epoch %05d: early stopping THR" % epoch)
            # Add one to counter
            self.patience_counter+=1
            if (self.patience == 0):
                print(self.string_to_print)
                self.model.stop_training = True
                return None
            if (self.patience_counter == self.patience):
                print(self.string_to_print)
                self.model.stop_training = True


class EarlyStoppingAfterTreshold(Callback):
    """Will stop if the selected monitor stops increasing or decreasing (mode max/min respectively) for the specified number of times (patience), but only if it has reached the specified treshold). For now only for maximizers!"""
    def __init__(self, monitor='val_sparse_categorical_accuracy', trsh=0.2, patience = 0, verbose=0, mode = "max"):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.treshold = trsh
        self.verbose = verbose
        self.patience = patience
        self.patience_counter = 0
        if (mode == "max"):
            self.maximize = True
        else:
            self.maximize = False
        self.number_called = 0
        self.string_to_print = "\n Early stopping: Metric has reached specified treshold and has stopped improving!"
        self.metric_history = []
        self.enough_data = False
        self.top_value = 0
        self.earlystop = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                         min_delta=0,
        patience=patience,
        verbose=0,
        mode=mode,
        baseline=None,
        restore_best_weights=True)

    def on_epoch_end(self, epoch, logs={}):
        # # self.number_called += 1
        # # current = logs.get(self.monitor)
        # # self.metric_history.append(current)
        # # if (self.number_called > self.patience):
            # # self.enough_data = True
        # # if (not (self.enough_data)):
            # stops if it has been called for less than patience
            # # return None
        # # if self.maximize:
            # # if (current < self.treshold):
                # If the metric hasn't reached the specified treshold, do nothing.
                # # return None
            # # if (current > self.top_value):
                # # self.top_value = current
                # # self.patience_counter = 0
                # # return None
            # # else:
                # Now do things
                # # self.patience_counter+=1
                # # if (self.patience == 0):
                    # # print(self.string_to_print)
                    # # self.model.stop_training = True
                    # # return None
                # # if (self.patience_counter == self.patience):
                    # # print(self.string_to_print)
                    # # self.model.stop_training = True
        self.number_called += 1
        current = logs[self.monitor]
        if (self.maximize):
            if (current < self.treshold):
                self.model.stop_training = False
            else:
                return None