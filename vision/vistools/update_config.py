try:
    from keras.callbacks import Callback
except ImportError:
    from tensorflow.python.keras.callbacks import Callback

import pickle
import time
import datetime
import json

class UpdateConfig(Callback):
    def __init__(self, cfg, config_path, checkpoint_prefix=None):
        self.config = cfg
        self.config_path = config_path
        self.checkpoint_prefix = checkpoint_prefix

        self.best_loss = float('inf')

    def on_train_begin(self, logs={}):
        self.epoch_num = 0
        self.batch_num = 0
    
    def on_epoch_end(self, epoch, logs={}):
        if logs['val_loss'] < self.best_loss:
            ts = time.time()
            timestamp = datetime.datetime.fromtimestamp(ts).strftime('%m_%d_%H%M')

            self.best_loss = logs['val_loss']
            self.config['performance']['best_loss'] = self.best_loss
            self.config['performance']['best_weights'] = self.checkpoint_prefix + '-{:03d}-{:.0f}.hdf5'.format(epoch+1, self.best_loss)
            with open(self.config_path, 'wb') as f:
                json.dump(self.config, f)
