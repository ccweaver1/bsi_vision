from optparse import OptionParser
from vistools.data import Data
from vistools.model_helper import get_model_and_load_weights, get_callbacks, compile_model
import time
import json
import datetime
import os

'''
For local training.
Set configuration in configs/
'''

parser = OptionParser()
parser.add_option("-c", "--config", dest="config", help="Path to config.json file", default=None)
(options, args) = parser.parse_args()

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M')

if options.config:
    config_file = options.config
else:   # if filename is not given
    config_file = sorted(os.listdir('configs/'), key=lambda x: os.path.getmtime(os.path.join('configs', x)), reverse=True)[0]
    if not config_file:
        print('Unable to load config file.')
        exit()
    else:
        config_file = os.path.join('configs', config_file)        
        print('Loading most recent config file.  Found: {}'.format(config_file))
config = Config(config_file, folder='configs/' always_write=True)
cfg = config.get_json()
cfg['performance']['last_opened'] = timestamp
config.write()


# Get data generators
data = Data(cfg['training']['train_path'], config, train_test='train')
train_generator = data.train_test_generator('train')
test_generator = data.train_test_generator('test')

# Get model with weights loaded
model = get_model_and_load_weights(config)

# Compile model
compile_model(model, config)

# Parameter for fit_gen
batches_per_epoch = (data.train_len * cfg['training']['train_test_ratio']) // cfg['hyperparameters']['batch_size']    

# Get list of callback functions
callbacks_list = get_callbacks(config, config_file, timestamp)

X, y = next(train_generator)
pred = model.predict(X)
print pred
print y

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=batches_per_epoch,
    epochs=cfg['hyperparameters']['nb_epochs'],
    verbose=1,
    callbacks=callbacks_list,
    validation_data=test_generator,
    validation_steps=data.test_len)    
