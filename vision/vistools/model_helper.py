try:
    from keras import losses
    from keras.optimizers import Adam
    from keras.models import Model, load_model
    from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
except ImportError:
    from tensorflow.python.keras import losses
    from tensorflow.python.keras.optimizers import Adam
    from tensorflow.python.keras.models import Model, load_model
    from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger


from update_config import UpdateConfig
from research_models import ResearchModels
import os


def get_model_and_load_weights(cfg):
    '''
     Get model and load weights based on config file
     cfg: dictionary of config values
    '''
    # First try to load best weights from this config
    weights = cfg.get('training', {}).get('init_weights', {})
    # if not weights:
    #     # If no best weights, look to initial weights
    #     weights = cfg['model']['init_weights']
    #     if weights == 'imagenet':
    #         # If inital weights are imagenet, load through keras
    #         models = ResearchModels(cfg, load_imagenet=True)
    #         print("Loading imagenet weights") 
    #         model = models.get_model()    
    #         return model
    #     else:
    #         # not loading any weights
    #         print("Weights un-initialized")
    #         models = ResearchModels(cfg,)
    #         return models.get_model()
    # # Load up a locally stored weights file
    # print("Using weights found in {}".format(weights))
    models = ResearchModels(cfg)
    model = models.get_model()
    # model.load_weights(weights, by_name=True)
    
    return model


def compile_model(model, cfg):
    '''
    model: keras model
    config: dict of config values
    '''
    learning_rate = cfg['hyperparameters']['learning_rate']
    decay = cfg['hyperparameters']['decay']
    loss_name = cfg['model']['loss']
    optimizer_name = cfg['model']['optimizer']
    if loss_name == 'mean_squared_error':
        loss = losses.mean_squared_error
    if optimizer_name == 'Adam':
        optimizer = Adam(lr=learning_rate, decay=decay) #Adam is introducing a *= warning. FYI
    model.compile(loss=loss, optimizer=optimizer)

def get_callbacks(cfg, config_file, timestamp):
    # edit json from config instance directl
    #Callback Functions
    callbacks = []
    early_stopper = EarlyStopping(patience=cfg['training']['patience'])
    csv_logger = CSVLogger(os.path.join('data', 'logs', cfg['model']['model_name'], 'csv', 'training-' + \
        str(timestamp) + '.log'))
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', cfg['model']['model_name'], 'tb'))
    checkpointer = ModelCheckpoint(filepath=os.path.join('data', 'checkpoints', cfg['model']['model_name'] + '-' + timestamp + \
        '-{epoch:03d}-{val_loss:.0f}.hdf5'), verbose=1, save_best_only=True)
    config_updater = UpdateConfig(cfg, config_file, 
        checkpoint_prefix = os.path.join('data', 'checkpoints', cfg['model']['model_name'] + '-' + timestamp))
    
    callbacks = [early_stopper, csv_logger, tb, checkpointer, config_updater]
        
    return callbacks

