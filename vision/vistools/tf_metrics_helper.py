from tensorflow.tools.api.generator.api.metrics import mean_squared_error

def metrics_list_to_dict(metrics):
    ret = {}
    for m in metrics:
        if m == 'mean_squared_error':
            ret[m] = mean_squared_error
    
    return ret