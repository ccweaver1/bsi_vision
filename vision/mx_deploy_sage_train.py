from sagemaker.mxnet import MXNet
import boto3
import time
import datetime
import os
from vistools.config import ConfigManager
from vistools.finding_file_helper import *
from vistools.tf_metrics_helper import *
from eval_warper import *
# Using boto3 to load role arn due to arn construction bug
# https://github.com/aws/sagemaker-python-sdk/pull/68
client = boto3.client('iam')
role_arn = client.get_role(RoleName='AmazonSageMaker-ExecutionRole-20180425T132698')['Role']['Arn']
print("Role: {}".format(role_arn))

config_name = choose_file_with_stdin('configs/', 'Choose desired config file')
# starting_point = choose_file_with_stdin('train_history/', 'Choose a checkpoint to continue training on (hit Enter to start new)')
# if starting_point:
#     starting_point_manager = ConfigManager(starting_point, 'train_history/')
#     init_ckpt = starting_point_manager.get('sagemaker_job_info', 'ckpt_dir')
# else:
init_ckpt = None

print("Training based on {}".format(config_name))
# print("Starting from {}".format(starting_point if starting_point else "scratch"))
config_manager = ConfigManager(config_name, always_write=False)
cfg = config_manager.get_json()

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
model_name_dashes = cfg['training']['model']['model_name'].replace('_', '-')

# Load training parameters from config file
output_path = cfg['sagemaker']['output_path']
training_steps = cfg['sagemaker']['training_steps']
eval_steps = cfg['sagemaker']['evaluation_steps']
instance_type = cfg['sagemaker']['train_instance_type']
train_max_run = int(cfg['sagemaker']['train_max_run'])
eval_metrics_list = cfg['sagemaker']['eval_metrics']
eval_metrics = metrics_list_to_dict(eval_metrics_list)

job_name = 'bsi-training-' + model_name_dashes + '-' + timestamp
# ckpt_dir = job_name + '/' + 'checkpoints/'
channel_input_dirs = {"train":"", "test": ""}

# if init_ckpt:
#     ckpt_dir = init_ckpt
#     mx_estimator = MXNet(entry_point='mx_sage_train.py',
#                             role=role_arn, # AWS IAM role
#                             # training_steps=training_steps, # Training steps. None means train forever
#                             # evaluation_steps=eval_steps, # number of steps of evaluation
#                             train_instance_count=1, # number of machines to use in training (keras limited to 1)
#                             train_instance_type=instance_type, #instance type
#                             hyperparameters=cfg['training'], # hyperparameters passed to training funcs
#                             train_max_run=train_max_run, # max training seconds before termination
#                             source_dir='vistools/',
#                             output_path=output_path, # output bucket name
#                             checkpoint_path = init_ckpt # train starting point
#                         )
#     config_manager.put('sagemaker_job_info', 'init_ckpt', init_ckpt)                    
# else:
ckpt_dir = job_name + '/' + 'checkpoints/'  
mx_estimator = MXNet(entry_point='mx_sage_train.py',
                        role=role_arn, # AWS IAM role
                        # training_steps=training_steps, # Training steps. None means train forever
                        # evaluation_steps=eval_steps, # number of steps of evaluation
                        train_instance_count=1, # number of machines to use in training (keras limited to 1)
                        train_instance_type=instance_type, #instance type
                        hyperparameters=cfg['training'], # hyperparameters passed to training funcs
                        train_max_run=train_max_run, # max training seconds before termination
                        source_dir='vistools/',
                        # job_name=job_name,
                        # channel_input_dirs=channel_input_dirs,
                        output_path=output_path # output bucket name
                    )


# adding information that is job/runtime specific
# note: this isn't being written back to the config file
config_manager.put('sagemaker_job_info', 'job_name', job_name)
config_manager.put('sagemaker_job_info', 'ckpt_dir', ckpt_dir)
config_manager.put('sagemaker_job_info', 'timestamp', timestamp)


# write a log for this job run to a new file
history_save_path = 'train_history/'
config_manager.write_copy(os.path.join(history_save_path, job_name + '.json'))

# Call evaluate in seperate process
# Process(evaluate_on_timer, ('csvs/test_data.csv', ))


# Call Fit.  Train path expected to contain both a train_data.csv and test_data.csv file
train_path = cfg['train_path']
mx_estimator.fit({"train": str(train_path), "test": str(train_path)})

# Write termination time
end = time.time()
print("Total traintime: {}".format(end - ts))
config_manager.put('sagemaker_job_info', 'train_runtime', end - ts)
config_manager.write_copy(os.path.join(history_save_path, job_name + '.json'))
