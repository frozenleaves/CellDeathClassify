import os
import time

model = 'resnet50'

image_height    = 120
image_width     = 120
channels        = 2
EPOCHS          = 200
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-6
TIMES           = 20  # Image magnification


SEG_DEV = False  # 是否启用开发版本的segment model

segment_model_name_20x = 'segment_20x_model'
segment_model_saved_dir_20x = './models/segment/20x/'

segment_dev_model_name = 'segment_dev'
segment_dev_model_basedir = './models/segment_dev/'

train_process_20x_detail_data_savefile = f'./logs/{time.strftime("%Y-%m-%d_%H-%M-%S")}-train_detail_20x.csv'

save_model_dir_20x = './models/classify/model'
save_model_dir_20x_best = './models/best/model'

# save_model_dir_20x = './models/classify/20x/final-dev/model'
# save_model_dir_20x_best = './models/classify/20x/best-dev/model'

dataset_dir_20x = r'G:\DeathDataset\train'
dataset_dir_mcy_20x = os.path.join(dataset_dir_20x, 'train_mcy')
dataset_dir_dic_20x = os.path.join(dataset_dir_20x, 'train_dic')

train_dir_mcy_20x = os.path.join(dataset_dir_mcy_20x, "train")
valid_dir_mcy_20x = os.path.join(dataset_dir_mcy_20x, "valid")
test_dir_mcy_20x = os.path.join(dataset_dir_mcy_20x, "test")
# dataset_dir_dic_20x = './CCDeep_train_data/classify/train_dic'
train_dir_dic_20x = os.path.join(dataset_dir_dic_20x, "train")
valid_dir_dic_20x = os.path.join(dataset_dir_dic_20x, "valid")
test_dir_dic_20x = os.path.join(dataset_dir_dic_20x, "test")
