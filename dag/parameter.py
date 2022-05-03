# main
WORKER_NUM = 10  # total number of  local worker in shard
TOTAL_ROUND = 5  # total round (number of for loop)
TRAINING_EPOCH = 1  # training epochs each round

ASYNC_TRAINING = True

RANDOM_TRAINING_EPOCH = False  # setup random training epoch
MIN_EPOCH = 1
MAX_EPOCH = 5

# tip selection algorithmn
TIP_SELECT_ALGO = 'high_accuracy'
LEARNING_MEASURE = "f1 score"
SIMILARITY = "cosine"

# possion parameter
LAM = 7
SIZE = TOTAL_ROUND

# file save path
SAVE_SHARD_MODEL_PATH = './model/'
SAVE_MIGRATION_INFO_PATH = './migrate/shard1.txt'

# dag pos difficulity
DIFFICULTY = 0

# setup for worker
MINI_BATCH_SIZE = 64
LEARNING_RATE = 0.001

# socket connection
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 9025

MIGRATION_SERVER_HOST = '127.0.0.1'
MIGRATION_SERVER_PORT = 9045

SHARD_HOST = '127.0.0.1'
SHARD_PORT = 9010

SHARD_ID = "shard1"

SHARD_NUM = 1

# all shard list
# shard_list = {"shard1": [], "shard2": [], "shard3": [], "shard4": [], "shard5": []}
shard_list = {"shard1": []}

# data set labels
# label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'frog', 'ship', 'truck'] # cifar-10
# label = ['Bag', 'Boot', 'Coat', 'Dress', 'Pullover', 'Sandal', 'Shirt', 'Sneaker', 'Top', 'Trouser'] # fashion-mnist
label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Mnist

# sub_label = ['airplane', 'automobile']
sub_label = ['0', '1']

# model weight attack
POISON_WORKER = ['worker0', 'worker1']
GAUSSIAN_MEAN = 0
GAUSSIAN_SIGMA = 2
