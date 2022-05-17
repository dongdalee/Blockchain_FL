# server receiver parameter
HOST = '127.0.0.1'
PORT = 9025
SHARD_NUM = 1

# server sender parameter
SHARD_ADDR_LIST = [
     {"ip": '127.0.0.1', "port": 9010}
]

FILE_LIST = ["shard1.pt", "shard2.pt", "shard3.pt", "shard4.pt", "shard5.pt", "aggregation.pt"]

MINI_BATCH = 64
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

WORKER_DATA = "./data/"
LEARNING_MEASURE = "f1 score"

UPLOAD_MODEL_NUM = 1  # number of upload modle form shard
SHARD_LIST = ['shard1', 'shard2', 'shard3', 'shard4', 'shard5']  # using for select shard ramdomly

