import shutil
import os
import torch
import warnings
from model import CNN
from round_checker import model_loader
from parameter import SHARD_ID
warnings.filterwarnings('ignore')

# BLOCK_NUM = input("Input Block Number: ")
BLOCK_NUM = max(model_loader())

"""
SAVE_MODEL_PATH = "./model/"
PATH = "./model/aggregation.pt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# FedAvg ================================================================================================================
class ShardModel:
    def __init__(self, model_name):
        self.model = CNN().to(device)
        self.model.load_state_dict(torch.load(SAVE_MODEL_PATH + str(model_name)), strict=False)

aggregation_model = CNN().to(device)

model_file_list = os.listdir(SAVE_MODEL_PATH)
if ".DS_Store" in model_file_list:
    model_file_list.remove(".DS_Store")
if "aggregation.pt" in model_file_list:
    model_file_list.remove("aggregation.pt")

model_dict = {}

def handler():
    model_num = len(model_file_list)
    print(model_file_list, model_num)

    aggregation_model.layer1[0].weight.data.fill_(0.0)
    aggregation_model.layer1[0].bias.data.fill_(0.0)

    aggregation_model.layer2[0].weight.data.fill_(0.0)
    aggregation_model.layer2[0].bias.data.fill_(0.0)

    aggregation_model.layer3[0].weight.data.fill_(0.0)
    aggregation_model.layer3[0].bias.data.fill_(0.0)

    aggregation_model.fc1.weight.data.fill_(0.0)
    aggregation_model.fc1.bias.data.fill_(0.0)

    aggregation_model.fc2.weight.data.fill_(0.0)
    aggregation_model.fc2.bias.data.fill_(0.0)

    for index, model_file in enumerate(model_file_list):
        model_dict["model" + str(index+1)] = ShardModel(str(model_file))

    for model in model_dict:
        aggregation_model.layer1[0].weight.data += model_dict[model].model.layer1[0].weight.data
        aggregation_model.layer1[0].bias.data += model_dict[model].model.layer1[0].bias.data

        aggregation_model.layer2[0].weight.data += model_dict[model].model.layer2[0].weight.data
        aggregation_model.layer2[0].bias.data += model_dict[model].model.layer2[0].bias.data

        aggregation_model.layer3[0].weight.data += model_dict[model].model.layer3[0].weight.data
        aggregation_model.layer3[0].bias.data += model_dict[model].model.layer3[0].bias.data

        aggregation_model.fc1.weight.data += model_dict[model].model.fc1.weight.data
        aggregation_model.fc1.bias.data += model_dict[model].model.fc1.bias.data

        aggregation_model.fc2.weight.data += model_dict[model].model.fc2.weight.data
        aggregation_model.fc2.bias.data += model_dict[model].model.fc2.bias.data

    aggregation_model.layer1[0].weight.data = aggregation_model.layer1[0].weight.data / model_num
    aggregation_model.layer1[0].bias.data = aggregation_model.layer1[0].bias.data / model_num

    aggregation_model.layer2[0].weight.data = aggregation_model.layer2[0].weight.data / model_num
    aggregation_model.layer2[0].bias.data = aggregation_model.layer2[0].bias.data / model_num

    aggregation_model.layer3[0].weight.data = aggregation_model.layer3[0].weight.data / model_num
    aggregation_model.layer3[0].bias.data = aggregation_model.layer3[0].bias.data / model_num

    aggregation_model.fc1.weight.data = aggregation_model.fc1.weight.data / model_num
    aggregation_model.fc1.bias.data = aggregation_model.fc1.bias.data / model_num

    aggregation_model.fc2.weight.data = aggregation_model.fc2.weight.data / model_num
    aggregation_model.fc2.bias.data = aggregation_model.fc2.bias.data / model_num
    
    torch.save(aggregation_model.state_dict(), PATH)
"""


# 실험 데이터 복사 =========================================================================================================
if SHARD_ID not in os.listdir("./../"):
    print("{0} Directory produced !".format(SHARD_ID))
    os.mkdir("./../" + SHARD_ID)

# ./../s1/1 -> ./../shard1/1
if str(BLOCK_NUM) not in os.listdir("./../" + SHARD_ID + "/"):
    print("Block {0} Directory produced !".format(BLOCK_NUM))
    os.mkdir("./../" + SHARD_ID + "/" + str(BLOCK_NUM))

shutil.copytree("./acc_graph_data", "./../" + SHARD_ID + "/" + str(BLOCK_NUM) + "/acc_graph_data/")
shutil.copytree("./graph", "./../" + SHARD_ID + "/" + str(BLOCK_NUM) + "/graph/")
shutil.copytree("./logs", "./../" + SHARD_ID + "/" + str(BLOCK_NUM) + "/logs/")
shutil.copytree("./migrate", "./../" + SHARD_ID + "/" + str(BLOCK_NUM) + "/migrate/")
shutil.copytree("./model/" + str(BLOCK_NUM) + "/", "./../" + SHARD_ID + "/" + str(BLOCK_NUM) + "/model/")

"""
# aggregation model
handler()
print("model copy !")
shutil.copytree("./model", "./../" + SHARD_ID + "/" + str(BLOCK_NUM) + "/model/")
"""

shutil.rmtree('./logs', ignore_errors=True)
shutil.rmtree('./migrate', ignore_errors=True)
shutil.rmtree('./graph', ignore_errors=True)
shutil.rmtree('./acc_graph_data', ignore_errors=True)

os.mkdir('./logs')
os.mkdir('./migrate')
os.mkdir('./graph')
os.mkdir('./acc_graph_data')






