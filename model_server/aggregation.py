import torch
import os
import warnings
from model import CNN
warnings.filterwarnings('ignore')

round = 1

SAVE_MODEL_PATH = "./model/"+str(round)+"/"
PATH = "./model/"+str(round)+"/aggregation.pt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

class ShardModel:
    def __init__(self, model_name):
        self.model = CNN().to(device)
        self.model.load_state_dict(torch.load(SAVE_MODEL_PATH + str(model_name)), strict=False)

aggregation_model = CNN().to(device)
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


model_file_list = os.listdir(SAVE_MODEL_PATH)
if ".DS_Store" in model_file_list:
    model_file_list.remove(".DS_Store")
if "aggregation.pt" in model_file_list:
    model_file_list.remove("aggregation.pt")

model_dict = {}

def handler():
    model_num = len(model_file_list)
    print(model_file_list, model_num)

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
