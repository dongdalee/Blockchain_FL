import torch.nn.init
import matplotlib.pyplot as plt
from numpy.linalg import norm
from model import CNN
import numpy as np
from numpy import dot

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sgd_model_path = "./weight/sgd_model100.pt"

BLOCK_NUM = 10

sgd_model = CNN().to(device)
sgd_model.load_state_dict(torch.load(sgd_model_path), strict=False)


def weight_divergence(fedavg, sgd):
    return norm(fedavg - sgd)/norm(sgd)

def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def plot_bar(mode, shard_name, block_num, bar_width):
    fed_model = CNN().to(device)
    fed_model.load_state_dict(torch.load("./experiment_data/s1/" + str(block_num) + "/model/" + shard_name + ".pt"), strict=False)

    if mode == "weight divergence":
        layer1 = weight_divergence(fed_model.layer1[0].weight.data, sgd_model.layer1[0].weight.data)
        layer2 = weight_divergence(fed_model.layer2[0].weight.data, sgd_model.layer2[0].weight.data)
        layer3 = weight_divergence(fed_model.layer3[0].weight.data, sgd_model.layer3[0].weight.data)
        fc1 = weight_divergence(fed_model.fc1.weight.data, sgd_model.fc1.weight.data)
        fc2 = weight_divergence(fed_model.fc2.weight.data, sgd_model.fc2.weight.data)

    elif mode == "cosine similarity":
        layer1 = cos_sim(fed_model.layer1[0].weight.data.numpy().reshape(-1, ), sgd_model.layer1[0].weight.data.numpy().reshape(-1, ))
        layer2 = cos_sim(fed_model.layer2[0].weight.data.numpy().reshape(-1, ), sgd_model.layer2[0].weight.data.numpy().reshape(-1, ))
        layer3 = cos_sim(fed_model.layer3[0].weight.data.numpy().reshape(-1, ), sgd_model.layer3[0].weight.data.numpy().reshape(-1, ))
        fc1 = cos_sim(fed_model.fc1.weight.data.numpy().reshape(-1, ), sgd_model.fc1.weight.data.numpy().reshape(-1, ))
        fc2 = cos_sim(fed_model.fc2.weight.data.numpy().reshape(-1, ), sgd_model.fc2.weight.data.numpy().reshape(-1, ))

    label = ["layer1", "layer2", "layer3", "fc1", "fc2"]
    xaxis = np.arange(len(label))
    yaxis = [layer1, layer2, layer3, fc1, fc2]

    plt.bar(xaxis + bar_width, yaxis, width=0.1, label=shard_name)
    plt.xticks(xaxis, label)
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.legend(bbox_to_anchor=(1, 1))


shard_list = ["shard1", "shard2", "shard3", "shard4", "shard5", "aggregation"]


for block in np.arange(1, BLOCK_NUM+1):
    width = 0.0
    similarity_mode = "weight divergence"
    for shard in shard_list:
        plot_bar(similarity_mode, shard, block, width)
        width += 0.1

    plt.savefig("./data_similarity/"+similarity_mode+"_block"+str(block)+shard)
    plt.show()