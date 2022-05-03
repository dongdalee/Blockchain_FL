from model import CNN
import torch
from scipy.spatial.distance import cityblock, cosine

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.manual_seed_all(777)



def load_model(model_name):
    shard_path = "./model/" + str(model_name) + ".pt"
    model = CNN().to(device)
    model.load_state_dict(torch.load(shard_path), strict=False)

    return model


model1 = load_model("r2_15_mal0")
model2 = load_model("r2_14_mal0")
model3 = load_model("r2_15_mal1")
print(id(model1))

mode = "cosine"
def vector_similarity(model1, model2):
    print(id(model1))
    if mode == "cosine":  # 클 수록 비슷하다.
        layer1 = 1 - cosine(model1.layer1[0].weight.data.numpy().reshape(-1, ), model2.layer1[0].weight.data.numpy().reshape(-1, ))
        layer2 = 1 - cosine(model1.layer2[0].weight.data.numpy().reshape(-1, ), model2.layer2[0].weight.data.numpy().reshape(-1, ))
        layer3 = 1 - cosine(model1.layer3[0].weight.data.numpy().reshape(-1, ), model2.layer3[0].weight.data.numpy().reshape(-1, ))
        fc1 = 1 - cosine(model1.fc1.weight.data.numpy().reshape(-1, ), model2.fc1.weight.data.numpy().reshape(-1, ))
        fc2 = 1 - cosine(model1.fc2.weight.data.numpy().reshape(-1, ), model2.fc2.weight.data.numpy().reshape(-1, ))
        # print(layer1, layer2, layer3, fc1, fc2)
        return layer1 + layer2 + layer3 + fc1 + fc2

    elif mode == "manhattann":  # 작을수록 비슷
        layer1 = cityblock(model1.layer1[0].weight.data.numpy().reshape(-1, ), model2.layer1[0].weight.data.numpy().reshape(-1, ))
        layer2 = cityblock(model1.layer2[0].weight.data.numpy().reshape(-1, ), model2.layer2[0].weight.data.numpy().reshape(-1, ))
        layer3 = cityblock(model1.layer3[0].weight.data.numpy().reshape(-1, ), model2.layer3[0].weight.data.numpy().reshape(-1, ))
        fc1 = cityblock(model1.fc1.weight.data.numpy().reshape(-1, ), model2.fc1.weight.data.numpy().reshape(-1, ))
        fc2 = cityblock(model1.fc2.weight.data.numpy().reshape(-1, ), model2.fc2.weight.data.numpy().reshape(-1, ))
        # print(layer1, layer2, layer3, fc1, fc2)

        if (layer1 + layer2 + layer3 + fc1 + fc2) == 0:
            return 100
        else:
            return 1/(layer1 + layer2 + layer3 + fc1 + fc2)


re1 = vector_similarity(model1, model2)
print(re1)

re2 = vector_similarity(model1, model3)
print(re2)

if re1 < re2:
    print("re1 < re2")
else:

    print("re1 > re2")


