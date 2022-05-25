from model import CNN
import torch
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(load_path):
    model = CNN().to(device)
    model.load_state_dict(torch.load(load_path), strict=False)
    return model


def load_worker(shard):
    workers = os.listdir("./data/" + shard + "/")
    for worker in workers:
        if "worker" not in worker:
            workers.remove(worker)

    return workers


def model_fraction(model, numerator, denominator):
    model.layer1[0].weight.data = numerator * model.layer1[0].weight.data / denominator
    model.layer1[0].bias.data = numerator * model.layer1[0].bias.data / denominator

    model.layer2[0].weight.data = numerator * model.layer2[0].weight.data / denominator
    model.layer2[0].bias.data = numerator * model.layer2[0].bias.data / denominator

    model.layer3[0].weight.data = numerator * model.layer3[0].weight.data / denominator
    model.layer3[0].bias.data = numerator * model.layer3[0].bias.data / denominator

    model.fc1.weight.data = numerator * model.fc1.weight.data / denominator
    model.fc1.bias.data = numerator * model.fc1.bias.data / denominator

    model.fc2.weight.data = numerator * model.fc2.weight.data / denominator
    model.fc2.bias.data = numerator * model.fc2.bias.data / denominator


def add_model(*models):
    fed_add_model = CNN().to(device)

    fed_add_model.layer1[0].weight.data.fill_(0.0)
    fed_add_model.layer1[0].bias.data.fill_(0.0)

    fed_add_model.layer2[0].weight.data.fill_(0.0)
    fed_add_model.layer2[0].bias.data.fill_(0.0)

    fed_add_model.layer3[0].weight.data.fill_(0.0)
    fed_add_model.layer3[0].bias.data.fill_(0.0)

    fed_add_model.fc1.weight.data.fill_(0.0)
    fed_add_model.fc1.bias.data.fill_(0.0)

    fed_add_model.fc2.weight.data.fill_(0.0)
    fed_add_model.fc2.bias.data.fill_(0.0)

    for model in models:
        fed_add_model.layer1[0].weight.data += model.layer1[0].weight.data
        fed_add_model.layer1[0].bias.data += model.layer1[0].bias.data

        fed_add_model.layer2[0].weight.data += model.layer2[0].weight.data
        fed_add_model.layer2[0].bias.data += model.layer2[0].bias.data

        fed_add_model.layer3[0].weight.data += model.layer3[0].weight.data
        fed_add_model.layer3[0].bias.data += model.layer3[0].bias.data

        fed_add_model.fc1.weight.data += model.fc1.weight.data
        fed_add_model.fc1.bias.data += model.fc1.bias.data

        fed_add_model.fc2.weight.data += model.fc2.weight.data
        fed_add_model.fc2.bias.data += model.fc2.bias.data

    return fed_add_model


def sub_model(model1, model2):
    fed_sub_model = CNN().to(device)

    fed_sub_model.layer1[0].weight.data.fill_(0.0)
    fed_sub_model.layer1[0].bias.data.fill_(0.0)

    fed_sub_model.layer2[0].weight.data.fill_(0.0)
    fed_sub_model.layer2[0].bias.data.fill_(0.0)

    fed_sub_model.layer3[0].weight.data.fill_(0.0)
    fed_sub_model.layer3[0].bias.data.fill_(0.0)

    fed_sub_model.fc1.weight.data.fill_(0.0)
    fed_sub_model.fc1.bias.data.fill_(0.0)

    fed_sub_model.fc2.weight.data.fill_(0.0)
    fed_sub_model.fc2.bias.data.fill_(0.0)

    fed_sub_model.layer1[0].weight.data = model1.layer1[0].weight.data - model2.layer1[0].weight.data
    fed_sub_model.layer1[0].bias.data = model1.layer1[0].bias.data - model2.layer1[0].bias.data

    fed_sub_model.layer2[0].weight.data = model1.layer2[0].weight.data - model2.layer2[0].weight.data
    fed_sub_model.layer2[0].bias.data = model1.layer2[0].bias.data - model2.layer2[0].bias.data

    fed_sub_model.layer3[0].weight.data = model1.layer3[0].weight.data - model2.layer3[0].weight.data
    fed_sub_model.layer3[0].bias.data = model1.layer3[0].bias.data - model2.layer3[0].bias.data

    fed_sub_model.fc1.weight.data = model1.fc1.weight.data - model2.fc1.weight.data
    fed_sub_model.fc1.bias.data = model1.fc1.bias.data - model2.fc1.bias.data

    fed_sub_model.fc2.weight.data = model1.fc2.weight.data - model2.fc2.weight.data
    fed_sub_model.fc2.bias.data = model1.fc2.bias.data - model2.fc2.bias.data

    return fed_sub_model


def fed_avg(*models):
    fed_avg_model = CNN().to(device)

    fed_avg_model.layer1[0].weight.data.fill_(0.0)
    fed_avg_model.layer1[0].bias.data.fill_(0.0)

    fed_avg_model.layer2[0].weight.data.fill_(0.0)
    fed_avg_model.layer2[0].bias.data.fill_(0.0)

    fed_avg_model.layer3[0].weight.data.fill_(0.0)
    fed_avg_model.layer3[0].bias.data.fill_(0.0)

    fed_avg_model.fc1.weight.data.fill_(0.0)
    fed_avg_model.fc1.bias.data.fill_(0.0)

    fed_avg_model.fc2.weight.data.fill_(0.0)
    fed_avg_model.fc2.bias.data.fill_(0.0)

    for model in models:
        fed_avg_model.layer1[0].weight.data += model.layer1[0].weight.data
        fed_avg_model.layer1[0].bias.data += model.layer1[0].bias.data

        fed_avg_model.layer2[0].weight.data += model.layer2[0].weight.data
        fed_avg_model.layer2[0].bias.data += model.layer2[0].bias.data

        fed_avg_model.layer3[0].weight.data += model.layer3[0].weight.data
        fed_avg_model.layer3[0].bias.data += model.layer3[0].bias.data

        fed_avg_model.fc1.weight.data += model.fc1.weight.data
        fed_avg_model.fc1.bias.data += model.fc1.bias.data

        fed_avg_model.fc2.weight.data += model.fc2.weight.data
        fed_avg_model.fc2.bias.data += model.fc2.bias.data

    fed_avg_model.layer1[0].weight.data = fed_avg_model.layer1[0].weight.data / len(models)
    fed_avg_model.layer1[0].bias.data = fed_avg_model.layer1[0].bias.data / len(models)

    fed_avg_model.layer2[0].weight.data = fed_avg_model.layer2[0].weight.data / len(models)
    fed_avg_model.layer2[0].bias.data = fed_avg_model.layer2[0].bias.data / len(models)

    fed_avg_model.layer3[0].weight.data = fed_avg_model.layer3[0].weight.data / len(models)
    fed_avg_model.layer3[0].bias.data = fed_avg_model.layer3[0].bias.data / len(models)

    fed_avg_model.fc1.weight.data = fed_avg_model.fc1.weight.data / len(models)
    fed_avg_model.fc1.bias.data = fed_avg_model.fc1.bias.data / len(models)

    fed_avg_model.fc2.weight.data = fed_avg_model.fc2.weight.data / len(models)
    fed_avg_model.fc2.bias.data = fed_avg_model.fc2.bias.data / len(models)

    return fed_avg_model