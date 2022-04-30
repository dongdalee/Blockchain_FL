import torch
import os
import warnings
from model import CNN
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score

import dataloader as d
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

def load_model(model_name):
    model = CNN().to(device)
    model.load_state_dict(torch.load("./weight/" + model_name + ".pt"), strict=False)

    return model


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


def test_label_predictions(model, device, testloader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]


def test_model(model):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    tuples = []
    origin_testloader = d.testloader
    label_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # 클래스별 정확도
    with torch.no_grad():
        for data in origin_testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        label_accuracy = round(100 * class_correct[i] / class_total[i], 2)
        tuples.append([label_list[i], label_accuracy])
        print('accuracy of {0} : {1:.2f}'.format(label_list[i], label_accuracy))

    # 모델 Accuracy, F1 Score 출력
    actuals, predictions = test_label_predictions(model, device, origin_testloader)
    
    accuracy = round(accuracy_score(actuals, predictions) * 100, 2)
    f1 = round(f1_score(actuals, predictions, average='weighted') * 100, 2)

    print(" ")
    print('Accuracy: {0:.5f}'.format(accuracy))
    print('F1: {0:.5f}'.format(f1))


model1 = load_model("local1")
model2 = load_model("local2")
model3 = load_model("local3")
model4 = load_model("local4")
model5 = load_model("local5")

avg1 = fed_avg(model1, model2, model3)

avg2 = fed_avg(model3, model4, model5)

avg = fed_avg(avg1, avg2, model1, model2, model4, model5)

test_model(avg)

