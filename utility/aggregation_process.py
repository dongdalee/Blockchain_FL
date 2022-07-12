from model import CNN
import torch
import torch.nn as nn
from dataloader import testloader
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = nn.CrossEntropyLoss()

labels = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

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


def test_label_predictions(model):
    model.eval()
    actuals = []
    predictions = []

    labels_of_acc = []

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        # train_data_loader, test_data_loader = testloader

        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)

            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)

            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()
            for i in range(4):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        for i in range(10):
            label_accuracy = round(100 * class_correct[i] / class_total[i], 2)
            print('accuracy of {0} : {1:.2f}'.format(labels[i], label_accuracy))
            labels_of_acc.append(label_accuracy)

    return labels_of_acc


if __name__ == '__main__':
    model1 = CNN().to(device)
    model2 = CNN().to(device)
    model3 = CNN().to(device)
    model4 = CNN().to(device)
    model5 = CNN().to(device)

    model1.load_state_dict(torch.load("./model/local1.pt"), strict=False)
    model2.load_state_dict(torch.load("./model/local2.pt"), strict=False)
    model3.load_state_dict(torch.load("./model/local3.pt"), strict=False)
    model4.load_state_dict(torch.load("./model/local4.pt"), strict=False)
    model5.load_state_dict(torch.load("./model/local5.pt"), strict=False)

    fed_avg_model = fed_avg(model1, model2, model3, model4, model5)

    labels_of_acc = test_label_predictions(fed_avg_model)

    plt.bar(labels, labels_of_acc, width=0.4)
    plt.xticks(labels)
    plt.show()


