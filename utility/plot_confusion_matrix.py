import torch
import torch.nn.init
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import warnings
import pandas as pd

import dataloader as d
from model import CNN

warnings.filterwarnings(action='ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

START_BLOCK_NUM = 1
BLOCK_NUM = 10

data_loader = d.trainloader
test_loader = d.testloader

label_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'frog', 'ship', 'truck']
# label_list = ['Bag', 'Boot', 'Coat', 'Dress', 'Pullover', 'Sandal', 'Shirt', 'Sneaker', 'Top', 'Trouser'] # fashion-mnist


np.set_printoptions(precision=2)

def test_label_predictions(model, device, test_loader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    plt.figure(figsize=(50, 50))

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),

           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig("./matrix/"+title+".png")
    return ax


def class_accuracy(model):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (label_list[i], 100 * class_correct[i] / class_total[i]))


def create_eval_dataset(shard_name, block_num):
    shard_path = "./experiment_data/s1/" + str(block_num) + "/model/" + shard_name + ".pt"
    model = CNN().to(device)
    model.load_state_dict(torch.load(shard_path), strict=False)
    model.eval()
    actuals, predictions = test_label_predictions(model, device, test_loader)
    plot_confusion_matrix(actuals, predictions, normalize=True, classes=label_list, title="Block" + str(block_num) + "_" + shard_name)

    accuracy = accuracy_score(actuals, predictions) * 100
    f1 = f1_score(actuals, predictions, average='weighted') * 100

    print("Block Number: {0}".format(block_num))
    print(shard_name, ":", accuracy, f1)
    class_accuracy(model)
    print("===================================")
    return [accuracy, f1]


acc_dict = {}

for block in np.arange(START_BLOCK_NUM, BLOCK_NUM+1):
    s1_data = create_eval_dataset("shard1", block)
    s2_data = create_eval_dataset("shard2", block)
    s3_data = create_eval_dataset("shard3", block)
    s4_data = create_eval_dataset("shard4", block)
    s5_data = create_eval_dataset("shard5", block)
    fedavg_data = create_eval_dataset("aggregation", block)

    acc_dict[block] = s1_data, s2_data, s3_data, s4_data, s5_data, fedavg_data

tuples = []
s1_row_data = ["shard1"]
s2_row_data = ["shard2"]
s3_row_data = ["shard3"]
s4_row_data = ["shard4"]
s5_row_data = ["shard5"]
fedavg_row_data = ["FedAvg"]


for block in np.arange(START_BLOCK_NUM, BLOCK_NUM+1):
    s1_row_data.extend(acc_dict[block][0])
    s2_row_data.extend(acc_dict[block][1])
    s3_row_data.extend(acc_dict[block][2])
    s4_row_data.extend(acc_dict[block][3])
    s5_row_data.extend(acc_dict[block][4])
    fedavg_row_data.extend(acc_dict[block][5])

column = ["accuracy", "f1"]
columns = ["shard"]

for block in np.arange(START_BLOCK_NUM, BLOCK_NUM+1):
    columns.extend(column)

tuples = [
    s1_row_data,
    s2_row_data,
    s3_row_data,
    s4_row_data,
    s5_row_data,
    fedavg_row_data
]

df = pd.DataFrame(tuples, columns=columns)
df.to_excel("./acc_data/Block_data.xlsx", index=False)
