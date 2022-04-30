import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import warnings
import matplotlib.pyplot as plt
import numpy as np
from model import CNN

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

warnings.filterwarnings(action='ignore')

# GPU가 없을 경우, CPU를 사용한다.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# Hyper parameter
learning_rate = 0.001
training_epochs = 5
batch_size = 100
PATH = "./model/aggregation.pt"

mnist_train = dsets.MNIST(root='MNIST_data/',  # 다운로드 경로 지정
                          train=True,  # True를 지정하면 train data로 다운로드
                          transform=transforms.ToTensor(),  # 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',  # 다운로드 경로 지정
                         train=False,  # False를 지정하면 validate data로 다운로드
                         transform=transforms.ToTensor(),  # 텐서로 변환
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)  # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("model load!")
model.load_state_dict(torch.load(PATH), strict=False)
"""
total_batch = len(data_loader)
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:  # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


print("model saved")
torch.save(model.state_dict(), PATH)
"""

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


actuals, predictions = test_label_predictions(model, device, test_loader)

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),

           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)

label_dict = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plt.figure(figsize=(50, 50))
plot_confusion_matrix(actuals, predictions, normalize=True, classes=label_dict, title='Confusion matrix')

plt.show()

print('Accuracy: {0:.3f}'.format(accuracy_score(actuals, predictions)*100))
print('F1: {0:.3f}'.format(f1_score(actuals, predictions, average='weighted')*100))