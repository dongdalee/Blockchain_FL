import os
from model import CNN
from dataloader import get_dataloader
from sklearn.metrics import f1_score, accuracy_score
import torch
from util import variable_name
from parameter import WORKER_DATA, LEARNING_MEASURE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL_PATH = "./model/"+str(round)+"/"


def load_model(load_path):
    model = CNN().to(device)
    model.load_state_dict(torch.load(load_path), strict=False)
    return model

def test_model(model):
    testloader = get_dataloader("all")

    # 모델 Accuracy, F1 Score 출력
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
    actuals, predictions = [i.item() for i in actuals], [i.item() for i in predictions]

    accuracy = round(accuracy_score(actuals, predictions) * 100, 2)
    f1 = round(f1_score(actuals, predictions, average='weighted') * 100, 2)

    print(" ")
    print('Accuracy: {0:.5f}'.format(accuracy))
    print('F1: {0:.5f}'.format(f1))

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


class Worker:
    def __init__(self, *_models, _shard, _worker):
        self.models = _models
        self.testloader = get_dataloader(str(_shard), _worker)
        self.ballot = {}

    def test_global_model(self):
        for model in self.models:
            model.eval()
            actuals = []
            predictions = []
            with torch.no_grad():
                for data, target in self.testloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    prediction = output.argmax(dim=1, keepdim=True)
                    actuals.extend(target.view_as(prediction))
                    predictions.extend(prediction)
            actuals, predictions = [i.item() for i in actuals], [i.item() for i in predictions]

            if LEARNING_MEASURE == "accuracy":
                accuracy = round(accuracy_score(actuals, predictions) * 100, 2)
                # print('Accuracy: {0:.5f}'.format(accuracy))
            elif LEARNING_MEASURE == "f1 score":
                accuracy = round(f1_score(actuals, predictions, average='weighted') * 100, 2)
                # print('F1: {0:.5f}'.format(f1))

            model_name = variable_name(model)
            self.ballot[model_name] = accuracy

        max(self.ballot, key=self.ballot.get)
        elected = ''.join([k for k, v in self.ballot.items() if max(self.ballot.values()) == v][0])

        return elected


class ModelCombinator:
    def __init__(self, _round, _mode, _model=None):
        self.round = _round
        self.mode = _mode

        self.previous_A_previous_B = None
        self.previous_A_current_B = None
        self.current_A_previous_B = None
        self.current_A_current_B = None

        self.global_model_previous_model = None
        self.global_model_current_model = None

    def combinator(self):
        if self.mode == "A+B":
            previous_round_model_A = load_model("./model/"+str(self.round-1)+"/shard1.pt")
            current_round_model_A = load_model("./model/"+str(self.round)+"/shard1.pt")
            previous_round_model_B = load_model("./model/"+str(self.round-1)+"/shard2.pt")
            current_round_model_B = load_model("./model/"+str(self.round)+"/shard2.pt")

            model_fraction(previous_round_model_A, 1, 5)
            model_fraction(current_round_model_A, 1, 5)
            model_fraction(previous_round_model_B, 1, 5)
            model_fraction(current_round_model_B, 1, 5)

            self.previous_A_previous_B = add_model(previous_round_model_A, previous_round_model_B)
            model_fraction(self.previous_A_previous_B, 5, 2)
            self.previous_A_current_B = add_model(previous_round_model_A, current_round_model_B)
            model_fraction(self.previous_A_current_B, 5, 2)
            self.current_A_previous_B = add_model(current_round_model_A, previous_round_model_B)
            model_fraction(self.current_A_previous_B, 5, 2)
            self.current_A_current_B = add_model(current_round_model_A, current_round_model_B)
            model_fraction(self.current_A_current_B, 5, 2)

            return self.previous_A_previous_B, self.previous_A_current_B, self.current_A_previous_B, self.current_A_current_B

        elif self.mode == "A+B+C": # G1 + C
            previous_global_model = load_model("./model/" + str(self.round) + "/g1.pt")
            previous_round_model_C = load_model("./model/" + str(self.round - 1) + "/shard3.pt")
            current_round_model_C = load_model("./model/" + str(self.round) + "/shard3.pt")

            model_fraction(previous_round_model_C, 1, 5)
            model_fraction(current_round_model_C, 1, 5)

            self.global_model_previous_model = add_model(previous_global_model, previous_round_model_C)
            model_fraction(self.global_model_previous_model, 5, 3)
            self.global_model_current_model = add_model(previous_global_model, current_round_model_C)
            model_fraction(self.global_model_current_model, 5, 3)

            return self.global_model_previous_model, self.global_model_current_model

        elif self.mode == "A+B+C+D": # G2 + D
            previous_global_model = load_model("./model/" + str(self.round) + "/g2.pt")
            previous_round_model_D = load_model("./model/" + str(self.round - 1) + "/shard4.pt")
            current_round_model_D = load_model("./model/" + str(self.round) + "/shard4.pt")

            model_fraction(previous_round_model_D, 1, 5)
            model_fraction(current_round_model_D, 1, 5)

            self.global_model_previous_model = add_model(previous_global_model, previous_round_model_D)
            model_fraction(self.global_model_previous_model, 5, 4)
            self.global_model_current_model = add_model(previous_global_model, current_round_model_D)
            model_fraction(self.global_model_current_model, 5, 4)

            return self.global_model_previous_model, self.global_model_current_model

        elif self.mode == "A+B+C+D+E": # G2 + D
            previous_global_model = load_model("./model/" + str(self.round) + "/g3.pt")
            previous_round_model_E = load_model("./model/" + str(self.round - 1) + "/shard5.pt")
            current_round_model_E = load_model("./model/" + str(self.round) + "/shard5.pt")

            model_fraction(previous_round_model_E, 1, 5)
            model_fraction(current_round_model_E, 1, 5)

            self.global_model_previous_model = add_model(previous_global_model, previous_round_model_E)
            model_fraction(self.global_model_previous_model, 5, 5)
            self.global_model_current_model = add_model(previous_global_model, current_round_model_E)
            model_fraction(self.global_model_current_model, 5, 5)

            return self.global_model_previous_model, self.global_model_current_model


    def aggregator(self, elected_model):
        if self.mode == "A+B":
            if elected_model == "model1":
                model_fraction(self.previous_A_previous_B, 2, 5)
                torch.save(self.previous_A_previous_B.state_dict(), "./model/" + str(self.round) + "/g1.pt")
            elif elected_model == "model2":
                model_fraction(self.previous_A_current_B, 2, 5)
                torch.save(self.previous_A_current_B.state_dict(), "./model/" + str(self.round) + "/g1.pt")
            elif elected_model == "model3":
                model_fraction(self.current_A_previous_B, 2, 5)
                torch.save(self.current_A_previous_B.state_dict(), "./model/" + str(self.round) + "/g1.pt")
            elif elected_model == "model4":
                model_fraction(self.current_A_current_B, 2, 5)
                torch.save(self.current_A_current_B.state_dict(), "./model/" + str(self.round) + "/g1.pt")

            return

        elif self.mode == "A+B+C": # G1 + C
            if elected_model == "model1":
                model_fraction(self.global_model_previous_model, 3, 5)
                torch.save(self.global_model_previous_model.state_dict(), "./model/" + str(self.round) + "/g2.pt")
            elif elected_model == "model2":
                model_fraction(self.global_model_current_model, 3, 5)
                torch.save(self.global_model_current_model.state_dict(), "./model/" + str(self.round) + "/g2.pt")

            return

        elif self.mode == "A+B+C+D": # G2 + D
            if elected_model == "model1":
                model_fraction(self.global_model_previous_model, 4, 5)
                torch.save(self.global_model_previous_model.state_dict(), "./model/" + str(self.round) + "/g3.pt")
            elif elected_model == "model2":
                model_fraction(self.global_model_current_model, 4, 5)
                torch.save(self.global_model_current_model.state_dict(), "./model/" + str(self.round) + "/g3.pt")

            return

        elif self.mode == "A+B+C+D+E": # G3 + E
            if elected_model == "model1":
                model_fraction(self.global_model_previous_model, 5, 5)
                torch.save(self.global_model_previous_model.state_dict(), "./model/" + str(self.round) + "/aggregation.pt")
            elif elected_model == "model2":
                model_fraction(self.global_model_current_model, 5, 5)
                torch.save(self.global_model_current_model.state_dict(), "./model/" + str(self.round) + "/aggregation.pt")

            return

class Voting:
    def __init__(self, _round, _global_model_type):
        self.global_model_type = _global_model_type
        self.round = _round
        self.combinator_class = None

        if self.global_model_type == "A+B":
            self.voting_result = {'model1' : 0, 'model2' : 0, 'model3' : 0, 'model4' : 0}
        else:
            self.voting_result = {'model1': 0, 'model2': 0}

    def handler(self):
        if self.round == 1:
            A = load_model("./model/1/shard1.pt")
            B = load_model("./model/1/shard2.pt")
            C = load_model("./model/1/shard3.pt")
            D = load_model("./model/1/shard4.pt")
            E = load_model("./model/1/shard5.pt")

            model_fraction(A, 1, 5)
            model_fraction(B, 1, 5)
            model_fraction(C, 1, 5)
            model_fraction(D, 1, 5)
            model_fraction(E, 1, 5)

            model = add_model(A, B, C, D, E)

            torch.save(model.state_dict(), "./model/" + str(self.round) + "/aggregation.pt")

            return
        else:
            if self.global_model_type == "A+B":
                self.combinator_class = ModelCombinator(self.round, self.global_model_type)
                model1, model2, model3, model4 = self.combinator_class.combinator()

                for i in range(2):
                    shard = os.listdir(WORKER_DATA + "shard" + str(i+1) + "/")
                    if ".DS_Store" in shard:
                        shard.remove(".DS_Store")

                    for worker in shard:
                        worker_model = Worker(model1, model2, model3, model4, _shard=i+1, _worker=int(worker[-1]))
                        # print("worker"+worker[-1])
                        vote_result = worker_model.test_global_model()
                        self.voting_result[vote_result] += 1

            elif self.global_model_type == "A+B+C":
                self.combinator_class = ModelCombinator(self.round, self.global_model_type)
                model1, model2 = self.combinator_class.combinator()

                for i in range(3):
                    shard = os.listdir(WORKER_DATA + "shard" + str(i + 1) + "/")
                    if ".DS_Store" in shard:
                        shard.remove(".DS_Store")

                    for worker in shard:
                        worker_model = Worker(model1, model2, _shard=i + 1, _worker=int(worker[-1]))
                        vote_result = worker_model.test_global_model()
                        self.voting_result[vote_result] += 1

            elif self.global_model_type == "A+B+C+D":
                self.combinator_class = ModelCombinator(self.round, self.global_model_type)
                model1, model2 = self.combinator_class.combinator()

                for i in range(3):
                    shard = os.listdir(WORKER_DATA + "shard" + str(i + 1) + "/")
                    if ".DS_Store" in shard:
                        shard.remove(".DS_Store")

                    for worker in shard:
                        worker_model = Worker(model1, model2, _shard=i + 1, _worker=int(worker[-1]))
                        vote_result = worker_model.test_global_model()
                        self.voting_result[vote_result] += 1

            elif self.global_model_type == "A+B+C+D+E":
                self.combinator_class = ModelCombinator(self.round, self.global_model_type)
                model1, model2 = self.combinator_class.combinator()

                for i in range(3):
                    shard = os.listdir(WORKER_DATA + "shard" + str(i + 1) + "/")
                    if ".DS_Store" in shard:
                        shard.remove(".DS_Store")

                    for worker in shard:
                        worker_model = Worker(model1, model2, _shard=i + 1, _worker=int(worker[-1]))
                        vote_result = worker_model.test_global_model()
                        self.voting_result[vote_result] += 1

            print(self.voting_result)

            max(self.voting_result, key=self.voting_result.get)
            elected = ''.join([k for k, v in self.voting_result.items() if max(self.voting_result.values()) == v][-1])
            print("elected model", elected)
            self.combinator_class.aggregator(elected)

            return