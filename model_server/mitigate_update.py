import os
import random
from dataloader import get_dataloader
from sklearn.metrics import f1_score, accuracy_score
import torch
from util import variable_name
from itertools import product
from parameter import WORKER_DATA, LEARNING_MEASURE, SHARD_LIST, UPLOAD_MODEL_NUM
from util import Logger
from MachineLearningUtility import device, load_model, load_worker, model_fraction, add_model, fed_avg, sub_model

class Worker:
    def __init__(self, *_models, _shard, _worker, _current_round):
        self.models = _models
        self.testloader = get_dataloader(str(_shard), _worker)
        self.ballot = {}
        self.current_round = _current_round


    def test_global_model(self):
        for index, model in enumerate(self.models):
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

            if LEARNING_MEASURE == "f1 score":
                accuracy = round(f1_score(actuals, predictions, average='weighted') * 100, 2)  # f1 score
            else:
                accuracy = round(accuracy_score(actuals, predictions) * 100, 2)  # accuracy

            model_name = variable_name(model)
            Logger("server_logs" + str(self.current_round)).log("Model {0} Accuracy: {1}".format(model_name, accuracy))
            self.ballot[model_name] = accuracy

        max(self.ballot, key=self.ballot.get)
        elected = ''.join([k for k, v in self.ballot.items() if max(self.ballot.values()) == v][0])

        return elected


class Voting:
    def __init__(self, _current_round):
        self.first_voting = True
        self.voting_order = 2

        self.current_round = _current_round
        self.SAVE_MODEL_PATH = "./model/" + str(self.current_round) + "/"
        self.voting_committee = []


    def model_voter(self):
        if self.voting_order == 2:
            Logger("server_logs" + str(self.current_round)).log("################ first voting ################")
            shard_A_models = []
            shard_B_models = []

            input_models = []

            voting_result = {"preA_preB" : 0, "preA_currB" : 0, "currA_preB" : 0, "currA_currB" : 0}

            # ?????? ????????? ???????????? 2?????? ????????? ???????????? ????????? ????????? ??????????????? ?????????.
            random_shards = random.sample(SHARD_LIST, 2)
            SHARD_LIST.remove(random_shards[0])
            SHARD_LIST.remove(random_shards[1])
            self.voting_committee.extend([random_shards[0], random_shards[1]])

            # ???????????? ????????? shard??? ?????? round??? model??? ?????? round??? model??? ????????????.
            # ???????????? ????????? ????????? 1/5??? ?????????. 5 = number of shard
            pre_model_A = load_model("./model/" + str(self.current_round - 1) + "/" + random_shards[0] + ".pt")
            model_fraction(pre_model_A, 1, 5)
            curr_model_A = load_model("./model/" + str(self.current_round) + "/" + random_shards[0] + ".pt")
            model_fraction(curr_model_A, 1, 5)
            shard_A_models.extend([pre_model_A, curr_model_A])

            pre_model_B = load_model("./model/" + str(self.current_round - 1) + "/" + random_shards[1] + ".pt")
            model_fraction(pre_model_B, 1, 5)
            curr_model_B = load_model("./model/" + str(self.current_round) + "/" + random_shards[1] + ".pt")
            model_fraction(curr_model_B, 1, 5)
            shard_B_models.extend([pre_model_B, curr_model_B])

            Logger("server_logs" + str(self.current_round)).log("{0}, {1} Model Loaded".format(random_shards[0], random_shards[1]))

            # model combination??? ?????? input_models ???????????? ???????????? ????????? ???????????? ?????????. -> [[], []] 2?????? ?????????
            input_models.extend([shard_A_models, shard_B_models])

            # ?????? ?????? combination
            combination_models = list(product(*input_models))
            combination_model_names = list(product(*[["pre_" + random_shards[0], "curr_" + random_shards[0]], ["pre_" + random_shards[1], "curr_" + random_shards[1]]]))
            Logger("server_logs" + str(self.current_round)).log("Combination model list: {0} length {1}".format(combination_model_names, len(combination_models)))

            # ????????? ?????? FedAsyncAvg
            preA_preB = add_model(combination_models[0][0], combination_models[0][1])
            model_fraction(preA_preB, 5, 2)

            preA_currB = add_model(combination_models[1][0], combination_models[1][1])
            model_fraction(preA_currB, 5, 2)

            currA_preB = add_model(combination_models[2][0], combination_models[2][1])
            model_fraction(currA_preB, 5, 2)

            currA_currB = add_model(combination_models[3][0], combination_models[3][1])
            model_fraction(currA_currB, 5, 2)

            # ?????? ????????? worker ?????? ???????????? ?????? ??????
            shard1_worker_length = len(load_worker(random_shards[0]))
            shard2_worker_length = len(load_worker(random_shards[1]))

            Logger("server_logs" + str(self.current_round)).log("Voting Committee: {0}".format(self.voting_committee))

            Logger("server_logs" + str(self.current_round)).log("=============== Voting Shard: {0} ===============".format(random_shards[0]))
            for worker_id in range(shard1_worker_length):
                Logger("server_logs" + str(self.current_round)).log("#-----  Worker{0} -----#".format(worker_id))
                worker = Worker(preA_preB, preA_currB, currA_preB, currA_currB, _shard=random_shards[0], _worker=worker_id, _current_round=self.current_round)
                elect_result = worker.test_global_model()
                voting_result[elect_result] += 1
                Logger("server_logs" + str(self.current_round)).log("<----- elected: {0} ----->\n".format(elect_result))

            Logger("server_logs" + str(self.current_round)).log("=============== Voting Shard: {0} ===============".format(random_shards[1]))
            for worker_id in range(shard2_worker_length):
                Logger("server_logs" + str(self.current_round)).log("#----- Worker{0} -----#".format(worker_id))
                worker = Worker(preA_preB, preA_currB, currA_preB, currA_currB, _shard=random_shards[1], _worker=worker_id, _current_round=self.current_round)
                elect_result = worker.test_global_model()
                voting_result[elect_result] += 1
                Logger("server_logs" + str(self.current_round)).log("<----- elected: {0} ----->\n".format(elect_result))

            Logger("server_logs" + str(self.current_round)).log("Voting Result: {0}".format(voting_result))
            max(voting_result, key=voting_result.get)
            elected = ''.join([k for k, v in voting_result.items() if max(voting_result.values()) == v][-1])
            Logger("server_logs" + str(self.current_round)).log("(--------------- Elected Model: {0} ---------------)".format(elected))


            if elected == "preA_preB":
                model_fraction(preA_preB, 2, 5)
                torch.save(preA_preB.state_dict(), "./model/" + str(self.current_round) + "/g1.pt")
            elif elected == "preA_currB":
                model_fraction(preA_currB, 2, 5)
                torch.save(preA_currB.state_dict(), "./model/" + str(self.current_round) + "/g1.pt")
            elif elected == "currA_preB":
                model_fraction(currA_preB, 2, 5)
                torch.save(currA_preB.state_dict(), "./model/" + str(self.current_round) + "/g1.pt")
            elif elected == "currA_currB":
                model_fraction(currA_currB, 2, 5)
                torch.save(currA_currB.state_dict(), "./model/" + str(self.current_round) + "/g1.pt")

            self.voting_order += 1

            return

        else:
            input_models = []
            voting_result = {"global_pre" : 0, "global_curr" : 0}
            # votted_model_list = os.listdir("model/" + str(self.current_round) + "/")

            print("Voting Order: {0}".format(self.voting_order))

            # ?????? ????????? ?????? ?????? ????????? ????????? ????????????.
            if self.voting_order == 5:
                Logger("server_logs" + str(self.current_round)).log("################ load model g3 ################")
                pre_global_model = load_model("model/" + str(self.current_round) + "/g3.pt")
                save_model_name = "aggregation.pt"
            elif self.voting_order == 4:
                Logger("server_logs" + str(self.current_round)).log("################ load model g2 ################")
                pre_global_model = load_model("model/" + str(self.current_round) + "/g2.pt")
                save_model_name = "g3.pt"
            else:
                Logger("server_logs" + str(self.current_round)).log("################ load model g1 ################")
                pre_global_model = load_model("model/" + str(self.current_round) + "/g1.pt")
                save_model_name = "g2.pt"

            # ????????? ???????????? ????????? ????????? ???????????? ???????????? ????????? ????????????.
            random_shards = random.sample(SHARD_LIST, 1)
            SHARD_LIST.remove(random_shards[0])
            self.voting_committee.append(random_shards[0])

            # ?????? ????????? ????????? ????????? ????????? ????????? combination?????? ?????? ???????????? ?????? ????????? ?????? ????????? ????????????.
            input_models.append([pre_global_model])

            pre_model = load_model("./model/" + str(self.current_round - 1) + "/" + random_shards[0] + ".pt")
            model_fraction(pre_model, 1, 5)

            curr_model = load_model("./model/" + str(self.current_round) + "/" + random_shards[0] + ".pt")
            model_fraction(curr_model, 1, 5)

            # model combination??? ?????? input_models ???????????? ???????????? ????????? ???????????? ?????????. -> [[], []] 2?????? ?????????
            input_models.append([pre_model, curr_model])

            # ?????? ?????? combination
            combination_models = list(product(*input_models))
            combination_model_names = list(product(*[["g" + str(self.voting_order - 2)], ["pre_" + random_shards[0], "curr_" + random_shards[0]]]))

            global_pre = add_model(combination_models[0][0], combination_models[0][1])
            model_fraction(global_pre, 5, self.voting_order)

            global_curr = add_model(combination_models[1][0], combination_models[1][1])
            model_fraction(global_curr, 5, self.voting_order)

            Logger("server_logs" + str(self.current_round)).log("Combination model list: {0} length {1}".format(combination_model_names, len(combination_models)))

            Logger("server_logs" + str(self.current_round)).log("Voting Committee: {0}".format(self.voting_committee))


            for shard in self.voting_committee:
                # voting committee??? ?????? ??? ???????????? worker ?????? ????????????.
                worker_length = len(load_worker(shard))

                Logger("server_logs" + str(self.current_round)).log("=============== Voting Shard: {0} ===============".format(shard))
                for worker_id in range(worker_length):
                    Logger("server_logs" + str(self.current_round)).log("#----- Worker{0} -----#".format(worker_id))
                    worker = Worker(global_pre, global_curr, _shard=shard, _worker=worker_id, _current_round=self.current_round)
                    elect_result = worker.test_global_model()
                    voting_result[elect_result] += 1
                    Logger("server_logs" + str(self.current_round)).log("<----- elected: {0} ----->\n".format(elect_result))

            Logger("server_logs" + str(self.current_round)).log("Voting Result: {0}".format(voting_result))
            max(voting_result, key=voting_result.get)
            elected = ''.join([k for k, v in voting_result.items() if max(voting_result.values()) == v][0])
            Logger("server_logs" + str(self.current_round)).log(elected)

            if elected == "global_pre":
                model_fraction(global_pre, self.voting_order, 5)
                torch.save(global_pre.state_dict(), "./model/" + str(self.current_round) + "/" + save_model_name)
            elif elected == "global_curr":
                model_fraction(global_curr, self.voting_order, 5)
                torch.save(global_curr.state_dict(), "./model/" + str(self.current_round) + "/" + save_model_name)

            self.voting_order += 1

            return



