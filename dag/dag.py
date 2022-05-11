from uuid import uuid4
from collections import OrderedDict
import numpy as np
import random
import hashlib
import torch

import wallet
from worker import Worker
import transaction as tx
import parameter as p
from util import Logger, vector_similarity


GENESIS_ID = str(uuid4()).replace('-', '')

GENESIS_KEYS = wallet.generate_wallet(inital=True)
GENESIS_WORKER = Worker('genesis_worker', 0.001, False)

class Tangle:
    def __init__(self):
        self.genesis_worker = GENESIS_WORKER
        self.genesis_model = self.genesis_worker.model
        self.transactions = {GENESIS_ID: tx.transaction_block(GENESIS_KEYS['public_key'],
                                                              GENESIS_KEYS['public_key'],
                                                              self.genesis_model,
                                                              wallet.sign(GENESIS_KEYS['private_key'], transaction_dict=OrderedDict({'sender_public_key': GENESIS_KEYS['public_key'],
                                                                                                                                     'recipient_public_key': GENESIS_KEYS['public_key'],
                                                                                                                                     'payload': self.genesis_model})),
                                                              GENESIS_ID, None, None, None, self.genesis_worker.worker_id)}

        self.edges = {GENESIS_ID: []}
        self.reverse_edges = {GENESIS_ID: []}
        self.worker_cumulative_weight_dict = {}
        for id_index in np.arange(p.WORKER_NUM):
            self.worker_cumulative_weight_dict["worker" + str(id_index)] = 0


    def add_transaction(self, transaction: tx.transaction_block):
        self.transactions[transaction.tx_id] = transaction
        self.add_edges(transaction)


    def add_edges(self, transaction: tx.transaction_block):
        # Creating the forward (in time) edge dict
        approved = transaction.approved_tx[0]
        self.reverse_edges[transaction.tx_id] = []
        if transaction.tx_id not in self.edges:
            self.edges[transaction.tx_id] = approved

        if approved[0] not in self.reverse_edges:
            self.reverse_edges[approved[0]] = [transaction.tx_id]
        else:
            self.reverse_edges[approved[0]].append(transaction.tx_id)

        if approved[1] not in self.reverse_edges:
            self.reverse_edges[approved[1]] = [transaction.tx_id]
        else:
            self.reverse_edges[approved[1]].append(transaction.tx_id)


    def random_walk_weighted(self, current_node=GENESIS_ID):
        if len(self.reverse_edges[current_node]) == 0:
            return current_node

        if len(self.reverse_edges[current_node]) < 3:
            option = np.random.choice(np.arange(0, 2))
            if option == 0:
                return current_node

        prob = []
        for next_node in self.reverse_edges[current_node]:
            prob.append(self.transactions[next_node].cumulative_weight)

        prob = prob / np.sum(prob)

        choice = np.random.choice(np.arange(0, len(self.reverse_edges[current_node])), p=prob)
        return self.random_walk_weighted(self.reverse_edges[current_node][choice])


    def find_tips(self, algo='weighted_random_walk', local_worker=None):
        if algo == 'recently_added':
            tips_list = list(random.sample(set(list(self.transactions.keys())[-2:]), 2))
            local_worker.approve_list[local_worker.round] = tips_list
            return tips_list

        elif algo == 'weighted_random_walk':
            tips_list = []
            for n in range(2):
                tips_dict = {}
                for i in range(1):
                    tip = self.random_walk_weighted()
                    if tip not in tips_dict:
                        tips_dict[tip] = 1
                    else:
                        tips_dict[tip] += 1
                temp_max = 0
                max_tip = ''
                for i in tips_dict:
                    if tips_dict[i] > temp_max:
                        temp_max = tips_dict[i]
                        max_tip = i
                print(max_tip, type(max_tip))
                tips_list.append(max_tip)
            return tips_list

        elif algo == 'high_accuracy':
            tips_list = []
            model_dict = {}
            search_list = []
            widen_search_space = 2

            for i, tx_id in enumerate(self.transactions):
                if i/p.WORKER_NUM < local_worker.round-2:
                    continue
                else:
                    search_list.append(self.transactions[tx_id])

            if len(search_list) < 2:
                for i, tx_id in enumerate(self.transactions):
                    search_list.append(self.transactions[tx_id])


            # cumulative weight를 기준으로 내림차순 정렬을 한다.
            search_list.sort(key=lambda x: x.cumulative_weight, reverse=True)

            Logger(str(local_worker.worker_id)).log("|-------------------- Search List TX --------------------|")
            # for target in search_list[:p.SEARCH_SPACE_RANGE]:
            for target in search_list:
                model = target.get_payload()
                own_worker_id = target.tx_worker_id
                accuracy = local_worker.evaluation(model, False)
                if p.SIMILARITY == "cosine":
                    similarity = vector_similarity(local_worker.model, model)
                    model_dict[target.tx_id] = accuracy + similarity, own_worker_id
                    Logger(str(local_worker.worker_id)).log("Worker: {0}, F1 Score: {1:.5f} {2} Similarity: {3:.2f}".format(target.tx_worker_id, accuracy, p.SIMILARITY, similarity))
                else:
                    model_dict[target.tx_id] = accuracy, own_worker_id
                    Logger(str(local_worker.worker_id)).log("Worker: {0}, F1 Score: {1:.5f}".format(target.tx_worker_id, accuracy))

            Logger(str(local_worker.worker_id)).log("|---------------- total search list: {0} ----------------|".format(len(search_list)))

            sorted_by_value = sorted(model_dict.items(), key=lambda x: x[1][0], reverse=True)

            if p.LEARNING_MEASURE == "accuracy":
                Logger(str(local_worker.worker_id)).log("TIP1: Tx ID: {0} | Worker ID: {1} | Accuracy: {2:.5f}".format(sorted_by_value[0][0], sorted_by_value[0][1][1], sorted_by_value[0][1][0]))
                Logger(str(local_worker.worker_id)).log("TIP2: Tx ID: {0} | Worker ID: {1} | Accuracy: {2:.5f}".format(sorted_by_value[1][0], sorted_by_value[1][1][1], sorted_by_value[1][1][0]))
                Logger(str(local_worker.worker_id)).log("{0} -> {1} | {0} -> {2}".format(local_worker.worker_id, sorted_by_value[0][1][1], sorted_by_value[1][1][1]))
            elif p.LEARNING_MEASURE == "f1 score":
                Logger(str(local_worker.worker_id)).log("TIP1: Tx ID: {0} | Worker ID: {1} | F1 Score: {2:.5f}".format(sorted_by_value[0][0], sorted_by_value[0][1][1], sorted_by_value[0][1][0]))
                Logger(str(local_worker.worker_id)).log("TIP2: Tx ID: {0} | Worker ID: {1} | F1 Score: {2:.5f}".format(sorted_by_value[1][0], sorted_by_value[1][1][1], sorted_by_value[1][1][0]))
                Logger(str(local_worker.worker_id)).log("{0} -> {1} | {0} -> {2}".format(local_worker.worker_id, sorted_by_value[0][1][1], sorted_by_value[1][1][1]))

            tips_list.append(sorted_by_value[0][0])
            tips_list.append(sorted_by_value[1][0])

            # update cumulative weight
            self.transactions[tips_list[0]].cumulative_weight += 1
            self.transactions[tips_list[1]].cumulative_weight += 1

            local_worker.approve_list[local_worker.round] = tips_list

            # code to get Worekr's all cumulative weight
            if sorted_by_value[0][1][1] in self.worker_cumulative_weight_dict.keys():
                self.worker_cumulative_weight_dict[sorted_by_value[0][1][1]] += 1
            if sorted_by_value[1][1][1] in self.worker_cumulative_weight_dict.keys():
                self.worker_cumulative_weight_dict[sorted_by_value[1][1][1]] += 1

            return tips_list

        else:
            tips_list = random.sample(list(self.transactions.keys()), 2)
            local_worker.approve_list[local_worker.round] = tips_list
            return tips_list


tangle = Tangle()


def get_previous_hashes(tips):
    previous_hashes = [tangle.transactions[tips[0]].get_hash(), tangle.transactions[tips[0]].get_hash()]
    return previous_hashes


def valid_proof(previous_hashes, transaction_dict, nonce):
    guess = (str(previous_hashes)+ str(transaction_dict) + str(nonce)).encode('utf-8')
    h = hashlib.new('sha256')
    h.update(guess)
    guess_hash = h.hexdigest()
    return guess_hash[:p.DIFFICULTY] == '0'*p.DIFFICULTY


def proof_of_work(previous_hashes, transaction_dict):
    nonce = 0
    while not valid_proof(previous_hashes, transaction_dict, nonce):
        nonce = nonce + 1

    return nonce


def generate_transactions(initial=False, initial_count=5, tip_selection_algo='weighted_random_walk', payload=None, local_worker=None):
    if initial:
        for i in range(initial_count):
            # print("genesis transaction created")
            keys = wallet.generate_wallet()
            transaction_dict = OrderedDict({
                'sender_public_key': GENESIS_KEYS['public_key'],
                'recipient_public_key': keys['public_key'],
                'payload': tangle.genesis_model
            })

            tips = [GENESIS_ID, GENESIS_ID]
            previous_hashes = get_previous_hashes(tips)
            transaction = tx.transaction_block(
                sender_public_key=transaction_dict['sender_public_key'],
                recipient_public_key=transaction_dict['recipient_public_key'],
                payload=transaction_dict['payload'],
                signature=wallet.sign(GENESIS_KEYS['private_key'], transaction_dict),
                tx_id=str(uuid4()).replace('-', ''),
                approved_tx=tips,
                nonce=proof_of_work(previous_hashes, transaction_dict),
                previous_hashes=previous_hashes,
                tx_worker_id=tangle.genesis_worker.worker_id)

            tangle.add_transaction(transaction)
    else:
        keys = random.choice(wallet.WALLET_LIST)
        transaction_dict = OrderedDict({
            'sender_public_key': GENESIS_KEYS['public_key'],
            'recipient_public_key': keys['public_key'],
            'payload': payload
        })
        tips = tangle.find_tips(algo=tip_selection_algo, local_worker=local_worker)
        previous_hashes = get_previous_hashes(tips)
        transaction = tx.transaction_block(
            sender_public_key=transaction_dict['sender_public_key'],
            recipient_public_key=transaction_dict['recipient_public_key'],
            payload=transaction_dict['payload'],
            signature=wallet.sign(GENESIS_KEYS['private_key'], transaction_dict),
            tx_id=str(uuid4()).replace('-', ''),
            approved_tx=tips,
            nonce=proof_of_work(previous_hashes, transaction_dict),
            previous_hashes=previous_hashes,
            tx_worker_id=local_worker.worker_id)
        tangle.add_transaction(transaction)


def save_shard_global_model(current_round):
    model_dict = {}

    model_acc_dict = {}

    total_length = len(tangle.transactions)
    search_space = np.arange(total_length, total_length-((2*p.WORKER_NUM)+(p.WORKER_NUM/2)), -1)

    for i, tx_id in enumerate(tangle.transactions):
        if i in search_space and tangle.transactions[tx_id].tx_worker_id != "genesis_worker":
            # sort by cumulative weight
            transaction = tangle.transactions[tx_id]
            cumulative_weight = transaction.cumulative_weight
            model_dict[tangle.transactions[tx_id].tx_id] = cumulative_weight
        else:
            continue

    for i, tx_id in enumerate(tangle.transactions):
        if i in search_space:
            # sort by accuracy
            model = tangle.transactions[tx_id].get_payload()
            accuracy = tangle.genesis_worker.evaluation(model, False)
            model_acc_dict[tangle.transactions[tx_id].tx_id] = accuracy
        else:
            continue

    """
    # high cumulative weight model
    # print("cumulative weight:", model_dict)
    max_cumulative_weight_model_tx_id = max(model_dict, key=lambda i: model_dict[i])
    max_cumulative_weight_model = tangle.transactions[max_cumulative_weight_model_tx_id].get_payload()
    """

    # high cumulative weight model
    descending_cumulative_weight = sorted(model_dict.items(), key=lambda x: x[1], reverse=True)
    upload_model_tx_id = descending_cumulative_weight[:p.UPLOAD_MODEL_NUM]
    print(upload_model_tx_id)
    # high accuracy model
    # print("acc weight:", model_acc_dict)
    max_accuracy_model_tx_id = max(model_acc_dict, key=lambda i: model_acc_dict[i])
    max_accuracy_model = tangle.transactions[max_accuracy_model_tx_id].get_payload()

    for index, tx_id_tuple in enumerate(upload_model_tx_id):
        tx_id = tx_id_tuple[0]
        top_cumulative_weight_model = tangle.transactions[tx_id].get_payload()
        torch.save(top_cumulative_weight_model.state_dict(), p.SAVE_SHARD_MODEL_PATH + str(current_round) + "/" + p.SHARD_ID + "_" + str(index) + ".pt") # shard1_0.pt, shard1_1.pt
        Logger(str("transaction")).log("Sort by Cumulative Weight: save shard model: {0} | Worker ID: {1} | Accuracy: {2} | Cumulative Weight {3}".format(tx_id, tangle.transactions[tx_id].tx_worker_id, tangle.genesis_worker.evaluation(top_cumulative_weight_model, True), tangle.transactions[tx_id].cumulative_weight))
    # torch.save(max_cumulative_weight_model.state_dict(), p.SAVE_SHARD_MODEL_PATH + str(current_round) + "/" + p.SHARD_ID + ".pt")

    print(" ")
    # Logger(str("transaction")).log("Sort by Cumulative Weight: save shard model: {0} | Worker ID: {1} | Accuracy: {2} | Cumulative Weight {3}".format(max_cumulative_weight_model_tx_id, tangle.transactions[max_cumulative_weight_model_tx_id].tx_worker_id, tangle.genesis_worker.evaluation(max_cumulative_weight_model, True), tangle.transactions[max_cumulative_weight_model_tx_id].cumulative_weight))
    Logger(str("transaction")).log("Sort by Accuracy: save shard model: {0} | Worker ID: {1} | Accuracy: {2}".format(max_accuracy_model_tx_id, tangle.transactions[max_accuracy_model_tx_id].tx_worker_id, tangle.genesis_worker.evaluation(max_accuracy_model, True)))

    # return max_cumulative_weight_model
