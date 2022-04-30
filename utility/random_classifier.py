import random
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil
from data_classifier import create_training_data, check_len

WORKER_NUM = 10

DROP_COUNT = 8

# train_dataset_size = [5293, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949] # Mnist
# test_dataset_size = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009] # Mnist

train_dataset_size = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000] # Cifar10
test_dataset_size = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000] # Cifar10

# label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

TRAIN_DATA_DISTRIBUTION = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
TEST_DATA_DISTRIBUTION = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

REMOVE_LABEL_INDEX = [2, 3, 4, 5, 6, 7, 8, 9]
# make random dataset distribution
train_dataset_list = []
test_dataset_list = []


for i in range(0, WORKER_NUM):
    dataset = random.sample(range(10, 100, 5), 10)
    # print(dataset)
    for index, value in enumerate(dataset):
        TRAIN_DATA_DISTRIBUTION[index] = int(value * train_dataset_size[index] / 100)
        TEST_DATA_DISTRIBUTION[index] = int(value * test_dataset_size[index] / 100)

    train_dataset_list.append(TRAIN_DATA_DISTRIBUTION.copy())
    test_dataset_list.append(TEST_DATA_DISTRIBUTION.copy())


def random_number():
    value_list = []
    ran_num = random.randint(0, 9)
    for i in range(DROP_COUNT):
        while ran_num in value_list:
            ran_num = random.randint(0, 9)
        value_list.append(ran_num)

    value_list.sort()

    return value_list


def drop_dataset(mode):
    if mode == "random":
        for worker_index in range(0, WORKER_NUM):
            random_index = random_number()

            for label_index in random_index:
                random_data_size = random.randint(1, 20)
                train_dataset_list[worker_index][label_index] = random_data_size

                test_dataset_list[worker_index][label_index] = random_data_size


            print("worker: ", worker_index)
            print(train_dataset_list[worker_index])
            print(test_dataset_list[worker_index])
            print(" ")
    if mode == "selection":
        drop_index_list = REMOVE_LABEL_INDEX
        for worker_index in range(0, WORKER_NUM):

            for label_index in drop_index_list:
                random_data_size = random.randint(1, 20)
                train_dataset_list[worker_index][label_index] = 0

                test_dataset_list[worker_index][label_index] = 0


            print("worker: ", worker_index)
            print(train_dataset_list[worker_index])
            print(test_dataset_list[worker_index])
            print(" ")


drop_dataset("selection")

def plot_dataset_distribution(data_type):
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 10))
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    # plt.tight_layout()
    origin_data = plt.subplot(4, 4, 1)

    if data_type == "train":
        origin_data.set_title("Train Origin Data")
        origin_data.bar(x, train_dataset_size)
        plt.xticks(np.arange(0, 10), labels)
        plt.yticks([1000, 3000, 5000, 7000])
    elif data_type == "test":
        origin_data.set_title("Test Origin Data")
        origin_data.bar(x, test_dataset_size)
        plt.xticks(np.arange(0, 10))
        plt.yticks([100, 400, 800, 1200])

    if data_type == "train":
        dataset_list = train_dataset_list
    elif data_type == "test":
        dataset_list = test_dataset_list

    i = 2

    for index, data in enumerate(dataset_list):
        sub_data = plt.subplot(4, 4, i)
        sub_data.set_title("worker" + str(index))
        sub_data.bar(x, data)
        plt.xticks(np.arange(0, 10))
        if data_type == "train":
            plt.yticks([1000, 3000, 5000, 7000])
        elif data_type == "test":
            plt.yticks([100, 400, 800, 1200])
        i += 1

    if data_type == "train":
        plt.tight_layout()
        plt.savefig("train_distribution.png")
    elif data_type == "test":
        plt.tight_layout()
        plt.savefig("test_distribution.png")

    plt.show()


plot_dataset_distribution("train")
plot_dataset_distribution("test")

for i in range(0, WORKER_NUM):
    create_training_data("worker" + str(i), train_dataset_list[i][0], 'train', "airplane")
    create_training_data("worker" + str(i), train_dataset_list[i][1], 'train', "automobile")
    create_training_data("worker" + str(i), train_dataset_list[i][2], 'train', "bird")
    create_training_data("worker" + str(i), train_dataset_list[i][3], 'train', "cat")
    create_training_data("worker" + str(i), train_dataset_list[i][4], 'train', "deer")
    create_training_data("worker" + str(i), train_dataset_list[i][5], 'train', "dog")
    create_training_data("worker" + str(i), train_dataset_list[i][6], 'train', "frog")
    create_training_data("worker" + str(i), train_dataset_list[i][7], 'train', "horse")
    create_training_data("worker" + str(i), train_dataset_list[i][8], 'train', "ship")
    create_training_data("worker" + str(i), train_dataset_list[i][9], 'train', "truck")
    check_len("worker" + str(i), "train")

for i in range(0, WORKER_NUM):
    create_training_data("worker" + str(i), test_dataset_list[i][0], 'test', "airplane")
    create_training_data("worker" + str(i), test_dataset_list[i][1], 'test', "automobile")
    create_training_data("worker" + str(i), test_dataset_list[i][2], 'test', "bird")
    create_training_data("worker" + str(i), test_dataset_list[i][3], 'test', "cat")
    create_training_data("worker" + str(i), test_dataset_list[i][4], 'test', "deer")
    create_training_data("worker" + str(i), test_dataset_list[i][5], 'test', "dog")
    create_training_data("worker" + str(i), test_dataset_list[i][6], 'test', "frog")
    create_training_data("worker" + str(i), test_dataset_list[i][7], 'test', "horse")
    create_training_data("worker" + str(i), test_dataset_list[i][8], 'test', "ship")
    create_training_data("worker" + str(i), test_dataset_list[i][9], 'test', "truck")
    check_len("worker" + str(i), "test")


remove_label_list = []

for label in labels:
    if labels.index(label) not in REMOVE_LABEL_INDEX:
        pass
    else:
        remove_label_list.append(label)


for worker_index in range(0, WORKER_NUM):
    train_path = os.listdir("./data/worker" + str(worker_index) + "/train/")
    for remove_label in remove_label_list:
        if remove_label in train_path:
            shutil.rmtree("./data/worker" + str(worker_index) + "/train/"+remove_label, ignore_errors=True)

    test_path = os.listdir("./data/worker" + str(worker_index) + "/test/")
    for remove_label in remove_label_list:
        if remove_label in test_path:
            shutil.rmtree("./data/worker" + str(worker_index) + "/test/" + remove_label, ignore_errors=True)