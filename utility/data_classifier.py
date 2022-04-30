import shutil
import os
import random

from math import factorial, exp

# WORKER_NUM = 30

# for iid data distribution
# DATA_SET_SIZE = 120
# DATA_DISTRIBUTION = [DATA_SET_SIZE] * WORKER_NUM

# # Mnist train data set
# train_dataset_size = [5293, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
#
# # Mnist test data set
# test_dataset_size = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]

train_dataset_size = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000] # Cifar10
test_dataset_size = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000] # Cifar10

# for non-iid data distribution
DATA_DISTRIBUTION_RATIO = [90, 90, 90, 90, 90, 90, 90, 90, 90, 90]
TRAIN_DATA_DISTRIBUTION_RATIO = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
TEST_DATA_DISTRIBUTION_RATIO = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# # for train data set
for index, value in enumerate(DATA_DISTRIBUTION_RATIO):
    TRAIN_DATA_DISTRIBUTION_RATIO[index] = int(value * train_dataset_size[index] / 100)

# # for test data set
for index, value in enumerate(DATA_DISTRIBUTION_RATIO):
    TEST_DATA_DISTRIBUTION_RATIO[index] = int(value * test_dataset_size[index] / 100)
"""
cifar10_train_path = {'0': "./mnist_png/train/0",
                      '1': "./mnist_png/train/1",
                      '2': "./mnist_png/train/2",
                      '3': "./mnist_png/train/3",
                      '4': "./mnist_png/train/4",
                      '5': "./mnist_png/train/5",
                      '6': "./mnist_png/train/6",
                      '7': "./mnist_png/train/7",
                      '8': "./mnist_png/train/8",
                      '9': "./mnist_png/train/9"}

cifar10_test_path = {'0': "./mnist_png/test/0",
                     '1': "./mnist_png/test/1",
                     '2': "./mnist_png/test/2",
                     '3': "./mnist_png/test/3",
                     '4': "./mnist_png/test/4",
                     '5': "./mnist_png/test/5",
                     '6': "./mnist_png/test/6",
                     '7': "./mnist_png/test/7",
                     '8': "./mnist_png/test/8",
                     '9': "./mnist_png/test/9"}
"""
cifar10_train_path = {'airplane': "./CIFAR_10/train/airplane",
                      'automobile': "./CIFAR_10/train/automobile",
                      'bird': "./CIFAR_10/train/bird",
                      'cat': "./CIFAR_10/train/cat",
                      'deer': "./CIFAR_10/train/deer",
                      'dog': "./CIFAR_10/train/dog",
                      'frog': "./CIFAR_10/train/frog",
                      'horse': "./CIFAR_10/train/horse",
                      'ship': "./CIFAR_10/train/ship",
                      'truck': "./CIFAR_10/train/truck"}

cifar10_test_path = {'airplane': "./CIFAR_10/test/airplane",
                      'automobile': "./CIFAR_10/test/automobile",
                      'bird': "./CIFAR_10/test/bird",
                      'cat': "./CIFAR_10/test/cat",
                      'deer': "./CIFAR_10/test/deer",
                      'dog': "./CIFAR_10/test/dog",
                      'frog': "./CIFAR_10/test/frog",
                      'horse': "./CIFAR_10/test/horse",
                      'ship': "./CIFAR_10/test/ship",
                      'truck': "./CIFAR_10/test/truck"}

directory_list = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# directory_list = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


def create_folder(worker_directory, class_name, d_type):
    try:
        if not os.path.exists("./data/" + worker_directory + "/" + d_type + "/" + class_name):
            os.makedirs("./data/" + worker_directory + "/" + d_type + "/" + class_name)
    except OSError:
        print("Error: Creating directory. " + "./data/" + worker_directory + "/" + class_name)


def data_copy(file_path, worker_directory, class_name, data_set_size, data_type):
    data_list = os.listdir(file_path)
    print("list:", len(data_list), "       size:", data_set_size)
    sample_list = random.sample(data_list, data_set_size)
    for index in sample_list:
        shutil.copy2(file_path + "/" + str(index), "./data/" + str(worker_directory) + "/" + data_type + "/" + "/" + str(class_name) + "/" + str(index))


def create_training_data(worker_directory, data_set_size, data_type, in_class_name):
    if data_type == 'train':
        for class_name in directory_list:
            create_folder(str(worker_directory), str(class_name), 'train')

        # for index, class_key in enumerate(cifar10_train_path):
        data_copy(cifar10_train_path[in_class_name], worker_directory, str(in_class_name), data_set_size, 'train')
    elif data_type == 'test':
        for class_name in directory_list:
            create_folder(str(worker_directory), str(class_name), 'test')

        # for index, class_key in enumerate(cifar10_test_path):
        data_copy(cifar10_test_path[in_class_name], worker_directory, str(in_class_name), data_set_size, 'test')


def check_len(worker_directory, data_type):
    if data_type == "train":
        print("========= train data =========")
        for i, class_name in enumerate(directory_list):
            print(i, " ", class_name, ": ",
                  len(os.listdir("./data/" + worker_directory + "/train/" + "/" + class_name)))
    elif data_type == "test":
        print("========= test data =========")
        for i, class_name in enumerate(directory_list):
            print(i, " ", class_name, ": ", len(os.listdir("./data/" + worker_directory + "/test/" + "/" + class_name)))


"""
for i in range(0, WORKER_NUM):
    create_training_data("worker" + str(i), TRAIN_DATA_DISTRIBUTION_RATIO[0], 'train', "0")
    create_training_data("worker" + str(i), TRAIN_DATA_DISTRIBUTION_RATIO[1], 'train', "1")
    create_training_data("worker" + str(i), TRAIN_DATA_DISTRIBUTION_RATIO[2], 'train', "2")
    create_training_data("worker" + str(i), TRAIN_DATA_DISTRIBUTION_RATIO[3], 'train', "3")
    create_training_data("worker" + str(i), TRAIN_DATA_DISTRIBUTION_RATIO[4], 'train', "4")
    create_training_data("worker" + str(i), TRAIN_DATA_DISTRIBUTION_RATIO[5], 'train', "5")
    create_training_data("worker" + str(i), TRAIN_DATA_DISTRIBUTION_RATIO[6], 'train', "6")
    create_training_data("worker" + str(i), TRAIN_DATA_DISTRIBUTION_RATIO[7], 'train', "7")
    create_training_data("worker" + str(i), TRAIN_DATA_DISTRIBUTION_RATIO[8], 'train', "8")
    create_training_data("worker" + str(i), TRAIN_DATA_DISTRIBUTION_RATIO[9], 'train', "9")
    check_len("worker" + str(i), "train")

for i in range(0, WORKER_NUM):
    create_training_data("worker" + str(i), TEST_DATA_DISTRIBUTION_RATIO[0], 'test', "0")
    create_training_data("worker" + str(i), TEST_DATA_DISTRIBUTION_RATIO[1], 'test', "1")
    create_training_data("worker" + str(i), TEST_DATA_DISTRIBUTION_RATIO[2], 'test', "2")
    create_training_data("worker" + str(i), TEST_DATA_DISTRIBUTION_RATIO[3], 'test', "3")
    create_training_data("worker" + str(i), TEST_DATA_DISTRIBUTION_RATIO[4], 'test', "4")
    create_training_data("worker" + str(i), TEST_DATA_DISTRIBUTION_RATIO[5], 'test', "5")
    create_training_data("worker" + str(i), TEST_DATA_DISTRIBUTION_RATIO[6], 'test', "6")
    create_training_data("worker" + str(i), TEST_DATA_DISTRIBUTION_RATIO[7], 'test', "7")
    create_training_data("worker" + str(i), TEST_DATA_DISTRIBUTION_RATIO[8], 'test', "8")
    create_training_data("worker" + str(i), TEST_DATA_DISTRIBUTION_RATIO[9], 'test', "9")
    check_len("worker" + str(i), "test")
"""