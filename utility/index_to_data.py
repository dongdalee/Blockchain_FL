import os
import shutil

ORIGIN_DATA = "./mnist_png"

TRAIN_INDEX_FILE_PATH = "./train_index"
TEST_INDEX_FILE_PATH = "./test_index"

def handler(mode):
    if mode == "train":
        INDEX_FILE_PATH = TRAIN_INDEX_FILE_PATH
    elif mode == "test":
        INDEX_FILE_PATH = TEST_INDEX_FILE_PATH

    with open(INDEX_FILE_PATH, "r") as file:
        index_data = file.readlines()

    index_data = "".join(index_data)
    index_data = eval(index_data)


    for worker in index_data:
        worker_label = index_data[worker]
        print(worker_label)

        print("{0} {1} data copy!".format(worker, mode))

        for label in worker_label:
            for index in worker_label[label]:
                if mode == "train":
                    copy_path = "./copy_data/" + worker + "/train/" + label + "/" # worker 구분할때
                    # copy_path = "./copy_data" + "/train/" + label + "/" # worker 구분 안할 때
                elif mode =="test":
                    copy_path = "./copy_data/" + worker + "/test/" + label + "/" # worker 구분할때
                    # copy_path = "./copy_data" + "/test/" + label + "/" # worker 구분 안할 때

                if not os.path.isdir(copy_path):
                    os.makedirs(copy_path)

                if mode == "train":
                    shutil.copy2(ORIGIN_DATA + "/" + "train/" + label + "/" + index, "./copy_data/" + worker + "/train/" + label + "/" + index) # worker 구분할때
                    # shutil.copy2(ORIGIN_DATA + "/" + "train/" + label + "/" + index, "./copy_data" + "/train/" + label + "/" + index) # worker 구분 안할 때
                elif mode == "test":
                    shutil.copy2(ORIGIN_DATA + "/" + "test/" + label + "/" + index,"./copy_data/" + worker + "/test/" + label + "/" + index) # worker 구분할때
                    # shutil.copy2(ORIGIN_DATA + "/" + "test/" + label + "/" + index,"./copy_data" + "/test/" + label + "/" + index) # worker 구분 안할 때

handler("train")
handler("test")

