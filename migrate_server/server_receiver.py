from socket import *
import socket, threading
import parameter as p
import concurrent.futures

migrate_model_list = []

lock = threading.Lock()

train_shard_list = {}
test_shard_list = {}

PAYLOAD = {
    "train": {},
    "test": {}
}


for i in range(1, p.SHARD_NUM + 1):
    train_shard_list.update({"shard" + str(i): []})
    test_shard_list.update({"shard" + str(i): []})


def set_payload(train_data, test_data):
    PAYLOAD["train"] = train_data
    PAYLOAD["test"] = test_data


def add_worker_to_shard(msg, data_type):
    for shard_id in range(1, len(msg[data_type])+1):
        for worker_id in range(0, len(msg[data_type]["shard" + str(shard_id)])):
            lock.acquire()
            if data_type == "train":
                train_shard_list["shard" + str(shard_id)].append(msg[data_type]["shard" + str(shard_id)][worker_id])
            elif data_type == "test":
                test_shard_list["shard" + str(shard_id)].append(msg[data_type]["shard" + str(shard_id)][worker_id])
            lock.release()


def dataleader(file_path):
    read_file = open(file_path, 'r')
    data = read_file.read()
    data = eval(data)

    return data


def datawriter():
    file = open("./migrate/migration_info.txt", 'w')
    file.write(str(PAYLOAD))
    file.close()


def binder(client_socket, addr):
    print('클라이언트 [ {0} ] 에서 접속'.format(addr))
    try:
        # while True:
        msg = client_socket.recv(4)
        length = int.from_bytes(msg, "little")
        msg = client_socket.recv(length)

        filename = msg.decode()
        if filename == '/exit':
            print("-> {0} 클라이언트에 의해 중단".format(client_socket.client_address[0]))

        data = client_socket.recv(1024)

        data_transferred = 0

        with open(filename, 'wb') as f:
            try:
                while data:
                    f.write(data)
                    data_transferred += len(data)
                    data = client_socket.recv(1024)
            except Exception as ex:
                print(ex)

        print('송신완료{0}, 송신량{1}'.format(filename, data_transferred))

        if filename not in migrate_model_list:
            lock.acquire()
            migrate_model_list.append(filename)
            lock.release()
            print(migrate_model_list)

        if len(migrate_model_list) == p.SHARD_NUM:
            for path in migrate_model_list:
                data = dataleader(path)
                add_worker_to_shard(data, "train")
                add_worker_to_shard(data, "test")

            set_payload(train_shard_list, test_shard_list)
            datawriter()
            print("server closed")
            return "exit"
            # server_socket.close()
            # break
        else:
            return "continue"

    except Exception as e:
        # print(e)
        print("except : ", e)
    finally:
        client_socket.close()


def runServer():
    print("-> migration server start")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((p.HOST, p.PORT))
    server_socket.listen()
    try:
        while True:
            client_socket, addr = server_socket.accept()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(binder, client_socket, addr)
                return_value = future.result()
                if return_value == "exit":
                    break
    except:
        print("migration server closed")
    finally:
        server_socket.close()


