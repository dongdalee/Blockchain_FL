import time

import server_receiver as receiver
import server_send as sender
import parameter as p


receiver.runServer()
print("receiver closed")

time.sleep(10)

for address in p.SHARD_ADDR_LIST:
    shard = sender.sendServer(address["ip"], address["port"])
    shard.send_file("./migrate/migration_info.txt")
    shard.clientSock.close()
    print("socket closed")