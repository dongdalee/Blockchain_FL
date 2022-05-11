import server_receiver as receiver
import server_send as sender
import parameter as p
from round_checker import current_round_checker
from model_voting import Voting

curren_round = current_round_checker()
receiver.runServer()

print("================== MODEL VOTING ==================")
handler = Voting(curren_round)
handler.model_voter()
handler.model_voter()
handler.model_voter()
handler.model_voter()

for address in p.SHARD_ADDR_LIST:
    for filename in p.FILE_LIST:
        shard = sender.sendServer(address["ip"], address["port"])
        if filename == ".DS_Store":
            pass
        shard.send_file("model/" + str(curren_round) + "/" + filename)
        shard.clientSock.close()
        print("socket closed")
