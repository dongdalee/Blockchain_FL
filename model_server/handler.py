import server_receiver as receiver
import server_send as sender
import parameter as p
from round_checker import current_round_checker
from model_voting import Voting, load_model, fed_avg
import torch

current_round = current_round_checker()
receiver.runServer()

"""
print("================== MODEL VOTING ==================")
handler = Voting(current_round)
handler.model_voter()
handler.model_voter()
handler.model_voter()
handler.model_voter()
"""
handler = Voting(current_round)
handler.model_voter()
handler.model_voter()
handler.model_voter()
handler.model_voter()

shard1 = load_model("./model/" + str(current_round) + "/shard1.pt")
shard2 = load_model("./model/" + str(current_round) + "/shard2.pt")
shard3 = load_model("./model/" + str(current_round) + "/shard3.pt")
shard4 = load_model("./model/" + str(current_round) + "/shard4.pt")
shard5 = load_model("./model/" + str(current_round) + "/shard5.pt")

print("aggregation model")
avg = fed_avg(shard1, shard2, shard3, shard4, shard5)
torch.save(avg.state_dict(), "./model/" + str(current_round) + "/aggregation.pt")

for address in p.SHARD_ADDR_LIST:
    for filename in p.FILE_LIST:
        shard = sender.sendServer(address["ip"], address["port"])
        if filename == ".DS_Store":
            pass
        shard.send_file("model/" + str(current_round) + "/" + filename)
        shard.clientSock.close()
        print("socket closed")
