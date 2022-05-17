import server_receiver as receiver
import server_send as sender
import parameter as p
from round_checker import current_round_checker
from model_voting import Voting, load_model, fed_avg
import torch

current_round = current_round_checker()
receiver.runServer()

"""
# global blockchain의 이전 모델과 업데이트된 모델을 비교투표 하여 global model 생성
if curren_round == 1:
    # print("=========== Round 1 Model Aggregation ===========")
    Logger("server_logs" + str(self.round)).log("=========== Round 1 Model Aggregation ===========")
    Voting(1, "A+B+C+D+E").handler()
else:
    # print("=========== s1+s2 voting ===========")
    Logger("server_logs" + str(self.round)).log(("=========== s1+s2 voting ===========")
    Voting(curren_round, "A+B").handler()

    # print("========== s1+s2+s3 voting ==========")
    Logger("server_logs" + str(self.round)).log("========== s1+s2+s3 voting ==========")
    Voting(curren_round, "A+B+C").handler()

    # print("========= s1+s2+s3+s4 voting =========")
    Logger("server_logs" + str(self.round)).log("========= s1+s2+s3+s4 voting =========")
    Voting(curren_round, "A+B+C+D").handler()

    # print("======== s1+s2+s3+s4+s5 voting ========")
    Logger("server_logs" + str(self.round)).log("======== s1+s2+s3+s4+s5 voting ========")
    Voting(curren_round, "A+B+C+D+E").handler()
"""

"""
# 각 shard로부터 3개의 모델이 업로드되면 투표를 통해 모델을 선택한 후 가장 좋은 모델을 이용하여 global model을 생성한다.
print("================== MODEL VOTING ==================")
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
"""

# 동기적으로 FedAvg
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
