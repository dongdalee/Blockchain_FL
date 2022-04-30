import server_receiver as receiver
import server_send as sender
import parameter as p
from round_checker import current_round_checker
from model_voting import Voting

curren_round = current_round_checker()
# curren_round = 1
receiver.runServer()

# aggregate model: make global model
# print("model aggregation")
# aggregation.handler()

if curren_round == 1:
    print("=========== Round 1 Model Aggregation ===========")
    Voting(1, "A+B+C+D+E").handler()
else:
    print("=========== s1+s2 voting ===========")
    Voting(curren_round, "A+B").handler()

    print("========== s1+s2+s3 voting ==========")
    Voting(curren_round, "A+B+C").handler()

    print("========= s1+s2+s3+s4 voting =========")
    Voting(curren_round, "A+B+C+D").handler()

    print("======== s1+s2+s3+s4+s5 voting ========")
    Voting(curren_round, "A+B+C+D+E").handler()


for address in p.SHARD_ADDR_LIST:
    for filename in p.FILE_LIST:
         shard = sender.sendServer(address["ip"], address["port"])
         if filename == ".DS_Store":
             pass
         shard.send_file("model/" + str(curren_round) + "/" + filename)
         shard.clientSock.close()
         print("socket closed")
