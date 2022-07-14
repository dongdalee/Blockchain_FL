import matplotlib.pyplot as plt
import numpy as np

BLOCK_NUM = 30

total_cumultive_weight = {'worker0' : 0, 'worker1' : 0, 'worker2' : 0, 'worker3' : 0, 'worker4' : 0, 'worker5' : 0, 'worker6' : 0, 'worker7' : 0, 'worker8' : 0, 'worker9' : 0}

for shard in range(1, 6):
    for block in range(1, BLOCK_NUM+1):
        shard_log_path = "experiment_data/shard" + str(shard) + "/" + str(block) + "/logs/transaction"

        with open(shard_log_path, 'r') as file:
            x = file.readlines()[-4]
            # print(block)
            temp = x.split("{")[1]
            cumulative_weight = eval("{" + temp)
            total_cumultive_weight['worker0'] += cumulative_weight['worker0']
            total_cumultive_weight['worker1'] += cumulative_weight['worker1']
            total_cumultive_weight['worker2'] += cumulative_weight['worker2']
            total_cumultive_weight['worker3'] += cumulative_weight['worker3']
            total_cumultive_weight['worker4'] += cumulative_weight['worker4']
            total_cumultive_weight['worker5'] += cumulative_weight['worker5']
            total_cumultive_weight['worker6'] += cumulative_weight['worker6']
            total_cumultive_weight['worker7'] += cumulative_weight['worker7']
            total_cumultive_weight['worker8'] += cumulative_weight['worker8']
            total_cumultive_weight['worker9'] += cumulative_weight['worker9']

            # print("{0}, {1}, {2}, {3}, {4}".format(cumulative_weight['worker0'], cumulative_weight['worker1'], cumulative_weight['worker2'], cumulative_weight['worker3'], cumulative_weight['worker4']))
            print("{0}".format(cumulative_weight['worker0'] + cumulative_weight['worker1'] + cumulative_weight['worker2'] + cumulative_weight['worker3'] + cumulative_weight['worker4']))
            # print(total_cumultive_weight)
            # print("===============================")
    print("-"*50)

print(total_cumultive_weight)


# =========================================================================================================================================
"""
malicious1 = [35, 11, 9, 0, 1, 1, 2, 4, 2, 1, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 5, 0]
malicious2 = [61, 11, 5, 20, 10, 26, 5, 20, 3, 13, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0]

round = np.arange(0, len(malicious1))

plt.title("MNIST Round5 lr=0.0001 lambda=8 malicious 5")
plt.plot(round, malicious1, 'x--', label="A+S+M")
plt.plot(round, malicious2, 'd--', label="A+S")
# plt.plot(round, accuracy4, 'p--', label="random order, one, mitigate")
plt.xlabel("Global Round")
plt.xticks(range(0, 36, 5))
plt.ylabel("Malicious TX Cumulative weight")
plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
# plt.legend(bbox_to_anchor=(1, 1))
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("graph")
plt.show()
"""
# =========================================================================================================================================
"""
# malicious transaction, honest transaction distribution
index = np.arange(5)
bar_width = 0.25

xtick_label = ["0.0", "0.1", "0.3", "0.5", "0.7"]
malicious_dis = [0.052, 0.071, 0.016, 0.068, 0.163]
honest_dis = [0.948, 0.929, 0.984, 0.932, 0.837]

plt.title("Transaction distribtuion")
plt.bar(index, malicious_dis, bar_width, color="orange")
plt.bar(index + bar_width, honest_dis, bar_width, color="dodgerblue")
plt.xticks(index, xtick_label)
plt.savefig("graph")
plt.show()
"""
