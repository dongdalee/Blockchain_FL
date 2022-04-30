import math
import matplotlib.pyplot as plt
import numpy as np

train_dataset_size = [5293, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
train_p_distribution = []
train_q_distribution = []
train_p_cross_entropy = 0
train_q_cross_entropy = 0

# data_distribution_ratio = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data_distribution_ratio = [10, 10, 10, 80, 10, 10, 10, 90, 90, 90]

for index, value in enumerate(data_distribution_ratio):
    data_distribution_ratio[index] = int(value * train_dataset_size[index] / 100)

for variable in train_dataset_size:
    p = variable / sum(train_dataset_size)
    train_p_distribution.append(p)
print("p probability: ", train_p_distribution)

for variable in data_distribution_ratio:
    p = variable / sum(data_distribution_ratio)
    train_q_distribution.append(p)
print("q probability: ", train_q_distribution)

# p cross entropy
for p in train_p_distribution:
    h = -1 * (p * math.log2(p))
    train_p_cross_entropy += h

# q corss entropy
for i, p in enumerate(train_q_distribution):
    h = -1 * (train_p_distribution[i] * math.log2(p))
    train_q_cross_entropy += h

kl_divergence = train_q_cross_entropy - train_p_cross_entropy

print("p crossentropy: ", train_p_cross_entropy)
print("q crossentropy: ", train_q_cross_entropy)
print("KL Divergence: ", kl_divergence)

# plt.plot(train_p_distribution)
# plt.plot(train_q_distribution)

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

plt.bar(x, train_p_distribution)
plt.bar(x, train_q_distribution)

# plt.xlim([0, 9])
plt.show()