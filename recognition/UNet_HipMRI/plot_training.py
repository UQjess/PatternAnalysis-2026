
import matplotlib.pyplot as plt
import os

# Standard U-Net training results (20 epochs)

epochs = list(range(1, 21))

train_losses = [
    0.17571, 0.05338, 0.04630, 0.04373, 0.04055,
    0.04145, 0.03737, 0.03876, 0.03974, 0.03660,
    0.03754, 0.03456, 0.03595, 0.03342, 0.03524,
    0.03266, 0.03092, 0.03379, 0.02956, 0.03023
]

val_losses = [
    0.08176, 0.06765, 0.05969, 0.06182, 0.06369,
    0.06316, 0.07499, 0.06129, 0.06558, 0.06435,
    0.06502, 0.08209, 0.06506, 0.06185, 0.08905,
    0.08668, 0.08890, 0.06856, 0.08627, 0.08910
]

val_dices = [
    0.8596, 0.8801, 0.9034, 0.9005, 0.8931,
    0.9010, 0.8915, 0.9037, 0.8999, 0.9045,
    0.9009, 0.8763, 0.9005, 0.8979, 0.8866,
    0.8819, 0.8841, 0.8991, 0.8935, 0.8866
]

os.makedirs("results", exist_ok=True)

# Graph 1: Training and Validation Loss


plt.plot(epochs, train_losses, label="Training Loss", marker="o")
plt.plot(epochs, val_losses, label="Validation Loss", marker="o")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Standard U-Net: Training and Validation Loss")
plt.xticks(epochs)
plt.legend()

plt.savefig("results/baseline_loss.png")
plt.show()
plt.close()


# Graph 2: Validation Dice


plt.plot(epochs, val_dices, label="Validation Dice", marker="o")

best_epoch = val_dices.index(max(val_dices)) + 1
best_dice = max(val_dices)

plt.scatter(best_epoch, best_dice, color="red")

plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Standard U-Net: Validation Dice")
plt.xticks(epochs)
plt.legend()

plt.savefig("results/baseline_dice.png", dpi=300)
plt.show()
plt.close()

print(f"Best Validation Dice: {best_dice:.4f}")
print(f"Best Epoch: {best_epoch}")
