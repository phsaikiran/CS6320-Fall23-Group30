import matplotlib.pyplot as plt

# val_accuracy = [
#     0.50375,
#     0.5275,
#     0.43625,
#     0.61,
#     0.585,
# ]
val_accuracy = [
    0.34625,
    0.43625,
    0.32125,
    0.175,
    0.40125,
    0.4125,
    0.465,
    0.4575,
]

# train_accuracy = [
#     0.4243125,
#     0.5096875,
#     0.5515625,
#     0.584875,
#     0.6021875,
# ]
train_accuracy = [
    0.2613125,
    0.3345,
    0.314,
    0.2433125,
    0.319625,
    0.3740625,
    0.3769375,
    0.3886875,
]

# train_loss = [
#     1.18208921,
#     1.090358019,
#     0.91771692,
#     0.945527792,
#     1.053645849,
# ]
train_loss = [
    1.57361424,
    1.478888869,
    1.510890961,
    1.592818975,
    1.497057199,
    1.402033687,
    1.39758265,
    1.382132173
]

# Subplot above two plots
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(train_accuracy, label="train_accuracy")
ax1.plot(val_accuracy, label="val_accuracy")
ax1.set_title("Train and Validation Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()

ax2.plot(train_loss, label="train_loss")
ax2.set_title("Train Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
plt.tight_layout()
plt.show()

# Save as eps
fig.savefig("rnn_train_val_loss.eps", format="eps")
