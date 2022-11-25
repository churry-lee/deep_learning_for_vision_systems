import matplotlib.pyplot as plot

result = pd.DataFrame({
    "acc": train_acc_list,
    "loss": train_loss_list
})
result.to_csv("./test_save", mode="a", sep=",", na_rep="NaN", float_format="%.4f", index=False)
# plt.figure(figsize=(10, 5))
# plt.title("Training and Validation Loss")
# plt.plot(train_loss_list, label="train")
# plt.xlabel("iterations")
# plt.ylabel("loss")
# plt.legend()
# plt.show()
