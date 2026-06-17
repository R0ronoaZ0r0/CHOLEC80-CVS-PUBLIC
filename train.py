from colenet.trainer import ColenetTrainer

root_dir = "/home/hemanth/Desktop/CHOLEC80-CVS-PUBLIC/data/cholec80_frames"
backbone = "resnet"
log_name = "resnet"

epochs = 10
batch_size = 32
learning_rate = 1e-5

trainer = ColenetTrainer(root_dir, backbone, log_name, "mean_f1")
trainer.train_colenet(epochs, batch_size, learning_rate)
