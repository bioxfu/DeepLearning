from cnn import RootNet

model = RootNet.build(32, 32, 3, 2)
model.summary()
