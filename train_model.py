# train_model.py

import numpy as np
from alexnet import alexnet
WIDTH = 240
HEIGHT = 180
LR = 1e-3
EPOCHS = 1
MODEL_NAME = 'alexnet.model'

model = alexnet(WIDTH, HEIGHT, LR)

hm_data = 1
for i in range(EPOCHS):
    for i in range(1,hm_data+1):
        train_data = np.load('model.npy')

        train = train_data[:-50]
        test = train_data[-50:]

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
        Y = [i[1] for i in train]
        print(Y)
        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
        test_y = [i[1] for i in test]
        model.fit({'input': X}, {'targets': Y}, n_epoch=20, validation_set=({'input': test_x}, {'targets': test_y}), 
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)



# tensorboard --logdir=foo:C:/path/to/log






