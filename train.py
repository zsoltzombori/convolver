import numpy as np
import argparse
import sys
sys.path.append('/home/zombori//k-arm')
sys.path.append('/Users/zsoltzombori/git/k-arm')
import data
from vis import *

from model import *
sys.setrecursionlimit(2**20)


parser = argparse.ArgumentParser(description="Image classifer using sparsifying arm layers embedded into convolution.")
parser.add_argument('--iteration', dest="iteration", type=int, default=1, help="Number of iterations in k-arm approximation")
parser.add_argument('--threshold', dest="threshold", type=float, default=0.03, help="Sparsity coefficient")
parser.add_argument('--reconsCoef', dest="reconsCoef", type=float, default=1, help="Reconstruction coefficient of the arm layers")
parser.add_argument('--dict', dest="dict_size", type=int, default=32, help="Size of the feature dictionary")
parser.add_argument('--epoch', dest="epoch", type=int, default=10, help="Number of epochs")
parser.add_argument('--lr', dest="lr", type=float, default=0.001, help="learning rate")
parser.add_argument('--batch', dest="batchSize", type=int, default=1, help="Batch size")
parser.add_argument('--armLayers', dest="armLayers", type=int, default=1, help="Arm layer count")
parser.add_argument('--denseLayers', dest="denseLayers", type=int, default=1, help="Dense layer count")
parser.add_argument('--convLayers', dest="convLayers", type=int, default=0, help="Convolution layer count")
parser.add_argument('--dataset', dest="dataset", default="mnist", help="mnist/cifar10")
parser.add_argument('--testSize', dest="testSize", type=int, default=5000, help="test size")
parser.add_argument('--trainSize', dest="trainSize", type=int, default=5000, help="train size")
args = parser.parse_args()

print "dataset: {}, convLayers: {}, armLayers: {}, denseLayers: {}, iteration: {}, threshold: {}, reconsCoef: {}, dict_size: {}, lr: {}, batch: {}, epoch: {}, trainSize:{}, testSize: {}".format(args.dataset,args.convLayers,args.armLayers,args.denseLayers,args.iteration,args.threshold,args.reconsCoef,args.dict_size,args.lr,args.batchSize,args.epoch,args.trainSize, args.testSize)
(X_train, Y_train), (X_test, Y_test), datagen, test_datagen, nb_classes = data.load_data(args.dataset)
X_train = X_train[:args.trainSize]
Y_train = Y_train[:args.trainSize]
X_test = X_test[:args.testSize]
Y_test = Y_test[:args.testSize]
 
#model = arm_model(X_train.shape, nb_classes, args.batchSize, args.lr, args.iteration, args.threshold, args.reconsCoef, args.dict_size)
model = conv_model(X_train.shape, nb_classes, args.batchSize, args.lr, args.dict_size)
model.summary()

# fit the model on the batches generated by datagen.flow()
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=args.batchSize, shuffle=True),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=args.epoch,
                        validation_data=test_datagen.flow(X_test, Y_test, batch_size=args.batchSize),
                        nb_val_samples=X_test.shape[0]
                        )

# lastArmLayer = model.get_layer(name="arm_1")
# y_fun = K.function([model.layers[0].input, K.learning_phase()], [lastArmLayer.output])
# Y_learned = y_fun([X_test,0])[0]
# W_learned = lastArmLayer.get_weights()[0]

# W_scaled = W_learned - np.min(W_learned)
# W_scaled /= np.max(W_scaled)
# W_scaled *= 255
# vis(W_scaled, "ArmDict.png", n=int(np.sqrt(args.dict_size)), w=3)
