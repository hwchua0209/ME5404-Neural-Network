import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='LLS', help='LLS, LMS')
parser.add_argument('--lr', default=0.01, type=float, help='Learning Rate')
parser.add_argument('--epoch', default=100, type=int, help='Epoch')

if __name__ == '__main__':

    args = parser.parse_args()

    # Input and Output Function
    x = np.array([[1,0], [1,0.8], [1,1.6], [1,3], [1,4], [1,5]])
    d = np.array([0.5 ,1 ,4 ,5, 6, 8])

    # LLS or LMS
    if args.method == 'LLS':
        w = np.linalg.inv(x.T@x)@x.T@d

        print(w)
        plt.plot(x[:,1], d, 'ro', label = 'Input Data')
        plt.plot(x[:,1], x@w, label = 'LLS Line')
        plt.legend()
        plt.title('Input Data vs Output Data with LLS Method')
        plt.xlabel('Input Data x')
        plt.ylabel('Output d')
        plt.show()
    
    elif args.method == 'LMS':

        # Weights Initialization
        np.random.seed(42)
        w = np.random.rand(2)

        # LMS
        epoch = 0
        bias = []
        weights = []

        for _ in range(args.epoch):
            for i in range(x.shape[0]):
                error = d[i] - np.dot(w, x[i])
                w += args.lr * error * x[i]

            bias.append(w[0])
            weights.append(w[1])
            epoch += 1

        print(w)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

        ax[0].plot(x[:,1], d, 'ro', label = 'Input Data')
        ax[0].plot(x[:,1], x@w, label = 'LMS Line')
        ax[0].legend()
        ax[0].set_title('Input Data vs Output Data with LMS Method')
        ax[0].set_xlabel('Input Data x')
        ax[0].set_ylabel('Output d')

        ax[1].plot(np.arange(epoch), bias, label='bias')
        ax[1].plot(np.arange(epoch), weights, label='weight')
        ax[1].legend()
        ax[1].set_title('Weights Across Epochs')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Weights')
        plt.show()
    