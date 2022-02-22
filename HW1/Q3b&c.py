import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logic', default='AND', help='AND, OR, COMPLEMENT, NAND, XOR')
parser.add_argument('--lr', default=1, type=float, help='Learning Rate')
parser.add_argument('--epoch', default=20, type=int, help='Epoch')

def hard_limiter(v):

    return 1 if v >= 0 else 0

# Plotting Weight Trajectory 
def plot(bias, weight1, weight2, epoch, lr, logic):

    lr = str(lr)
    x = np.linspace(0, epoch, len(bias))
    plt.plot(x, bias)
    plt.plot(x, weight1)
    plt.plot(x, weight2)
    plt.legend(["Bias", "Weight 1", "Weight 2"])
    plt.title(logic, loc='left')
    plt.title('Weights Trajectory ' + 'lr = ' + lr)
    plt.xlabel('Epochs')
    plt.ylabel('Weights')
    plt.show()

def plot_com(bias, weight1, epoch, lr, logic):

    lr = str(lr)
    x = np.linspace(0, epoch, len(bias))
    plt.plot(x, bias)
    plt.plot(x, weight1)
    plt.legend(["Bias", "Weight 1"])
    plt.title(logic, loc='left')
    plt.title('Weights Trajectory ' + 'lr = ' + lr)
    plt.xlabel('Epochs')
    plt.ylabel('Weights')
    plt.show()

if __name__ == '__main__':

    args = parser.parse_args()

    np.random.seed(42)

    # Logic Function Selection
    if args.logic == 'AND':
        x = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
        y = np.array([0,0,0,1])
        w = np.random.rand(3)

    elif args.logic == 'OR':
        x = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
        y = np.array([0,1,1,1])
        w = np.random.rand(3)

    elif args.logic == 'NAND':
        x = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
        y = np.array([1,1,1,0])
        w = np.random.rand(3)

    elif args.logic == 'XOR':
        x = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
        y = np.array([0,1,1,0])
        w = np.random.rand(3)

    elif args.logic == 'COMPLEMENT':
        x = np.array([[1,0], [1,1]])
        y = np.array([1,0])
        w = np.random.rand(2)
    
    # Implementation of Logic Functions Weight Selection via Learning Procedure
    weights = []
    weights.append(w.tolist())

    for _ in range(args.epoch):
        for i in range(x.shape[0]):
            v = np.dot(x[i], w)
            output = hard_limiter(v)
            error = y[i] - output
            w += args.lr * error * x[i]
            weights.append(w.tolist())

    # Collect All Weights Rollouts & Plot Results
    if args.logic in ['AND', 'OR', 'NAND', 'XOR']:
        bias, weight1, weight2 = [],[],[]
        for i in range(len(weights)):
            bias.append(weights[i][0])
            weight1.append(weights[i][1])
            weight2.append(weights[i][2])
        
        plot(bias, weight1, weight2, args.epoch, args.lr, args.logic)
        print(weights[-1])
    
    else:
        bias, weight1 = [],[]
        for i in range(len(weights)):
            bias.append(weights[i][0])
            weight1.append(weights[i][1])
    
        plot_com(bias, weight1, args.epoch, args.lr, args.logic)
        print(weights[-1])