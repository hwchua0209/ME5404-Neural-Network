import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='gradient', help='gradient, newton')
parser.add_argument('--lr', default=0.001, type=float, help='Learning Rate')
parser.add_argument('--min_f', default=0.00001, type=float, help='Min f to Consider as Convergence')

def rosenbrock(x, y):
    f = (1-x)**2 + 100*(y-x**2)**2
    return f

def gradient(x, y):
    dx = 2*x-2 - 400*x*y + 400*x**3
    dy = 200*y - 200*x**2
    return np.array([dx, dy])

def hessian(x, y):
    dxx = 2 - 400*y + 1200*x**2
    dxy = -400*x
    dyy = 200
    return np.array([[dxx, dxy],[dxy, dyy]])

def plot(x_value, y_value):

    fig = plt.figure(figsize=(12,4))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    X = np.arange(-5, 5, 0.05)
    Y = np.arange(-5, 5, 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = rosenbrock(X, Y)
    ax.plot_surface(X, Y, Z,linewidth=0, antialiased=False)
    ax.set_title('Function Line with First Order Method')
        
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_value, y_value)
    ax.set_title('trajectory of (x, y) in the 2D space')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()

if __name__ == '__main__':

    args = parser.parse_args()
    np.random.seed(42)

    # Record Values of x,y and f
    x_value, y_value, f_value = [], [], []
    counter = 0

    # Random Starting Value of x and y [-1,1]
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    f = rosenbrock(x, y)

    x_value.append(x)
    y_value.append(y)
    f_value.append(f)

    # First Order Method (Steepest Gradient) or Second Order Method (Newton Method)
    if args.method == 'gradient':
        while f_value[-1] >= args.min_f:
            g = gradient(x, y)
            x -= args.lr * g[0]
            y -= args.lr * g[1]
            f = rosenbrock(x, y)

            x_value.append(x)
            y_value.append(y)
            f_value.append(f)

            counter += 1
        
        print(f'The function converge after {counter} epochs')
        print(f'The x value after convergence is {x}')
        print(f'The y value after convergence is {y}')

        plot(x_value, y_value)
    
    elif args.method == 'newton':
        while f_value[-1] >= args.min_f:
            g = gradient(x, y)
            h = hessian(x, y)

            w = -np.linalg.inv(h) @ g

            x += w[0]
            y += w[1]
            f = rosenbrock(x, y)

            x_value.append(x)
            y_value.append(y)
            f_value.append(f)

            counter += 1
        
        print(f'The function converge after {counter} epochs')
        print(f'The x value after convergence is {x}')
        print(f'The y value after convergence is {y}')

        plot(x_value, y_value)

