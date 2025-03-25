import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 读取数据
train_data = pd.read_csv("TrainingData.csv")[['x', 'y_complex']].dropna()
test_data = pd.read_csv("TestData.csv")[['x_new', 'y_new_complex']].dropna()
test_data.columns = ['x', 'y']

x_train, y_train = train_data['x'].values, train_data['y_complex'].values
x_test, y_test = test_data['x'].values, test_data['y'].values

# 线性模型
def linear_model(x, w):
    return w[0] + w[1] * x

# 均方误差损失
def mse_loss(w, x, y):
    return np.mean((linear_model(x, w) - y) ** 2)

# 最小二乘法
X_train_matrix = np.vstack([np.ones_like(x_train), x_train]).T
w_ls = np.linalg.inv(X_train_matrix.T @ X_train_matrix) @ X_train_matrix.T @ y_train

# 梯度下降法
def gradient_descent(x, y, lr=0.01, epochs=10000):
    w = np.zeros(2)
    for _ in range(epochs):
        grad = np.array([np.mean(2 * (linear_model(x, w) - y)),
                         np.mean(2 * (linear_model(x, w) - y) * x)])
        w -= lr * grad
    return w

w_gd = gradient_descent(x_train, y_train)

# 牛顿法
res = minimize(lambda w: mse_loss(w, x_train, y_train), x0=np.zeros(2), method='Newton-CG',
               jac=lambda w: np.array([np.mean(2 * (linear_model(x_train, w) - y_train)),
                                       np.mean(2 * (linear_model(x_train, w) - y_train) * x_train)]),
               hess=lambda w: np.array([[2, 2 * np.mean(x_train)], [2 * np.mean(x_train), 2 * np.mean(x_train**2)]]))

w_newton = res.x

# 计算训练误差
train_mse_ls = mse_loss(w_ls, x_train, y_train)
train_mse_gd = mse_loss(w_gd, x_train, y_train)
train_mse_newton = mse_loss(w_newton, x_train, y_train)

print(f"Training MSE - Least Squares: {train_mse_ls:.5f}")
print(f"Training MSE - Gradient Descent: {train_mse_gd:.5f}")
print(f"Training MSE - Newton Method: {train_mse_newton:.5f}")

# 绘制训练集拟合
x_fit = np.linspace(min(x_train), max(x_train), 100)
y_ls, y_gd, y_newton = linear_model(x_fit, w_ls), linear_model(x_fit, w_gd), linear_model(x_fit, w_newton)

plt.scatter(x_train, y_train, label="Training Data", color='blue', alpha=0.5)
plt.plot(x_fit, y_ls, label=f"Least Squares: y={w_ls[1]:.5f}x+{w_ls[0]:.5f}", color='red')
plt.plot(x_fit, y_gd, label=f"Gradient Descent: y={w_gd[1]:.5f}x+{w_gd[0]:.5f}", color='green', linestyle='dashed')
plt.plot(x_fit, y_newton, label=f"Newton Method: y={w_newton[1]:.5f}x+{w_newton[0]:.5f}", color='purple', linestyle='dotted')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression on Training Data")
plt.legend()
plt.show()

# 计算测试误差
test_mse_ls = mse_loss(w_ls, x_test, y_test)
test_mse_gd = mse_loss(w_gd, x_test, y_test)
test_mse_newton = mse_loss(w_newton, x_test, y_test)

print(f"Test MSE - Least Squares: {test_mse_ls:.5f}")
print(f"Test MSE - Gradient Descent: {test_mse_gd:.5f}")
print(f"Test MSE - Newton Method: {test_mse_newton:.5f}")

# 绘制测试集拟合（分别画）
def plot_test_fit(w, method, color, linestyle):
    y_pred = linear_model(x_test, w)
    plt.scatter(x_test, y_test, label="Test Data", color='blue', alpha=0.5)
    plt.plot(x_test, y_pred, label=f"{method}: y={w[1]:.5f}x+{w[0]:.5f}", color=color, linestyle=linestyle)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{method} Regression on Test Data")
    plt.legend()
    plt.show()

plot_test_fit(w_ls, "Least Squares", 'red', '-')
plot_test_fit(w_gd, "Gradient Descent", 'green', 'dashed')
plot_test_fit(w_newton, "Newton Method", 'purple', 'dotted')



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
train_data = pd.read_csv("TrainingData.csv")[['x', 'y_complex']].dropna()
test_data = pd.read_csv("TestData.csv")[['x_new', 'y_new_complex']].dropna()
test_data.columns = ['x', 'y']

x_train, y_train = train_data['x'].values, train_data['y_complex'].values
x_test, y_test = test_data['x'].values, test_data['y'].values


# 定义目标函数
def model(x, params):
    a, b, c, d, e = params
    return a * np.sin(b * x + c) + d * x + e


# 定义均方误差
def mse_loss(params, x, y):
    return np.mean((model(x, params) - y) ** 2)


# 计算梯度
def compute_gradients(params, x, y):
    a, b, c, d, e = params
    y_pred = model(x, params)
    error = y_pred - y
    m = len(y)

    da = (2 / m) * np.sum(error * np.sin(b * x + c))
    db = (2 / m) * np.sum(error * a * x * np.cos(b * x + c))
    dc = (2 / m) * np.sum(error * a * np.cos(b * x + c))
    dd = (2 / m) * np.sum(error * x)
    de = (2 / m) * np.sum(error)

    return np.array([da, db, dc, dd, de])


# 梯度下降优化
def gradient_descent(x, y, lr=0.03, epochs=5000):
    params = np.random.randn(5)  # 初始化参数 [a, b, c, d, e]
    params[0]=-1
    for i in range(epochs):
        gradients = compute_gradients(params, x, y)
        params -= lr * gradients

        # 动态学习率衰减
        if i % 1000 == 0:
            lr *= 0.8

    return params


# 训练模型
optimized_params = gradient_descent(x_train, y_train)

# 计算训练误差
train_mse = mse_loss(optimized_params, x_train, y_train)
print(f"Training MSE: {train_mse:.5f}")

# 计算测试误差
test_mse = mse_loss(optimized_params, x_test, y_test)
print(f"Test MSE: {test_mse:.5f}")

# 生成拟合曲线
x_fit = np.linspace(min(x_train), max(x_train), 200)
y_fit = model(x_fit, optimized_params)

# 绘制训练集拟合结果
plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, label="Training Data", color='gray', alpha=0.5)
plt.plot(x_fit, y_fit, label=f"Fitted Curve\nParams: {optimized_params}", color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sinusoidal + Linear Fit on Training Data")
plt.legend()
plt.show()

# 生成测试拟合曲线
x_fit_test = np.linspace(min(x_test), max(x_test), 200)
y_fit_test = model(x_fit_test, optimized_params)

# 绘制测试集拟合结果
plt.figure(figsize=(8, 6))
plt.scatter(x_test, y_test, label="Test Data", color='blue', alpha=0.5)
plt.plot(x_fit_test, y_fit_test, label=f"Fitted Curve\nParams: {optimized_params}", color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sinusoidal + Linear Fit on Test Data")
plt.legend()
plt.show()
