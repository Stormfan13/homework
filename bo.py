import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def nirm_model(t, y, r1, r2, r3, alpha, pi, Q, p_A, e_A, p_D, e_D):
    """
    NIRM 模型的微分方程
    """
    N, I, R, M = y

    # 计算攻击和防御的期望效用
    a = np.dot(p_A, e_A)  # 期望攻击效用
    d = np.dot(p_D, e_D)  # 期望防御效用

    eta = a - d  # 攻防效用差异

    # 增强动态性：对效用差异引入非线性响应
    dynamic_factor = np.tanh(eta * alpha)

    # 微分方程
    dN = -dynamic_factor * pi * I * N / Q  # 正常节点减少
    dI = dynamic_factor * pi * I * N / Q - r1 * I  # 感染节点变化
    dR = r1 * I - r2 * R  # 恢复节点变化
    dM = r3 * R  # 失败节点变化

    return [dN, dI, dR, dM]


def plot_results(t_vals, y_vals, title):
    """
    绘制仿真结果
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, y_vals[0], 'k-', label='N(t) - Normal', marker='s', markevery=20)
    plt.plot(t_vals, y_vals[1], 'b-', label='I(t) - Infected', marker='o', markevery=20)
    plt.plot(t_vals, y_vals[2], 'g-', label='R(t) - Recovered', marker='^', markevery=20)
    plt.plot(t_vals, y_vals[3], 'r-', label='M(t) - Failed', marker='v', markevery=20)

    plt.xlabel('Time (t)')
    plt.ylabel('Number of Nodes')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # 初始化参数
    Q = 1000  # 网络节点总数
    t_span = (0, 20)  # 仿真时间范围
    y0 = [Q, 10, 0, 0]  # 初始状态

    # 参数组合推荐
    param_sets = [
        {"r1": 0.2, "r2": 0.1, "r3": 0.05, "alpha": 1.0, "pi": 3.14,
         "p_A": [0.6, 0.3, 0.1], "e_A": [0.8, 0.5, 0.2],
         "p_D": [0.7, 0.3], "e_D": [0.6, 0.2]},
        {"r1": 0.3, "r2": 0.15, "r3": 0.1, "alpha": 0.8, "pi": 3.14,
         "p_A": [0.5, 0.4, 0.1], "e_A": [0.9, 0.6, 0.3],
         "p_D": [0.5, 0.5], "e_D": [0.7, 0.4]},
        {"r1": 0.1, "r2": 0.05, "r3": 0.02, "alpha": 1.2, "pi": 3.14,
         "p_A": [0.4, 0.4, 0.2], "e_A": [1.0, 0.8, 0.4],
         "p_D": [0.6, 0.4], "e_D": [0.8, 0.3]},
    ]

    # 仿真并绘图
    for i, params in enumerate(param_sets):
        sol = solve_ivp(
            nirm_model,
            t_span,
            y0,
            args=(params["r1"], params["r2"], params["r3"],
                  params["alpha"], params["pi"], Q,
                  params["p_A"], params["e_A"],
                  params["p_D"], params["e_D"]),
            max_step=0.1,
            dense_output=True,
        )

        t_vals = np.linspace(t_span[0], t_span[1], 200)  # 生成时间点
        y_vals = sol.sol(t_vals)

        # 绘制每组参数的结果
        plot_results(t_vals, y_vals, f"Simulation {i+1}: Parameter Set {i+1}")


if __name__ == "__main__":
    main()
