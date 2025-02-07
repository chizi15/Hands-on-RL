"""
1. 不管是ε-greedy、DecayingEpsilonGreedy、UCB、Thompson Sampling，都是在当前状态下，采用某种方法并带有一定随机性地选择动作，达到平衡探索与利用，使累计懊悔尽可能地小的目的。
2. 多臂老虎机是一种状态不变的环境，self.probs和self.best_idx是一开始就定好的，不会随着时间的推移和动作的发生而改变，是证明算法数学性质的理想环境。比如ε-greedy的累积懊悔是线性增加，DecayingEpsilonGreedy、UCB、Thompson Sampling的累积懊悔是次线性增加，其中UCB、Thompson Sampling的次线性增加可以保证对数收敛。
"""
# %%
# 导入需要使用的库,其中numpy是支持数组和矩阵运算的科学计算库,而matplotlib是绘图库
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


plot = 1
np.random.seed(1)


class BernoulliBandit:
    """伯努利多臂老虎机,输入K表示拉杆个数"""

    def __init__(self, K):
        self.probs = np.random.uniform(
            size=K
        )  # 随机生成K个0～1的数,作为拉动每根拉杆的获奖
        # 概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # type: ignore  # 最大的获奖概率
        self.K = K

    def step(self, k):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未获奖）
        if np.random.rand() < self.probs[k]:  # type: ignore
            return 1
        else:
            return 0


K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print(
    "获奖概率最大的拉杆为%d号,其获奖概率为%.4f"
    % (bandit_10_arm.best_idx, bandit_10_arm.best_prob)
)

# 随机生成了一个10臂伯努利老虎机
# 获奖概率最大的拉杆为1号,其获奖概率为0.7203


# %%
class Solver:
    """多臂老虎机算法基本框架"""

    def __init__(self, bandit):
        # 初始化Solver类
        self.bandit = bandit  # 老虎机实例
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数，初始化为0
        self.regret = 0.0  # 当前步的累积懊悔，初始化为0
        self.actions = []  # 维护一个列表,记录每一步的动作
        self.regrets = []  # 维护一个列表,记录每一步的累积懊悔

    def update_regret(self, k):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        self.regret += (
            self.bandit.best_prob - self.bandit.probs[k]
        )  # 累积懊悔增加当前选择的拉杆与最佳拉杆的期望收益差
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


# %%
class EpsilonGreedy(Solver):
    """epsilon贪婪算法,继承Solver类"""

    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        # Solver.__init__(self, bandit)  # 如果类的继承结构发生变化（例如，EpsilonGreedy 类不再直接继承自 Solver 类），这种写法可能不再适用，需要手动更改。
        # super().__init__(bandit)  # super() 函数会自动查找并调用父类的 __init__ 方法，即使继承结构发生变化也能正常工作。
        super(EpsilonGreedy, self).__init__(
            bandit
        )  # 明确指定了当前类 (EpsilonGreedy) 和实例 (self)，在某些复杂的继承结构（如多重继承）中，可以更精确地控制调用哪个父类的方法。
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估值，每根杆的初始概率应该尽量大，最好等于1，而不是0。否则在逐步学习的过程容易陷入小回报的杆跳不出来，不容易尝试和看到其他大回报的杆。
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


# %%
def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
        plt.xlabel("Time steps")
        plt.ylabel("Cumulative regrets")
        plt.title("%d-armed bandit" % solver.bandit.K)
        plt.legend()
        plt.show()
    if len(solvers) > 1:
        for idx, solver in enumerate(solvers):
            time_list = range(len(solver.regrets))
            plt.plot(time_list, solver.regrets, label=solver_names[idx])
        plt.xlabel("Time steps")
        plt.ylabel("Cumulative regrets")
        plt.title("%d-armed bandit" % solvers[0].bandit.K)
        plt.legend()
        plt.show()


epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print("epsilon-贪婪算法的累积懊悔为：", epsilon_greedy_solver.regret)
if plot:
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

# epsilon-贪婪算法的累积懊悔为：25.526630933945313

# %%
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons]
epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)

if plot:
    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)


# %%
class DecayingEpsilonGreedy(Solver):
    """epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类"""

    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随步数衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print("epsilon值衰减的贪婪算法的累积懊悔为：", decaying_epsilon_greedy_solver.regret)
if plot:
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

# epsilon值衰减的贪婪算法的累积懊悔为：10.114334931260183


# %%
class UCB(Solver):
    """UCB算法,继承Solver类"""

    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1))
        )  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


coef = 1  # 控制不确定性比重的系数
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print("上置信界算法的累积懊悔为：", UCB_solver.regret)
if plot:
    plot_results([UCB_solver], ["UCB"])

# 上置信界算法的累积懊悔为： 70.45281214197854


# %%
class ThompsonSampling(Solver):
    """汤普森采样算法,继承Solver类"""

    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += 1 - r  # 更新Beta分布的第二个参数
        return k


thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print("汤普森采样算法的累积懊悔为：", thompson_sampling_solver.regret)
if plot:
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])

# 汤普森采样算法的累积懊悔为：57.19161964443925
