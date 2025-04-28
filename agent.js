class Agent {
    constructor(stateSize, actionSize) {
        // 创建策略网络
        this.network = new SimpleNN(stateSize, 24, actionSize);
        
        // 探索参数
        this.epsilon = 0.5;          // 初始探索率50%
        this.epsilonMin = 0.01;      // 最小探索率1%
        this.epsilonDecay = 0.98;    // 更慢的衰减以获得更好的探索
        
        // 收敛判断相关的属性
        this.recentRewards = [];
        this.rewardHistorySize = 10;  // 保存最近10轮的奖励
        this.convergenceThreshold = -5;  // 平均奖励大于-5认为已收敛

        // 添加训练轮数
        this.episodeCount = 0;
        
        // 添加轨迹存储
        this.clearTrajectory();
    }

    // 添加轨迹存储和清理函数
    clearTrajectory() {
        this.trajectory = {
            states: [],
            actions: [],
            rewards: [],
            logProbs: []
        };
    }

    // 记录一步轨迹
    recordStep(state, action, reward, logProb) {
        this.trajectory.states.push(state);
        this.trajectory.actions.push(action);
        this.trajectory.rewards.push(reward);
        this.trajectory.logProbs.push(logProb);
    }

    // 计算折扣回报
    computeDiscountedReturns(rewards, gamma = 0.99) {
        const returns = new Array(rewards.length).fill(0);
        let runningReturn = 0;
        
        for (let t = rewards.length - 1; t >= 0; t--) {
            runningReturn = rewards[t] + gamma * runningReturn;
            returns[t] = runningReturn;
        }
        
        // 标准化回报
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        const std = Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length + 1e-8);
        return returns.map(r => (r - mean) / std);
    }

    // 选择动作
    selectAction(state) {
        // 计算动作概率
        const actionProbs = this.network.forward(state);
        
        // Epsilon-greedy探索策略
        if (Math.random() < this.epsilon) {
            // 探索：随机选择动作
            const randomAction = Math.floor(Math.random() * actionProbs.length);
            this.recordStep(state, randomAction, null, Math.log(actionProbs[randomAction] + 1e-8));
            return randomAction;
        } else {
            // 利用：选择概率最高的动作
            const bestAction = actionProbs.indexOf(Math.max(...actionProbs));
            this.recordStep(state, bestAction, null, Math.log(actionProbs[bestAction] + 1e-8));
            return bestAction;
        }
    }

    // 添加专门的探索率更新函数
    updateExplorationRate() {
        // 随着训练进行逐渐减小探索率
        this.epsilon = Math.max(
            this.epsilonMin, 
            this.epsilon * this.epsilonDecay
        );
        console.log(`更新探索率: ${this.epsilon.toFixed(4)}`);
    }

    // 更新策略
    update(finalReward) {
        // 更新训练轮数
        this.episodeCount++;
        
        // 更新最后一步的奖励
        this.trajectory.rewards[this.trajectory.rewards.length - 1] = finalReward;
        
        // 计算折扣回报
        const returns = this.computeDiscountedReturns(this.trajectory.rewards);
        
        // 计算策略梯度
        for (let t = 0; t < this.trajectory.states.length; t++) {
            const state = this.trajectory.states[t];
            const action = this.trajectory.actions[t];
            const G = returns[t];
            
            // 计算梯度并更新权重
            this.network.backprop(state, action, G);
        }
        
        // 更新探索率
        this.updateExplorationRate();
        
        // 清理轨迹
        this.clearTrajectory();
    }

    // 添加检查是否收敛的方法
    checkConvergence(reward) {
        this.recentRewards.push(reward);
        if (this.recentRewards.length > this.rewardHistorySize) {
            this.recentRewards.shift();
        }
        
        if (this.recentRewards.length === this.rewardHistorySize) {
            const avgReward = this.recentRewards.reduce((a, b) => a + b, 0) / this.rewardHistorySize;
            return avgReward > this.convergenceThreshold;
        }
        return false;
    }
} 