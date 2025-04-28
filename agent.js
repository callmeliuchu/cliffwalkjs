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
        
        // 使用Memory类替代trajectory
        this.memory = new Memory();
    }

    // 选择动作
    selectAction(state) {
        // 计算动作概率
        const actionProbs = this.network.forward(state);
        
        let action;
        let logProb;
        
        // Epsilon-greedy探索策略
        if (Math.random() < this.epsilon) {
            // 探索：随机选择动作
            action = Math.floor(Math.random() * actionProbs.length);
            logProb = Math.log(actionProbs[action] + 1e-8);
        } else {
            // 利用：选择概率最高的动作
            action = actionProbs.indexOf(Math.max(...actionProbs));
            logProb = Math.log(actionProbs[action] + 1e-8);
        }
        
        // 记录状态和动作（奖励将在环境step后记录）
        this.memory.record(state, action, null, logProb);
        
        return action;
    }
    
    // 记录奖励
    recordReward(reward, nextState = null, done = false) {
        if (this.memory.rewards.length < this.memory.actions.length) {
            // 更新最后一个记录的奖励
            this.memory.rewards[this.memory.rewards.length - 1] = reward;
            
            // 如果提供了下一个状态和终止标志，也记录它们
            if (nextState !== null) {
                this.memory.nextStates.push(nextState);
            }
            
            if (done !== null) {
                this.memory.dones.push(done);
            }
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
        if (this.memory.rewards.length > 0 && this.memory.rewards[this.memory.rewards.length - 1] === null) {
            this.memory.rewards[this.memory.rewards.length - 1] = finalReward;
        }
        
        // 计算折扣回报
        const returns = this.memory.computeDiscountedReturns();
        
        // 计算策略梯度
        for (let t = 0; t < this.memory.states.length; t++) {
            const state = this.memory.states[t];
            const action = this.memory.actions[t];
            const G = returns[t];
            
            // 计算梯度并更新权重
            this.network.backprop(state, action, G);
        }
        
        // 更新探索率
        this.updateExplorationRate();
        
        // 清理记忆
        this.memory.clear();
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