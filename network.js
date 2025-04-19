class SimpleNN {
    constructor(inputSize, hiddenSize, outputSize) {
        console.log('Creating NN with sizes:', inputSize, hiddenSize, outputSize);
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenSize = hiddenSize;
        
        // 使用较小的初始权重值
        const initScale = 0.01;
        this.w1 = Array(inputSize).fill().map(() => 
            Array(hiddenSize).fill().map(() => (Math.random() * 2 - 1) * initScale)
        );
        this.w2 = Array(hiddenSize).fill().map(() => 
            Array(outputSize).fill().map(() => (Math.random() * 2 - 1) * initScale)
        );

        // 添加偏置项
        this.b1 = Array(hiddenSize).fill().map(() => 0);
        this.b2 = Array(outputSize).fill().map(() => 0);

        // 修改探索参数
        this.epsilon = 0.5;          // 初始探索率50%
        this.epsilonMin = 0.01;      // 最小探索率1%
        this.epsilonDecay = 0.98;    // 更慢的衰减以获得更好的探索
        
        // 降低学习率以获得稳定性
        this.learningRate = 0.001;
        
        // 调整收敛判断
        this.rewardHistorySize = 5;   // 减少判断窗口
        this.convergenceThreshold = -5;  // 调整阈值

        // 添加轨迹存储
        this.clearTrajectory();

        // 添加收敛判断相关的属性
        this.recentRewards = [];
        this.rewardHistorySize = 10;  // 保存最近10轮的奖励
        this.convergenceThreshold = -5;  // 平均奖励大于-5认为已收敛

        // 添加训练轮数
        this.episodeCount = 0;
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

    // ReLU激活函数
    relu(x) {
        return Math.max(0, x);
    }

    // 数值稳定的Softmax函数
    softmax(x) {
        try {
            const maxVal = Math.max(...x);
            const expValues = x.map(val => Math.exp(Math.min(val - maxVal, 20))); // 防止exp溢出
            const sumExp = expValues.reduce((a, b) => a + b, 0);
            if (sumExp === 0) {
                // 如果所有值都很小，返回均匀分布
                return Array(x.length).fill(1/x.length);
            }
            return expValues.map(val => val / sumExp);
        } catch (error) {
            console.error('Softmax error:', error);
            // 返回均匀分布作为后备方案
            return Array(x.length).fill(1/x.length);
        }
    }

    // 前向传播
    forward(state) {
        // 计算state的坐标表示
        const row = Math.floor(state / env.cols);
        const col = state % env.cols;
        
        // 使用两个独热编码表示行和列
        const input = Array(this.inputSize).fill(0);
        input[row * env.cols + col] = 1;
        
        // 隐藏层
        const hidden = Array(this.hiddenSize).fill(0);
        for (let i = 0; i < this.hiddenSize; i++) {
            let sum = this.b1[i];
            for (let j = 0; j < this.inputSize; j++) {
                sum += input[j] * this.w1[j][i];
            }
            hidden[i] = this.relu(sum);
        }

        // 输出层
        const output = Array(this.outputSize).fill(0);
        for (let i = 0; i < this.outputSize; i++) {
            let sum = this.b2[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += hidden[j] * this.w2[j][i];
            }
            output[i] = sum;
        }

        // 应用softmax得到动作概率
        return this.softmax(output);
    }

    // 反向传播更新权重
    backprop(state, action, G, logProb) {
        // 前向传播获取所有中间值
        const input = Array(this.inputSize).fill(0);
        input[state] = 1;

        // 隐藏层前向传播
        const hidden = Array(this.hiddenSize).fill(0);
        const hiddenRaw = Array(this.hiddenSize).fill(0);
        for (let i = 0; i < this.hiddenSize; i++) {
            hiddenRaw[i] = this.b1[i];
            for (let j = 0; j < this.inputSize; j++) {
                hiddenRaw[i] += input[j] * this.w1[j][i];
            }
            hidden[i] = this.relu(hiddenRaw[i]);
        }

        // 输出层前向传播
        const output = Array(this.outputSize).fill(0);
        for (let i = 0; i < this.outputSize; i++) {
            output[i] = this.b2[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                output[i] += hidden[j] * this.w2[j][i];
            }
        }

        // 计算策略概率
        const actionProbs = this.softmax(output);
        
        // 计算输出层梯度
        const outputGrads = Array(this.outputSize).fill(0);
        for (let i = 0; i < this.outputSize; i++) {
            // 真正的策略梯度
            if (i === action) {
                outputGrads[i] = G * (1 - actionProbs[i]);
            } else {
                outputGrads[i] = -G * actionProbs[i];
            }
        }
        
        // 更新输出层权重
        for (let i = 0; i < this.outputSize; i++) {
            this.b2[i] += this.learningRate * outputGrads[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                this.w2[j][i] += this.learningRate * outputGrads[i] * hidden[j];
            }
        }
        
        // 更新隐藏层权重 - 反向传播梯度
        for (let i = 0; i < this.hiddenSize; i++) {
            // 计算隐藏层梯度
            let hiddenGrad = 0;
            for (let j = 0; j < this.outputSize; j++) {
                hiddenGrad += outputGrads[j] * this.w2[i][j];
            }
            
            // 应用ReLU梯度
            const reluGrad = (hiddenRaw[i] > 0 ? 1 : 0);
            hiddenGrad *= reluGrad;
            
            // 更新权重
            this.b1[i] += this.learningRate * hiddenGrad;
            for (let j = 0; j < this.inputSize; j++) {
                this.w1[j][i] += this.learningRate * hiddenGrad * input[j];
            }
        }
    }

    // 修改selectAction函数，明确实现epsilon-greedy策略
    selectAction(state) {
        // 计算动作概率
        const actionProbs = this.forward(state);
        
        // Epsilon-greedy探索策略
        if (Math.random() < this.epsilon) {
            // 探索：随机选择动作
            const randomAction = Math.floor(Math.random() * this.outputSize);
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

    // 修改update函数，加入探索率更新
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
            const logProb = this.trajectory.logProbs[t];
            
            // 计算梯度并更新权重
            this.backprop(state, action, G, logProb);
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