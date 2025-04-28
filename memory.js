/**
 * Memory 类 - 用于存储智能体的经验轨迹
 * 将历史数据记录与智能体的动作选择逻辑解耦
 */
class Memory {
    constructor() {
        this.clear();
    }
    
    /**
     * 清空记忆
     */
    clear() {
        this.states = [];
        this.actions = [];
        this.rewards = [];
        this.logProbs = [];
        this.nextStates = [];
        this.dones = [];
    }
    
    /**
     * 记录一步经验
     * @param {number} state - 当前状态
     * @param {number} action - 执行的动作
     * @param {number} reward - 获得的奖励
     * @param {number} logProb - 动作的对数概率
     * @param {number} nextState - 下一个状态
     * @param {boolean} done - 是否终止
     */
    record(state, action, reward, logProb, nextState = null, done = false) {
        this.states.push(state);
        this.actions.push(action);
        this.rewards.push(reward);
        this.logProbs.push(logProb);
        
        if (nextState !== null) {
            this.nextStates.push(nextState);
        }
        
        if (done !== null) {
            this.dones.push(done);
        }
    }
    
    /**
     * 获取最近记录的动作的对数概率
     * @returns {number} 最近记录的动作的对数概率
     */
    getLastLogProb() {
        if (this.logProbs.length === 0) {
            return null;
        }
        return this.logProbs[this.logProbs.length - 1];
    }
    
    /**
     * 获取记忆的大小（记录的步数）
     * @returns {number} 记忆大小
     */
    size() {
        return this.states.length;
    }
    
    /**
     * 计算折扣回报
     * @param {number} gamma - 折扣因子
     * @returns {Array} 折扣回报数组
     */
    computeDiscountedReturns(gamma = 0.99) {
        const returns = new Array(this.rewards.length).fill(0);
        let runningReturn = 0;
        
        for (let t = this.rewards.length - 1; t >= 0; t--) {
            runningReturn = this.rewards[t] + gamma * runningReturn;
            returns[t] = runningReturn;
        }
        
        // 标准化回报
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        const std = Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length + 1e-8);
        return returns.map(r => (r - mean) / std);
    }
    
    /**
     * 获取最后一个奖励
     * @returns {number} 最后一个奖励
     */
    getLastReward() {
        if (this.rewards.length === 0) {
            return null;
        }
        return this.rewards[this.rewards.length - 1];
    }
    
    /**
     * 获取总奖励
     * @returns {number} 总奖励
     */
    getTotalReward() {
        return this.rewards.reduce((a, b) => a + b, 0);
    }
} 