/**
 * experiment.js - 训练智能体与环境交互的实验模块
 */

/**
 * 将状态转换为张量格式
 * @param {Number} state - 环境状态
 * @returns {Number} - 转换后的状态
 */
function convert2tensor(state) {
    // 在当前实现中，我们直接返回状态值
    // 因为我们的Agent和Environment都是使用简单的数值表示状态
    return state;
}

/**
 * 训练智能体在环境中的表现
 * @param {Agent} agent - 智能体实例
 * @param {CliffWalkEnv} env - 环境实例
 */
function train(agent, env) {
    const successCount = [];
    const maxSize = 1000;
    const epochs = 20000;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
        const rewards = [];
        const logProbs = [];
        let terminated = false;
        let success = true;
        
        // 重置环境，获取初始状态
        let state = env.reset();
        
        // 在单个回合中执行动作，直到终止或达到最大步数
        while (!terminated && logProbs.length < maxSize) {
            // 将状态转换为张量格式
            const stateArr = convert2tensor(state);
            
            // 智能体选择动作
            const action = agent.selectAction(stateArr);
            // 获取动作的对数概率
            const logProb = agent.memory.getLastLogProb();
            
            // 执行动作，获取下一个状态和奖励
            const [nextState, reward, done] = env.step(action);
            
            // 记录奖励和下一个状态
            agent.recordReward(reward, nextState, done);
            
            // 更新当前状态
            state = nextState;
            
            // 记录奖励和对数概率（用于统计）
            rewards.push(reward);
            logProbs.push(logProb);
            
            // 检查是否终止
            terminated = done;
        }
        
        // 计算最终奖励
        const finalReward = rewards[rewards.length - 1];
        
        // 更新智能体
        agent.update(finalReward);
        
        // 记录本回合是否成功
        successCount.push(finalReward > 0);
        
        // 定期输出训练信息
        if ((epoch + 1) % 10 === 0) {
            // 计算最近100回合的成功率
            const recentSuccesses = successCount.slice(-100);
            const successRate = recentSuccesses.filter(s => s).length / Math.min(recentSuccesses.length, 100);
            
            // 计算总奖励
            const totalReward = rewards.reduce((a, b) => a + b, 0);
            
            console.log(`成功率: ${successRate.toFixed(4)}`);
            console.log(`回合: ${epoch}, 奖励: ${totalReward.toFixed(2)}, 步数: ${rewards.length}`);
        }
    }
    
    return agent;
}

/**
 * 评估训练好的智能体
 * @param {Agent} agent - 训练好的智能体
 * @param {CliffWalkEnv} env - 环境实例
 * @param {number} episodes - 评估回合数
 */
function evaluate(agent, env, episodes = 100) {
    const rewards = [];
    const steps = [];
    const successes = [];
    
    // 保存当前探索率
    const originalEpsilon = agent.epsilon;
    // 评估时不进行探索
    agent.epsilon = 0;
    
    for (let i = 0; i < episodes; i++) {
        let state = env.reset();
        let totalReward = 0;
        let step = 0;
        let terminated = false;
        
        while (!terminated && step < 1000) {
            const stateArr = convert2tensor(state);
            const action = agent.selectAction(stateArr);
            const [nextState, reward, done] = env.step(action);
            
            state = nextState;
            totalReward += reward;
            step++;
            terminated = done;
        }
        
        rewards.push(totalReward);
        steps.push(step);
        successes.push(totalReward > 0); // 根据奖励判断是否成功
    }
    
    // 恢复探索率
    agent.epsilon = originalEpsilon;
    
    const successRate = successes.filter(s => s).length / episodes;
    const avgReward = rewards.reduce((a, b) => a + b, 0) / episodes;
    const avgSteps = steps.reduce((a, b) => a + b, 0) / episodes;
    
    console.log(`评估结果 (${episodes} 回合):`);
    console.log(`成功率: ${successRate.toFixed(4)}`);
    console.log(`平均奖励: ${avgReward.toFixed(2)}`);
    console.log(`平均步数: ${avgSteps.toFixed(2)}`);
    
    return { successRate, avgReward, avgSteps };
}

/**
 * 运行实验的主函数
 */
function runExperiment() {
    // 创建环境和智能体
    const rows = 4;
    const cols = 12;
    const env = new CliffWalkEnv(rows, cols);
    const agent = new Agent(rows * cols, 4); // 状态空间大小为rows*cols，动作空间大小为4
    
    console.log("开始训练...");
    const trainedAgent = train(agent, env);
    
    console.log("训练完成，开始评估...");
    evaluate(trainedAgent, env);
    
    return trainedAgent;
}

// 如果在浏览器环境中，将函数暴露给全局对象
if (typeof window !== 'undefined') {
    window.convert2tensor = convert2tensor;
    window.train = train;
    window.evaluate = evaluate;
    window.runExperiment = runExperiment;
} 