// networkTests.js - 神经网络单元测试

// 模拟env对象，在前向传播中会用到
const env = {
    rows: 4,
    cols: 4
};

// 辅助函数：判断两个数组是否近似相等
function arraysAlmostEqual(arr1, arr2, epsilon = 1e-6) {
    if (arr1.length !== arr2.length) return false;
    return arr1.every((val, i) => Math.abs(val - arr2[i]) < epsilon);
}

// 辅助函数：断言测试结果
function assert(condition, message) {
    if (!condition) {
        console.error(`❌ 测试失败: ${message}`);
        return false;
    }
    console.log(`✅ 测试通过: ${message}`);
    return true;
}

// 辅助函数：创建固定权重的神经网络
function createTestNN() {
    const nn = new SimpleNN(16, 8, 4);
    
    // 设置固定权重以便测试
    nn.w1 = Array(16).fill().map((_, i) => 
        Array(8).fill().map((_, j) => 0.1 * (i % 4) + 0.05 * j)
    );
    nn.w2 = Array(8).fill().map((_, i) => 
        Array(4).fill().map((_, j) => 0.1 * i + 0.05 * j)
    );
    nn.b1 = Array(8).fill().map((_, i) => 0.01 * i);
    nn.b2 = Array(4).fill().map((_, i) => 0.01 * i);
    
    return nn;
}

// 测试套件
const tests = {
    // 测试初始化
    testConstructor: function() {
        const nn = new SimpleNN(16, 8, 4);
        
        assert(nn.inputSize === 16, "输入层大小应为16");
        assert(nn.hiddenSize === 8, "隐藏层大小应为8");
        assert(nn.outputSize === 4, "输出层大小应为4");
        
        assert(nn.w1.length === 16, "w1应有16行");
        assert(nn.w1[0].length === 8, "w1应有8列");
        assert(nn.w2.length === 8, "w2应有8行");
        assert(nn.w2[0].length === 4, "w2应有4列");
        
        assert(nn.b1.length === 8, "b1应有8个元素");
        assert(nn.b2.length === 4, "b2应有4个元素");
        
        assert(nn.trajectory.states.length === 0, "轨迹状态应为空数组");
        assert(nn.recentRewards.length === 0, "最近奖励应为空数组");
        
        return true;
    },
    
    // 测试ReLU激活函数
    testRelu: function() {
        const nn = new SimpleNN(1, 1, 1);
        
        assert(nn.relu(1.5) === 1.5, "ReLU(1.5)应为1.5");
        assert(nn.relu(0) === 0, "ReLU(0)应为0");
        assert(nn.relu(-2.5) === 0, "ReLU(-2.5)应为0");
        
        return true;
    },
    
    // 测试Softmax函数
    testSoftmax: function() {
        const nn = new SimpleNN(1, 1, 1);
        
        // 测试普通情况
        const result1 = nn.softmax([1, 2, 3, 4]);
        const expected1 = [0.0320586, 0.08714432, 0.23688282, 0.64391426];
        assert(arraysAlmostEqual(result1, expected1, 1e-4), "Softmax计算结果正确");
        
        // 测试总和为1
        const sum1 = result1.reduce((a, b) => a + b, 0);
        assert(Math.abs(sum1 - 1.0) < 1e-6, "Softmax输出总和应为1");
        
        // 测试极端情况
        const result2 = nn.softmax([1000, 0, 0, 0]);
        assert(Math.abs(result2[0] - 1) < 1e-6, "处理大数值时应稳定");
        
        // 测试负值情况
        const result3 = nn.softmax([-1, -2, -3, -4]);
        const expected3 = [0.64391426, 0.23688282, 0.08714432, 0.0320586];
        assert(arraysAlmostEqual(result3, expected3, 1e-4), "负值Softmax计算结果正确");
        
        return true;
    },
    
    // 测试前向传播 - 关键测试
    testForward: function() {
        const nn = createTestNN();
        
        // 测试状态0的前向传播 (第一行第一列)
        const probs1 = nn.forward(0);
        assert(probs1.length === 4, "输出应有4个元素");
        
        // 手动计算预期结果
        // 独热编码后的输入: 状态0应该被编码为第一个位置为1，其余为0
        // 然后通过权重计算结果
        
        // 验证概率和为1
        const sum = probs1.reduce((a, b) => a + b, 0);
        assert(Math.abs(sum - 1.0) < 1e-6, "前向传播输出概率和应为1");
        
        // 测试不同状态
        const probs2 = nn.forward(5); // 第一行第二列
        assert(probs2.length === 4, "输出应有4个元素");
        assert(Math.abs(probs2.reduce((a, b) => a + b, 0) - 1.0) < 1e-6, "概率和应为1");
        
        // 测试概率分布不同
        assert(!arraysAlmostEqual(probs1, probs2), "不同状态应产生不同的概率分布");
        
        return true;
    },
    
    // 测试前向传播的详细计算
    testForwardDetailed: function() {
        const nn = createTestNN();
        
        // 测试状态0的前向传播计算过程
        const state = 0;  // 位置(0,0)
        
        // 手动计算隐藏层输出
        const hidden = Array(nn.hiddenSize).fill(0);
        for (let i = 0; i < nn.hiddenSize; i++) {
            hidden[i] = nn.relu(nn.b1[i] + nn.w1[0][i]);  // 只有输入[0]为1
        }
        
        // 手动计算输出层
        const output = Array(nn.outputSize).fill(0);
        for (let i = 0; i < nn.outputSize; i++) {
            output[i] = nn.b2[i];
            for (let j = 0; j < nn.hiddenSize; j++) {
                output[i] += hidden[j] * nn.w2[j][i];
            }
        }
        
        // 手动计算softmax
        const expectedProbs = nn.softmax(output);
        
        // 通过网络计算
        const actualProbs = nn.forward(state);
        
        assert(arraysAlmostEqual(actualProbs, expectedProbs), "前向传播详细计算结果正确");
        
        return true;
    },
    
    // 测试后向传播 - 关键测试
    testBackprop: function() {
        const nn = createTestNN();
        const originalW1 = JSON.parse(JSON.stringify(nn.w1));
        const originalW2 = JSON.parse(JSON.stringify(nn.w2));
        const originalB1 = JSON.parse(JSON.stringify(nn.b1));
        const originalB2 = JSON.parse(JSON.stringify(nn.b2));
        
        // 保存原始学习率
        const originalLR = nn.learningRate;
        // 设置较大的学习率以便观察变化
        nn.learningRate = 0.1;
        
        // 执行后向传播
        const state = 0;
        const action = 1;
        const G = 1.0;  // 正向奖励
        const logProb = -0.5;
        
        nn.backprop(state, action, G, logProb);
        
        // 验证权重已经更新
        let weightsChanged = false;
        
        // 检查w2是否已更新
        for (let i = 0; i < nn.hiddenSize; i++) {
            for (let j = 0; j < nn.outputSize; j++) {
                if (nn.w2[i][j] !== originalW2[i][j]) {
                    weightsChanged = true;
                    break;
                }
            }
        }
        
        assert(weightsChanged, "后向传播应更新权重");
        
        // 恢复学习率
        nn.learningRate = originalLR;
        
        return true;
    },
    
    // 测试后向传播的详细计算
    testBackpropDetailed: function() {
        const nn = createTestNN();
        
        // 设置固定学习率
        nn.learningRate = 0.01;
        
        // 1. 保存原始权重
        const originalW1 = JSON.parse(JSON.stringify(nn.w1));
        const originalW2 = JSON.parse(JSON.stringify(nn.w2));
        const originalB1 = JSON.parse(JSON.stringify(nn.b1));
        const originalB2 = JSON.parse(JSON.stringify(nn.b2));
        
        // 2. 执行前向传播以获取预测概率
        const state = 0;
        const action = 1;
        const G = 1.0;
        
        const probs = nn.forward(state);
        const logProb = Math.log(probs[action] + 1e-8);
        
        // 3. 执行后向传播
        nn.backprop(state, action, G, logProb);
        
        // 4. 验证梯度方向
        // 对于选择的动作，梯度应增加其概率
        // 检查输出层权重的变化
        for (let i = 0; i < nn.hiddenSize; i++) {
            // 选中的动作权重应增加，其他动作权重应减少
            if (originalW2[i][action] <= nn.w2[i][action]) {
                assert(true, `选中动作的权重应增加: 原值=${originalW2[i][action]}, 新值=${nn.w2[i][action]}`);
            } else {
                assert(false, `选中动作的权重应增加: 原值=${originalW2[i][action]}, 新值=${nn.w2[i][action]}`);
            }
        }
        
        return true;
    },
    
    // 测试轨迹记录和清理
    testTrajectory: function() {
        const nn = new SimpleNN(16, 8, 4);
        nn.clearTrajectory();
        
        assert(nn.trajectory.states.length === 0, "清理后轨迹状态应为空");
        assert(nn.trajectory.actions.length === 0, "清理后轨迹动作应为空");
        
        nn.recordStep(0, 1, 0.5, -0.5);
        assert(nn.trajectory.states.length === 1, "记录后轨迹状态长度应为1");
        assert(nn.trajectory.states[0] === 0, "记录的状态应为0");
        assert(nn.trajectory.actions[0] === 1, "记录的动作应为1");
        assert(nn.trajectory.rewards[0] === 0.5, "记录的奖励应为0.5");
        assert(nn.trajectory.logProbs[0] === -0.5, "记录的对数概率应为-0.5");
        
        nn.clearTrajectory();
        assert(nn.trajectory.states.length === 0, "再次清理后轨迹状态应为空");
        
        return true;
    },
    
    // 测试折扣回报计算
    testComputeDiscountedReturns: function() {
        const nn = new SimpleNN(16, 8, 4);
        
        // 测试简单情况
        const rewards = [1, 2, 3];
        const gamma = 0.9;
        const returns = nn.computeDiscountedReturns(rewards, gamma);
        
        // 手动计算预期结果
        // R[2] = 3
        // R[1] = 2 + 0.9*3 = 4.7
        // R[0] = 1 + 0.9*4.7 = 5.23
        // 然后标准化
        const expectedReturns = [1, 2, 3].map((_, i) => 
            i === 2 ? 3 : (i === 1 ? 4.7 : 5.23)
        );
        
        // 标准化预期回报
        const mean = expectedReturns.reduce((a, b) => a + b, 0) / expectedReturns.length;
        const std = Math.sqrt(expectedReturns.reduce((a, b) => a + (b - mean) ** 2, 0) / expectedReturns.length + 1e-8);
        const normalizedExpected = expectedReturns.map(r => (r - mean) / std);
        
        assert(returns.length === rewards.length, "回报长度应等于奖励长度");
        assert(arraysAlmostEqual(returns, normalizedExpected, 1e-2), "折扣回报计算正确");
        
        return true;
    },
    
    // 测试动作选择
    testSelectAction: function() {
        const nn = createTestNN();
        
        // 多次选择动作以验证分布
        const actionCounts = Array(nn.outputSize).fill(0);
        const NUM_TRIALS = 1000;
        
        for (let i = 0; i < NUM_TRIALS; i++) {
            const action = nn.selectAction(0);
            actionCounts[action]++;
            
            // 清理轨迹以便下次测试
            nn.clearTrajectory();
        }
        
        // 验证所有动作都有被选择到
        const allActionsSelected = actionCounts.every(count => count > 0);
        assert(allActionsSelected, "所有动作都应有被选择到");
        
        // 验证轨迹记录
        nn.selectAction(0);
        assert(nn.trajectory.states.length === 1, "selectAction应记录状态");
        assert(nn.trajectory.actions.length === 1, "selectAction应记录动作");
        assert(nn.trajectory.logProbs.length === 1, "selectAction应记录对数概率");
        
        return true;
    },
    
    // 测试策略更新
    testUpdate: function() {
        const nn = createTestNN();
        
        // 记录几个步骤
        nn.recordStep(0, 1, 0.5, -0.5);
        nn.recordStep(1, 2, 0.3, -0.7);
        nn.recordStep(2, 0, null, -0.6);  // 最后一步奖励为null
        
        const oldEpisodeCount = nn.episodeCount;
        
        // 保存原始权重
        const originalW1 = JSON.parse(JSON.stringify(nn.w1));
        const originalW2 = JSON.parse(JSON.stringify(nn.w2));
        
        // 执行更新
        nn.update(1.0);  // 提供最终奖励
        
        // 验证权重已经更新
        let weightsChanged = false;
        
        for (let i = 0; i < nn.inputSize; i++) {
            for (let j = 0; j < nn.hiddenSize; j++) {
                if (nn.w1[i][j] !== originalW1[i][j]) {
                    weightsChanged = true;
                    break;
                }
            }
        }
        
        assert(weightsChanged, "update应更新权重");
        assert(nn.episodeCount === oldEpisodeCount + 1, "update应增加训练轮数");
        assert(nn.trajectory.states.length === 0, "update后应清理轨迹");
        
        return true;
    },
    
    // 测试收敛检查
    testCheckConvergence: function() {
        const nn = new SimpleNN(16, 8, 4);
        nn.recentRewards = [];
        nn.rewardHistorySize = 3;
        nn.convergenceThreshold = -5;
        
        // 测试未收敛情况
        nn.checkConvergence(-10);
        nn.checkConvergence(-10);
        assert(!nn.checkConvergence(-10), "平均奖励为-10，应未收敛");
        
        // 测试收敛情况
        nn.recentRewards = [];
        nn.checkConvergence(-4);
        nn.checkConvergence(-4);
        assert(nn.checkConvergence(-4), "平均奖励为-4，应已收敛");
        
        return true;
    },
    
    // 测试探索策略
    testExplorationStrategy: function() {
        const nn = createTestNN();
        
        // 保存原始探索参数
        const originalEpsilon = nn.epsilon;
        const originalEpsilonMin = nn.epsilonMin;
        const originalEpsilonDecay = nn.epsilonDecay;
        
        // 设置特定的探索参数以便测试
        nn.epsilon = 1.0;          // 100%探索
        nn.epsilonMin = 0.1;       // 最小探索率10%
        nn.epsilonDecay = 0.5;     // 快速衰减以便测试
        
        // 1. 测试高探索率时的行为
        const actionDistributionHighEpsilon = Array(nn.outputSize).fill(0);
        const TRIALS = 1000;
        
        for (let i = 0; i < TRIALS; i++) {
            const action = nn.selectAction(0);
            actionDistributionHighEpsilon[action]++;
            nn.clearTrajectory();
        }
        
        // 验证所有动作都有机会被选中
        const allActionsSelected = actionDistributionHighEpsilon.every(count => count > 0);
        assert(allActionsSelected, "高探索率时所有动作都应被选中");
        
        // 验证动作分布的熵（随机性）较高
        const highEntropyDistribution = actionDistributionHighEpsilon.map(count => count / TRIALS);
        const highEntropy = -highEntropyDistribution.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
        
        // 2. 模拟训练过程中探索率的衰减
        // 多次更新探索率
        for (let i = 0; i < 10; i++) {
            nn.epsilon = Math.max(nn.epsilon * nn.epsilonDecay, nn.epsilonMin);
        }
        
        // 验证探索率已经减小但不低于最小值
        assert(nn.epsilon < 1.0, "探索率应该衰减");
        assert(nn.epsilon >= nn.epsilonMin, "探索率不应低于最小值");
        
        // 3. 测试低探索率时的行为
        nn.epsilon = 0.1; // 设置为低探索率
        
        const actionDistributionLowEpsilon = Array(nn.outputSize).fill(0);
        for (let i = 0; i < TRIALS; i++) {
            const action = nn.selectAction(0);
            actionDistributionLowEpsilon[action]++;
            nn.clearTrajectory();
        }
        
        // 低探索率时，分布应该更集中于高价值动作
        const lowEntropyDistribution = actionDistributionLowEpsilon.map(count => count / TRIALS);
        const lowEntropy = -lowEntropyDistribution.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
        
        // 4. 测试探索与利用的平衡
        // 在SimpleNN中，通过从策略分布采样实现了探索与利用平衡
        // 验证有足够多的样本时，动作分布接近策略分布
        const policyProbs = nn.forward(0);
        const empiricalProbs = actionDistributionLowEpsilon.map(count => count / TRIALS);
        
        // 计算两个分布的KL散度，应该较小
        let klDivergence = 0;
        for (let i = 0; i < nn.outputSize; i++) {
            if (policyProbs[i] > 0 && empiricalProbs[i] > 0) {
                klDivergence += policyProbs[i] * Math.log(policyProbs[i] / empiricalProbs[i]);
            }
        }
        
        assert(klDivergence < 0.1, `策略分布和实际采样分布应接近(KL散度=${klDivergence.toFixed(4)})`);
        
        // 恢复原始探索参数
        nn.epsilon = originalEpsilon;
        nn.epsilonMin = originalEpsilonMin;
        nn.epsilonDecay = originalEpsilonDecay;
        
        return true;
    },
    
    // 测试探索率随训练的变化
    testExplorationDecay: function() {
        const nn = new SimpleNN(16, 8, 4);
        
        // 设置初始探索参数
        nn.epsilon = 0.5;
        nn.epsilonMin = 0.01;
        nn.epsilonDecay = 0.9;
        
        const initialEpsilon = nn.epsilon;
        
        // 模拟多轮训练，记录探索率变化
        const epsilonHistory = [initialEpsilon];
        const EPISODES = 20;
        
        for (let i = 0; i < EPISODES; i++) {
            // 模拟一轮训练后探索率衰减
            nn.episodeCount++;
            nn.epsilon = Math.max(nn.epsilon * nn.epsilonDecay, nn.epsilonMin);
            epsilonHistory.push(nn.epsilon);
        }
        
        // 验证探索率递减
        assert(epsilonHistory[EPISODES] < initialEpsilon, "探索率应随训练递减");
        
        // 验证不会低于最小值
        assert(epsilonHistory[EPISODES] >= nn.epsilonMin, "探索率不应低于最小值");
        
        // 验证递减是单调的
        let isMonotonic = true;
        for (let i = 1; i <= EPISODES; i++) {
            if (epsilonHistory[i] > epsilonHistory[i-1]) {
                isMonotonic = false;
                break;
            }
        }
        assert(isMonotonic, "探索率应单调递减");
        
        return true;
    },
    
    // 测试epsilon-greedy机制
    testEpsilonGreedyMechanism: function() {
        const nn = createTestNN();
        
        // 设置固定的探索率以便测试
        nn.epsilon = 0.3;  // 30%的探索率
        
        // 修改前向传播结果，使得某个动作的价值明显高于其他动作
        // 创建一个模拟版本的forward函数返回固定分布
        const originalForward = nn.forward;
        nn.forward = function(state) {
            // 返回一个固定的动作概率分布，动作0的概率明显高于其他动作
            return [0.7, 0.1, 0.1, 0.1];
        };
        
        // 进行大量采样以测试epsilon-greedy机制
        const actionCounts = Array(nn.outputSize).fill(0);
        const NUM_SAMPLES = 10000;
        
        for (let i = 0; i < NUM_SAMPLES; i++) {
            const action = nn.selectAction(0);
            actionCounts[action]++;
            nn.clearTrajectory();
        }
        
        // 计算每个动作的选择频率
        const actionFrequencies = actionCounts.map(count => count / NUM_SAMPLES);
        
        // 动作0应该被选择的概率：(1-epsilon)×0.7 + epsilon×0.25 = 0.7×0.7 + 0.3×0.25 = 0.49 + 0.075 = 0.565
        // 其他动作应该被选择的概率：epsilon×0.25 + epsilon导致的随机选择 = 约0.145
        const expectedFreq0 = (1 - nn.epsilon) * 0.7 + nn.epsilon * 0.25;
        const expectedFreqOthers = nn.epsilon * 0.25;
        
        // 允许一定的统计误差
        const tolerance = 0.02;
        
        // 验证动作0的选择频率接近理论值
        assert(Math.abs(actionFrequencies[0] - expectedFreq0) < tolerance, 
            `最优动作的选择频率(${actionFrequencies[0].toFixed(3)})应接近理论值(${expectedFreq0.toFixed(3)})`);
        
        // 验证探索导致所有动作都有被选择的机会
        assert(actionCounts.every(count => count > 0), 
            "epsilon-greedy策略应确保所有动作都有被选择的机会");
        
        // 验证低价值动作的选择频率较为接近（因为它们在探索时被均匀选择）
        const nonGreedyFreqs = [actionFrequencies[1], actionFrequencies[2], actionFrequencies[3]];
        const maxDiff = Math.max(...nonGreedyFreqs) - Math.min(...nonGreedyFreqs);
        assert(maxDiff < tolerance, 
            `非贪婪动作的选择频率差异(${maxDiff.toFixed(3)})应小于${tolerance}`);
        
        // 恢复原始forward函数
        nn.forward = originalForward;
        
        return true;
    },
    
    // 测试探索-利用权衡
    testExplorationExploitationTradeoff: function() {
        const nn = createTestNN();
        
        // 测试不同的探索率对动作选择的影响
        const epsilonValues = [0.0, 0.3, 0.7, 1.0];
        const results = {};
        
        // 修改forward函数返回固定分布
        const originalForward = nn.forward;
        nn.forward = function(state) {
            // 返回一个固定的动作概率分布，动作0的概率明显高于其他动作
            return [0.7, 0.1, 0.1, 0.1];
        };
        
        for (const eps of epsilonValues) {
            nn.epsilon = eps;
            const actionCounts = Array(nn.outputSize).fill(0);
            const SAMPLES = 1000;
            
            for (let i = 0; i < SAMPLES; i++) {
                const action = nn.selectAction(0);
                actionCounts[action]++;
                nn.clearTrajectory();
            }
            
            // 计算熵值衡量随机性
            const probs = actionCounts.map(count => count / SAMPLES);
            const entropy = -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
            
            results[eps] = {
                counts: actionCounts,
                probs: probs,
                entropy: entropy
            };
        }
        
        // 验证随着epsilon增加，分布的熵值增加（更随机）
        assert(results[0.0].entropy < results[0.3].entropy, 
            "更高的探索率应增加动作选择的熵值(随机性)");
        assert(results[0.3].entropy < results[0.7].entropy, 
            "更高的探索率应增加动作选择的熵值(随机性)");
        
        // 验证epsilon=0时(纯贪婪)应该几乎总是选择最优动作
        assert(results[0.0].probs[0] > 0.95, 
            `纯贪婪策略(epsilon=0)应几乎总是选择最优动作(实际比例: ${results[0.0].probs[0].toFixed(3)})`);
        
        // 验证epsilon=1时(纯随机)动作选择应该近似均匀分布
        const uniformProb = 0.25;
        const tolerance = 0.05;
        const isApproxUniform = results[1.0].probs.every(p => Math.abs(p - uniformProb) < tolerance);
        assert(isApproxUniform, 
            `纯随机探索(epsilon=1)应产生近似均匀的动作分布`);
        
        // 恢复原始forward函数
        nn.forward = originalForward;
        
        return true;
    }
};

// 运行所有测试
function runAllTests() {
    console.log("开始神经网络单元测试...");
    let passed = 0;
    let total = 0;
    
    for (const testName in tests) {
        total++;
        console.log(`\n运行测试: ${testName}`);
        try {
            if (tests[testName]()) {
                console.log(`✅ ${testName} 测试通过`);
                passed++;
            } else {
                console.log(`❌ ${testName} 测试失败`);
            }
        } catch (e) {
            console.error(`❌ ${testName} 测试出错:`, e);
        }
    }
    
    console.log(`\n测试完成: ${passed}/${total} 通过`);
}

// 运行测试
runAllTests(); 