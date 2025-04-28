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
        
        // 学习率
        this.learningRate = 0.001;
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
        // 创建输入向量 - 使用独热编码表示状态
        const input = Array(this.inputSize).fill(0);
        input[state] = 1;
        
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
    backprop(state, action, G) {
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

    // 设置学习率
    setLearningRate(rate) {
        this.learningRate = rate;
    }
} 