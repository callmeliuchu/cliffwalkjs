// CliffWalk环境类
class CliffWalkEnv {
    constructor(rows=2, cols=3) {
        this.rows = rows;
        this.cols = cols;
        this.maxSteps = rows * cols * 3;  // 设置最大步数为格子数的3倍
        this.reset();
    }

    reset() {
        // Start at bottom-left corner
        this.state = this.cols * (this.rows - 1);
        this.done = false;
        this.steps = 0;
        this.totalReward = 0;
        return this.state;
    }

    step(action) {
        if (this.done) {
            console.log("Episode already done!");
            return [this.state, 0, true];
        }

        this.steps += 1;
        
        // 检查是否超过最大步数
        if (this.steps >= this.maxSteps) {
            console.log("Reached maximum steps!");
            this.done = true;
            return [this.state, -10, true];  // 给一个负奖励
        }

        let row = Math.floor(this.state / this.cols);
        let col = this.state % this.cols;
        let reward = -1;
        let done = false;

        // Move according to action
        switch(action) {
            case 0: // up
                if (row > 0) row -= 1;
                break;
            case 1: // right
                if (col < this.cols - 1) col += 1;
                break;
            case 2: // down
                if (row < this.rows - 1) row += 1;
                break;
            case 3: // left
                if (col > 0) col -= 1;
                break;
        }

        // Calculate new state
        this.state = row * this.cols + col;

        // Check if hit cliff or reached goal
        if (row === this.rows - 1 && col > 0 && col < this.cols - 1) {
            reward = -10;  // 悬崖惩罚
            // done = true;   // 掉悬崖立即结束
        } else if (row === this.rows - 1 && col === this.cols - 1) {
            reward = 10;   // 增加到达目标的奖励
            done = true;
        }

        this.done = done;
        this.totalReward += reward;

        return [this.state, reward, done];
    }
}

// 辅助函数
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
} 