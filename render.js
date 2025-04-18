// 全局状态变量
let env;
let agent;
let currentState;
let isPlaying = false;
let episodeCount = 0;
let totalRewards = [];
let bestReward = -Infinity;
let totalEpisodes = 0;
let successfulEpisodes = 0;

// 初始化环境和智能体
function initializeGame(rows = 4, cols = 8) {
    env = new CliffWalkEnv(rows, cols);
    currentState = env.reset();
    agent = new SimpleNN(env.rows * env.cols, 64, 4);
}

// 网格更新
function updateGrid() {
    const grid = document.getElementById('grid');
    grid.style.gridTemplateColumns = `repeat(${env.cols}, 60px)`;
    grid.innerHTML = '';
    
    for (let i = 0; i < env.rows; i++) {
        for (let j = 0; j < env.cols; j++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            const stateNum = i * env.cols + j;
            
            cell.title = `${stateNum}`;
            
            if (stateNum === currentState) {
                cell.classList.add('agent');
                cell.textContent = '👾';
            } else if (i === env.rows - 1 && j > 0 && j < env.cols - 1) {
                cell.classList.add('cliff');
                cell.textContent = '💀';
            } else if (i === env.rows - 1 && j === env.cols - 1) {
                cell.classList.add('goal');
                cell.textContent = '🎯';
            } else {
                cell.textContent = stateNum;
            }
            
            grid.appendChild(cell);
        }
    }
}

// 信息更新
function updateInfo(action, reward) {
    document.getElementById('state').textContent = currentState;
    document.getElementById('action').textContent = 
        action !== undefined ? ['Up', 'Right', 'Down', 'Left'][action] : 'None';
    document.getElementById('reward').textContent = reward !== undefined ? reward : 0;
    document.getElementById('steps').textContent = env.steps;
    document.getElementById('total-reward').textContent = env.totalReward;
}

// 重置环境
async function resetEnv() {
    currentState = env.reset();
    updateGrid();
    updateInfo();
}

// 添加训练信息显示
function addTrainingInfo() {
    const infoDiv = document.querySelector('.info');
    const trainingInfo = document.createElement('div');
    trainingInfo.className = 'training-info';
    trainingInfo.innerHTML = `
        <p>Episode: <span id="episode-count">0</span></p>
        <p>Best Reward: <span id="best-reward">0</span></p>
        <p>Average Reward: <span id="avg-reward">0</span></p>
        <p>Exploration Rate: <span id="epsilon">1.000</span></p>
        <p>Steps per Episode: <span id="steps-per-episode">0</span></p>
        <p>Success Rate: <span id="success-rate">0.00%</span></p>
        <p>Status: <span id="training-status" style="font-weight: bold;">Training...</span></p>
    `;
    infoDiv.appendChild(trainingInfo);
}

// 更新训练信息
async function updateTrainingInfo() {
    document.getElementById('episode-count').textContent = episodeCount;
    document.getElementById('best-reward').textContent = bestReward.toFixed(1);
    document.getElementById('epsilon').textContent = agent.epsilon.toFixed(3);
    document.getElementById('steps-per-episode').textContent = env.steps;
    
    // 更新成功率
    const successRate = (totalEpisodes > 0) ? (successfulEpisodes / totalEpisodes * 100).toFixed(2) : "0.00";
    document.getElementById('success-rate').textContent = `${successRate}% (${successfulEpisodes}/${totalEpisodes})`;
    
    if (totalRewards.length > 0) {
        const avgReward = totalRewards.reduce((a, b) => a + b, 0) / totalRewards.length;
        document.getElementById('avg-reward').textContent = avgReward.toFixed(1);
    }
}

// 运行一个episode
async function runEpisode() {
    if (!isPlaying) return;
    
    resetEnv();
    let episodeReward = 0;
    totalEpisodes++; // 增加总尝试次数
    
    while (isPlaying && !env.done) {
        if (env.steps >= env.maxSteps) {
            console.log("Episode terminated due to max steps");
            break;
        }
        const action = agent.selectAction(currentState);
        simulateKeyPress(action);
        await sleep(100);
    }

    // 检查是否成功到达终点
    const row = Math.floor(currentState / env.cols);
    const col = currentState % env.cols;
    if (row === env.rows - 1 && col === env.cols - 1) {
        successfulEpisodes++; // 增加成功次数
    }

    if (env.totalReward > bestReward) {
        bestReward = env.totalReward;
    }
    totalRewards.push(env.totalReward);
    episodeCount++;
    
    // 检查是否收敛
    const hasConverged = agent.checkConvergence(env.totalReward);
    const statusElement = document.getElementById('training-status');
    if (hasConverged) {
        statusElement.textContent = '✅ Trained Successfully!';
        statusElement.style.color = '#4CAF50';
    } else {
        statusElement.textContent = 'Training...';
        statusElement.style.color = '#666';
    }
    
    await updateTrainingInfo();
    
    if (isPlaying) {
        runEpisode();
    }
}

// 添加控制按钮
function addControls() {
    // 先检查是否已经存在controls
    let controls = document.querySelector('.controls');
    if (controls) {
        controls.remove();
    }

    controls = document.createElement('div');
    controls.className = 'controls';
    controls.innerHTML = `
        <button id="aiStep">AI Step</button>
        <button id="autoPlay">Auto Play</button>
        <button id="stopAutoPlay">Stop Auto Play</button>
    `;
    
    // 将controls添加到info div中
    const infoDiv = document.querySelector('.info');
    infoDiv.appendChild(controls);

    // 确保按钮存在后再添加事件监听器
    const aiStepBtn = document.getElementById('aiStep');
    const autoPlayBtn = document.getElementById('autoPlay');
    const stopAutoPlayBtn = document.getElementById('stopAutoPlay');

    if (aiStepBtn) {
        aiStepBtn.addEventListener('click', async () => {
            console.log('AI Step clicked');
            if (!env.done) {
                const action = agent.selectAction(currentState);
                console.log('Selected action:', action);
                simulateKeyPress(action);
            }
        });
    }

    if (autoPlayBtn) {
        autoPlayBtn.addEventListener('click', () => {
            console.log('Auto Play clicked');
            if (!isPlaying) {
                isPlaying = true;
                runEpisode();
            }
        });
    }

    if (stopAutoPlayBtn) {
        stopAutoPlayBtn.addEventListener('click', () => {
            console.log('Stop Auto Play clicked');
            isPlaying = false;
        });
    }
}

// 模拟按键
function simulateKeyPress(action) {
    const keyMap = {
        0: 'ArrowUp',
        1: 'ArrowRight',
        2: 'ArrowDown',
        3: 'ArrowLeft'
    };
    
    // 确保action是有效的数字
    if (action === undefined || action === null) {
        console.error('Invalid action:', action);
        action = Math.floor(Math.random() * 4);  // 随机选择一个动作作为后备
    }
    
    const key = keyMap[action];
    console.log('Simulating key press:', key, 'for action:', action);
    
    if (key) {
        const event = new KeyboardEvent('keydown', {
            key: key,
            bubbles: true,
            cancelable: true
        });
        document.dispatchEvent(event);
    } else {
        console.error('Invalid action number:', action);
    }
}

// 添加键盘事件监听
function setupKeyboardEvents() {
    document.addEventListener('keydown', async (event) => {
        if (env.done) return;

        let action;
        switch(event.key) {
            case 'ArrowUp':
                action = 0;
                break;
            case 'ArrowRight':
                action = 1;
                break;
            case 'ArrowDown':
                action = 2;
                break;
            case 'ArrowLeft':
                action = 3;
                break;
            default:
                return;
        }

        const [newState, reward, done] = env.step(action);
        
        // 获取当前和新的单元格
        const oldCell = document.querySelector(`.cell[title="${currentState}"]`);
        const newCell = document.querySelector(`.cell[title="${newState}"]`);
        
        if (oldCell && newCell) {
            // 移除旧位置的agent
            oldCell.classList.remove('agent');
            if (!oldCell.classList.contains('cliff') && !oldCell.classList.contains('goal')) {
                oldCell.style.backgroundColor = 'white';
                oldCell.textContent = currentState;
            } else if (oldCell.classList.contains('cliff')) {
                oldCell.textContent = '💀';
            } else if (oldCell.classList.contains('goal')) {
                oldCell.textContent = '🎯';
            }
            
            // 添加新位置的agent
            newCell.classList.add('agent');
            newCell.textContent = '👾';
        }
        
        currentState = newState;
        updateInfo(action, reward);

        if (done) {
            console.log("Episode finished!");
            console.log("Total Steps:", env.steps);
            console.log("Total Reward:", env.totalReward);
            
            // 增加总尝试次数
            totalEpisodes++;
            
            // 检查是否成功到达终点
            const row = Math.floor(currentState / env.cols);
            const col = currentState % env.cols;
            if (row === env.rows - 1 && col === env.cols - 1) {
                successfulEpisodes++;
            }
            
            // 更新成功率显示
            updateTrainingInfo();
            
            // 更新策略
            agent.update(reward);
            
            await sleep(300);  // 减少等待时间
        } else {
            // 记录非终止状态的奖励
            agent.trajectory.rewards.push(reward);
        }
    });
}

// 初始化页面
function initializePage() {
    initializeGame();
    updateGrid();
    updateInfo();
    addTrainingInfo();
    addControls();
    setupKeyboardEvents();
}

// 页面加载完成后初始化
window.addEventListener('load', initializePage); 