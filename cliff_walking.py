import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. 環境設定 (Environment Setup)
# ==========================================
ROWS = 4
COLS = 12
START = (3, 0)
GOAL = (3, 11)

# 動作: 0:上, 1:下, 2:左, 3:右
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def step(state, action_idx):
    """
    執行動作，回傳 (next_state, reward, done)
    """
    r, c = state
    dr, dc = ACTIONS[action_idx]
    nr, nc = r + dr, c + dc
    
    # 邊界處理
    nr = max(0, min(ROWS - 1, nr))
    nc = max(0, min(COLS - 1, nc))
    next_state = (nr, nc)
    
    # 懸崖區域: (3, 1) 到 (3, 10)
    if nr == 3 and 1 <= nc <= 10:
        return START, -100, False
    
    # 到達終點
    if next_state == GOAL:
        return next_state, -1, True
        
    return next_state, -1, False

# ==========================================
# 2. 策略 (Policy)
# ==========================================
def choose_action(state, q_table, epsilon):
    """
    ε-greedy 策略選擇動作
    """
    if np.random.rand() < epsilon:
        return np.random.randint(len(ACTIONS))
    else:
        values = q_table[state[0], state[1], :]
        # 處理多個相同最大值的情況，隨機選擇其中一個
        max_val = np.max(values)
        max_indices = np.where(values == max_val)[0]
        return np.random.choice(max_indices)

# ==========================================
# 3. 演算法實作 (Algorithms)
# ==========================================
def q_learning(episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = np.zeros((ROWS, COLS, len(ACTIONS)))
    rewards = []
    
    for _ in range(episodes):
        state = START
        total_reward = 0
        done = False
        
        while not done:
            action = choose_action(state, q_table, epsilon)
            next_state, reward, done = step(state, action)
            
            # Q-learning 更新 (Off-policy)
            # 基於 next_state 的最大可能 Q 值來更新
            best_next_action = np.argmax(q_table[next_state[0], next_state[1], :])
            td_target = reward + gamma * q_table[next_state[0], next_state[1], best_next_action] * (not done)
            td_error = td_target - q_table[state[0], state[1], action]
            
            q_table[state[0], state[1], action] += alpha * td_error
            
            state = next_state
            total_reward += reward
            
        rewards.append(total_reward)
        
    return q_table, rewards

def sarsa(episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = np.zeros((ROWS, COLS, len(ACTIONS)))
    rewards = []
    
    for _ in range(episodes):
        state = START
        action = choose_action(state, q_table, epsilon)
        total_reward = 0
        done = False
        
        while not done:
            next_state, reward, done = step(state, action)
            next_action = choose_action(next_state, q_table, epsilon)
            
            # SARSA 更新 (On-policy)
            # 基於實際採取的 next_action 的 Q 值來更新
            td_target = reward + gamma * q_table[next_state[0], next_state[1], next_action] * (not done)
            td_error = td_target - q_table[state[0], state[1], action]
            
            q_table[state[0], state[1], action] += alpha * td_error
            
            state = next_state
            action = next_action
            total_reward += reward
            
        rewards.append(total_reward)
        
    return q_table, rewards

def get_policy_path(q_table):
    """
    根據學到的 Q-table 取得純貪婪策略路徑 (epsilon=0)
    """
    state = START
    path = [state]
    done = False
    steps = 0
    # 限制最大步數防止無窮迴圈
    while not done and steps < 50:
        action = np.argmax(q_table[state[0], state[1], :])
        next_state, _, done = step(state, action)
        path.append(next_state)
        state = next_state
        if state == START: # 掉入懸崖
            break
        steps += 1
    return path

# ==========================================
# 4. 執行與評估 (Execution & Evaluation)
# ==========================================
if __name__ == '__main__':
    episodes = 500
    runs = 50 # 執行多次取平均使曲線更平滑
    
    q_rewards_all = np.zeros((runs, episodes))
    sarsa_rewards_all = np.zeros((runs, episodes))

    print(f"開始訓練... 總共將執行 {runs} 次取平均 (每次 {episodes} 回合)")
    
    final_q_table = None
    final_sarsa_table = None
    
    for i in range(runs):
        q_table, q_rewards = q_learning(episodes=episodes)
        sarsa_table, sarsa_rewards = sarsa(episodes=episodes)
        
        q_rewards_all[i] = q_rewards
        sarsa_rewards_all[i] = sarsa_rewards
        
        if i == runs - 1:
            final_q_table = q_table
            final_sarsa_table = sarsa_table
            
    print("訓練完成！")
            
    # 平均獎勵
    q_rewards_avg = np.mean(q_rewards_all, axis=0)
    sarsa_rewards_avg = np.mean(sarsa_rewards_all, axis=0)
    
    # 平滑處理 (Moving Average) 讓圖表更容易觀察
    def moving_average(a, n=10):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return np.concatenate((a[:n-1], ret[n - 1:] / n))

    plt.figure(figsize=(10, 6))
    plt.plot(moving_average(sarsa_rewards_avg, 10), label='SARSA (On-policy)', color='#17becf')
    plt.plot(moving_average(q_rewards_avg, 10), label='Q-learning (Off-policy)', color='#d62728')
    plt.ylim(-100, -10)
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.title('SARSA vs Q-learning on Cliff Walking (Averaged over 50 runs)')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png', dpi=300)
    print("\n已儲存學習曲線圖至 'learning_curve.png'")
    
    # 輸出最終策略路徑
    q_path = get_policy_path(final_q_table)
    sarsa_path = get_policy_path(final_sarsa_table)
    
    print("\n--- 最終學習到的路徑 (ε=0) ---")
    print(f"Q-learning 路徑長度: {len(q_path)-1} 步")
    print(f"Q-learning 走法: {q_path}")
    print(f"\nSARSA 路徑長度: {len(sarsa_path)-1} 步")
    print(f"SARSA 走法: {sarsa_path}")

    # ==========================================
    # 5. 繪製路徑視覺化圖 (Path Visualization)
    # ==========================================
    def draw_grid_path(ax, title, path, color):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim(0, COLS)
        ax.set_ylim(ROWS, 0) # 反轉 Y 軸，讓 (0,0) 在左上角
        ax.set_xticks(np.arange(COLS+1))
        ax.set_yticks(np.arange(ROWS+1))
        ax.grid(color='k', linestyle='-', linewidth=1)
        
        # 標示懸崖
        for c in range(1, 11):
            rect = patches.Rectangle((c, 3), 1, 1, linewidth=1, edgecolor='k', facecolor='skyblue')
            ax.add_patch(rect)
            ax.text(c + 0.5, 3.5, 'Cliff', ha='center', va='center', fontsize=10, color='blue', alpha=0.5)
        
        # 標示起點與終點
        ax.text(0.5, 3.5, 'Start', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(11.5, 3.5, 'Goal', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 畫出路徑
        if path:
            xs = [c + 0.5 for r, c in path]
            ys = [r + 0.5 for r, c in path]
            # 畫出點跟線
            ax.plot(xs, ys, color=color, marker='o', linewidth=3, markersize=6, alpha=0.8)
            # 加上方向箭頭
            for i in range(len(xs)-1):
                ax.annotate('', xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                            arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.8))
            
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    draw_grid_path(ax1, 'Q-learning Path (Off-policy, Risk-seeking)', q_path, '#d62728') # 紅色
    draw_grid_path(ax2, 'SARSA Path (On-policy, Safe/Conservative)', sarsa_path, '#17becf') # 青色
    plt.tight_layout()
    plt.savefig('cliff.jpg', dpi=300, bbox_inches='tight')
    print("\n已儲存路徑視覺化圖至 'cliff.jpg'")
