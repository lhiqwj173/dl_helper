from dl_helper.rl.rl_env.lob_env import LOB_trade_env

def paly():
    env = LOB_trade_env()
    
    # 回合的评价指标
    state, info = env.reset()
    done = False
    while not done:
        # 动作
        action = env.action_space.sample()

if __name__ == '__main__':
    paly()

