from dl_helper.rl.rl_env.cartpole_env import CartPoleEnv

import pygame, time
import numpy as np

delay = 0.05  # 设置延迟时间（秒），可以调整这个值来改变速度

def main():
    print("Initializing environment...")
    env = CartPoleEnv(render_mode="rgb_array")
    screen_size = (600, 400)
    
    print("Resetting environment...")
    observation, info = env.reset()
    print(f"Observation{observation.shape}: \n{observation}")
    print(f"Info: \n{info}")
    
    print("Initializing Pygame...")
    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption('CartPole - Use LEFT/RIGHT arrows to control, Q to quit')
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    frame_count = 0

    print("Controls:")
    print("LEFT ARROW  - Move cart left")
    print("RIGHT ARROW - Move cart right")
    print("Q          - Quit game")

    while not done:
        frame_count += 1
        
        # 渲染游戏画面
        frame = env.render()
        
        if frame is not None:
            try:
                if len(frame.shape) == 3:
                    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    frame_surface = pygame.transform.scale(frame_surface, screen_size)
                    screen.blit(frame_surface, (0, 0))
                    
                    # 显示得分和控制提示
                    font = pygame.font.Font(None, 36)
                    score_text = font.render(f'Score: {total_reward}', True, (255, 255, 255))
                    control_text = font.render('Use LEFT/RIGHT arrows', True, (255, 255, 255))
                    screen.blit(score_text, (10, screen_size[1]-60))
                    screen.blit(control_text, (10, screen_size[1]-30))
                    
                    pygame.display.flip()
            except Exception as e:
                print(f"Error during rendering: {e}")
        
        time.sleep(delay)
        action = None  # 初始化为 None
        
        # 处理键盘输入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Quit event received")
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    print("Q key pressed - quitting")
                    done = True
        
        # 获取当前按下的所有键
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0
            print("LEFT pressed")
        elif keys[pygame.K_RIGHT]:
            action = 1
            print("RIGHT pressed")
            
        # 如果没有按键输入，使用默认动作
        if action is None:
            action = 0
        
        # 执行动作
        observation, reward, done, truncated, info = env.step(action)
        print(f"Observation{observation.shape}: \n{observation}")
        print(f"Info: \n{info}")
        total_reward += reward

        if done:
            print(f"Episode finished with total reward: {total_reward}")
    
    print("Cleaning up...")
    env.close()
    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()