import gym
import retro
import pyglet
import time
from pyglet.window import key
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from baselines.common.retro_wrappers import Downsample, Rgb2gray, FrameStack, TimeLimit
from utilalien import *

#キーイベント用のウィンドウの生成
win = pyglet.window.Window(width=300, height=100, vsync=False)
key_handler = pyglet.window.key.KeyStateHandler()
win.push_handlers(key_handler)
pyglet.app.platform_event_loop.start()

#キー状態の取得
def get_key_state():
    key_state = set()
    win.dispatch_events()
    for key_code, pressed in key_handler.items():
        if pressed:
            key_state.add(key_code)
    return key_state
# キー入力待ち
while len(get_key_state()) == 0:
    time.sleep(1.0/30.0)

def main():
    # 環境の生成
    env = retro.make(game='AlienSoldier-Genesis', state='DefaultSettings.Level1')
    env = AlienDiscretizer(env) # 行動空間を離散空間に変換
    env = CustomRewardAndDoneEnv(env) # エピソード完了の変更
    env = Downsample(env, 2) # ダウンサンプリング
    env = Rgb2gray(env) # グレースケール
    env = TimeLimit(env, max_episode_steps=4500) # 5分タイムアウト

    # ベクトル環境の生成
    env = DummyVecEnv([lambda: env])

    # モデルの生成 rightモデル
    model = PPO2('CnnPolicy', env, verbose=1)

    # モデルの生成 shootモデル
    model2 = PPO2('CnnPolicy', env, verbose=1)

    # モデルの生成 jumpモデル
    model3 = PPO2('CnnPolicy', env, verbose=1)

    # モデルの読み込み rightモデル
    model = PPO2.load('alien_bc_right1000', env=env, verbose=0)

    # モデルの読み込み shootモデル
    model2 = PPO2.load('alien_bc_shoot1000', env=env, verbose=0)

    # モデルの読み込み jumpモデル
    model3 = PPO2.load('alien_bc_jump1000', env=env, verbose=0)

    # モデルのテスト
    state = env.reset()
    total_reward = 0
    judge = 0
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    countQ = 0
    countW = 0
    countE = 0
    active = 0

    while True:
        # 環境の描画
        env.render()

        # スリープ
        time.sleep(1/120)

        # モデルの推論
        # action, _ = model.predict(state)

        #キー入力によるモデルの切り替え
        key_state = get_key_state()
        action = [0]

        if key.Q in key_state: #rightモデルによる操作
            judge = 1
            countQ +=1        
        elif key.W in key_state: # shootモデルによる操作
            judge = 2
            countW +=1
        elif key.E in key_state: # jumpモデルよる操作
            judge = 3
            countE +=1
        
        if judge == 1:
            # rightモデルの推論
            action, _ = model.predict(state)
        elif judge == 2:
            # shootモデルの推論
            action, _ = model2.predict(state)
        elif judge == 3:
            # jumpモデルの推論
            action, _ = model3.predict(state)
            
        
        if (action == [0]):
            count0 += 1
        elif (action == [1]):
            count1 += 1
        elif (action == [2]):
            count2 += 1
        elif (action == [3]):
            count3 += 1
        elif (action == [4]):
            count4 += 1
        elif (action == [5]):
            count5 += 1
        elif (action == [6]):
            count6 += 1
        
        # 1ステップ実行
        state, reward, done, info = env.step(action)
        total_reward += reward[0]

        # エピソード完了
        if done:
            active += 1
            print('reward,', total_reward, ', action,',count0,',',count1,',',count2,',',count3,',',count4,',',count5,',',count6,', QWE',countQ,',',countW,',',countE,', count=',active)
            count0 = 0
            count1 = 0
            count2 = 0
            count3 = 0
            count4 = 0
            count5 = 0
            count6 = 0
            state = env.reset()
            total_reward = 0

#メインの実行
if __name__ == "__main__":
  main()