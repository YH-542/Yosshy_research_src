import gym
import retro
import time
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from baselines.common.retro_wrappers import Downsample, Rgb2gray, FrameStack, TimeLimit
from utilalien import *

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

    # デモの読み込み
    dataset = ExpertDataset(expert_path='alien_demo_normal.npz', verbose=1)

    # モデルの生成
    #model = PPO2('CnnPolicy', env, verbose=1)

    # モデルの読み込み
    model = PPO2.load('alien_bc_normal', env=env, verbose=0)

    # # モデルの事前学習
    # model.pretrain(dataset, n_epochs=100)

    # # モデルの学習(今回は必要なし)
    # # model.learn(total_timesteps=1000)

    # # モデルの保存
    # model.save('alien_bc_jump')

    # モデルのテスト
    state = env.reset()
    total_reward = 0
    while True:
        # 環境の描画
        env.render()

        # スリープ
        time.sleep(1/120)

        # モデルの推論
        action, _ = model.predict(state)

        # 1ステップ実行
        state, reward, done, info = env.step(action)
        total_reward += reward[0]

        # エピソード完了
        if done:
            print('reward:', total_reward)
            state = env.reset()
            total_reward = 0

#メインの実行
if __name__ == "__main__":
  main()