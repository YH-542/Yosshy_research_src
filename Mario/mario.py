import os
import time
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from baselines.common.retro_wrappers import *
from utilsmario import CustomRewardAndDoneEnv


def main():
    # 環境の生成
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomRewardAndDoneEnv(env) # エピソード完了の変更
    env = Downsample(env, 2) # ダウンサンプリング
    env = Rgb2gray(env) # グレースケール
    env = TimeLimit(env, max_episode_steps=4500) # 5分タイムアウト

    # ベクトル環境の生成
    env = DummyVecEnv([lambda: env])

    # デモの読み込み
    dataset = ExpertDataset(expert_path='mario_demo_normal.npz', verbose=1)

    # モデルの生成
    model = PPO2('CnnPolicy', env, verbose=1)

    # モデルの読み込み
    # model = PPO2.load('sonic_bc_model', env=env, verbose=0)

    # モデルの事前学習
    model.pretrain(dataset, n_epochs=100)

    # モデルの学習(今回は必要なし)
    # model.learn(total_timesteps=1000)

    # モデルの保存
    model.save('mario_bc_normal')

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