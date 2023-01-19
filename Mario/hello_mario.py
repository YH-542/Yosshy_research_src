from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time 

#環境の生成
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#ランダム行動による動作確認
state = env.reset()
while True:
  #環境の描画
  env.render()

  #スリープ
  time.sleep(1/60)

  #1ステップ実行
  state, reward, done, info = env.step(env.action_space.sample())
  print('reward:', reward)
  print('info:', info)

  if done :
    print('done')
    state = env.reset()