import pyglet
import retro
import time
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from pyglet.window import key
from stable_baselines.gail import generate_expert_traj
from baselines.common.retro_wrappers import *
from utilsmario import *

# 環境の生成 (1)
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = CustomRewardAndDoneEnv(env) # エピソード完了の変更
env = Downsample(env, 2) # ダウンサンプリング
env = Rgb2gray(env) # グレースケール
env = TimeLimit(env, max_episode_steps=4500) # 5分タイムアウト
env.reset()
env.render()

# キーイベント用のウィンドウの生成
win = pyglet.window.Window(width=300, height=100, vsync=False)
key_handler = pyglet.window.key.KeyStateHandler()
win.push_handlers(key_handler)
pyglet.app.platform_event_loop.start()

# キー状態の取得
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

# # actions for very simple movement
# SIMPLE_MOVEMENT = [
#     ['NOOP'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['right', 'A', 'B'],
#     ['A'],
#     ['left'],
# ]


# # actions for more complex movement
# COMPLEX_MOVEMENT = [
#     ['NOOP'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['right', 'A', 'B'],
#     ['A'],
#     ['left'],
#     ['left', 'A'],
#     ['left', 'B'],
#     ['left', 'A', 'B'],
#     ['down'],
#     ['up'],
# ]

# 人間のデモを収集するコールバック
def human_expert(_state):
    # キー状態の取得
    key_state = get_key_state()

    # キー状態を行動に変換
    action = 0
    if key.RIGHT in key_state:
        if key.D in key_state:
            if key.S in key_state:
                action = 4
            else:
                action = 2
        if key.S in key_state:
            if key.D in key_state:
                action = 4
            else:
                action = 3
        else:
            action = 1
    elif key.D in key_state:
        if key.RIGHT in key_state:
            if key.S in key_state:
                action = 4
            else:
                action = 2
        else:
            action = 5
    elif key.LEFT in key_state:
        action = 6

    # スリープ
    time.sleep(1.0/120.0)

    # 環境の描画
    env.render()

    # 行動の選択
    return action

# 人間のデモの収集の開始
generate_expert_traj(human_expert, 'mario_demo_normal', env, n_episodes=10)