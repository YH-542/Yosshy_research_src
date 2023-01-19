import random
import pyglet
import retro
import time
from pyglet.window import key
from stable_baselines.gail import generate_expert_traj
from baselines.common.retro_wrappers import Downsample, Rgb2gray, FrameStack, TimeLimit
from utilalien import *

# 環境の生成 (1)
env = retro.make(game='AlienSoldier-Genesis', state='DefaultSettings.Level1')
env = AlienDiscretizer(env) # 行動空間を離散空間に変換
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

# 人間のデモを収集するコールバック
def human_expert(_state):
    # キー状態の取得
    key_state = get_key_state()

    # キー状態を行動に変換
    action = 0
    if key.LEFT in key_state:
        action = 1
    elif key.RIGHT in key_state:
        action = 2
    elif key.A in key_state:
        action = 3
    elif key.S in key_state:
        if key.LEFT in key_state:
            action = 7
        elif key.RIGHT in key_state:
            action = 8
        else:
            action = 4
    elif key.D in key_state:
        if key.DOWN in key_state:
            action = 6
        else:
            action = 5

    # スリープ
    time.sleep(1.0/120.0)

    # 環境の描画
    env.render()

    # 行動の選択
    return action

# 人間のデモの収集の開始
generate_expert_traj(human_expert, 'alien_demo_jump', env, n_episodes=5)