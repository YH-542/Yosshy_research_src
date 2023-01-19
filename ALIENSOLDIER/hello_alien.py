import retro
import time 
import pyglet
from pyglet.window import key
from utilalien import *

#環境の生成
env = retro.make(game='AlienSoldier-Genesis', state='DefaultSettings.Level1')
env = AlienDiscretizer(env) # 行動空間を離散空間に変換

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

# 0:なし
# 1:LEFT
# 2:RIGHT
# 3:A PC:A
# 4:B PC:S
# 5:C PC:D
# 6:DOWN+C
# 7:LEFT+B
# 8:RIGHT+B


#ランダム行動による動作確認
state = env.reset()
while True:
  #環境の描画
  env.render()

  #スリープ
  time.sleep(1/60)
  key_state = get_key_state()
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
  
  #1ステップ実行
  state, reward, done, info = env.step(action)
  # print('reward:', reward)
  print('info:', info)

  if done :
    print('done')
    state = env.reset()