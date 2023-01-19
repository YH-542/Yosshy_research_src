import gym
import numpy as np

# SonicDiscretizerラッパー
class AlienDiscretizer(gym.ActionWrapper):
    # 初期化
    def __init__(self, env):
        super(AlienDiscretizer, self).__init__(env)
        buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        actions = [[],['LEFT'],['RIGHT'],['A'],['B'],['C'],['DOWN','C'],['LEFT','B'],['RIGHT','B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    # 行動の取得
    def action(self, a):
        return self._actions[a].copy()

# 0:なし
# 1:LEFT
# 2:RIGHT
# 3:A PC:A
# 4:B PC:S
# 5:C PC:D
# 6:DOWN+C
# 7:LEFT+B
# 8:RIGHT+B

# CustomRewardAndDoneラッパー
class CustomRewardAndDoneEnv(gym.Wrapper):
    # ステップ
    def step(self, action):
        state, rew, done, info = self.env.step(action)

        # エピソード完了の変更
        if info['health'] == 0 or info['time'] == 130:
            # print('time,', info['time'])
            done = True

        return state, rew, done, info