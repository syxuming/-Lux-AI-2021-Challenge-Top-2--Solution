# Lux-AI-2021-Challenge-Top-2%-Solution

![header](https://user-images.githubusercontent.com/22366914/145206873-cab4c152-16da-4476-acbc-30d17e6f0ddb.png)

https://www.kaggle.com/c/lux-ai-2021



## 比赛介绍

本次竞赛是由Lux AI举办的，参赛者将开发一个会玩策略有些的AI Bot，两个队伍分别控制一队单位，收集资源为他们的城市提供燃料，主要目标是在回合制游戏结束时尽可能多地拥有城市地块。两队都有关于整个游戏状态的完整信息，并需要利用这些信息来优化资源收集，与对手争夺稀缺资源，并建造城市以获得积分。

- 竞赛类型：本次竞赛属于 **深度学习/强化学习** ，所以推荐使用相关的模型 Imitation Learning，DQN等。
- 评估标准：排行榜会公布每一个Bot的实时的天梯分和排名，截止时间的最终排名，来决定谁能获得各种基于排名的奖项。
- 游戏规则：[Lux AI Challenge (lux-ai.org)](https://www.lux-ai.org/specs-2021) # 附中文翻译版本 [Lux AI中文规则介绍\] | Kaggle](https://www.kaggle.com/nin7a1/lux-ai-rules-lux-ai)
- 如何在Kaggle平台上开发一个最简的Lux AI Bot：[Lux AI Season 1 Jupyter Notebook Tutorial | Kaggle](https://www.kaggle.com/stonet2000/lux-ai-season-1-jupyter-notebook-tutorial)
- 高分Bot的游戏视频：[Lux AI Season 1 Sprint 3 Livestream - YouTube](https://www.youtube.com/watch?v=De4VTxn7bes)
- 录像播放器：https://2021vis.lux-ai.org/



## 如何开始

你需要的唯一先决条件是安装NodeJS v12或以上，以及你选择的编程语言的任何工具。Lux AI团队为本次比赛提供了Python、Typescript/Javascript、C++和Java的技术支持。更多语言待定。

除了你开始编写机器人并提交给比赛所需的启动工具包外，所有的说明都在https://github.com/Lux-AI-Challenge/Lux-Design-2021。

对于希望得到使用教程的python用户，请随时通过复制和编辑这个[模板](https://www.kaggle.com/stonet2000/lux-ai-season-1-jupyter-notebook-tutorial)，开始使用Jupyter笔记本或Kaggle笔记本（一个互动的在线编辑器）。

对于那些熟悉使用笔记本参加Kaggle模拟竞赛的人来说，可以随时复制这个快速入门[模板](https://www.kaggle.com/stonet2000/lux-ai-season-1-jupyter-notebook-quickstart)。



## 解决方案思路

本次比赛的解决方案有三大流派：Rule-based， Imitation Learning， Reinforcement Learning。

我们本次选择的方案是 Imitation Learning 模仿学习。顾名思义是通过深度神经网络，学习高分Bot的replays，模仿高分Bot的行为。

 Imitation Learning 方案参考的 [Lux AI with Imitation Learning | Kaggle](https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning)

- 数据上，将replay中每个回合的observation数据和action数据 整理成20*32的矩阵数据。
- 模型采用的是：2DCNN+MLP

- Loss函数: CrossEntropy
- 其他trick：尝试了数据增强，样本标签平衡，更换Focal Loss，采用升温cosine调度器。
- 单独讲一下数据增强（即便最终方案没有使用）：如果把地图看成X-Y坐标轴，unit会根据observation来决策作出action，那么如果将X轴对称变换，observation的信息也就随之对称变换，那么action会根据新的observation做出决策即可。举个例子：如果在某一个observation下，unit move east，那么将X轴对称变换后，unit move west即可。



## 比赛上分历程

1. 直接使用 [参考方案](https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning) 后，Rank : 1320+
2. 原方案基础上尝试降低学习率，Rank : 1370+
3. 换成升温cosine调度器，Rank : 1390+
3. 原方案中只向rank排名第一的Team: Toad Brigade 学习，改成不限队伍，只要replay分数是1800+即可，Rank : 1420+
4. 尝试加入数据增强，没有提升
5. 使用新特征，并改成全量的训练数据，跑10个seed并简单平均融合，Public LB : 0.1322
6. 再对 batch_size、learning_rate、optimizer调参，10个seed并简单平均融合，Public LB : 0.1303
7. 20个seed，Public LB : 0.1292
8. 让unit学习不动 (move center)，Rank : 1490+ 
8. 因为move center指令太多，尝试平衡样本标签，Rank : 1530+
8. 尝试Focal Loss ，没有提升
8. 尝试XAVIER初始化，没有提升
8. 将epoch固定在25，并减小 batch size 至128，Rank : 1570+



## 矩阵数据生成

```python
# 将replay中每个回合的observation数据和action数据 整理成20*32的矩阵数据
def make_input(obs, unit_id):
    width, height = obs['width'], obs['height'] # 12, 16, 24, or 32
    # 让小棋盘移动到32*32的正中间玩
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    b = np.zeros((20, 32, 32), dtype=np.float32)
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            if unit_id == strs[3]:
                # Position and Cargo
                # unit自己: 0,1
                b[:2, x, y] = (
                    1,
                    (wood + coal + uranium) / 100
                )
            else:
                # Units
                team = int(strs[2])
                cooldown = float(strs[6])
                idx = 2 + (team - obs['player']) % 2 * 3
                # 我方的unit: 2,3,4 对方的unit: 5,6,7
                b[idx:idx + 3, x, y] = (
                    1,
                    cooldown / 6,
                    (wood + coal + uranium) / 100
                ) # todo 是不是没有考虑重叠的unit?
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            idx = 8 + (team - obs['player']) % 2 * 2
            # 我方的citytiles:8,9, 对方的citytiles:10,11 
            b[idx:idx + 2, x, y] = (
                1,
                cities[city_id] # 城市燃料够用多少天（最大10天），且除以10归一化(0,1]
            )
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 12, 'coal': 13, 'uranium': 14}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            # 我方的研究点数: 15, 我方的研究点数: 16
            b[15 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
    
    # Day/Night Cycle # 日夜循环中的几点，除以40归一化
    b[17, :] = obs['step'] % 40 / 40
    # Turns # 第几个turn，除以360归一化
    b[18, :] = obs['step'] / 360 
    # Map Size # 地图范围
    b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    return b
```



## 模型代码

```python
# Neural Network for Lux AI
class BasicConv2d(nn.Module):
    '''
    2D CNN层: Conv2d + BatchNorm2d
    '''
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, 
            kernel_size=kernel_size, 
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class LuxNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = BasicConv2d(20, filters, (3, 3), True) # 20 * 32
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, len(actions_fullname), bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        p = self.head_p(h_head)
        return p
```



## 代码、数据集

+ 代码
  lux-ai-with-imitation-learning.ipynb
  
  
## TL;DR
竞赛是由Lux AI举办的，参赛者将开发一个会玩策略有些的AI Bot，两个队伍分别控制一队单位，收集资源为他们的城市提供燃料，主要目标是在回合制游戏结束时尽可能多地拥有城市地块。本次竞赛中我们团队选择的方案是 Imitation Learning，我们通过将replay中每个回合的observation数据和action数据 整理成20*32的矩阵数据。然后输入给一个 2DCNN+MLP 的深度神经网络模型进行学习。我们还尝试了数据增强，样本标签平衡，更换Focal Loss，采用升温cosine调度器等trick，最终让Rank分保持在了1570 (Top 2%) 的成绩。

