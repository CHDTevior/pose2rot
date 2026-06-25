# pose2rot 成功训练讲解
## ——「关节位置 → 关节旋转」模型,从数据到训练完整串讲

> 这是 MoCapAnything V2「视频→位置→旋转」管线的**后半段(Pose→Rotation)**,也是 V2 相对 V1 的核心创新:用一个**学习式**的位置→旋转网络,取代 V1 里那套**解析式逆运动学(IK)**。

---

## 0. 任务定位:这到底在做什么

**输入**:一段动作的**关节 3D 位置序列**(每帧每个关节的 xyz)。
**输出**:每帧每个关节的**旋转**(用 6D 表示),再经前向运动学(FK)还原成完整骨架动画。

**为什么这件事难?**
位置**不能唯一决定**旋转——这是一个 ill-posed(欠定)问题:
- 同一组关节位置,绕骨头轴向的 twist(扭转)是任意的,位置约束不了它;
- 局部坐标系的轴向约定也不唯一。

所以模型不能"硬猜",必须靠**参考姿态(reference) + 骨架结构先验**把这个多值映射约束成可学的条件预测。

---

## 1. 数据:Truebones Zoo 动物动作库

### 1.1 原始数据
目录结构:`datasets/zoo1030/bvh/<物种>#<动作>/y<角度>.bvh`,例如 `Alligator#BigMouth/y0.bvh`。
- **72 个物种**(从鳄鱼、马、大象到蜘蛛、蝎子、龙;关节数 J 跨度 25–143);
- 每个动作绕 y 轴旋转成 **12 个 yaw 视角**各存一个文件(数据增强);
- 实际进训练(有处理好的 pose npz)的规模 = **823 个动作 / 9876 个 clip**。

每个 BVH 含:**骨架层级**(关节名 + 每关节的父节点 parents + parent-local 的骨长 offset)+ **逐帧旋转**(欧拉角→四元数)+ root 平移。

### 1.2 模型实际吃的 batch(关键 shape)
记号:`B`=batch(每卡4,DDP 双卡 global 8)、`T`=48 帧(训练随机截窗)、`J`=150(padding;真实关节数按物种不同)、`N`=32(记忆样本)。

| 字段 | shape | 语义 |
|---|---|---|
| **position** | `[B,48,150,3]` | **输入**:逐帧逐关节 3D 位置(已 root 中心化 + 归一) |
| **rot6d_a** | `[B,48,150,6]` | **监督目标**:GT 局部旋转(6D) |
| joint_t5embed | `[B,150,768]` | 每关节**名字的 T5 文本嵌入**(跨物种关节身份) |
| offset_a | `[B,150,3]` | rest pose 骨长(**减根**版,给模型做条件) |
| fk_offset | `[B,150,3]` | **裸 bvh offset(不减根),专给 FK loss** |
| ref_rot6d_a | `[B,150,6]` | 参考帧旋转(锚点) |
| graph_hop / graph_edge / ancestor_mask | `[B,150,150]` | 骨架图:hop 距离 / 边类型 / 祖先链 |
| joint_mask | `[B,150]` | 标记真实关节(其余是 padding) |
| static_rot_joint_mask | `[B,150]` | 钉死不动的旋转关节 |
| memory_pose / memory_rot6d | `[B,32,150,3]` / `[B,32,150,6]` | per-物种记忆库(**本配方消融,不用**) |
| global_scale | `[B]` | per-物种缩放标量(反归一化/FK 用) |

---

## 2. 预处理:BVH 怎么变成模型输入(7 步)

1. **位置是 FK 算出来的,不是直接读的**。对每帧,沿骨架链逐关节连乘 4×4 齐次变换矩阵(`transforms_global`),取平移分量得全局位置。
2. **6D 旋转 = 局部旋转矩阵的前两行**(Zhou et al. CVPR 2019 的连续旋转表示)。注意是 **parent-local**(相对父节点)的局部旋转,不是全局。
3. **归一化 = per-物种 root 中心化 + 除以单一 `global_scale` 标量**。
   - root 中心化:每帧减去 0 号关节(root)位置,去掉全局平移;
   - `global_scale` = 该物种关节包围盒(bbox)最大边长(范围约 12–826)。
   - **关键:这不是归一到 [-1,1]³ 或统一 1m³ cube**,而是 per-物种单一标量缩放,让不同体型物种量级可比;
   - **自洽性**:`position` 和 `offset` 必须除以**同一个** global_scale 才一致。
4. **reference(参考帧)**:**训练时随机取一帧**(不是第一帧!),测试时取第一帧。取它的位置和旋转当锚点——告诉模型"这具骨架、这个坐标约定下,某个姿态对应某组旋转"。
5. **T5 关节名嵌入**:每个关节名(如 "LeftFrontLeg")离线用 `t5-base`(768 维)编成向量。这是**跨拓扑的关节身份**——不同物种关节数/连接不同,但"这是个左前腿"由文本嵌入统一表达。**这是"任意骨架"的关键设计。**
6. **fk_offset**(2026-06-23 修 FK bug 后的产物):直接读裸 bvh `offsets / global_scale`,**不减根**,专给 FK loss 跑前向运动学,使 `FK(GT旋转) ≈ position`。
7. **图结构 + static mask + 记忆 + padding 到 150**:骨架树的 hop 距离/边类型/祖先链;经验静止的关节标记(geodesic < 2° 的关节);每物种用 FPS(最远点采样)挑 32 个最分散姿态当记忆池;所有 J 维 pad 到 150。

---

## 3. 模型设计:`Pose2RotMemoryRestModel`(约 29.7M 参数)

### 3.1 核心思想:一个模型处理任意物种的任意骨架
靠三样东西:
- **T5 关节名嵌入** → 跨物种认出同名关节;
- **图注意力** → 按骨架树结构传播信息;
- **rest-FiLM** → 用静止姿态(rest pose)调制特征,适配具体骨架。

### 3.2 四个子模块

| 模块 | 角色 | 输入 → 输出 | 参数 |
|---|---|---|---|
| **RestPoseEncoder** | 编码"骨架是谁" | offset + 关节名 → `rest_embed[B,J,256]` | 3.4M |
| **PoseEncoderLite**(主干) | 编码"动作长啥样" | position 序列 → `q_feat[B,T,J,256]` | 5.0M |
| **MemoryEncoder** | 编码记忆库(**本配方消融,死路径**) | 记忆 → `mem_feat` | 3.9M |
| **RotDecoder**(最大,58%) | 解码到旋转 | q_feat → `pred_rot6d[B,T,J,6]` | 17.4M |

### 3.3 四个关键设计

**(a) 图注意力(GraphMultiHeadAttention)** —— 把骨架图压进 attention
- 标准全 J×J 点积注意力 **加上两类骨架先验**:
  1. 按**关节对的最短路 hop 数 + 边类型**(self/parent/child/sibling/…)学的 **attention bias**;
  2. 偶数层的**祖先掩码**:关节只能 attend 自己 + 祖先链(符合自顶向下的运动学依赖),奇数层放开全图。
- 这就是 "Global-Local":Global=任意关节对可交互,Local=骨架拓扑先验注入。

**(b) 时序注意力** —— per-joint、RoPE、窗口化
- 把 `[B,T,J,D]` 重排成 `[B*J,T,D]`,**每个关节独立沿时间**做 self-attention(关节间交互交给图块,解耦);
- 用 **RoPE** 旋转位置编码 + **窗口化(±2 帧)**:只建模动作的短时连续性。

**(c) rest-FiLM 条件调制**
- 每层用 `rest_embed` 算出 (scale, shift) 调制动作特征:`x*(1 + 0.1*scale) + shift`;
- 同一段动作放到不同骨架,rest_embed 不同 → FiLM 给不同调制(`0.1*scale` 限幅是稳定性设计)。

**(d) static 关节覆盖**
- `pred = pred·(~static_mask) + ref_rot6d·static_mask`;
- 钉死不动的关节,预测直接用参考旋转顶替,不让网络在它们上浪费容量、引入噪声梯度。

### 3.4 为什么用 6D 旋转表示
- 欧拉角有万向锁、四元数有双重覆盖+不连续,对神经网络回归不友好;
- **6D 是连续无奇异**的旋转表示(Zhou et al. 2019);
- 重建用 **Gram-Schmidt 正交化**:取前 3 维 a1、后 3 维 a2,正交化得 b1,b2,b3,`R = stack([b1,b2,b3], dim=-2)`(**行约定**)。
- ⚠️ **行约定是项目踩过的坑**(见第 6 节):用错成列约定会重建出转置矩阵。

### 3.5 模型规模
总 **29.7M** 参数,主算力在 10 层的 **RotDecoder(17.4M,58%)**。共享超参:`q_dim=256`、`num_heads=8`、`joint_embed_dim=768`(T5)、`temporal_window=2`、所有 attention 块是 pre-norm Transformer、FFN 4× 扩展。

---

## 4. Tensor information flow(位置→旋转怎么流)

```
position[B,48,150,3] + 关节名T5[B,150,768] + offset[B,150,3]
   │                          │                      │
   │ in_proj(3→256)    joint_t5proj(768→256)   RestPoseEncoder
   ▼                          ▼                  (图注意力)
 pose_feat ──(+)── joint_sem                        │
   │                                                ▼
   ▼                                          rest_embed[B,150,256]
 ┌────────────────────────────────┐◄──rest-FiLM────┤
 │ PoseEncoderLite ×4层:           │                │
 │  rest-FiLM → 时序块 → 图块       │◄──骨架图────────┤
 └────────────────────────────────┘                │
   │ q_feat[B,48,150,256]                           │
   ▼                                                │
 ┌────────────────────────────────┐◄──rest-FiLM─────┤
 │ RotDecoder ×10层:               │◄──骨架图────────┘
 │  FiLM→时序→图→(cross-attn消融)   │
 │  head: LayerNorm→Linear→Linear(256→6)
 └────────────────────────────────┘
   │ pred_rot6d[B,48,150,6]
   ▼
 static关节覆盖: pred = pred·(~static) + ref·static
   ▼
 pred_rot6d[B,48,150,6]  →  6 项 loss(对比 GT rot6d_a)
```

一句话流:**位置 + 关节名身份 → 拼起来 → 时序/图/FiLM 反复精炼 → 解码到旋转 → static 关节用参考值覆盖**。

---

## 5. 训练:损失 + 抗塌缩 + fk 渐增 + 配方

### 5.1 六项损失(都只在"会动的非-static 关节"上算)
```
total = 1·rot + 1·vel + 1·acc + 0.1·root + 2·tvar + fk_wt·fk
```
- **rot**(SmoothL1):逐帧 6D 旋转主监督;
- **vel / acc**(SmoothL1):旋转的一阶 / 二阶时间差分(运动速度 / 加速度平滑);
- **root**(0.1):根关节单独再监督(根朝向决定整体姿态);
- **fk**:`FK(pred_rot6d, fk_offset)` 还原出关节位置,对比 GT 位置(SmoothL1)——把旋转误差换算到**末端位置**,压住长肢累积漂移;
- **tvar**(2.0,**抗塌缩核心**):去均值时序 L1——减掉每个 clip 的时间均值后,**只比时变部分**。

### 5.2 两大难点 + 解法(讲给老师最硬的部分)

#### 难点① 后验塌缩(posterior collapse)
**现象**:模型偷懒,不管输入什么都输出该物种的一个**几乎恒定的姿态**(时间维上几乎不变),loss 上"安全"但完全没用,视觉上表现为冻结 / 塌成一团。普通 rot 项压不住它(一个"输出时间均值"的冻结预测在 L1 上已能拿到不错数字——**metric 骗人**)。

**双管齐下解决:**
- **(a) memory 消融**(`decoder_use_cross_layers=0`):记忆库里存的是物种级近似常量姿态,decoder 一旦能 cross-attn 读它,就会走捷径直接"抄"出来当输出(species-constant leakage)。砍掉这条捷径,逼模型从输入位置真正推理逐帧旋转。
- **(b) tvar 去均值项**:数学核心是去均值算子的雅可比 **`I - 1/T ≠ 0`** —— **即使预测是冻结的,这个梯度也非零**,主动把模型推向"产生时变运动";而且不能靠加随机噪声蒙混(必须复现 GT 真实的时序模式)。

**监控指标 `ratio_DYN`** = 预测时变能量 / GT 时变能量;> 0.3 = 破塌缩。最终 v10 ratio_DYN ≈ 0.78,确实在产生时变运动。**配套必看 GT-vs-pred 多帧动画做视觉 QA,数值不能单独定案。**

#### 难点② fk 权重的 regime 矛盾 → fk 渐增 ramp
- 够纠偏的 fk(权重 ~30)**从头就上会压垮抗塌缩**:模型会用一组**恒定旋转**去凑位置,把塌缩固化下来;
- fk=30 **恒定续训会自发发散**:诊断根因——fk 的梯度 ≈ grad_clip 阈值(0.99 ≈ 1.0)= **亚稳态**,被某个 hard batch 一掀就塌进坏吸引子(不是权重/Adam 爆炸);
- **解法 = fk 从 0 线性升到 10**:前 5 epoch fk=0 让 tvar 先破塌缩 → epoch 5–15 线性升到 10 纠偏 → 之后恒定 10。fk=10 的梯度 ~0.33 ≪ grad_clip,不再发散。

### 5.3 DDP 训练配方
- **DDP 双卡,global batch 8**(每卡 4);
- **lr 2e-4** —— 抗塌缩配方对 LR 脆弱(linear-scale 到 8e-4 会破塌缩,2e-4 保住);
- **500 步线性 LR warmup**(Goyal 2017,稳定 Adam 启动);
- **bf16 autocast**(fp16 曾发散到 NaN 权重,改 bf16 修复);
- **grad_clip 1.0**;固定 **60 epoch**(对齐 MoCapAnything 协议);
- 优化器:配置 `weight_decay=0`,所以**实际是 Adam 而非 AdamW**(代码细节)。

---

## 6. 我们怎么一步步训成功的(踩坑 → 修复)

这条线**不是一把训成的**,是一串 bug 修出来的——最能体现工程严谨:

1. **提速**:单卡 batch4(2×)→ DDP 双卡 global8(3×)。发现抗塌缩配方**对 LR 脆弱**(lr8e-4 破、lr2e-4 保);bf16 修了 fp16 的 NaN 发散。
2. **破塌缩**:memory 消融 + tvar 双管,配 ratio_DYN 监控。
3. **★FK 约定 bug(关键)**:加 FK loss 时发现训练 FK 用了**转置矩阵(列约定)+ 错的 offset(减了根)** → `FK(GT) ≠ position` → 开 FK loss 反而把 MPJPE 搞炸 3–11 倍。**训练 loss 还在降,是靠视觉 / MPJPE 才抓出来 = metric 骗人**。修正成**行约定 + 裸 offset**,**硬验 `FK(GT) ≈ position` 误差 ≈ 0** 才敢用,经 codex 审。
   - **教训:加任何 FK-based loss 前,必须先验 `FK(GT) ≈ target`。**
4. **★fk 渐增 ramp**:解决"强 fk 压塌缩 / 恒定 fk 发散"(发散根因诊断到梯度 ≈ grad_clip 亚稳)。
5. **对齐 MoCapAnything 协议**:读 V1 / V2 论文——他们固定 60ep 训完报最终模型、无验证集、无 early-stop、无 checkpoint 选择、单次运行、超参在 test 集上调。改成固定 60ep + geodesic 度数 metric + seen/rare/unseen 测试划分。
6. **★缓存泄漏 bug(关键)**:建好 held-out 划分后,发现**训练缓存文件名没含划分标识** → 新旧划分撞同一缓存 → 测试动作泄漏进训练。靠**核验缓存条目数**抓出来,修成缓存名带 split 标识,codex 独立验证 **0 泄漏**。

### 最终结果
- **见过物种时** geodesic **6.53° ≈ MoCapAnything 的 6.54°**(同水平);
- **真 held-out 跨拓扑泛化是天花板**:seen 留出 clip 9.8°,unseen(完全没见过的物种)双峰——有近亲的(Goat 17° / Coyote 19° / Cat 25°)泛化一般,拓扑独特的(Pigeon 67° / Spider 73°)完全迁移不了;
- 逐物种对比证明:**unseen 失败 100% 是泛化代价,不是物种本身难**(模型见过这些物种时全 5–8°)。

---

## 7. 几个"诚实点"(讲课加分,显得真懂)

1. **位置是 FK 算的、6D 是矩阵前两行、归一化是 per-物种 scale 不是 unit-cube**;
2. **FK loss 必须先硬验 `FK(GT) ≈ target`**,否则 metric 会骗你;
3. **抗塌缩 = 堵捷径(memory 消融)+ 给梯度(tvar 的 `I−1/T` 非零)**,双管互补;
4. **配置里 weight_decay=0,所以优化器实际是 Adam 而非 AdamW**;
5. **MemoryEncoder 当前是死路径**(算了 mem_feat 但 decoder 不读),讲"4 个子模块"要点明;
6. **核心方法论:CV 任务可视化 / 视觉验证 > metric** —— FK bug 和缓存泄漏都是靠核验抓出来的,光看 loss 发现不了。

---

*关键代码:模型 `MocapAnything/models/v2/pose2rot/model.py`;损失 `utils/loss.py`;训练 `train/pose2rot.py`;修正 FK `utils/rotation.py:rot6d_to_fk_positions_correct`;数据 `data/loader_v2.py`。配置 `configs/train/train_pose2rot_v10_split_heldout.yaml`。*
