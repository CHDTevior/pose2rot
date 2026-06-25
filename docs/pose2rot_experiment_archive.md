# pose2rot 实验归档 (2026-06-24)

**STATE**: pose2rot (MoCapAnything V2 的 Pose→Rotation 组件,位置→6D旋转) 实验**完成并归档**。这条线吃透了 V2 的核心创新(学习式旋转恢复,取代 V1 的解析 IK),训出了可工作的模型,并得到一个诚实可辩护的跨拓扑泛化结论。下一步:用户定下一项(候选:复现 V2 完整端到端 video2pose2rot 联合管线)。

---

## 一、最终交付物

- **v10 决定性 held-out 模型**(论文用): `checkpoints/pose2rot/_BACKUP_v10_heldout_FINAL/pose2rot_ckpt_epoch60.pt`(md5 0c648a52)。从头60ep + fk渐增0→10,在 seen/rare/unseen held-out 划分上训(训练集8868,排除84测试动作,codex验证0泄漏)。
- **v9 全数据/oracle 模型**(demo/上限参考): `exp_pose2rot_v9_fk10ramp_60ep` epoch60;另有更早的两段式最佳 `_BACKUP_v8b_converged_BEST/epoch40`。
- **最终 held-out geodesic(对标 MoCapAnything V2 unseen 6.54° / V1 ~17°)**: seen **9.8°** / rare **12.7°** / unseen **40.9°** / ALL 28.0°(Ang all-joints,度)。

## 二、核心发现(论文叙事)

v9-oracle(模型见过所有物种)整体 geodesic **6.53° ≈ MoCapAnything 6.54°** —— 即**我们的 pose2rot 只要见过该物种,就和他们同水平(SOTA)**。但 v10 真 held-out 的跨拓扑泛化是天花板:seen 留出 clip 9.8°、unseen(完全没见过的物种)双峰——有近亲的(Goat 17°/Coyote 19°/Cat 25°)泛化一般,拓扑独特的(Pigeon 67°/Spider 73°)完全迁移不了。v9-oracle vs v10 逐物种对比证明:**unseen 失败 100% 是泛化代价不是物种难**(v9 见过这些物种时全 5-8°)。MoCapAnything 的 6.54° "unseen" 他们自己承认是"走跑为主、reference 一锚定就简单"的易 split,接近我们 v9-oracle 而非 v10 真 held-out。可辩护命题:**见过=SOTA;我们做了比他们更硬更诚实的 held-out,暴露了这类方法跨拓扑泛化的真实天花板(开放难题)**。

## 三、技术历程(踩坑→修复,工程严谨证据链)

1. **提速**: 单卡 batch4(2×)→ DDP 双卡 global8(3×)。发现抗塌缩配方**对 LR 脆弱**(lr8e-4 破塌缩,lr2e-4 保住);bf16(fp16 曾发散 NaN)。
2. **后验塌缩**(模型输出物种常量姿态、不跟运动): 双管解决 —— (a) **memory 消融** `decoder_use_cross_layers=0` 砍掉"从记忆库抄物种常量旋转"的捷径; (b) **tvar 去均值时序项**(雅可比 `I-1/T≠0`,冻结预测也有非零梯度,主动推向产生时变运动)。`ratio_DYN`(预测时变能量/GT)>0.3 = 破塌缩,监控用。
3. **★FK 约定 bug**(关键): 加 FK-position loss 压漂移时,发现训练 FK 用了**转置矩阵(列约定 stack dim=-1)+ 错 offset(减了root的 offset_a)** → `FK(GT)≠position` → 开 FK loss 反而把 MPJPE 搞炸 3-11×。**训练 loss 还在降,是靠视觉/MPJPE 才抓出来 = metric 骗人**。修正成**行约定 `rotation_6d_to_matrix`(dim=-2)+ 裸 bvh offset(fk_offset,不减root)**,**硬验 `FK(GT)≈position` err≈0** 才用,codex PASS。教训:加任何 FK-based loss 前必先验 FK(GT)≈target。
4. **★fk 权重 regime 矛盾 → fk 渐增 ramp**: 够纠偏的 fk(~30)从头就压垮抗塌缩;fk=30 恒定续训会**自发发散**(诊断根因: fk 梯度 0.99≈grad_clip 1.0 = 亚稳态,被某 batch 一掀就塌进坏吸引子;非权重/Adam爆炸)。解法 = **fk 从 0 线性升到 10**(epoch5-15),前期 fk=0 让 tvar 先破塌缩,后期渐增纠偏;fk=10 梯度 ~0.33 ≪ clip 不发散。诊断脚本 `zoo1030_build/scripts/diagnose_divergence{,_grad}.py`。
5. **对齐 MoCapAnything 协议**: 读 V1(arXiv 2512.10881)/V2(2604.28130)论文 —— 他们固定60ep训完报最终模型、**无val/无early-stop/无checkpoint选择**、单次运行、超参在test集上调(select-on-test)。改成固定60ep + geodesic度数metric + seen/rare/unseen划分。
6. **★缓存泄漏 bug**(关键): 建好 held-out 划分后,训练 items 缓存文件名 `__mesh2pose1002_train_None_...` **不含 split 标识** → v9(空划分9876)和 v10(新划分8868)撞同一缓存 → v10 加载了旧的全量缓存、测试动作泄漏进训练。靠**核验缓存条目数**抓出(9876含Cat/Coyote等测试物种)。修成缓存名加 `split_tag`(loader_v2.py),codex 独立扫描验证 **8868训练/0泄漏**。

## 四、方法/数据/模型速查(细节见串讲稿)

- **数据**: Truebones Zoo, 72物种(Dog因无scale被丢), 823动作×12yaw = 9876 clips。位置=FK算(链乘齐次矩阵), 6D旋转=局部矩阵前两行(Zhou 2019), 归一=per物种 root中心化+除 global_scale(关节bbox边长,非unit-cube)。reference=训练随机帧。T5关节名嵌入(768d)=跨拓扑关节身份。
- **模型** `Pose2RotMemoryRestModel` ~29.7M: 4子模块(RestPoseEncoder/PoseEncoderLite主干/MemoryEncoder消融死路径/RotDecoder 10层58%)。关键: T5关节名 + 图注意力(hop/edge bias + 祖先mask)+ rest-FiLM + per-joint RoPE窗口化时序 + static关节覆盖。
- **训练**: 6项loss(rot/vel/acc/root0.1/tvar2.0抗塌缩/fk-ramp), DDP global8, lr2e-4, warmup500, bf16, grad_clip1.0, 60ep。compute_rot_loss 唯一损失函数。

## 五、关键文件

- 模型 `MocapAnything/models/v2/pose2rot/model.py`; loss `utils/loss.py`; 训练 `train/pose2rot.py`; FK修正 `utils/rotation.py:rot6d_to_fk_positions_correct`; 数据 `data/loader_v2.py`(含 split_tag 缓存修复 + fk_offset)。
- 配置 `configs/train/train_pose2rot_v10_split_heldout.yaml`(决定性)、`train_pose2rot_v9_fk10ramp.yaml`(全数据)。
- 划分 `datasets/zoo1030/test_split_seen_rare_unseen.json`(84测试动作 seen24/rare12/unseen48)。
- 脚本 `zoo1030_build/scripts/`: geodesic_eval.py(三档度数metric)/check_collapse.py(ratio_DYN)/pose2rot_qa.py(gif QA)/diagnose_divergence*.py/build_split.py/enum_dataset.py。
- gif: `artifacts/20260619_014725_MocapAnything/pose2rot_qa_v10_heldout_FINAL/`(v10 held-out)、`_v9_ep60_alldata/`(v9)。
