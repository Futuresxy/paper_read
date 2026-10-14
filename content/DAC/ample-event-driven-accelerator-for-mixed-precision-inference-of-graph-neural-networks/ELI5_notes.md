# AMPLE: Event-Driven Accelerator for Mixed-Precision Inference of Graph Neural Networks 通俗讲解

### 0. 整体创新点通俗解读

**痛点直击**

GNN 加速这件事，硬件早就有人在做了（HyGCN、GenGNN），但它们集体卡在三个“顾头不顾尾”的地方：

- **Batch 式调度撞上歪瓜裂枣的图**：图里节点度数分布极不均匀——Reddit 平均度数 99.6，Yelp 只有 19.5，同一张图里更是既有 1 度的闲云野鹤，也有上千度的社交达人。传统 accelerator 用 **double buffering**（凑一批节点一起算、算完换下一批），结果是低度数节点算完了还得干等高度数节点，流水线上全是气泡。
- **量化只能“一刀切”**：CPU/GPU 对低精度格式支持差，之前的 FPGA 加速器又只支持单一精度。但 Degree-Quant 论文早就指出：GNN 的量化误差主要来自 **Aggregation 阶段**，且和节点度数强相关——高度数节点容易量化崩，低度数节点完全扛得住 int8。这个先验知识在硬件层面一直没人用上。
- **大图根本塞不进片上**：现有加速器要求整个图的 embedding 缓存在 on-chip memory 里，超过 10 万节点的社交图谱直接出局。

---

**通俗比方**

把 AMPLE 想成一个**急诊室**，而不是流水线工厂。

- **传统 accelerator（double buffering）是流水线工厂**：一批 100 个病人凑齐了才一起进车间，等最慢的那个手术做完，下一批才能上。哪怕 99 个只是包扎创可贴，也得陪着重症病人耗着。
- **AMPLE 是急诊室**：病人（节点）到了就分诊（program 进 nodeslot），轻症分给 int8 护士站，重症分给 float 主任医生，**谁做完谁滚蛋，床位立刻让给下一个**——这就是 **event-driven**。
- 更妙的是，急诊室还配了**专门的护工团队**：不等病人进屋，就提前去仓库（HBM）把他的病历和邻居资料搬过来，搬了一部分就先让医生开刀，剩下的边做边搬。床位因此不需要囤着整套资料，一晚上能接 70 万个病人。

![](images/84463d803a53ab0b6a76fd7aad3315d03156097b84731932812d5a48957f0e94.jpg) *Fig. 1. AMPLE Top Level Diagram. Packets propagate through dimensionorder routing in the Aggregation Engine’s Network-on-Chip (shown in green), and are driven diagonally into the Transformation Engine’s systolic array (shown in red). Dashed lines represent control flow interfaces, while solid lines represent data flow between units. Node embeddings are fetched through HBM, while instructions are stored in DRAM.*

---

**关键一招**

作者没有去改 GNN 模型本身，也没有去堆更大的片上缓存，而是**把“调度的粒度”从 batch 砍到了单个 node**，然后围绕这个决定重新设计了整条数据通路：

- **第一步：把调度控制权交给 host 的“登记表”**。硬件侧只有一个 **Node Instruction Decoder（NID）**——本质是一组 memory-mapped 寄存器（nodeslot），每个 slot 登记一个节点的邻接表指针、精度、状态机。Host 异步地往空 slot 里塞节点，硬件算完一个就独立拉低 mask 位宣告“床位空了”。没有全局同步，没有 batch barrier，调度自然消除了流水线气泡。
- **第二步：把“量化粒度”从整张网砍到单个节点**。Aggregation Engine 做成 **NoC 上的异构 PE 阵列**——float 核和 int8 核物理隔离成子网，节点按 DegreeQuant 的保护概率被路由到对应的子网。由于保护节点通常不到 3%，分配**一个 float nodeslot** 就够，剩下的资源全铺 int8 核，节点并行度直接拉满。
- **第三步：把“等数据”改成“边算边搬”**。Prefetcher 里的 **Fetch Tag** 有个 partial response 机制：高度数节点的邻居搬不完？先搬够一队列就放行 AGE 开算，剩下的在计算期间继续偷偷搬。片上不需要缓存整个图，大图推理才成为可能。

![](images/29cb96496a01f841cfda9f1ac6a7524f7044211e1b73f53fea4e29de9d5d2556.jpg) *Fig. 2. Microarchitecture of AGE configured with three supported precisions. NID requests drive the Aggregation Managers (AGMs), which receive fetched embeddings from the Feature Bank (See Figure 1). These are then transferred to the Aggregation Cores (AGCs) through the network. Aggregation results are then bufered by the Bufering Managers (BMs).*

---

**成果一句话**

| 对比基线 | 平均加速比 |
|---|---|
| CPU (Intel Xeon) | **243×** |
| GPU (RTX A6000) | **7.2×** |

注意 PubMed 这种数据集上 GPU 加速比骤降到 3×——这正是 GPU 静态并行在大图 + 长尾度数分布下崩掉的地方，而 AMPLE 的急诊室模式把吞吐稳定在 **1 万+ nodes/ms** 量级，六个数据集全场景碾压。核心洞察就一句：**图的不规则性不是敌人，而是调度的信号——顺着节点的度数和精度去动态分资源，不规则性反而变成了免费的并行度。**

### 1. 基于节点指令解码器（NID）的事件驱动编程模型

**痛点直击**

- 传统加速器（如 HyGCN）用的是**双缓冲**机制：一批节点进 pipeline，必须等这一批里**最慢的那个节点**算完，整个批次才能往前走。
- 问题在于，图数据的**节点度分布极度不均匀**——有的节点只有 1 个邻居，有的有几千个。低度节点算聚合可能只要几拍，高度节点要磨蹭几百拍。
- 结果就是：**低度节点干等高度节点**，pipeline 里出现大量气泡，硬件大部分时间在空转。
- 更糟的是，这种“批处理”调度模式下，硬件资源（计算核、buffer）按批次整体分配和释放，粒度太粗，完全没利用好“图里大部分节点其实很快就能算完”这个事实。

**通俗比方**

- 把 GNN 推理想象成一家**快递分拣站**，每个节点是一件包裹（里面的邻居数 = 包裹里的物品数）。
- 双缓冲模式就像：**一整车包裹必须全部卸完才能发下一辆车**。你车上有一件超大的冰箱（高度节点），旁边几十封轻飘飘的信（低度节点）早就分拣好了，却只能陪着冰箱一起等下一趟。
- AMPLE 的做法是改成**流水线式单件分拣**：每封信一到就立刻处理、立刻发出，分拣台一空就接下一件，冰箱再慢也不堵别人的路。
- 这就是从“批同步”到“**事件驱动**”的思维转变——不是等一个全局时钟拍子，而是“谁好了谁走”。

**关键一招**

- 作者没有去优化缓冲策略本身，而是**把调度的所有权从“批次”下沉到了“单个节点”**——引入了一个叫 **Node Instruction Decoder (NID)** 的控制单元。
- 具体的逻辑转换在于：

- NID 本质上是一组**内存映射寄存器**，里面分成若干个 **nodeslot**——每个 slot 就是单个节点的“工位牌”，记录着它的 Node ID、精度、状态机、邻接表指针、输出指针（参考论文 Table 2）。
- Host 不再一次性打包一批节点扔给加速器，而是**异步地**往空闲的 nodeslot 里“塞”节点，塞完就干别的去。
- 加速器内部，NID 通过一个 **available_nodeslots 掩码** 独立维护每个 slot 的生命周期：某个节点算完了，NID 自己把对应的 mask 位**拉低**（deassert），发个中断告诉 Host：“这个工位空了，来新人”。
- Host 一看到有坑位，立刻补一个新节点进去。**Host 和加速器是并发跑的**，形成了一个类似“生产者-消费者”的流水补给循环。
- 注意一个反直觉的细节：节点塞进 nodeslot 的**顺序不代表执行优先级或时间关系**——slot 之间完全独立，谁先算完谁先腾地方，这正好吃掉了节点度不均匀带来的调度死锁。

![](images/84463d803a53ab0b6a76fd7aad3315d03156097b84731932812d5a48957f0e94.jpg) *Fig. 1. AMPLE Top Level Diagram. Packets propagate through dimensionorder routing in the Aggregation Engine’s Network-on-Chip (shown in green), and are driven diagonally into the Transformation Engine’s systolic array (shown in red). Dashed lines represent control flow interfaces, while solid lines represent data flow between units. Node embeddings are fetched through HBM, while instructions are stored in DRAM.*

- 上图是 AMPLE 的顶层结构，NID 是全局的“总调度台”：它按 Table 3 的顺序驱动 Prefetcher（取邻居特征）→ AGE（聚合）→ ABF（缓存聚合结果）→ FTE（变换写回 DRAM），每一步完成后自动推进节点的状态机，**全程不需要 Host 再插手**。
- 多个并发节点抢共享资源时，NID 用**轮询仲裁**兜底，保证公平性。

**一句话总结**：作者把“按批同步”换成了“按节点独立记账”——每个 nodeslot 是一个自带状态机的小任务，硬件资源在节点完成的瞬间即可回收复用，彻底消灭了“等最慢邻居”的 pipeline 气泡。

### 2. NoC互联的异构混合精度聚合引擎与节点级动态资源分配

**痛点直击 (The "Why")**

传统 GNN 加速器（如 HyGCN）的架构逻辑是“**流水线双缓冲（Double Buffering）**”——一批节点全部聚合完，才进入下一阶段。这个设计在图数据上会“死得很惨”，根源在于两个被忽视的现实：

- **节点度的分布极度不均匀**：在 Reddit 这种社交图里，可能 90% 的节点只有几个邻居，但存在个别“网红节点”挂着成百上千条边。双缓冲机制下，几十个“轻量级”节点算完了，却必须**干等**那个最重的节点，因为硬件资源（PE）是按批次静态绑定的，不能中途释放。
- **精度需求也是不均匀的**：按 DegreeQuant 的结论，聚合误差集中在**高节点**，而占 97% 以上的低节点用 int8 就够了。但传统架构只有一套同质算术单元——要么全用 float（浪费面积），要么全用 int8（精度崩掉），**没有硬件级的“因材施教”能力**。

一句话总结：老架构是“**批发模式**”——整批进、整批出、统一规格；而图数据天生是“**零售模式**”——每个节点的活儿量、精度要求都不一样。

---

**通俗比方 (The Analogy)**

把聚合引擎想象成一家**机场的行李处理系统**：

- 每个节点 = 一件行李；行李的大小（节点度）差异巨大——有小背包，也有超规的滑雪板包。
- **双缓冲的旧机场**：只有一条传送带，规定“这一批行李必须**全部**送到才能开下一批”。结果一个小背包装完后，卡在等那个滑雪板包，传送带大量空转（pipeline gaps）。
- **AMPLE 的新机场**：改成**分拣中心 + 无轨道穿梭车**模式——
  - 不同承载能力的穿梭车（**异构精度的 Aggregation Core**：float 车装载大、成本高、数量少；int8 车便宜、数量多）各自待命；
  - 每件行李落地时（节点被编程进 nodeslot），调度系统**实时**派一辆对应规格的车去拉它，拉完立刻回来接下一单；
  - 穿梭车之间不需要互相串门（float 数据和 int8 数据互不来往），所以干脆**分成独立的两片车道网络**，天然避免堵车（packet congestion）。

这个“车道网络”就是 **Network-on-Chip (NoC)**——它不是花哨的装饰，而是实现“任意节点 → 任意空闲算术单元”动态路由的物理前提。

![](images/29cb96496a01f841cfda9f1ac6a7524f7044211e1b73f53fea4e29de9d5d2556.jpg) *Fig. 2. Microarchitecture of AGE configured with three supported precisions. NID requests drive the Aggregation Managers (AGMs), which receive fetched embeddings from the Feature Bank (See Figure 1). These are then transferred to the Aggregation Cores (AGCs) through the network. Aggregation results are then bufered by the Bufering Managers (BMs).*

上图可以清晰看到这个结构：NID 发出请求驱动 **Aggregation Manager (AGM)**，特征数据从 Feature Bank 流入，经过 NoC 路由分发到不同精度的 **Aggregation Core (AGC)**，算完由 **Buffering Manager (BM)** 收尾。三种精度是**三个隔离的子网**，互不干扰。

---

**关键一招 (The "How")**

作者的巧妙之处不在于发明新算术单元，而在于把“**资源分配的决策权**”从编译期搬到了运行期，具体做了三步逻辑转换：

- **替换①：从“批次同步”到“节点事件”**——作者并没有改造 PE 本身，而是把硬件调度的最小粒度从“batch”压缩到“单节点”。每个 nodeslot 维护一个独立状态机（Prefetch → Aggregation → Transformation），谁先算完谁先释放资源，**不等任何人**。这就是事件驱动（event-driven）的本质。
- **替换②：从“同构阵列”到“按需配比的异构池”**——用 Equation 6 的逻辑，先根据 DegreeQuant 统计出的保护节点比例（实测均低于 3%），反推出 float 核与 int8 核各自的资源预算，然后**编译期一次性烧死这个配比**（如绝大多数 nodeslot 给 int8，一两个给 float）。节点运行时根据自己的精度标签，被路由到对应子网。
- **替换③：用 NoC 替代固定连线**——这是让前两个替换成立的物理基础。传统固定连线的 PE 阵列里，单元和节点是焊死的；而 NoC 的 packet 路由让“哪个 PE 服务哪个节点”变成**运行时可变**的，天然支持动态分配。

![](images/84463d803a53ab0b6a76fd7aad3315d03156097b84731932812d5a48957f0e94.jpg) *Fig. 1. AMPLE Top Level Diagram. Packets propagate through dimensionorder routing in the Aggregation Engine’s Network-on-Chip (shown in green), and are driven diagonally into the Transformation Engine’s systolic array (shown in red). Dashed lines represent control flow interfaces, while solid lines represent data flow between units. Node embeddings are fetched through HBM, while instructions are stored in DRAM.*

从顶层架构图可以看到整体数据流：聚合阶段的 packet 在 NoC 中按**维度序路由（dimension-order routing）**传播（绿色通路），聚合结果再对角馈入 Transformation Engine 的脉动阵列（红色通路）——聚合侧“乱序、动态”，变换侧“规整、流水”，两段各取所长。

**效果验证**——这套设计的收益直接体现在吞吐上：

| 平台 | 平均吞吐 (GCN, nodes/ms) | 相对 CPU 加速比 |
|---|---|---|
| Intel Xeon CPU | 242.2 | 1× |
| RTX A6000 GPU | 3,129.3 | ~13× |
| **AMPLE @200MHz** | **11,469.9** | **361.1×** |

![](images/74a50c8065173764b1c67d073eb6ac2c7754ddfde91c75b8ac03bbb394185a9d.jpg)

---

**一句话收束**

这个设计的哲学是：**图数据的不均匀性不是要被“抹平”的噪声，而是要被“利用”的信号**——节点度不均匀，所以资源要动态分配；精度需求不均匀，所以算力要异构配比。NoC 只是把这两个“不均匀”翻译成硬件语言的那座桥。

### 3. 节点中心的两级预取与部分响应机制（支持大规模图）

**痛点直击**

先想清楚一个矛盾：**图的邻居数据在内存里是“散装”的**。

- 一个节点的邻居 ID 列表存在一个地址，但每个邻居的 Feature Embedding 又散落在完全不同的内存位置。
- 传统做法（如 HyGCN）要求：把一个节点的**全部邻居 Embedding 都取回来、塞进片上 Buffer**，才能开始聚合。
- 这在**小图**上没问题，但论文要跑 Reddit（23 万节点）、Yelp（71 万节点）这种规模——片上 BRAM 根本装不下“在飞行中”的所有消息。
- 更难受的是**长尾效应**：图里总有少数“超级节点”（degree 高达几百上千），它一个人的邻居数据就能把 Buffer 撑爆。所有其他节点都得等它。这就是典型的**“一颗老鼠屎坏一锅粥”**——最慢的那个大度数节点，决定了整个流水线的下限。

---

**通俗比方**

把 AMPLE 想象成一个**火锅店的后厨**：

- 每个节点是一桌客人，聚合运算就是“涮这桌的菜”。
- 邻居 Embedding 是**散落在各个冷库里的食材**——不是打包好的，得一件一件去取。
- 传统后厨的规矩是：“一桌的菜全部到齐了才开涮”。碰到一桌点了 300 道菜的VIP，后面十桌小客户全在干等，灶台（计算单元）闲置。
- AMPLE 的做法是：**“菜够下第一锅了就先开涮，剩下的菜边涮边送”**。厨师（AGE）和送菜员（Fetch Tag）两条线并行，客人完全感觉不到等菜时间。

---

**关键一招**

作者没有去加大 Buffer（那会烧掉 BRAM、减少 nodeslot 数量），而是把原来**“取全 → 算”的刚性串行**，扭转成了**“取一部分 → 先算 → 边算边续杯”的弹性流水**：

- **两级预取**：Fetch Tag 先做第一级——把邻居的 **ID 列表**取回来放进 Address Queue；再用这些 ID 当指针，做第二级——去 HBM 把真正的邻居 Embedding 拉回来，塞进 Message Queue。
- **部分响应**：当某个节点的邻居数超过 Fetch Tag 的容量时，不再傻等凑齐，而是**Message Queue 一满就立刻“放行”**，直接解锁 AGE 开干。
- **延迟隐藏**：AGE 聚合手头这批邻居的同时，那个 Fetch Tag **被重新激活**，继续去 HBM 捞剩下的邻居。等 AGE 算完这一批，下一批已经躺在队列里了——**访存延迟被计算完全盖住**。

![](images/193fe8a78d412d11fc810e09ae7a67038e5111ed0d0aaeed058a8fce4f64a319.jpg)

这个机制的精妙之处在于一个**正反馈**：

- 因为每个 nodeslot 不需要缓存“全部”邻居，只需要的 **Message Queue 深度**变小 →
- 同样的片上存储，能塞下**更多 Fetch Tags / nodeslots** →
- 节点级并行度（node parallelism）反而**随图规模增大而提升**。

看实验数据就很明显——大图上 AMPLE 的吞吐不降反升：

| 数据集 | 节点数 | AMPLE 吞吐 |
|---|---|---|
| Cora | 2,708 | 11,022 |
| Flickr | 89,250 | 12,350 |
| Yelp | 716,847 | **12,471** |

---

**一句话总结**

传统架构是“**数据等计算**”——把 Buffer 塞满才敢动；AMPLE 是“**计算追数据**”——有多少算多少，剩下的在路上。代价几乎为零（只是加了一个状态机切换），却把“大规模图必须片上全缓存”这条铁律直接拆掉了。
