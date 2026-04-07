# 语义引导的建筑外立面点云补全框架 — 技术方案

---

## 一、推荐模型清单

### 1. AdaPoinTr（首选基线）

- **出处**：Yu et al., "AdaPoinTr: Diverse Point Cloud Completion with Adaptive Geometry-Aware Transformers", T-PAMI 2023（扩展自PoinTr, ICCV 2021 Oral）
- **代码**：https://github.com/yuxumin/PoinTr
- **核心机制**：将点云补全重构为set-to-set translation问题，采用Transformer encoder-decoder架构。引入Geometry-Aware Block显式建模局部几何关系，AdaPoinTr在此基础上增加自适应query生成机制（动态query bank）和辅助去噪任务，训练效率提升15倍以上，补全质量提升超20%。
- **适用性分析**：
  - 原生支持多通道输入——point proxy本身可拼接任意维度特征（XYZ + 语义embedding + 法向量等），只需修改input embedding层的维度。
  - 不依赖任何平面假设，Geometry-Aware Block通过KNN graph建模局部拓扑，对凸起结构（空调、栏杆）天然友好。
  - Coarse-to-fine解码策略适合大规模点云的分块补全：先生成粗糙代理点（proxy），再逐步细化。
  - PCN上CD达到6.53，ShapeNet-55上CD 0.81，KITTI上MMD 0.392，均为当前最优。
  - **风险**：原始设计面向单个物体（~8192点），需要工程改造适配建筑级别的分块输入（每块数万点）。

### 2. Point Transformer V3（特征提取backbone）

- **出处**：Wu et al., "Point Transformer V3: Simpler, Faster, Stronger", CVPR 2024 Oral
- **代码**：https://github.com/Pointcept/Pointcept
- **核心机制**：通过空间填充曲线（space-filling curves）序列化点云，用patch-based attention替代传统KNN attention，大幅提升效率。采用U-Net式encoder-decoder结构，配合FlashAttention实现大规模点云高效处理。
- **适用性分析**：
  - 专为大规模场景设计，在ScanNet、nuScenes等数据集上达到SOTA语义分割性能，非常适合作为你的语义分支backbone。
  - 序列化+patch attention机制使其能够处理百万级点云（配合grid pooling），比PTv2的KNN attention内存开销小一个量级。
  - Pointcept框架提供完善的数据预处理、训练和评估pipeline，包含grid sampling、color augmentation等建筑场景适用的功能。
  - **定位**：不直接用于补全，而是作为语义特征提取器，为补全网络提供逐点语义embedding。可以预训练后frozen，或作为联合训练的一个分支。

### 3. SeedFormer

- **出处**：Zhou et al., "SeedFormer: Patch Seeds based Point Cloud Completion with Upsample Transformer", ECCV 2022
- **代码**：https://github.com/hrzhou2/seedformer
- **核心机制**：提出Patch Seeds表示——一组分布在形状表面的种子点，每个种子携带局部patch信息。通过Upsample Transformer逐步从种子生成高密度输出，保留局部几何细节。
- **适用性分析**：
  - Patch Seeds的概念与建筑外立面的语义组件天然对应：可以将每个语义类别（墙体、窗户、空调）的采样点作为种子初始化。
  - 种子到完整点云的上采样过程不依赖全局形状假设，逐patch独立细化，对非平面细节保留能力强。
  - 在PCN和ShapeNet-55上性能接近AdaPoinTr，但架构更轻量，推理速度更快。
  - **风险**：种子数量固定，需要根据建筑规模动态调整；原始设计同样面向单物体。

### 4. SVDFormer

- **出处**：Zhu et al., "SVDFormer: Complementing Point Cloud via Self-View Augmentation and Self-Structure Dual-Generator", ICCV 2023
- **代码**：https://github.com/czvvd/SVDFormer
- **核心机制**：提出自视图增强（self-view augmentation）从多个视角提取互补几何信息，以及自结构双生成器（self-structure dual-generator）分别处理已知区域和缺失区域的结构恢复。
- **适用性分析**：
  - 多视角信息聚合的思路非常适合建筑外立面——立面本身就有明确的正面/侧面结构，自视图增强可以利用这一先验。
  - 双生成器分离已知/缺失区域的处理，避免已有几何被"覆盖"，对保持真实扫描区域的精度非常有利。
  - **风险**：自视图渲染模块引入额外计算开销，且依赖深度渲染，对大规模点云需要分块处理。

### 5. DiffComplete（扩散方法备选）

- **出处**：Chu et al., "DiffComplete: Diffusion-based Generative 3D Shape Completion", NeurIPS 2023
- **代码**：https://github.com/dvlab-research/DiffComplete
- **核心机制**：在TSDF/TUDF体素空间进行条件扩散，通过层次化特征聚合注入条件特征，支持occupancy-aware融合策略处理多个部分输入。
- **适用性分析**：
  - 扩散模型在缺失区域较大时能生成合理的多模态补全结果，对遮挡严重的建筑立面有独特优势。
  - 在体素空间操作，天然兼容密度差异巨大的输入——通过TSDF归一化。
  - 在PatchComplete benchmark上泛化性显著优于确定性方法（包括未见类别）。
  - **风险**：推理速度慢（扩散采样需要多步去噪）；体素分辨率限制几何精度上限；训练成本高。不建议作为首选，适合在确定性方法达到瓶颈后作为精细化或后处理模块。

### 模型选型建议

对于你的场景，我建议的主方案是：**PTv3（语义backbone）+ AdaPoinTr变体（补全主干）**的双分支融合架构。SeedFormer作为轻量替代方案，DiffComplete作为后期精细化模块备选。SVDFormer的多视角思路可以作为数据增强的灵感来源（从不同视角模拟遮挡）。

---

## 二、输入端设计方案对比：语义聚类前置分组 vs. 语义Embedding端到端融合

### 方案A：语义聚类前置分组补全（Semantic Clustering → Per-Group Completion）

**流程**：先用预训练的语义分割网络（如PTv3）对残缺输入进行逐点语义预测，按语义类别将点云分组（墙体组、窗户组、空调组等），对每组独立或分组调用补全网络，最后合并全部补全结果。

**优势**：

每组内的几何模式单一（墙体趋向平面、窗户趋向矩形凹陷、空调趋向凸出盒体），补全网络的学习难度大幅降低。对训练数据量有限的类别（如栏杆）可以单独设计增强策略或loss权重。推理阶段可以并行处理各组，吞吐量高。各组补全结果可以独立评估，便于诊断问题。

**劣势**：

语义分割的错误会直接级联到补全阶段。残缺点云上的语义预测本身精度有限（尤其是缺失区域边界），错误分组会导致补全网络"看到"错误的局部几何上下文。各组独立补全后，接缝处（如窗户与墙体交界）容易出现间隙或重叠，需要额外的后处理对齐模块。分组丢失了跨语义类别的几何上下文——比如窗户的位置依赖于墙面的整体结构，单独补全窗户组缺少这个约束。计算开销：需要运行一次完整的语义分割网络 + K次补全网络（K为类别数），总计算量可能高于端到端方案。

**对几何精度的影响**：组内精度可能较高（受益于简化的几何模式），但组间一致性和全局几何连贯性是主要短板。在建筑立面这种高度结构化但组件紧密耦合的场景中，这个短板可能是致命的。

### 方案B：语义Embedding端到端融合（End-to-End Semantic Feature Fusion）

**流程**：将语义信息编码为逐点embedding向量（通过one-hot编码、预训练特征、或可学习的语义token），与几何坐标拼接后直接送入补全网络，在网络内部通过attention机制实现语义-几何的隐式交互。

**优势**：

避免语义分割错误的级联放大——即使某些点的语义预测有偏差，补全网络仍可通过全局attention自行修正。保留完整的跨语义几何上下文，网络可以学习"窗户旁边通常是墙体"这类结构先验。端到端梯度传播允许语义特征和几何特征相互优化，语义分支可以"学会"提取对补全最有用的语义表示（而非通用分割特征）。单次前向推理完成补全+语义预测，无需后处理对齐。

**劣势**：

训练难度更高——网络需要同时学习几何补全和语义理解两个任务，loss平衡是关键挑战。语义embedding增加了特征维度，对attention机制的计算/显存开销有直接影响（尤其在大规模点云上）。如果语义loss权重不当，可能出现"语义正确但几何变形"的问题（网络为了降低CE loss而牺牲几何精度），这需要仔细的loss设计来避免。调试困难——当补全质量不佳时，难以区分是几何能力不足还是语义信号干扰。

**对几何精度的影响**：在loss设计得当的前提下，端到端方案的全局几何一致性通常优于分组方案。但需要严格的几何优先loss策略来防止语义任务"劫持"网络容量。

### 结论与推荐

**推荐方案B（端到端融合）为主方案**，但采用一种折中设计来缓解其劣势：

1. **语义backbone预训练+冻结**：先在语义标注数据上预训练PTv3，然后冻结权重作为特征提取器，而非从头联合训练。这样语义分支不会被补全loss干扰，同时补全网络可以利用高质量的语义特征。
2. **分阶段解冻**：训练后期（如最后30%的epoch）解冻语义backbone的最后几层，进行小学习率的端到端微调，让语义特征适配补全任务。
3. **语义信息以soft embedding（而非hard label）形式注入**：使用PTv3输出的倒数第二层特征向量（如256维）而非argmax后的one-hot标签，保留语义不确定性信息。

---

## 三、网络架构流程

```
输入: 残缺点云 P_in ∈ R^{N×3}（可选: +RGB, +法向量）

┌───────────────────────────────────────────┐
│ Stage 0: 预处理                             │
│  · 体素下采样（0.05m）+ 固定点数采样         │
│  · 局部坐标归一化（减min, 保留height）        │
│  · 拼接预计算法向量（Open3D estimate_normals）│
└───────────────┬───────────────────────────┘
                │
    ┌───────────▼───────────┐
    │ 语义特征提取器（Frozen PTv3）│
    │  输入: P_in (N×6: XYZ+Normal)│
    │  输出: F_sem (N×256)         │
    │  + 语义预测 S_pred (N×C)     │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────────────────┐
    │ 特征拼接                           │
    │  F_input = [XYZ; Normal; F_sem]   │
    │  维度: N × (3 + 3 + 256) = N×262 │
    └───────────┬───────────────────────┘
                │
    ┌───────────▼───────────────────────────┐
    │ 补全主干网络（Modified AdaPoinTr）        │
    │                                        │
    │ Encoder:                               │
    │  · Input Embedding: MLP(262→512)       │
    │  · FPS + KNN 分组 → Point Proxies      │
    │  · Geometry-Aware Transformer Blocks ×6│
    │  · 输出: Latent Proxies Z (M×512)      │
    │                                        │
    │ Adaptive Query Generator:              │
    │  · Q_I: 从encoder输出抽象              │
    │  · Q_O: 自适应生成缺失区域queries        │
    │  · Scoring → 选择 Top-M' queries        │
    │                                        │
    │ Decoder:                               │
    │  · Transformer Decoder Blocks ×6       │
    │  · Cross-attention(queries, Z)         │
    │  · 输出: Decoded Proxies (M'×512)      │
    │                                        │
    │ FoldingNet Head:                       │
    │  · 每个proxy展开为局部patch             │
    │  · 粗→细两阶段上采样                    │
    └───────┬─────────┬─────────────────────┘
            │         │
    ┌───────▼──┐  ┌───▼──────────────┐
    │ 几何输出  │  │ 语义分类头        │
    │ P_out    │  │ MLP(512→C)       │
    │ (N'×3)   │  │ → S_out (N'×C)   │
    └──────────┘  └──────────────────┘
```

**关键设计细节**：

Geometry-Aware Block中，KNN graph的构建同时考虑欧氏距离和语义特征相似度。具体地，边权重由两部分组成：空间距离权重（主导）和语义cosine相似度权重（辅助），比例约为0.8:0.2。这使得同一语义区域内的点在attention中获得更强连接，而不需要显式分组。

Decoder阶段的cross-attention自然实现了"语义感知的几何生成"——query带有位置信息和隐式语义信息，通过与encoder输出的注意力交互，生成的每个proxy point已经隐含了语义上下文。

---

## 四、多任务Loss函数详解

### 总体Loss公式

$$\mathcal{L}_{\text{total}} = \lambda_{\text{geo}} \cdot \mathcal{L}_{\text{geo}} + \lambda_{\text{sem}} \cdot \mathcal{L}_{\text{sem}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}} + \lambda_{\text{denoise}} \cdot \mathcal{L}_{\text{denoise}}$$

### 4.1 几何损失 $\mathcal{L}_{\text{geo}}$（最高优先级）

采用分层Chamfer Distance + F-Score联合驱动：

$$\mathcal{L}_{\text{geo}} = \mathcal{L}_{\text{CD}}^{\text{coarse}} + \alpha \cdot \mathcal{L}_{\text{CD}}^{\text{fine}} + \beta \cdot \mathcal{L}_{\text{F}}$$

其中Chamfer Distance（L1版本，比L2对离群点更鲁棒）：

$$\mathcal{L}_{\text{CD}}(P, Q) = \frac{1}{|P|}\sum_{p \in P} \min_{q \in Q} \|p - q\|_1 + \frac{1}{|Q|}\sum_{q \in Q} \min_{p \in P} \|q - p\|_1$$

F-Score损失（可微近似）：

$$\mathcal{L}_{\text{F}} = 1 - \text{F-Score}(P_{\text{out}}, P_{\text{gt}}, \tau)$$

$$\text{F-Score} = \frac{2 \cdot \text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau)}$$

其中 $\tau$ 为距离阈值（建议 $\tau = 0.05\text{m}$）。Precision和Recall使用sigmoid soft assignment实现可微：

$$\text{Precision}(\tau) = \frac{1}{|P|}\sum_{p \in P} \sigma\left(\frac{\tau - d(p, Q)}{t}\right), \quad t = 0.01$$

**初始值**：$\alpha = 1.0$，$\beta = 0.5$

### 4.2 语义损失 $\mathcal{L}_{\text{sem}}$

加权交叉熵 + Dice Loss组合（处理类别不平衡）：

$$\mathcal{L}_{\text{sem}} = \mathcal{L}_{\text{WCE}} + \gamma \cdot \mathcal{L}_{\text{Dice}}$$

$$\mathcal{L}_{\text{WCE}} = -\frac{1}{N'}\sum_{i=1}^{N'} w_{y_i} \cdot \log \frac{e^{z_{i,y_i}}}{\sum_{c=1}^{C} e^{z_{i,c}}}$$

类别权重 $w_c$ 按逆频率计算：$w_c = \frac{1}{\ln(1.2 + f_c)}$，其中 $f_c$ 为类别 $c$ 在训练集中的点数占比。

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C}\sum_{c=1}^{C} \frac{2\sum_i p_{i,c} \cdot g_{i,c} + \epsilon}{\sum_i p_{i,c} + \sum_i g_{i,c} + \epsilon}$$

**初始值**：$\gamma = 1.0$

### 4.3 正则项 $\mathcal{L}_{\text{reg}}$

包含三个部分：

$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{uniform}} + \delta \cdot \mathcal{L}_{\text{normal}} + \mu \cdot \mathcal{L}_{\text{boundary}}$$

均匀性正则（防止补全点聚集）：

$$\mathcal{L}_{\text{uniform}} = \sum_{p \in P_{\text{out}}} \left( d_{\text{KNN}}(p) - \bar{d}_{\text{KNN}} \right)^2$$

其中 $d_{\text{KNN}}(p)$ 是点 $p$ 到其K近邻的平均距离，$\bar{d}_{\text{KNN}}$ 是全局均值。

法向量一致性正则（鼓励局部曲面光滑）：

$$\mathcal{L}_{\text{normal}} = \frac{1}{|P_{\text{out}}|}\sum_{p \in P_{\text{out}}} \sum_{q \in \text{KNN}(p)} \left| (p - q)^T \cdot n_q \right|$$

语义边界锐化正则（防止语义过渡区域的几何模糊）：

$$\mathcal{L}_{\text{boundary}} = -\frac{1}{|B|}\sum_{p \in B} \max_c \, s_{p,c} \cdot \log(\max_c \, s_{p,c})$$

其中 $B$ 是语义边界点集（KNN邻域内存在不同语义标签的点），$s_{p,c}$ 是语义softmax概率。此项鼓励边界点的语义预测更确定。

**初始值**：$\delta = 0.1$，$\mu = 0.05$

### 4.4 去噪辅助损失 $\mathcal{L}_{\text{denoise}}$

继承AdaPoinTr的设计，对ground truth中心点加噪后由decoder恢复：

$$\mathcal{L}_{\text{denoise}} = \mathcal{L}_{\text{CD}}(P_{\text{denoised}}, P_{\text{gt\_local}})$$

### 4.5 权重初始值与动态调整策略

| 权重 | 初始值 | 最终值 | 调整策略 |
|------|--------|--------|----------|
| $\lambda_{\text{geo}}$ | 1.0 | 1.0 | **固定不变**，作为锚点 |
| $\lambda_{\text{sem}}$ | 0.0 | 0.3 | 线性warmup：前20% epoch从0升至0.3 |
| $\lambda_{\text{reg}}$ | 0.1 | 0.1 | 固定 |
| $\lambda_{\text{denoise}}$ | 1.0 | 0.1 | 余弦退火：从1.0衰减至0.1 |

**核心原则：几何绝对优先。** 具体做法：

**几何主导warmup**（epoch 0–20%）：仅训练几何损失 + 去噪损失，语义loss权重为0。确保网络首先建立正确的几何感知能力。

**语义渐入**（epoch 20%–60%）：语义loss从0线性增长到目标值0.3。同时监控val CD指标——如果某个epoch的val CD相比上一个epoch恶化超过5%，立即将 $\lambda_{\text{sem}}$ 减半并保持5个epoch。

**联合微调**（epoch 60%–100%）：所有权重固定，解冻语义backbone最后2层，学习率设为主网络的0.1倍。

$$\lambda_{\text{sem}}(t) = \begin{cases} 0 & t < 0.2T \\ 0.3 \cdot \frac{t - 0.2T}{0.4T} & 0.2T \leq t < 0.6T \\ 0.3 & t \geq 0.6T \end{cases}$$

其中 $T$ 为总训练epoch数。

**自动安全阀**：每个epoch结束时检查，若 val CD > 1.2 × best val CD，则 $\lambda_{\text{sem}} \leftarrow 0.5 \cdot \lambda_{\text{sem}}$，持续5个epoch后恢复。

---

## 五、数据处理策略

### 5.1 密度归一化

你的数据密度差异巨大（数万到数千万点），需要标准化处理流程：

**体素下采样**：统一使用0.05m体素网格下采样（比你之前用的0.1m更精细，适配建筑细节）。每个体素内取重心而非随机选点，保证几何准确性。对于语义标签，取体素内出现频率最高的标签（majority voting）。

**分块策略**：将每栋建筑切分为重叠块。块尺寸建议 $8\text{m} \times 8\text{m} \times 8\text{m}$，重叠率 50%（即步长4m）。每块体素下采样后固定采样为 $N = 80{,}000$ 点（与你之前的facade数据集设置一致）。推理时对重叠区域取加权平均（权重与到块中心的距离成正比）。

**GT对齐**：BIM/CAD采样的完整GT点云需要与残缺扫描严格对齐。建议使用ICP精配准后再进行体素下采样，确保GT和输入的体素网格一致。

### 5.2 数据增强

建筑外立面的增强策略需要尊重物理约束——不能随意旋转（建筑有重力方向），增强应保持结构合理性：

- **绕Z轴随机旋转**（0°/90°/180°/270° 四选一，非连续旋转，因为建筑通常与坐标轴对齐）
- **随机镜像翻转**（沿X或Y轴）
- **高度方向缩放**（0.9–1.1，模拟不同楼层高度）
- **随机遮挡模拟**：在完整GT上随机放置3–5个椭球遮挡体，生成模拟残缺输入（增加训练对中的多样性）
- **点级jitter**：XYZ坐标加 $\mathcal{N}(0, 0.01\text{m})$ 高斯噪声（模拟扫描噪声）
- **随机dropout**：随机移除5%–15%的点（模拟稀疏区域）
- **语义标签扰动**：对5%的点随机替换语义标签为邻近类别（提升对语义噪声的鲁棒性）

### 5.3 课程学习方案

**Phase 1（epoch 1–30%）：简单→困难，按残缺程度排序**

训练初期只使用残缺率 < 30% 的样本（即保留点数占GT的70%以上），让网络先学会"小面积补全"。同时仅启用几何loss。

**Phase 2（epoch 30%–70%）：逐步引入高残缺样本**

每10个epoch将残缺率上限提高10%，直到覆盖所有样本（残缺率可达70%+）。此阶段开始引入语义loss。

**Phase 3（epoch 70%–100%）：全数据 + 困难样本挖掘**

使用全量数据。每个epoch结束后，根据val CD对所有样本排序，下一个epoch中对CD最高的20%样本进行2x过采样（即出现两次），增加困难样本的训练频率。

---

## 六、训练超参数建议

| 参数 | 值 | 说明 |
|------|-----|------|
| Optimizer | AdamW | weight_decay=0.05 |
| Base LR | 5e-4 | 补全主干 |
| Semantic backbone LR | 5e-5 | PTv3解冻后（补全LR的0.1x） |
| LR Schedule | Cosine Annealing | warmup 10 epochs（linear从1e-6到5e-4）|
| Batch Size | 8 per GPU | 8× RTX 4090 → 有效batch 64 |
| Total Epochs | 300 | ~3000栋 × 分块 ≈ 每栋平均10块 → 30k样本 |
| 输入点数 N | 80,000 | 下采样后固定 |
| 输出点数 N' | 80,000 | 与输入等量 |
| Coarse Proxy数 | 512 | AdaPoinTr默认 |
| Fine倍率 | 4× → 2× | 两阶段上采样 |
| KNN K值 | 32 | Geometry-Aware Block |
| 语义类别数 C | 12 | 你的数据集定义 |
| 块尺寸 | 8m × 8m × 8m | 重叠50% |
| 体素大小 | 0.05m | 下采样网格 |
| Gradient Clip | max_norm=1.0 | 防止梯度爆炸 |
| Mixed Precision | FP16（AMP） | 节省显存，4090支持 |

---

## 七、评估体系（几何指标最高优先级）

### 主指标（决策依据）

1. **Chamfer Distance-L1（CD）**：全局几何精度，越低越好。分coarse和fine两级报告。
2. **F-Score@0.05m**：在5cm阈值下的精度-召回调和均值。直接反映补全点云的"可用性"——F-Score高意味着大部分补全点都在GT表面5cm范围内。

### 辅助指标

3. **per-class CD**：按语义类别分别计算CD，识别哪些类别是瓶颈（预期空调/栏杆类CD最高）。
4. **语义 mIoU**：补全输出上的语义分割质量，作为辅助参考，不作为模型选择的主要依据。
5. **OA（Overall Accuracy）**：语义预测的整体准确率。
6. **均匀性指标**：$\text{CV}(d_{\text{KNN}}) = \sigma(d_{\text{KNN}}) / \mu(d_{\text{KNN}})$，变异系数越小越均匀。

### 评估协议

对每栋建筑：先分块补全→重叠区域融合→与完整GT计算CD和F-Score。报告所有建筑的均值±标准差，以及按建筑规模（小/中/大）分组的结果。

---

## 八、潜在风险与应对

### 风险1：语义loss主导导致几何变形

**症状**：训练中CE loss快速下降但CD停滞或恶化；可视化发现补全输出在语义边界处出现"台阶"或错位。

**应对**：上述loss权重动态调整策略（几何warmup + 自动安全阀）。若仍然严重，考虑完全分离语义头——在frozen encoder输出上独立训练一个语义MLP，不通过decoder反传。

### 风险2：分块补全的接缝伪影

**症状**：相邻块的补全结果在边界处不连续，出现明显的密度突变或几何错位。

**应对**：增大重叠率至60%–70%；重叠区域使用高斯权重融合（中心权重高、边缘权重低）；训练时对块边界区域的点施加额外的一致性loss。

### 风险3：稀有类别（栏杆、空调）补全质量差

**症状**：per-class CD显示这些类别远高于墙体/窗户。

**应对**：类别感知的过采样（稀有类别所在块出现频率×3）；在loss中加入class-balanced权重；必要时为稀有类别收集额外的合成数据（从BIM模型中变形生成）。

### 风险4：真实扫描与BIM采样的分布差距

**症状**：训练loss低但在真实残缺扫描上泛化差。

**应对**：在BIM→扫描的数据配对中引入domain randomization——对BIM采样点加噪声、随机稀疏化、模拟扫描仪特有的噪声模式。训练后期用少量真实扫描-BIM配对做fine-tune。

### 风险5：显存溢出（80K点 × 8 batch）

**症状**：OOM错误，尤其在attention层。

**应对**：使用FlashAttention-2（4090原生支持）；采用PTv3的patch attention策略而非full attention；必要时减小输入点数到40K–60K或使用gradient checkpointing。

---

## 九、推荐论文（已验证存在）

1. **AdaPoinTr** — Yu et al., "AdaPoinTr: Diverse Point Cloud Completion with Adaptive Geometry-Aware Transformers", T-PAMI 2023. [arXiv:2301.04545](https://arxiv.org/abs/2301.04545) | [GitHub](https://github.com/yuxumin/PoinTr)
   ↳ 首选补全backbone，SOTA性能 + 代码成熟。

2. **Point Transformer V3** — Wu et al., "Point Transformer V3: Simpler, Faster, Stronger", CVPR 2024 Oral. [GitHub (Pointcept)](https://github.com/Pointcept/Pointcept)
   ↳ 语义特征提取backbone，大规模点云效率最优。

3. **SeedFormer** — Zhou et al., "SeedFormer: Patch Seeds based Point Cloud Completion with Upsample Transformer", ECCV 2022. [GitHub](https://github.com/hrzhou2/seedformer)
   ↳ 种子点概念适合语义分组初始化。

4. **SVDFormer** — Zhu et al., "SVDFormer: Complementing Point Cloud via Self-View Augmentation and Self-Structure Dual-Generator", ICCV 2023. [GitHub](https://github.com/czvvd/SVDFormer)
   ↳ 双生成器分离已知/缺失区域思路值得借鉴。

5. **DiffComplete** — Chu et al., "DiffComplete: Diffusion-based Generative 3D Shape Completion", NeurIPS 2023. [GitHub](https://github.com/dvlab-research/DiffComplete)
   ↳ 扩散方法备选，尤其适合大面积缺失场景。

6. **ProxyFormer** — Li et al., "ProxyFormer: Proxy Alignment Assisted Point Cloud Completion with Missing Part Sensitive Transformer", CVPR 2023.
   ↳ 缺失区域敏感的attention设计，对遮挡模式建模有参考价值。

7. **SnowflakeNet** — Xiang et al., "SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer", ICCV 2021 / T-PAMI 2023.
   ↳ 你已尝试过但效果差——主要原因可能是数据量不足和未做分块处理，而非模型本身不适用。其Skip-Transformer和SPD模块的设计思路仍有参考价值。

8. **Sonata** — Wu et al., "Sonata: Self-Supervised Learning of Reliable Point Representations", CVPR 2025 Highlight. [GitHub (Pointcept)](https://github.com/Pointcept/Pointcept)
   ↳ PTv3团队最新的自监督预训练方法，可用于初始化语义backbone。

9. **SDS-Complete** — Kasten et al., "Point Cloud Completion with Pretrained Text-to-Image Diffusion Models", NeurIPS 2024. [GitHub](https://github.com/NVlabs/sds-complete)
   ↳ 利用2D扩散先验进行3D补全，对OOD建筑类型有潜在泛化优势（无需3D训练数据）。

---

## 十、实施路线图（建议）

| 阶段 | 时间 | 内容 |
|------|------|------|
| 1 | 第1–2周 | 数据pipeline搭建：体素下采样 + 分块 + GT对齐 + 数据加载器 |
| 2 | 第3–4周 | AdaPoinTr基线复现：在你的数据上跑通原版AdaPoinTr（仅几何loss），确认基线CD |
| 3 | 第5–6周 | PTv3语义预训练：在标注数据上训练语义分割，冻结后输出语义embedding |
| 4 | 第7–9周 | 融合架构实现：修改AdaPoinTr输入层接受语义embedding，实现多任务loss，调通训练 |
| 5 | 第10–12周 | 超参调优 + 课程学习 + 消融实验 |
| 6 | 第13–14周 | 全量数据训练 + 评估 + 可视化 |
