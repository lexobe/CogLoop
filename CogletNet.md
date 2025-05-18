# CogletNet 认知思考模型：数学表达形式

我们定义一个动态系统，通过语义向量检索、认元权重调整、时间追踪与认知生成，构建具备记忆、偏好与自生长能力的智能体认知循环系统。

---

## 1. 基本定义

认元集合定义为

$$
\mathcal{C} = \{c_i = (u_i, w_i, t_i, x_i) \mid i = 1, \dots, n\}
$$

其中

- $u_i \in \mathbb{R}^d$：认元语义向量（embedding）
- $w_i \in \mathbb{R}$：认元权重
- $t_i \in \mathbb{R}$：认元最后更新时间戳
- $x_i \in \mathcal{X}$：认元文本内容


当前输入文本编码为

$$
I = \text{Encode}(x) \in \mathbb{R}^d
$$

---

## 2. 认元激活

向量数据库检索相似认元。

相似度计算

$$
\text{sim}(u_i, I) = \frac{u_i \cdot I}{\lVert u_i \rVert\, \lVert I \rVert}
$$

检索相似度最高的 \$n\$ 个认元

$$
\mathcal{R}_n = \text{Top}_n\bigl(\{\text{sim}(u_i, I)\}_{i=1}^n\bigr)
$$

按权重 \$w\_i\$ 重新排序，取前 \$k=\lfloor n/2 \rfloor\$ 个认元作为激活集合

$$
\mathcal{A} = \text{Top}_k(\mathcal{R}_n,\; \text{by } w_i)
$$

---

## 3. 权重更新方程

仅对 **实际被使用** 的认元集合 `activated_cog_ids` 更新权重：

$$
\Delta t_i = t_{\text{now}} - t_i,\qquad
w_i \leftarrow e^{-b\,\Delta t_i}\bigl(\beta w_i + \gamma\,\Delta t_i\bigr),\quad
 t_i \leftarrow t_{\text{now}}
$$

---

## 4. 上下文构造

拼接激活认元文本

$$
P_{\mathcal{A}} = \bigl\Vert_{c_i\in\mathcal{A}} x_i
$$

完整 prompt 输入

$$
P = P_{\mathcal{A}} \;\Vert\; x
$$

系统指令（结构性指引）

```json
{
  "system_prompt": "你正在模拟一个结构性思考系统，请将对输入的回应结构化为：包含下一轮输入（如有）、认元列表（只给出文本，不含权重）、以及需要调用的函数。认元应当是基于你对输入内容的抽象性观点与模式总结。返回标准 JSON 格式。"
}
```

---

## 5. 响应结构

语言模型生成结构化输出

```json
{
  "next_thought": "继续分析以下文本：...",
  "activated_cog_ids": [101, 105],
  "log": "生成回应时排除了认元107（主张 meme 正在复兴），因其与当前叙事张力不符。",
  "generated_cog_texts": [
    "meme币炒作机制正在失效，背后是叙事耗尽而非资金枯竭。",
    "项目成功的关键在于叙事能否结构化嵌入共识框架。"
  ],
  "function_calls": [
    {
      "name": "X_reply",
      "args": {
        "post_id": "1234567890",
        "reply_text": "This suggests that meme coins may no longer be sustained by narrative alone. We might be entering a phase where structure matters more than momentum."
      }
    }
  ]
}
```

---

## 6. 函数调用与执行

若 $\text{function\_calls}\neq\varnothing$，则逐一执行

$$
\forall f_j\in\text{function\_calls},\quad \text{Execute}(f_j)
$$

函数可包括外部调用、模块跳转、认知链路扩展等。

---

## 7. 认元更新和写入

采用 **向量初筛 + LLM 精筛**：

1. **编码新认元**：\$u\_j = \text{Encode}(x\_j^{\text{new}})\$
2. **检索候选**：\$\mathcal{R}\_5 = \text{Top}\_5\bigl({\text{sim}(u\_i, u\_j)}\bigr)\$
3. **LLM 判断是否重复**
4. **若匹配则仅更新权重**，否则新建认元

新认元写入

$$
\mathcal{C}\leftarrow\mathcal{C}\cup\bigl\{(u_j, w_{\text{init}}, t_{\text{now}}, x_j^{\text{new}})\bigr\}
$$

---

## 8. 认知循环公式

$$
\boxed{\text{CogletNet}(x,\,\mathcal{C}) = \text{LLM}(S,\,P_{\mathcal{A}}\;\Vert\;x) \Rightarrow \text{JSON}}
$$

---

## 9. 系统能力摘要

| 能力    | 数学机制                                                               |
| ----- | ------------------------------------------------------------------ |
| 记忆强化  | \$w\_i \leftarrow e^{-b,\Delta t}(\beta w\_i + \gamma ,\Delta t)\$ |
| 语义检索  | \$\text{sim}(u\_i,I)\$ 与 Top‑\$k\$ 检索                              |
| 上下文构造 | \$P = P\_{\mathcal{A}} ;\Vert; x\$                                 |
| 新认元生成 | `generated_cog_texts`                                              |
| 流程控制  | `next_thought`, `function_calls`                                   |

---

## 10. 认知路径追踪（Cognitive Path Tracing）

CogletNet 运行中自然生成认知路径

$$
\pi = \left( N_1 \rightarrow N_2 \rightarrow \dots \rightarrow N_k \right)
$$

节点 \$N\_i\$ 包含：输入、激活认元、使用认元、新认元、函数调用等。

路径用途：

* 思考链可视化与调试
* 认元活跃度与冗余分析
* 逻辑一致性检测与回滚
* 多分支探索与对比

---

该框架为认元关系图（Coglet Graph）、逆向推理树（Backward Reasoning Tree）等高级认知模块提供数学基础。
