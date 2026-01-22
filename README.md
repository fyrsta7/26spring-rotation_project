# 结合大模型的程序优化

本仓库是熊英飞老师组为 2026 年北京大学图灵班轮转设计的实践项目之一，主要涉及和大模型相关的程序优化技术。

如果在完成项目的过程中遇到了任何问题，请随时通过微信或者邮件 (zhaoyuwei@stu.pku.edu.cn) 与负责的同学联系。另外，请大家善用大模型，可以解决大家在写代码时遇到的很多问题。

## 项目概述

### 动机

程序优化在现代计算领域具有重要意义，它直接关系到软件的执行效率、资源利用率以及用户体验。在性能需求不断提高和硬件环境日益复杂的背景下，传统的优化手段往往面临规则设计繁琐、自动化程度不足等瓶颈。而大模型凭借强大的自然语言理解与生成能力，以及在模式识别和知识推理中的卓越表现，为解决这些问题提供了新的可能。通过将大模型应用于程序优化任务，我们可以探索更加智能化、自动化的优化手段，从而提升程序开发效率并实现更高效的代码性能。

### 主要构成

- part1：尝试使用 api 来调用大模型，了解 api 相关的设置。尝试获取代码库中的 commit 信息。了解提示工程技术。
- part2：尝试使用大模型来完成给定的程序优化任务，在具体的任务上对比不同 prompt 的效果。
- part3：阅读相关论文。
- part4：选择一个路线并进一步探索。可以根据给定的路线来探索，或者尝试复现论文并提出改进。

注意：part2 和 part3 的顺序不是固定的，可以交叉完成。

### 时间安排

完成基础任务：
- part1：1周
- part2：1.5周
- part3：1.5周
- part4：1周

分配更多时间给后续探索：
- part1：1周
- part2：1-1.5周
- part3：1周
- part4：1.5-2周


## part 1

在开展具体的项目之前，我们需要了解并尝试使用基本的工具。该部分的构成如下：
- 1.0：简单学习 Git 以及命令行的使用
- 1.1：尝试使用 api 来调用大模型，了解 api 相关的设置
- 1.2：了解提示工程技术
- 1.3：尝试获取代码库中的 commit 信息

### 1.0

（有任何问题都可以先问大模型）

- 学习命令行的基本使用方式
    - 参考 https://missing-semester-cn.github.io/2020/shell-tools/
- 运行 python 脚本
    - 在本地配置 python 环境，并尝试在命令行运行 `python hello_world.py`
    - 如果需要的话，可以尝试使用 pyenv 工具管理 python 版本。
- 学习 Git
    - 参考 https://missing-semester-cn.github.io/2020/version-control/
    - 了解如何进行简单的版本管理
- 了解 GitHub 的主要用法，例如可以从以下几个问题入手
    - 配置环境，并在命令行使用 `git clone` 来下载代码，例如可以下载本项目。尝试使用 `git pull` 来获取代码库的更新。
    - 尝试在 GitHub 网页上查看 commit 信息。
    - 尝试在 GitHub 上创建仓库，并使用 `git push` 将本地的代码同步更新到 GitHub。
        - 注意：如果你需要在本项目的基础上进行修改，并希望将代码保存到 GitHub，请另外新开一个仓库来存储你自己的代码，而不要直接修改本项目对应的仓库。例如你可以使用 GitHub 上的 fork。

### 1.1

#### 1.1.1 - 尝试使用 api 来调用大模型

我们组使用[大模型门户](https://llm.xmcp.ltd/)来调用各类模型，包括 Claude、OpenAI、DeepSeek、通义千问 等模型，每位同学的限额是每月100刀。组内统一报销费用，只能用于自己的科研轮转项目，不能转借或者倒卖。

你需要先修改根目录下的 `config.py` 文件，其中包括以下变量：
- `GITHUB_TOKEN`：使用 GitHub api 时需要的配置，在 GitHub - Settings - Developer Settings - Personal access tokens - Tokens (classic) 中生成一个然后复制进来就行
- `root_path`：项目代码所在的根目录的路径
- `xmcp_base_url`：调用大模型相关，在大模型门户上的“API 调用秘钥/BASE_URL”一栏获取，已设置好
- `xmcp_api_key`：调用大模型相关，在大模型门户上的“API 调用秘钥/API_KEY”一栏获取自己账号的 api key 并放到这里
- `xmcp_model`：调用大模型相关，在大模型门户上查看模型列表，选择想要调用的模型，然后将“模型名称”复制到这里，例如 `ali/deepseek-v3`

注意保护好自己的 api key，防止泄露。如果要在 GitHub 上存放项目，注意不要将 api key 同步更新上去。如果发现 api key 有可能已经泄露，请立刻联系管理员重置秘钥。

`part1/test_api.py` 中是一个简单的大模型调用示例。设置好 `config.py` 后，你可以使用 `python test_api.py` 来运行 `test_api.py`。


#### 1.1.2 - 了解 api 相关的设置

参考以下资料了解如何通过 api 调用大模型以及相关的参数设置（OpenAI 和 DeepSeek 用的同一个库，所以可以先阅读 DeepSeek 文档，了解基本设置）：
- https://api-docs.deepseek.com/zh-cn
- https://platform.openai.com/docs
- https://cookbook.openai.com/

主要的设置包括：
- message：https://api-docs.deepseek.com/zh-cn/
    - role
    - content
- temperature：https://api-docs.deepseek.com/zh-cn/quick_start/parameter_settings
- logprobs: https://cookbook.openai.com/examples/using_logprobs & https://api-docs.deepseek.com/zh-cn/api/create-chat-completion
- 对话补全：https://api-docs.deepseek.com/zh-cn/api/create-chat-completion
- JSON Output：https://api-docs.deepseek.com/zh-cn/guides/json_mode
- 上下文硬盘缓存：https://api-docs.deepseek.com/zh-cn/guides/kv_cache
- ...

尝试基于 `test_api.py` 来调用大模型并解决简单的问题，尝试调整输入的参数以及获取输出的各类信息。


### 1.2 - 了解提示工程技术

参考以下资料了解提示工程技术：
- https://www.promptingguide.ai/zh

主要技术包括：
- 零样本提示
- 少样本提示
- 链式思考（CoT）提示
- 检索增强生成 (RAG)
    - https://www.promptingguide.ai/zh/techniques/rag
    - https://www.zhihu.com/tardis/zm/art/675509396?source_id=1003
    - https://arxiv.org/abs/2005.11401: 提出 RAG 技术的论文
        - https://blog.csdn.net/weixin_43221845/article/details/142610477: 随便找的一个论文解读
- ...


### 1.3 - 尝试获取代码库中的 commit 信息

编写脚本，在给定代码库以及 commit hash 后，自动获取 commit 中的所有信息。主要有以下两种思路：
- （后续主要用到的方案）如果 commit 集中来自于一个或多个代码库，可以考虑将整个代码库下载到本地，并且直接从 git 信息中获取需要的部分
- 如果 commit 分散在许多不同的代码库，可以考虑直接调用 GitHub api，获取对应的 commit 信息。GitHub api 有调用频率的限制，主要有以下几种解决方法。
    - 多注册几个账号获得更多的 key
    - 纯等待，例如每调用一次 api 之后 sleep(n)
    - 改用第一种方法

可以使用[RocksDB](https://github.com/facebook/rocksdb)代码库来尝试实现上述功能

## part 2

在这一部分，我们将在给定的 task 上尝试使用一些 prompt 技术，并对比效果。该部分的构成如下：
- 2.1：使用大模型来判断一个 commit 的主要目的是否为性能优化
- 2.2：使用大模型尝试优化一段代码

该部分将继续使用[RocksDB](https://github.com/facebook/rocksdb)代码库。

### 2.1

写 python 脚本，判断一个 commit 中代码修改的主要目的是否为性能优化（只包括提高代码运行效率 / 减少运行所需资源，不包括改善代码可读性和可维护性等）

具体步骤：
- 从代码库的 git 信息中获取 commit 信息
- 给出 commit 的具体信息（例如 commit message，具体的代码修改信息），让大模型给出答案
    - 如何让大模型只回答 true / false / unknown（在大模型不确定的时候就回答 unknown）
    - 如何提高大模型的回答准确率
    - 是否有一些类型的任务大模型认为不是性能优化，但在人类判断的结果上属于性能优化（比如优化内存访问的效率），尝试通过修改 prompt 来改善这一问题
- 提高大模型回答的置信度（选做，不重要）
    - 尝试获得输出 token 的置信度，并根据置信度判断，大模型是否在某些时刻实际上比较确定但回答了 unknown，在某些时刻实际上不确定但回答了 true/false
    - 如何优化上述问题

### 2.2

现在我们只考虑那些只修改了一个函数并且主要实现性能优化的 commit，然后尝试手动调用大模型来优化里面的代码（即尝试复现 commit 中的修改），主要有以下两种优化思路：
- 2.2.1：在所有任务上使用统一的 prompt，例如给出需要优化的函数，并写一句话让大模型实现性能优化，并给出优化后的完整函数。
- 2.2.2：在 prompt 中加入一些针对性的提示，也就是针对不同的 commit 使用不同的 prompt，以此来改善大模型的表现。

在这一部分中，提供两类 commit 可用于优化。你可以先从第一类中挑选几个并尝试优化，然后再考虑第二类中的 commit。
- 第一类 commit 来自[RAPGen](https://arxiv.org/abs/2306.17077)，这里的每个 commit 中都只做了 api 调用层面的修改（包括删除 / 增加 / 替换 api，以及进行必要的其他修改），并且保证所有被修改的文件都是只修改了一处。具体内容放在文件 `part2/rapgen_benchmark.json` 中，每个 commit 包含三方面信息，分别是所在的代码库，对应的 commit hash，以及被修改的文件名。
- 第二类 commit 来自[RocksDB](https://github.com/facebook/rocksdb)，这里的每个 commit 都只修改了一个函数。具体内容放在文件 `part2/rocksdb_benchmark.json` 中（可能有一些不符合要求的 commit，可以选择性直接跳过）。每个 commit 对应的 `github_commit_url` 字段是该 commit 所在的网址，可以直接在网页上确认该 commit 一共修改了哪些内容。）理论上来说，你可以在 part2.1 的基础上进一步做筛选，最终得到这里用到的 commit，例如筛选出只修改了一个文件中一个函数的 commit，但这些步骤比较繁琐，在此就直接跳过。

另外，在 2.2.3 中我们会判断优化后代码的正确性，包括语义是否保持不变以及是否实现了性能提升这两部分。
无论我们以何种方式实现代码优化，都需要在得到结果后进行判断，所以 2.2.3 的任务应该与 2.2.1 & 2.2.2 相结合。

#### 2.2.0

虽然我们给出的 commit 都是只修改了一个函数甚至一行代码的，但部分 commit 的修改内容依然可能非常复杂，对于一个并不了解该代码库的程序员来说，也很难快速读懂每一处修改的目的。所以在开始手动优化之前，你可以首先浏览一下 benchmark 中给出的 commit 具体都是做了什么样的修改，例如可以结合 commit message 和代码修改内容分析，或者可以让大模型辅助分析。

在分析一个 commit 时，你可以关注以下几个方面：
- 人工分析后是否觉得代码修改确实能够提升性能，还是说人类也无法理解这样的改动是否有效。如果是后者，可以暂时先不考虑这样的 commit。
- 是否将几个独立的修改内容放在了一个 commit 中，还是所有代码修改都为一个共同的目的服务。如果是前者，也可以暂时不考虑。
- 是否需要提前知道项目特定的一些信息，才能做出这些修改
    - 依赖项目特定信息：例如项目中的一些 api 接口，或者项目运行时的瓶颈是什么以及需要如何改进
    - 几乎不依赖项目特定信息：例如尝试调整 if 条件中的子条件顺序这种不太依赖项目特定环境，而是相对比较通用的修改
- 具体修改是否过分复杂（例如涉及几十甚至上百行代码，或者需要处理一个几百行的函数并且在里面修改很多地方），导致想要让大模型给出正确的优化结果较为困难
- ...

在后续的手动优化过程中，你可以根据上面几个方面对 commit 进行分类，并考虑什么样的 commit 是否让大模型来自动复现优化，什么样的 commit 是人类也很难理解或者复现的。

#### 2.2.1

在所有任务上使用统一的prompt，作为 baseline 方法。所以我们不会在prompt中告诉大模型当前代码可以在什么方面进行优化，而是完全让大模型自行决定每一步的结果。

可能的技术方案：
- 只进行一轮对话。直接在prompt中给出需要被优化的代码，并要求大模型分析代码中可能存在的性能优化问题并给出优化结果
- 使用 CoT。引导大模型先阅读并理解给出的代码，然后分析有什么性能优化的可能（大模型可能会给出多个方向，也可以在prompt中加入限制，要求大模型给出不超过三个的优化方向），然后依次要求大模型给出每种方向上的优化结果。

需要手动判断的内容：
- 分析大模型给出的优化方式有哪些类型，是否有一些是我们不想要的（例如提升代码可读性、可维护性等），调整prompt来规避这些可能。
- 判断大模型给出的优化是否是正确的方向，以及优化的结果是否保持语义不变并且有真正的性能提升。

#### 2.2.2

在 prompt 中加入针对当前代码的一些提示。具体来说，你需要手动分析原 commit 中具体是如何修改代码的，以及为什么要这样修改，然后总结出整体的优化方向，后续加入到提示中。请尝试排列组合不同的提示内容以及不同的提示策略，并分析最终效果。

增加的提示内容可以分为以下两种：
- 只包含优化方向，不包含任何示例代码或者具体的修改方式
- 包含优化方向和对应的示例代码（该方向上优化前后的代码对比，可以手动写一个简单的例子），并且该示例代码中的优化方式和我们现在想实现的优化方式基本类似，通过这种途径直接告诉大模型具体的修改方式

提示策略可以分为以下两种：
- 只进行一轮对话。人工判断里面有哪些可能的性能优化，并直接告诉大模型请你重点考虑xx方面的优化
- 使用 CoT。具体流程类似 2.2.1，例如先让大模型给出一些性能优化的方向，然后手动判断哪一种是我们最希望实现的，并引导大模型采用该种优化。

需要手动判断的内容：
- 判断大模型给出的优化是否为我们想要的方向，以及优化的结果是否保持语义不变并且有真正的性能提升。

#### 2.2.3

判断优化后代码的正确性，主要包含以下两个方面：
- 语义是否保持不变
    - 若代码库中包含 unit test，尝试在优化前后分别运行
    - 人工判断
- 是否真正实现了性能提升
    - 若代码库中包含 performance test，尝试在优化前后分别运行
    - 人工判断

第一类 commit 没有直接可以使用的测试；第二类 commit 请参考 `part2/rocksdb_test.md` 来尝试运行 RocksDB 代码库中的 unit test 和 performance test，主要是运行 performance test（但 RocksDB 的代码运行起来可能需要配置环境，以及比较古早的版本运行环境较难配置，如果运行不起来的话手动判断就可以了）。


## part 3

给定一些现有工作，阅读论文。可以先粗略过一遍所有论文，然后在优化小规模代码 / 大规模代码中选择一个感兴趣的方向进行精读。后续的 part4 也将在选择的方向上继续进行。

现有论文：
- 优化大规模代码
    - 同一个作者的前后两篇连续的工作
        - DeepDev-PERF：https://dl.acm.org/doi/abs/10.1145/3540250.3549096
        - RAPGen：https://arxiv.org/abs/2306.17077
    - performance bug
        - TANDEM: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9110902
        - A Large-Scale Empirical Study of Real-Life Performance Issues in Open Source Projects：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9757842
- 优化小规模代码（算法竞赛题） - PIE 和 SBLLM 是比较主要的论文。
    - PIE：https://arxiv.org/abs/2302.07867
    - Learning to Improve Code Efficiency：https://arxiv.org/abs/2208.05297
    - Supersonic：https://ieeexplore.ieee.org/abstract/document/10606318/
    - SBLLM：https://arxiv.org/abs/2408.12159



## part 4（在 4.1.1 / 4.1.2 / 4.2 中选择一个方向完成）

该部分的构成如下：
- 4.1 考虑优化大规模代码
- 4.2 考虑优化小规模代码（算法竞赛题）

### 4.1

考虑优化大规模代码，主要方式有以下两种：
- 4.1.1：遵循现有路线，进一步向下探索
- 4.1.2：复现论文中的工具，自行提出一些改进策略并实现

#### 4.1.1
- 在不同 commit 之间交叉对比，寻找共同点
    - 分析不同 commit 实现的优化类型
    - 寻找优化方向类似的 commit
    - 根据上述结果，尝试搭建一个简单的分类系统，例如性能优化可以根据目的分成优化时间复杂度 / 空间复杂度，每一个大类下可以继续细分。
- 根据上述构建的分类系统，涉及prompt引导大模型给出每个 commit 的具体分类
    - 例如可以按照树结构，一层层询问大模型，并逐渐确定 commit 具体实现的优化类型
- 根据具体分类，引导大模型实现优化
    - 搭建一个通用的 prompt 框架
    - 根据具体分类，在 prompt 中加入针对性的提示，具体有以下两种形式
        - 该优化方式的总结，例如将 A 替换成 B
        - 该类优化的一个或多个具体例子，例如一段现有的代码（包含 A）和对应的优化后代码（包含 B）

#### 4.1.2

参考 part3 中给出的论文，复现现有的优化大规模代码的工具。

尝试组合不同论文中的技术，或改进现有技术，判断效果是否有所提升。

目前想要复现可能还有一些额外的困难
- RAPGen 完全没有开源代码，用到的 benchmark 是 DeepDev-PERF 中的 benchmark。所以需要从头开始构建知识库，然后实现后续优化。

DeepDev-PERF：https://github.com/glGarg/DeepDev-PERF

### 4.2

参考 part3 中给出的论文，复现现有的优化小规模代码（算法竞赛题）的工具

尝试组合不同论文中的技术，或改进现有技术，判断效果是否有所提升。

PIE: https://github.com/LearningOpt/pie

SBLLM: https://github.com/shuzhenggao/sbllm

目前想要复现可能还有一些额外的困难
- PIE 的代码我没运行过暂时不清楚
- SBLLM 的代码风格比较奇怪
    - 源代码中调用不同的模型使用不同的 api key，但我们需要使用相同的 api key & base url，并需要修改里面调用的模型名，可能需要浏览项目文件，并找到正确的修改位置。
    - 运行时用到的输入格式，和我们给出的 benchmark 格式不同，如果想要优化我们给出的代码，可能需要做进一步的调整和对接。

