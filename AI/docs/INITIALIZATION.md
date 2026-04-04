# 项目初始化说明文档

## 本地初始化

### 安装 .venv 虚拟环境

**.venv** 将这个项目需要的 Python 版本和第三方插件全部隔离在项目的文件夹里、不影响系统，也不受其他项目干扰。

```shell
python3 -m venv .venv
```

### 安装 Jupyter Notebook 内核

```shell
pip install ipykernel
```

### 本地环境变量设置

将环境变量保存在 `.env` 文件中（一般保存在项目的根目录中）

**安装依赖包（包含 Cli）：**

```shell
pip install "python-dotenv[cli]"
```

**1. 在代码中加载 .env 文件：**

需要在代码中显式加载

```python
from dotenv import load_dotenv

# 加载 .env 文件中的变量到系统的环境变量中
load_dotenv()
```

**2. 命令行执行时加载 .env 文件：**

不需要在代码中显式加载，在命令行设置加载的 `.env` 文件

```shell
# 默认使用当前目录的 .env
dotenv run -- python main.py
```

```shell
# 指定 .env
dotenv -f .env run -- python main.py
```

---

## Google Colab

使用 `Google Colab` 环境测试

[https://colab.research.google.com/](<https://colab.research.google.com/>)

创建 `*.ipynb` 文件保存测试代码

```shell
# 安装依赖包
!pip install openai numpy tiktoken

# OPENAI_API_KEY 存放在环境变量中
%env OPENAI_API_KEY=[填写您的 OpenAI API Key]

# 读取环境变量
import os
os.environ.get("OPENAI_API_KEY")
```

```shell
# OPENAI_API_KEY 存放在 Google Colab Secret 中
# 读取 Secret
from google.colab import userdata
userdata.get("OPENAI_API_KEY")
```

使用其他 AI 则修改为对应的 API KEY

| AI     | API KEY        |
| ------ | -------------- |
| Gemini | GOOGLE_API_KEY |
| Kimi   | KIMI_API_KEY   |

---

## Model 使用的编码

参考：[https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb](<https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb>)

使用 tiktoken 统计 token 数量时，需要使用与 Model 一样的编码

**获取 Model 使用的编码：**

例如 Model=`gpt-4o-mini`，则使用下面代码获取使用的编码

```python3
encoding = tiktoken.encoding_for_model('gpt-4o-mini')
```

**设置 tiktoken 编码：**

例如 Model=`gpt-4o-mini`，则 `tiktoken` 配置以下的编码

```python3
encoding = tiktoken.get_encoding("o200k_base")
```
