# Tool 注册与自动发现

本模块说明 `@local_tool` / `@mcp_tool` 如何把函数注册进全局表，以及 `ToolBox` 如何在初始化时发现它们。运行时用法见 [ToolBox](tool_box.md)。

## 机制概览（按时间顺序读）

下面用 `RAG_tool.py` 举例，把整条链路串起来。

### ① 单个工具文件被 import 时 — 装饰器注册

`tools/LocalTool/RAG_tool.py` 里有：

```python
from tools.registry import local_tool   # 只是 import 装饰器，还不注册任何工具

@local_tool(context_keys=(...))         # 模块被 import 时，这一行立刻执行
def RAG_search_tool(...): ...
```

当 Python **第一次 import** 这个文件时，`@local_tool` 运行，把 `RAG_search_tool` 的 name / tool_path / description 写进 `tools/registry.py` 里的模块级变量 `_TOOL_REGISTRY`：

```python
_TOOL_REGISTRY["RAG_search_tool"] = ToolInfo(
    name="RAG_search_tool",
    source="local",
    tool_path="tools.LocalTool.RAG_tool.RAG_search_tool",
    ...
)
```

**要点**：写了 `@local_tool` 还不够，文件必须被 import，注册才会发生。

### ② `discover_packages` — 为什么要批量 import？

工具散落在 `tools/LocalTool/`、`tools.MCPTool/` 等多个 `.py` 文件里，Python **不会**自动加载它们。  
`discover_packages("tools.LocalTool", "tools.MCPTool")` 的工作就是：**把这两个包下的子模块逐个 import 一遍**。

它自己**不**往 `_TOOL_REGISTRY` 里写数据；它只是触发上一步——每个被 import 的文件里，**该文件自己的** `@local_tool` / `@mcp_tool` 装饰器会运行并完成注册。

```
discover_packages("tools.LocalTool", ...)
  └─ import tools.LocalTool.RAG_tool    → @local_tool 注册 RAG_index_tool, RAG_search_tool
  └─ import tools.LocalTool.math_tool   → @local_tool 注册 integrate_function
  └─ import tools.MCPTool.tavily_tool   → @mcp_tool 注册 tavily_search, ...
  └─ 跳过 tools.MCPTool._client 等内部模块
```

### ③ `ToolBox()` 构造前 — `_TOOL_REGISTRY` 往往是空的

`import tools.registry` 时，`_TOOL_REGISTRY = {}` 被创建，但此时**还没有** import 任何 tool 文件，所以是空字典。  
同理，`import tools.tool_box` 也不会自动发现工具。

直到 `ToolBox(autodiscover=True)` 调用 `discover_packages(...)`，`_TOOL_REGISTRY` 才被填充。

### ④ `get_registered_tools` + 过滤 — ToolBox 拿到自己的 registry

`_TOOL_REGISTRY` 和 `ToolBox` 在**同一进程、同一份内存**里，并非互相隔离。  
ToolBox 仍通过 `get_registered_tools()` 读取，原因有二：

1. 公开 API，不直接碰 `_TOOL_REGISTRY` 私有变量  
2. 返回拷贝，避免外部误改全局注册表  

然后按 `packages` 前缀过滤（只要 `tools.LocalTool.*` 等），结果写入 **`self._registry`**——这是 ToolBox **实例私有**的工具表，供 `list_tools()` / `ainvoke()` 使用。

```
ToolBox()
  ├─ discover_packages()        → import 子模块 → 各文件的装饰器填充 _TOOL_REGISTRY（全局）
  ├─ get_registered_tools()      → 读 _TOOL_REGISTRY 拷贝
  └─ 前缀过滤 → self._registry  → ToolBox 实例可用的工具表
```

→ ToolBox 构造细节见 [tool_box.md § 初始化](tool_box.md#初始化)

## 核心组件（`tools/registry.py`）

### 装饰器

```python
from tools.registry import local_tool, mcp_tool

@local_tool
def my_tool(...): ...

@mcp_tool(name="custom_name", description="...")
async def mcp_search(...): ...
```

装饰器在函数定义时调用 `_register_tool`，写入进程内单例 `_TOOL_REGISTRY`，并在函数上附加 `__tool_info__`。

每条注册记录为 `ToolInfo`：

| 字段 | 说明 |
|------|------|
| `name` | 注册名，默认函数名，可用装饰器参数覆盖 |
| `source` | `"local"` 或 `"mcp"` |
| `tool_path` | `{模块}.{函数名}`，如 `tools.LocalTool.math_tool.integrate_function` |
| `description` | 装饰器参数，或函数 docstring |
| `context_keys` | 部署 slot 名元组；`ainvoke` 前须在 `ToolContextBundle` 中 `bind`，见 [tool_box.md § 部署 Context](tool_box.md#部署-context) |

### `_TOOL_REGISTRY`

模块级字典，全进程唯一。同一工具名后注册者覆盖先注册者。

### `discover_packages(*packages)`

1. `importlib.import_module` 导入包
2. `pkgutil.walk_packages` 递归列出子模块
3. 跳过 `._config`、`._client` 结尾的内部模块
4. 对其余模块执行 `importlib.import_module`，触发顶层装饰器

默认扫描包（`DEFAULT_TOOL_PACKAGES`）：

- `tools.LocalTool`
- `tools.MCPTool`

### `get_registered_tools()`

返回 `_TOOL_REGISTRY` 的**拷贝**，作为对外读接口，避免调用方直接修改全局注册表。`ToolBox` 在此基础上再按 `packages` 前缀过滤。

## 新增工具

1. 在 `tools/LocalTool/` 或 `tools/MCPTool/` 新建或编辑 `.py` 文件
2. 在函数上添加 `@local_tool` 或 `@mcp_tool`
3. 确保文件会被 `walk_packages` 扫到（不要命名为 `._config` / `._client` 后缀的内部模块）
4. 新建 `ToolBox()` 或重启进程后即可发现

示例（local）：

```python
# tools/LocalTool/my_tool.py
from tools.registry import local_tool

@local_tool
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"
```

`local` 与 `mcp` 在发现与调用路径上一致；区别仅在于 `ToolInfo.source` 及实现是否走 MCP 客户端。

## 常见误区

- **写了 `@local_tool` ≠ 已注册**：文件必须被 import；通常由 `ToolBox()` 里的 `discover_packages` 触发。
- **`discover_packages` 自己不注册**：它只 import 子模块；真正注册的是**每个 tool 文件里**的装饰器。
- **`get_registered_tools` 不是因为访问不到 `_TOOL_REGISTRY`**：全局表与 ToolBox 同进程；用公开 API 读拷贝，再过滤进 `self._registry`。

---

→ 回到 [ToolBox](tool_box.md)
