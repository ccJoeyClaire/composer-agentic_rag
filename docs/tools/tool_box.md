# ToolBox

`ToolBox` 是装饰器发现工具的**注册表 + 运行时**：初始化时收集已注册工具，对外提供 schema 列表与异步调用接口。

工具如何被注册见 [Tool 注册与自动发现](tool_autodiscovery.md)。

## 初始化

```python
from tools.tool_box import ToolBox

box = ToolBox()  # 默认：自动发现 tools.LocalTool + tools.MCPTool
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `autodiscover` | `True` | 为 `True` 时在构造时扫描 `packages` 并填充内部 registry |
| `packages` | `("tools.LocalTool", "tools.MCPTool")` | 限定扫描的包；见 [自动发现](tool_autodiscovery.md) |

常见变体：

```python
# 只加载 LocalTool（测试或精简部署）
box = ToolBox(packages=("tools.LocalTool",))

# 空 registry，不触发自动发现（单元测试）
box = ToolBox(autodiscover=False)
```

### 构造时发生了什么（`_load_registered_tools`）

`ToolBox()` 刚创建时，内部 `self._registry` 是空的 `{}`——此时还没有任何工具可用。  
若 `autodiscover=True`，会立刻执行下面三步（对应 `tools/tool_box.py` 的 `_load_registered_tools`）：

**Step 1 — `discover_packages(*self._packages)`**

批量 import 包下的子模块，例如 `tools.LocalTool.RAG_tool`、`tools.LocalTool.math_tool` 等。  
每个文件被 import 时，文件顶部的 `@local_tool` / `@mcp_tool` 会**当场执行**，把函数写进 `tools.registry` 里的全局字典 `_TOOL_REGISTRY`（详见 [自动发现](tool_autodiscovery.md)）。

**Step 2 — `get_registered_tools()`**

读取 `_TOOL_REGISTRY` 的一份拷贝。这是 registry 提供的公开读接口；ToolBox 不直接改全局表。

**Step 3 — 按前缀过滤，写入 `self._registry`**

只保留 `tool_path` 落在 `packages` 下的条目（如 `tools.LocalTool.RAG_tool.RAG_search_tool`），以 `{注册名: ToolInfo}` 的形式存进 `self._registry`。

**构造完成后的变化**

| 对象 | 构造前 | 构造后 |
|------|--------|--------|
| `registry._TOOL_REGISTRY`（全局） | 可能仍为空，或已有其他模块提前 import 过的工具 | 包含所有已被 import 过的工具 |
| `box._registry`（本实例） | `{}` | 仅含 `packages` 范围内的 `{name: ToolInfo}` |
| `box.list_tools()` / `box.ainvoke(...)` | 无工具可用 | 可列出、可调用已注册工具 |

之后 `list_tools()` 和 `ainvoke()` 只认 `self._registry`，不再重新扫描包。

## API

### `list_tools() -> list[dict]`

返回 OpenAI function-calling 格式的 schema 列表，供 LLM 绑工具。

```python
for schema in box.list_tools():
    fn = schema["function"]
    print(fn["name"], fn.get("description", ""))
```

内部按 `ToolInfo.tool_path` 懒加载函数，并用 docstring（或装饰器上的 `description`）生成描述。

### `ainvoke(name, args) -> ToolResult`

按**注册名**（非 `tool_path`）调用工具。同步与 `async def` 均支持。

```python
result = await box.ainvoke("integrate_function", {"func_str": "x**2", "a": 0, "b": 1})
if result.error:
    print(result.error)
else:
    print(result.output)
```

- 工具不存在或加载失败：`result.error` 有值，不抛异常
- 执行异常：捕获后写入 `result.error`
- `result.source` 为 `"local"` 或 `"mcp"`（见 [自动发现](tool_autodiscovery.md)）

### `resolve(tool_path) -> Callable`

按 `模块路径.函数名` 解析并缓存 callable，例如 `tools.LocalTool.math_tool.integrate_function`。一般供内部或调试使用；正常调用走 `ainvoke`。

## ToolResult

| 字段 | 说明 |
|------|------|
| `name` | 注册名 |
| `args` | 传入参数 |
| `output` | 成功时的返回值 |
| `error` | 失败时的错误信息 |
| `source` | `"local"` 或 `"mcp"` |
| `meta` | 扩展元数据（默认 `{}`） |

## CLI

仓库根目录下可直接试用：

```bash
python -m tools.tool_box list
python -m tools.tool_box invoke integrate_function --args '{"func_str":"x**2","a":0,"b":1}'
```
