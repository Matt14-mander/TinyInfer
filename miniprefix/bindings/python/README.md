# Python 接口

状态：client.py 提供 NativeEngine 本地 IPC 客户端，可通过仓库 namespace package
导入；不在 Python 代跑模型或调度。原生扩展绑定技术仍未选定。

负责文本/配置适配、C++ Engine 生命周期与事件迭代器，支持参考校验和实验编排。
迁移初期可用原 worker IPC 适配已有场景；最终接口不在 Python 中再维护一套调度器。
绑定技术在原生 Engine 可用后确定，错误、取消与 shutdown 遵循统一生命周期契约。

请求支持 `timeout_ms`，构造函数支持等待队列、事件槽和历史 ID 容量/TTL；`lifecycle_stats()` 提供当前计数。超时在 step 边界检查，历史 ID 在消费终态后进入有限去重窗口，详见[生命周期语义](../../docs/request_lifecycle.md)。
