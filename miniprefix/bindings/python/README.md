# Python 接口

状态：规划目录，尚无可 import 的 miniprefix Python 包，也不引入绑定依赖。

负责文本/配置适配、C++ Engine 生命周期与事件迭代器，支持参考校验和实验编排。
迁移初期可用原 worker IPC 适配已有场景；最终接口不在 Python 中再维护一套调度器。
绑定技术在原生 Engine 可用后确定，错误、取消与 shutdown 遵循统一生命周期契约。
