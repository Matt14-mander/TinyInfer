# Scheduler

状态：规划目录，当前参考在 examples/serving_scheduler 的两个 Python scheduler。

负责 active/waiting/suspended、iteration-boundary refill、decode 优先、chunk/token
预算、动态准入和抢占选择。输出 SchedulePlan，不执行 kernel，不直接改物理 block table。
容量试算通过 KVManager；重算也是显式有预算的工作。

首批交付：确定性计划接口与现有手算轨迹回归，随后接入动态容量和恢复状态。
参考 Python 实现继续作策略 oracle，不在新目录复制一个长期独立版本。
