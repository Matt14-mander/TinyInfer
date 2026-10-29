# 非整页前缀复用成本与收益

每个组合 6 次无耗时插桩测量，另有 1 次 profiling；模型 FP32，页长 16，输出 8 tokens。
时间包含 submit/step IPC 和事件交付，不包含加载、tokenization、warmup、prime、统计查询与 clear。
吞吐是单请求输出 token/s，不代表并发饱和吞吐。CPU/CUDA 顺序执行；下表为中位数，原始 JSON 保留范围与逐次结果。

|后端|负载|模式|复用 tokens|TTFT ms|TPOT ms|输出 tok/s|COW 页|COW MiB|复制 ms*|查找 µs*|发布 µs*|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|cuda|identical-tail|off|0|18.976|7.869|108.67|0|0.000|0.000|0.20|2.40|
|cuda|identical-tail|cold|0|22.802|8.428|97.80|6|1.318|6.377|0.60|25.60|
|cuda|identical-tail|hot|30|8.471|7.396|131.20|1|0.615|1.612|3.90|115.50|
|cuda|multi-turn|off|0|27.420|9.001|88.70|0|0.000|0.000|0.30|2.80|
|cuda|multi-turn|cold|0|26.987|9.324|87.35|7|3.691|8.459|1.20|32.00|
|cuda|multi-turn|hot|38|9.414|8.769|112.76|8|3.955|14.811|5.50|44.50|
|cuda|in-page-branch|off|0|19.160|8.195|104.61|0|0.000|0.000|0.30|3.00|
|cuda|in-page-branch|cold|0|18.520|9.259|96.00|6|1.318|7.578|0.80|31.40|
|cuda|in-page-branch|hot|17|11.635|9.011|106.09|7|1.362|6.934|2.60|40.30|
|cuda|zero-sharing|off|0|18.215|8.141|106.38|0|0.000|0.000|0.30|2.80|
|cuda|zero-sharing|cold|0|18.835|8.443|102.66|6|1.318|5.677|0.70|17.40|
|cuda|zero-sharing|hot|0|17.407|8.543|103.12|6|1.318|13.508|1.70|38.30|

*耗时来自单独 profiling 轮次，不能与无插桩延迟直接相减；CUDA 复制区间由同一 stream 的 event 测量，可能包含提交间隙。
查找/发布为 KVManager 主机时间，包含对应元数据及引用操作；复制字节为所有层 K+V 的有效部分，不包含未使用尾部。

|后端|负载|off/hot TTFT 配对加速比中位数|hot/off 吞吐配对加速比中位数|
|---|---|---:|---:|
|cuda|identical-tail|2.372|1.320|
|cuda|multi-turn|2.984|1.159|
|cuda|in-page-branch|1.443|0.985|
|cuda|zero-sharing|0.995|0.963|

加速比小于 1 表示退化。hot 的预填充/生成 prime 成本未计入，热缓存收益需要后续请求摊销。
cold 相对 off 同时反映缓存发布、COW 和索引开销；本实验没有旧版整页缓存对照，不能将全部热缓存收益归因于非整页新增能力。

原始数据：[JSON](partial-prefix-benefit-cuda-confirm.json)。模型 SHA256：`fe826f487690854ac806197b60edffc2c76c6c446eede6b188e47763deae7aef`。
