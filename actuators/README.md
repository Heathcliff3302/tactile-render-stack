# Layer 5 Actuators

该目录存放执行器命令、测量、传递函数、逆模型和嵌入式运行时。Phase 2 先完成单执行器闭环，再扩展到多通道输出。

K1 没有接硬件。`single_axis.py` 的测量值由声明的传递模型产生，记录带 `model_id = single_axis_ideal_first_order_v0` 与 `status = modelled_not_measured`。它的作用是让命令路径、逆模型、饱和、延迟和看门狗成为真实代码与真实记录，K5 只需替换传递模型，不必改动层 1 到层 4。

传递模型：`measured(t) = gain * commanded(t - latency_steps) + bias`，并按声明量程饱和。逆模型是它的精确逆，因此 K1 唯一可见的误差来自声明的延迟。
