# 命令行运行本地RWKV进行聊天

## 基本说明

说明内容都以注释的方式写在 rwkv_chat_command_line.py 里面了。
rwkv_chat_command_line.py 是一个不到100行的Python脚本，里面是对话的逻辑。

文件 prompt.txt 是初始化时提供给 RWKV 模型的提示信息。

仓库的代码量少，必要的参数现在是写死的，需要的话直接修改就行，应该不难。

## 脚本执行效果

以下是我本地执行，采用 1.5B 参数的 RWKV5 模型的对话效果：

![对话内容截图](review.png)

完整的命令行输出：

```
$ python rwkv_chat_command_line.py
RWKV_JIT_ON 1 RWKV_CUDA_ON 0 RESCALE_LAYER 0

Loading RWKV-5-World-1B5-v2-20231025-ctx4096.pth ...
Model detected: v5.2
Strategy: (total 24+1=25 layers)
* cpu [float32, float32], store 25 layers
0-cpu-float32-float32 1-cpu-float32-float32 2-cpu-float32-float32 3-cpu-float32-float32 4-cpu-float32-float32 5-cpu-float32-float32 6-cpu-float32-float32 7-cpu-float32-float32 8-cpu-float32-float32 9-cpu-float32-float32 10-cpu-float32-float32 11-cpu-float32-float32 12-cpu-float32-float32 13-cpu-float32-float32 14-cpu-float32-float32 15-cpu-float32-float32 16-cpu-float32-float32 17-cpu-float32-float32 18-cpu-float32-float32 19-cpu-float32-float32 20-cpu-float32-float32 21-cpu-float32-float32 22-cpu-float32-float32 23-cpu-float32-float32 24-cpu-float32-float32
emb.weight                        f32      cpu  65536  2048
blocks.0.ln1.weight               f32      cpu   2048
blocks.0.ln1.bias                 f32      cpu   2048
blocks.0.ln2.weight               f32      cpu   2048
blocks.0.ln2.bias                 f32      cpu   2048
blocks.0.att.time_mix_k           f32      cpu   2048
blocks.0.att.time_mix_v           f32      cpu   2048
blocks.0.att.time_mix_r           f32      cpu   2048
blocks.0.att.time_mix_g           f32      cpu   2048
blocks.0.att.time_decay           f32      cpu     32    64
blocks.0.att.time_first           f32      cpu     32    64
blocks.0.att.receptance.weight    f32      cpu   2048  2048
blocks.0.att.key.weight           f32      cpu   2048  2048
blocks.0.att.value.weight         f32      cpu   2048  2048
blocks.0.att.output.weight        f32      cpu   2048  2048
blocks.0.att.gate.weight          f32      cpu   2048  2048
blocks.0.att.ln_x.weight          f32      cpu   2048
blocks.0.att.ln_x.bias            f32      cpu   2048
blocks.0.ffn.time_mix_k           f32      cpu   2048
blocks.0.ffn.time_mix_r           f32      cpu   2048
blocks.0.ffn.key.weight           f32      cpu   2048  7168
blocks.0.ffn.receptance.weight    f32      cpu   2048  2048
blocks.0.ffn.value.weight         f32      cpu   7168  2048
....................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
blocks.23.ln1.weight              f32      cpu   2048
blocks.23.ln1.bias                f32      cpu   2048
blocks.23.ln2.weight              f32      cpu   2048
blocks.23.ln2.bias                f32      cpu   2048
blocks.23.att.time_mix_k          f32      cpu   2048
blocks.23.att.time_mix_v          f32      cpu   2048
blocks.23.att.time_mix_r          f32      cpu   2048
blocks.23.att.time_mix_g          f32      cpu   2048
blocks.23.att.time_decay          f32      cpu     32    64
blocks.23.att.time_first          f32      cpu     32    64
blocks.23.att.receptance.weight   f32      cpu   2048  2048
blocks.23.att.key.weight          f32      cpu   2048  2048
blocks.23.att.value.weight        f32      cpu   2048  2048
blocks.23.att.output.weight       f32      cpu   2048  2048
blocks.23.att.gate.weight         f32      cpu   2048  2048
blocks.23.att.ln_x.weight         f32      cpu   2048
blocks.23.att.ln_x.bias           f32      cpu   2048
blocks.23.ffn.time_mix_k          f32      cpu   2048
blocks.23.ffn.time_mix_r          f32      cpu   2048
blocks.23.ffn.key.weight          f32      cpu   2048  7168
blocks.23.ffn.receptance.weight   f32      cpu   2048  2048
blocks.23.ffn.value.weight        f32      cpu   7168  2048
ln_out.weight                     f32      cpu   2048
ln_out.bias                       f32      cpu   2048
head.weight                       f32      cpu   2048 65536

我： 用“晴朗”造句。
机器人： 今天的天空晴朗，阳光明媚。

我： 用你造的句子扩充为一段小作文。
机器人： 今天，阳光明媚，天空湛蓝。鸟儿在树枝间欢快地歌唱，微风轻拂着我的头发，我感到自己的心情也变得轻松愉悦。我走在路 上，看着周围的美景，感到心情愉悦，仿佛整个世界都在为我欢迎。这是一个美好的早晨，让我心情舒畅，充满了生命的活力。

我： 继续你的创作，增加一些对路上的风景的详细描述。
机器人： 今天的阳光明媚，微风轻拂着我的头发，我感到心情愉悦，仿佛整个世界都在为我欢迎。沿着小路向前走，伴随着清新的空气 ，我看见了许多美丽的景色。路上有许多花园和树木，它们在微风中摇曳着，给人们带来了一种宁静和放松。我走过一条小溪，看见了一些小鱼在水里游来游去，它们的身影在水面上显得格外清晰。我感到自己的心情变得更加愉悦，仿佛整个世界在为我欢迎。

我： 
```
