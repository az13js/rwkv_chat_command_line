# 此脚本适配RWKV4/5 World系列模型，可以在命令行执行与大模型的连续对话功能。
# 依赖：
# python -m pip install rwkv numpy torch torchvision
# 模型可以从这里下载：
# https://huggingface.co/BlinkDL/rwkv-4-world/tree/main
# 脚本默认加载RWKV-4-World-0.1B-v1-20230520-ctx4096，需要加载别的模型就看情况改下代码吧。
#
# 更多内容请看最下面的注释。
import re
from ChatRWKV import ChatRWKV

def input_clean(human_input):
    return re.sub(r'\(temperature=\d+\.\d+\)$', '', human_input)

def get_temperature_from_input(human_input):
    temperature = 0.5 # default
    for math_content in re.findall(r'\(temperature=\d+\.\d+\)$', human_input):
        for temp in re.findall(r'\d+\.\d+', math_content):
            temperature = temp
    return float(temperature)

def read_prompt(file):
    with open(file, 'r', encoding='utf8') as fp:
        return fp.read()

def init_rwkv_with_prompt(rwkv):
    rwkv.clear()
    prompt = read_prompt('prompt.txt').strip()
    if len(prompt) > 0:
        rwkv.run_with(prompt)

if __name__ == '__main__':

    try:

        # 示例：加载不同的模型。模型的扩展名得是.pth。
        #rwkv = ChatRWKV() 这种默认加载当前目录的RWKV-4-World-0.1B-v1-20230520-ctx4096.pth
        #rwkv = ChatRWKV('E:\\RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096')
        #rwkv = ChatRWKV('E:\\RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096')
        rwkv = ChatRWKV(
            'RWKV-5-World-1B5-v2-20231025-ctx4096',
            mirostat_enable=True, # 默认False。改成True使用Mirostat算法进行采样。用False时使用常见的Top-k和Top-p采样。
            mirostat_tau=1.0 # τ值，默认3。值越大回答的内容用词越丰富。Mirostat算法的作者在github仓库默认此值为3。但是作者仓库代码是基于GPT2模型，不同模型这个值效果是不同的，不能横向比较。
        )
        init_rwkv_with_prompt(rwkv)

        while True:
            human_input = input('\x1b[0m%s： '%('\n我'))

            # 你在对话中想要输入换行的话，输入“\”然后按下回车键。不在行末加“\”就直接按下回车键的话内容是直接输入给模型的。
            while human_input.endswith('\\'):
                human_input = re.sub(r'\\$', '\n', human_input) + input()

            # 你直接输入“clear”会清除模型的对话历史记忆。相当于重置到初始状态。
            if 'clear' == human_input:
                init_rwkv_with_prompt(rwkv)
                continue

            if rwkv.mirostat_enable:
                print('\x1b[32m%s：'%('机器人'), flush=True, end='')
                rwkv.process_human_input(human_input, callback=lambda word: print(word, flush=True, end=''))

            else:
                # 当使用常见的Top-k和Top-p采样的时候，你可以在对话的结尾用“(temperature=0.2)”的格式指定采样时使用的温度
                # （0.2可以改成你需要指定的具体温度值，默认是0.5），温度是大于0的浮点数。
                temperature = get_temperature_from_input(human_input)
                print('\x1b[32m%s\x1b[33m(temperature=%.1f)\x1b[32m：'%('机器人', temperature), flush=True, end='')
                rwkv.process_human_input(input_clean(human_input), temperature=temperature, callback=lambda word: print(word, flush=True, end=''))

    except KeyboardInterrupt:
        # 对话中你想要退出的话，按Ctrl+C，会抛出异常然后走这里，退出脚本。
        print('\x1b[0m%s'%('已退出。'), flush=True)
