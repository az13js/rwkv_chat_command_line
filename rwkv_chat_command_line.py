# 此脚本适配RWKV4 World系列模型，可以在命令行执行与大模型的连续对话功能。
# 依赖：
# python -m pip install rwkv numpy torch torchvision
# 模型可以从这里下载：
# https://huggingface.co/BlinkDL/rwkv-4-world/tree/main
# 脚本默认加载RWKV-4-World-0.1B-v1-20230520-ctx4096，需要加载别的模型就看情况改下代码吧。
#
# 更多内容请看最下面的注释。
import os
import re
from chat_models import Mirostat # 我拿https://github.com/basusourya/mirostat的代码过来用的。Mirostat是一种对模型输出内容进行采样的算法。

class ChatRWKVEndOfText(Exception):
    pass

class ChatRWKVDoubleEnter(Exception):
    pass

class ChatRWKV:

    RWKV_TOKEN_ENDOFTEXT = 0
    RWKV_TOKEN_DOUBLEENTER = '\n\n'

    def __init__(self, model='RWKV-4-World-0.1B-v1-20230520-ctx4096', strategy='cpu fp32', mirostat_enable=False, mirostat_tau=3.0, mirostat_lr=1.0):

        if 'RWKV_JIT_ON' not in os.environ:
            os.environ['RWKV_JIT_ON'] = '1'
        if 'RWKV_CUDA_ON' not in os.environ:
            os.environ['RWKV_CUDA_ON'] = '0'

        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE

        self._rwkv_pipeline = PIPELINE(RWKV(model=model,strategy=strategy), 'rwkv_vocab_v20230424')
        self._rwkv_last_output = None
        self._rwkv_last_state = None

        self.mirostat_enable = mirostat_enable
        self.mirostat_tau = mirostat_tau
        self.mirostat_lr = mirostat_lr
        self._sample = Mirostat(self.mirostat_tau, self.mirostat_lr)

    def clear(self):
        """
        清除RWKV的状态。重置模型回到初始状态。
        """
        self._rwkv_last_output = None
        self._rwkv_last_state = None

        self._sample = Mirostat(self.mirostat_tau, self.mirostat_lr)

    def process_human_input(self, human_input, temperature=0.5, top_p=0.95, top_k=20, callback=None):
        prompt = "User: %s\n\nAssistant:"%(human_input.strip())
        bot_output = ''
        token_buffer = []
        self._rwkv_last_output, self._rwkv_last_state = self._rwkv_pipeline.model.forward(self._rwkv_pipeline.encode(prompt), self._rwkv_last_state)
        try:
            while True:
                token = self.convert_output_to_token(self._rwkv_last_output, temperature, top_p, top_k)
                token_buffer.append(token)
                maybe_utf8_str = self._rwkv_pipeline.decode(token_buffer)
                if self.str_is_ready(maybe_utf8_str):
                    bot_output = bot_output + maybe_utf8_str
                    token_buffer = []
                    if bot_output.endswith(self.RWKV_TOKEN_DOUBLEENTER):
                        raise ChatRWKVDoubleEnter()
                    if callback is not None:
                        callback(maybe_utf8_str)
                self._rwkv_last_output, self._rwkv_last_state = self._rwkv_pipeline.model.forward([token], self._rwkv_last_state)
        except ChatRWKVEndOfText:
            exit_with = '<|endoftext|>'
        except ChatRWKVDoubleEnter:
            exit_with = '\\n\\n'
        #self.save_chat_record(prompt + bot_output + exit_with)
        return bot_output.strip()

    def convert_output_to_token(self, rwkv_output, temperature, top_p, top_k):
        token = None
        if self.mirostat_enable:
            token = self._sample.choise(rwkv_output.view(rwkv_output.numel()))
        else:
            token = self._rwkv_pipeline.sample_logits(rwkv_output, temperature=temperature, top_p=top_p, top_k=top_k)
        if token == self.RWKV_TOKEN_ENDOFTEXT:
            raise ChatRWKVEndOfText()
        return token

    def str_is_ready(self, maybe_utf8_str):
        return "\ufffd" not in maybe_utf8_str

    def save_chat_record(self, new_record):
        with open('rwkv_chat_record.txt', 'a', encoding='utf8') as fp:
            fp.write(new_record)

def input_clean(human_input):
    return re.sub(r'\(temperature=\d+\.\d+\)$', '', human_input)

def get_temperature_from_input(human_input):
    temperature = 0.5 # default
    for math_content in re.findall(r'\(temperature=\d+\.\d+\)$', human_input):
        for temp in re.findall(r'\d+\.\d+', math_content):
            temperature = temp
    return float(temperature)

if __name__ == '__main__':

    try:

        # 示例：加载不同的模型。模型的扩展名得是.pth。
        #rwkv = ChatRWKV() 这种默认加载当前目录的RWKV-4-World-0.1B-v1-20230520-ctx4096.pth
        #rwkv = ChatRWKV('E:\\RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096')
        #rwkv = ChatRWKV('E:\\RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096')
        rwkv = ChatRWKV(
            'E:\\RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096',
            mirostat_enable=True, # 默认False。改成True使用Mirostat算法进行采样。用False时使用常见的Top-k和Top-p采样。
            mirostat_tau=1.1 # τ值，默认3。值越大回答的内容用词越丰富。Mirostat算法的作者在github仓库默认此值为3。但是作者仓库代码是基于GPT2模型，不同模型这个值效果是不同的，不能横向比较。
        )

        while True:
            human_input = input('\x1b[0m%s： '%('\n我'))

            # 你在对话中想要输入换行的话，输入“\”然后按下回车键。不在行末加“\”就直接按下回车键的话内容是直接输入给模型的。
            while human_input.endswith('\\'):
                human_input = re.sub(r'\\$', '\n', human_input) + input()

            # 你直接输入“clear”会清除模型的对话历史记忆。相当于重置到初始状态。
            if 'clear' == human_input:
                rwkv.clear()
                continue

            if rwkv.mirostat_enable:
                print('\x1b[32m%s：'%('机器人'), flush=True, end='')
                rwkv.process_human_input(human_input, callback=lambda word: print(word, flush=True, end=''))

            else:
                # 当mirostat_enable=False的时候，你可以在对话的结尾用“(temperature=0.2)”指定采样时使用的温度
                # （0.2可以改成你需要指定的具体温度值，默认是0.5），温度是大于0的浮点数。
                temperature = get_temperature_from_input(human_input)
                print('\x1b[32m%s\x1b[33m(temperature=%.1f)\x1b[32m：'%('机器人', temperature), flush=True, end='')
                rwkv.process_human_input(input_clean(human_input), temperature=temperature, callback=lambda word: print(word, flush=True, end=''))

    except KeyboardInterrupt:
        # 对话中你想要退出的话，按Ctrl+C，会抛出异常然后走这里，退出脚本。
        print('\x1b[0m%s'%('已退出。'), flush=True)
