# 我自己封装的用于对话，提供给rwkv_chat_command_line.py脚本。

import os

# 我拿https://github.com/basusourya/mirostat的代码过来用的。Mirostat是一种对模型输出内容进行采样的算法。
from chat_models import Mirostat

class ChatRWKVEndOfText(Exception):
    pass

class ChatRWKVDoubleEnter(Exception):
    pass

class ChatRWKVStopWordUser(Exception):
    pass

class ChatRWKVUnknowCharacterTooLong(Exception):
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
        self._rwkv_last_state = None

        self.mirostat_enable = mirostat_enable
        self.mirostat_tau = mirostat_tau
        self.mirostat_lr = mirostat_lr
        self._sample = Mirostat(self.mirostat_tau, self.mirostat_lr)

    def clear(self):
        """
        清除RWKV的状态。重置模型回到初始状态。
        """
        self._rwkv_last_state = None

        self._sample = Mirostat(self.mirostat_tau, self.mirostat_lr)

    def process_human_input(self, human_input, temperature=0.5, top_p=0.95, top_k=20, callback=None):
        prompt = "User: %s\n\nAssistant:"%(human_input.strip())
        if self._rwkv_last_state is not None:
            prompt = '\n\n' + prompt
        bot_output = ''
        token_buffer = []
        self.save_chat_record(prompt)
        rwkv_last_output, self._rwkv_last_state = self._rwkv_pipeline.model.forward(self._rwkv_pipeline.encode(prompt), self._rwkv_last_state)
        try:
            while True:
                token = self.convert_output_to_token(rwkv_last_output, temperature, top_p, top_k)
                token_buffer.append(token)
                maybe_utf8_str = self._rwkv_pipeline.decode(token_buffer)
                if self.str_is_ready(maybe_utf8_str):
                    bot_output = bot_output + maybe_utf8_str
                    token_buffer = []
                    if bot_output.endswith(self.RWKV_TOKEN_DOUBLEENTER):
                        raise ChatRWKVDoubleEnter()
                    if bot_output.strip().endswith('User') or bot_output.strip().endswith('User:'):
                        raise ChatRWKVStopWordUser()
                    self.save_chat_record(maybe_utf8_str)
                    if callback is not None:
                        callback(maybe_utf8_str)
                elif len(token_buffer) > 20:
                    raise ChatRWKVUnknowCharacterTooLong()

                rwkv_last_output, self._rwkv_last_state = self._rwkv_pipeline.model.forward([token], self._rwkv_last_state)
        except ChatRWKVEndOfText:
            pass
        except ChatRWKVDoubleEnter:
            pass
        except ChatRWKVStopWordUser:
            pass
        except ChatRWKVUnknowCharacterTooLong:
            pass
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
        # 需要保存聊天记录的时候可以不return
        return
        with open('rwkv_chat_record.txt', 'a', encoding='utf8') as fp:
            fp.write(new_record)

    def run_with(self, prompt):
        self.save_chat_record(prompt)
        rwkv_last_output, self._rwkv_last_state = self._rwkv_pipeline.model.forward(self._rwkv_pipeline.encode(prompt), self._rwkv_last_state)
        return rwkv_last_output, self._rwkv_last_state

