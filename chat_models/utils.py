import os
import re

class ChatRWKVEndOfText(Exception):
    pass

class ChatRWKVDoubleEnter(Exception):
    pass

class ChatRWKV:

    RWKV_TOKEN_ENDOFTEXT = 0
    RWKV_TOKEN_DOUBLEENTER = '\n\n'

    def __init__(self, model='RWKV-4-World-0.1B-v1-20230520-ctx4096', strategy='cpu fp32'):

        if 'RWKV_JIT_ON' not in os.environ:
            os.environ['RWKV_JIT_ON'] = '1'
        if 'RWKV_CUDA_ON' not in os.environ:
            os.environ['RWKV_CUDA_ON'] = '0'

        try:
            from rwkv.model import RWKV
            from rwkv.utils import PIPELINE
        except ImportError:
            raise ImportError(
                "Could not import rwkv python package. "
                "Please install it with `pip install rwkv`."
            )

        self._rwkv_pipeline = PIPELINE(RWKV(model=model,strategy=strategy), 'rwkv_vocab_v20230424')
        self._rwkv_last_output = None
        self._rwkv_last_state = None

    def clear(self):
        self._rwkv_last_output = None
        self._rwkv_last_state = None

    def process_input(self, input_prompt, **kwargs):
        bot_output = ''
        for output_str in self.process_input_stream(input_prompt, **kwargs):
            bot_output = bot_output + output_str
        return bot_output

    def process_input_stream(self, input_prompt, temperature=0.5, top_p=0.95, top_k=20, add_assistant=True, max_len=None, stop=None):
        prompt = '%s\n\nAssistant:'%(re.sub(r'[\n|\r]+', '\n', input_prompt).strip()) if add_assistant else input_prompt
        bot_output = ''
        token_buffer = []
        self.save_chat_record(prompt)
        self._rwkv_last_output, self._rwkv_last_state = self._rwkv_pipeline.model.forward(self._rwkv_pipeline.encode(prompt), self._rwkv_last_state)
        try:
            while True:
                token = self.convert_output_to_token(self._rwkv_last_output, temperature, top_p, top_k)
                token_buffer.append(token)
                maybe_utf8_str = self._rwkv_pipeline.decode(token_buffer)
                if self.str_is_ready(maybe_utf8_str):
                    bot_output = bot_output + maybe_utf8_str
                    token_buffer = []
                    if stop is not None and stop == maybe_utf8_str:
                        exit_with = '<|stop|>'
                        break
                    if max_len is not None and len(bot_output) > max_len:
                        exit_with = '<|maxlen|>'
                        break
                    if add_assistant and bot_output.endswith(self.RWKV_TOKEN_DOUBLEENTER):
                        raise ChatRWKVDoubleEnter()
                    self.save_chat_record(maybe_utf8_str)
                    yield maybe_utf8_str
                self._rwkv_last_output, self._rwkv_last_state = self._rwkv_pipeline.model.forward([token], self._rwkv_last_state)
        except ChatRWKVEndOfText:
            exit_with = '<|endoftext|>'
        except ChatRWKVDoubleEnter:
            exit_with = '\\n\\n'
        self.save_chat_record(exit_with)

    def convert_output_to_token(self, rwkv_output, temperature, top_p, top_k):
        token = self._rwkv_pipeline.sample_logits(rwkv_output, temperature=temperature, top_p=top_p, top_k=top_k)
        if token == self.RWKV_TOKEN_ENDOFTEXT:
            raise ChatRWKVEndOfText()
        return token

    def str_is_ready(self, maybe_utf8_str):
        return "\ufffd" not in maybe_utf8_str

    def save_chat_record(self, new_record):
        with open('rwkv_chat_record.txt', 'a', encoding='utf8') as fp:
            fp.write(new_record)
