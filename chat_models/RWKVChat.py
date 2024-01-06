# 这是我参考LangChain的代码后自己实现的可以在LangChain中使用的对话模型类。
from .utils import ChatRWKV
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from typing import Any, List, Optional, Iterator
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

def _convert_message_to_str(message: BaseMessage) -> str:
    if isinstance(message, ChatMessage):
        message_str = f'{message.role}: {message.content}'
    elif isinstance(message, HumanMessage):
        message_str = f'User: {message.content}'
    elif isinstance(message, AIMessage):
        message_str = f'Assistant: {message.content}'
    elif isinstance(message, SystemMessage):
        message_str = f'Instruction: {message.content}'
    else:
        raise ValueError(f"Got unknown type {message}")
    return message_str

class RWKVChat(BaseChatModel):

    streaming: bool = False
    model: str = 'RWKV-4-World-0.1B-v1-20230520-ctx4096'
    rwkv: Optional[ChatRWKV] = None

    def __init__(self, ai_model_path: Optional[str]=None, **kwargs: Any):
        super().__init__(**kwargs)
        if ai_model_path is not None:
            self.model = ai_model_path
        if self.rwkv is None:
            self.rwkv = ChatRWKV(model=self.model)

    @property
    def _llm_type(self) -> str:
        return "my-rwkv-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        input_str = '\n\n'.join([_convert_message_to_str(m) for m in messages])
        self.rwkv.clear()
        response = self.rwkv.process_input(input_str)
        generation = ChatGeneration(message=AIMessage(content=response))
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        input_str = '\n\n'.join([_convert_message_to_str(m) for m in messages])
        self.rwkv.clear()

        for output_str in self.rwkv.process_input_stream(input_str):
            chunk = AIMessageChunk(content=output_str)
            yield ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content)
