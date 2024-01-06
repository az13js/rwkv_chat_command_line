# 这是我参考LangChain的代码后自己实现的可以在LangChain中使用的模型类。
from typing import Any, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel
from .utils import ChatRWKV

class RWKVLLM(LLM, BaseModel):

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
        """Return the type of llm."""
        return "rwkv-llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        self.rwkv.clear()
        return self.rwkv.process_input(prompt, stop=stop, add_assistant=False, **kwargs)
