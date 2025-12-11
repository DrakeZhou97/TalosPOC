from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

load_dotenv()


class PromptLogger(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        # messages 是一批对话, 每个元素是 List[BaseMessage]
        for i, batch in enumerate(messages):
            print(f"[batch {i}] === outgoing messages ===")
            for m in batch:
                print(f"{m.type}: {getattr(m, 'content', m)}")
        print("END OF PROMPTS")
        # 对于 LLM (非 chat) 可用 on_llm_start, 拿 kwargs["prompts"]


MAX_MODEL = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    # base_url="https://api.openai.com/v1",
    top_p=0.95,
    seed=42,
    callbacks=[PromptLogger()],
)

MEDIUM_MODEL = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    base_url="https://yunwu.ai/v1",
    top_p=0.95,
    seed=42,
)

SMALL_MODEL = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    base_url="https://yunwu.ai/v1",
    top_p=0.95,
    seed=42,
)
