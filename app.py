from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")


from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

# print(model.invoke(messages))

# for token in model.stream(messages):
#     print(token.content, end="|")

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]  # 롤 공부하기
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

response = model.invoke(prompt)
print(response.content)