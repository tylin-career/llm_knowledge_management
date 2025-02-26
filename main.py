from langchain_openai import ChatOpenAI
from config import LLM_PROVIDER, OPENAI_API_KEY


def get_llm():
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=OPENAI_API_KEY,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True,
            # base_url="...",
            # organization="...",
            # other params...
        )
    elif LLM_PROVIDER == "ollama":
        return ChatOpenAI(
            model="llama3.1",
            openai_api_key="ollama",
            openai_api_base="http://10.96.196.63:11434/v1/",
            streaming=True,
        )
    else:
        raise ValueError("不支援的 LLM 提供者，請使用 'ollama' 或 'openai'")


user_inputs = f'''
Christmas
A Christian holiday signifying the birth of Jesus, Christmas is widely celebrated and enjoyed across the United States and the world. The holiday always falls on 25 December (regardless of the day of the week), and is typically accompanied by decorations, presents, and special meals.

Specifically, the legend behind Christmas (and the one that most children are told) is that Santa Claus, a bearded, hefty, jolly, and red-jacket-wearing old man who lives in the North Pole, spends the year crafting presents with his elves, or small, festive, excited Santa-assistants. All the children who behave throughout the year are admitted to the Good List, and will presumably receive their desired gifts on Christmas, while those who don't behave are placed on the Naughty List, and will presumably (although the matter is determined by parents) receive a lump of coal.

Santa Claus is said to fly around the Christmas sky in a sled powered by his magical reindeer, or cold-resistant, mythically powered, individually named animals, delivering presents to each child's house in the process. Santa is also expected to slide through chimneys to deliver these presents (homes not equipped with chimneys might "leave the front door cracked open"), and children sometimes arrange cookies or other treats on a plate for him to enjoy.

Gifts are placed underneath a Christmas tree, or a pine tree that's decorated with ornaments and/or lights and is symbolic of the holiday. Additionally, smaller gifts may be placed inside a stocking, or a sock-shaped, holiday-specific piece of fabric that's generally hung on the mantle of a fireplace (homes without fireplaces might use the wall). A Christmas tree's ornaments, or hanging, typically spherical decorations, in addition to the mentioned lights, may be accompanied by a star, or a representation of the Star of Jerusalem that the Three Apostles followed while bringing Baby Jesus gifts and honoring him, in the Bible."""),
'''
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to 繁體中文. Translate the user sentence.",
    ),
    ("human", user_inputs)
]


# 呼叫模型並逐步輸出
llm = get_llm()
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)  # 逐字輸出