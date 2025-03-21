import base64
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
from typing import Any
from pydantic import BaseModel



class ImageSummarizer:

    def __init__(self, image_path) -> None:
        self.image_path = image_path
        self.prompt = """
            You are an assistant tasked with summarizing images for retrieval.
            These summaries will be embedded and used to retrieve the raw image.
            Give a concise summary of the image that is well optimized for retrieval.
            請用繁體中文
        """

    def base64_encode_image(self):
        with open(self.image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def summarize(self, prompt = None):
        base64_image_data = self.base64_encode_image()
        chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000)

        # gpt4 vision api doc - https://platform.openai.com/docs/guides/vision
        response = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": prompt if prompt else self.prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image_data}"},
                        },
                    ]
                )
            ]
        )
        return base64_image_data, response.content
    

class Element(BaseModel):
    type: str
    text: Any