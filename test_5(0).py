from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client()

prompt = (
    "이미지와 어울리는 배경 이미지 생성"
    "전체 무드를 주어진 이미지와 잘 어울리도록 조정",
)

image = Image.open("/Users/jieunchoi/Documents/GitHub/generate-to-image/123.jpeg")

response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents=[prompt, image],
)

for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save("generated_image.png")