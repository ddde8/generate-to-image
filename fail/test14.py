from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client()

prompt = (
    "광고 문구 1: 다시 피어날"
    "따뜻하고 부드러운 빛이 감도는 파스텔 톤의 꽃밭에서 만개하기 직전의 꽃봉오리처럼 펜던트가 피어나는 모습입니다.",

    "광고 문구 2: 시간을 넘어"
    "고요하고 신비로운 분위기의 고대 유적지, 이끼 낀 돌담 사이에서 펜던트가 시간을 초월한 보물처럼 빛나는 모습입니다.",

    "광고 문구 3: 찬란한 귀환"
    "우아하고 세련된 모던한 공간, 혹은 잔잔한 물결이 일렁이는 수면 위에서 펜던트가 찬란하게 빛을 반사하며 시선을 사로잡는 모습입니다.",

    "광고 문구 3가지에 해당하는 이미지를 각각 생성해줘"
)

response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents=[prompt],
)

for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save("generated_image2.png")