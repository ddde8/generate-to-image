from openai import OpenAI
from dotenv import load_dotenv
import os
from PIL import Image
import base64

# Load API Key
load_dotenv()
api_key = os.getenv('OPEN_API_KEY')
client = OpenAI(api_key=api_key)

# 사용자 입력 받기
product_name = input("제품 이름을 입력하세요: ")
image_path = input("제품 이미지 파일 경로를 입력하세요 (예: './image.jpg'): ")

# 이미지 파일을 base64로 인코딩
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 이미지 인코딩
base64_image = encode_image_to_base64(image_path)

# GPT 요청
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.7,
    messages=[
        {
            "role": "system",
            "content": (
                '오직 다음 JSON만 출력해. 다른 설명/코멘트/마크다운 금지.\n'
                '형식: {\n'
                '  "product": {\n'
                '    "type": "<제품 종류>",\n'
                '    "material": "<재질>",\n'
                '    "design": "<디자인 요약>",\n'
                '    "features": "<브랜드/각인/색상 등 특징>"\n'
                '  },\n'
                '  "background": {\n'
                '    "ideal_color": "<배경 색상>",\n'
                '    "texture": "<배경 질감>",\n'
                '    "lighting": "<조명 스타일>",\n'
                '    "style": "<연출 스타일>"\n'
                '  },\n'
                '  "layout": {\n'
                '    "subject_layout": {"center":[cx,cy],"ratio":[rw,rh]},\n'
                '    "nongraphic_layout":[{"type":"headline","bbox":[x,y,w,h]},...],\n'
                '    "graphic_layout":[{"type":"logo","content":"...","bbox":[x,y,w,h]},...]\n'
                '  }\n'
                '}\n\n'
                '이미지 안에 텍스트나 로고가 없어도,\n'
                '나중에 headline이나 logo 같은 요소를 추가할 수 있는\n'
                '적절한 배치 위치를 layout에 유추해서 채워라.\n'
                '빈 배열은 절대 출력하지 말고, 예측해서 제안된 위치를 포함해라.'
            )

        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"제품 이름: {product_name}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    max_tokens=1000
)

# 응답 출력
print(response.choices[0].message.content)