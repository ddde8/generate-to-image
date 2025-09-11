import os
import base64
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

# 이미지 파일을 바이트로 읽는 함수
def read_image_to_bytes(image_path):
    """
    주어진 경로의 이미지 파일을 바이트로 읽어 반환합니다.
    파일을 찾을 수 없는 경우 None을 반환합니다.
    """
    try:
        with open(image_path, "rb") as image_file:
            return image_file.read()
    except FileNotFoundError:
        print(f"오류: '{image_path}' 파일을 찾을 수 없습니다.")
        return None

# 사용자 입력 받기
product_name = input("제품 이름을 입력하세요: ")
image_path = input("제품 이미지 파일 경로를 입력하세요 (예: './image.jpg'): ")

# 이미지 파일을 바이트로 읽기
image_data = read_image_to_bytes(image_path)
if image_data is None:
    exit()

# Gemini 모델 설정
model_name = "models/gemini-2.5-flash-image-preview"
model = genai.GenerativeModel(model_name)

# GPT 요청의 'system' 역할을 'text'로 통합
system_instruction = (
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

# prompt에 시스템 프롬프트, 사용자 입력 텍스트, 이미지를 함께 포함
prompt_parts = [
    system_instruction,
    f"제품 이름: {product_name}",
    {
        "mime_type": "image/jpeg",
        "data": image_data
    }
]

# 응답 생성
print("Gemini API에 요청 중...")
response = model.generate_content(prompt_parts)

# 응답 출력
print(response.text)
