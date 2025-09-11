import google.generativeai as genai
import os
import mimetypes
from dotenv import load_dotenv
from PIL import Image #pillow 라이브러리
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI

#환경 변수 로드 
load_dotenv()
#API 키 불러오기 
api_key = os.getenv("GEMINI_API_KEY")

'''
#api key 확인용 
assert api_key is not None, "GEMINI_API_KEY not found in environment variables"
print("API key loaded successfully.")
'''
# API 키 설정
genai.configure(api_key=api_key)

# 모델 생성
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-image-preview",
    system_instruction="당신은 광고 기획자입니다. 광고 기획자로써 이미지에 가장 적절한 광고 문구를 3가지 말해주세요. 그리고 광고 문구는 최대 8글자입니다. 그리고 어떤 배경 이미지와 잘 어울릴까요? 배경에 사용될 사물이나 질감, 컬러 등을 포함해서 말해주세요"
                # "오직 다음 JSON만 출력해. 다른 설명/코멘트/마크다운 금지.\n"
                # "형식: {\n"
                # '  "product": {\n'
                # '    "type": "<제품 종류>",\n'
                # '    "material": "<재질>",\n'
                # '    "design": "<디자인 요약>",\n'
                # '    "features": "<브랜드/각인/색상 등 특징>"\n'
                # "  },\n"
                # '  "background": {\n'
                # '    "ideal_color": "<배경 색상>",\n'
                # '    "texture": "<배경 질감>",\n'
                # '    "lighting": "<조명 스타일>",\n'
                # '    "style": "<연출 스타일>"\n'
                # "  },\n"
                # '  "layout": {\n'
                # '    "subject_layout": {"center": [cx, cy], "ratio": [rw, rh]},\n'
                # '    "nongraphic_layout": [{"type": "headline", "bbox": [x, y, w, h]}, ...],\n'
                # '    "graphic_layout": [{"type": "logo", "content": "...", "bbox": [x, y, w, h]}, ...]\n'
                # "  }\n"
                # "}\n\n"
                # "이미지 안에 텍스트나 로고가 없어도,\n"
                # "나중에 headline이나 logo 같은 요소를 추가할 수 있는\n"
                # "적절한 배치 위치를 layout에 유추해서 채워라.\n"
                # "빈 배열은 절대 출력하지 말고, 예측해서 제안된 위치를 포함해라."
                )

'''
image = Image.open("/Users/jieunchoi/Documents/GitHub/generate-to-image/123.jpeg")
# 콘텐츠 생성
response = model.generate_content([
    image,
    "이것은 어떤 객체인가요?"
])
print(response.text)
'''

def get_product_info():
    """
    사용자로부터 상품명과 이미지 경로를 입력받아 이미지와 함께 반환.
    이미지 파일이 유효하지 않으면 예외 처리 후 None 반환.
    
    Returns:
        tuple: (상품명(str), 이미지(PIL.Image 객체)) 또는 (None, None)
    """
    product_name = input("상품명을 입력하세요: ")
    image_path = input("광고 이미지 파일 경로를 입력하세요 (예: ./image.jpg): ").strip()
    image = Image.open(image_path)
    return product_name, image
    '''
    if not os.path.isfile(image_path):
        print("❌ 이미지 파일이 존재하지 않습니다. 경로를 다시 확인하세요.")
        return None, None

    try:
        image = Image.open(image_path)
        print(f"✅ '{product_name}' 이미지가 성공적으로 열렸습니다.")
        image.show()
        return product_name, image
    except Exception as e:
        print(f"❌ 이미지 파일을 열 수 없습니다: {e}")
        return None, None
    '''

if __name__ == "__main__":
    product_name, image = get_product_info()
    if product_name is None or image is None:
        print("입력이 유효하지 않습니다. 프로그램을 종료합니다.")
    else:
        response = model.generate_content([
            image,
            "이것은 어떤 객체인가요?"
        ])
        print("\n🧠 모델 응답:")
        print(response.text)