import os
import google.generativeai as genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import io

# 1. 환경 변수 로드 및 설정
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)


# 3. 사용자 입력 받기
def get_product_info():
    """
    사용자로부터 상품명과 이미지 경로를 입력받아 이미지와 함께 반환.
    """
    product_name = input("상품명을 입력하세요: ").strip()
    image_path = input("광고 이미지 파일 경로를 입력하세요 (예: ./image.jpg): ").strip()

    try:
        image = Image.open(image_path)
        print(f"✅ 이미지 불러오기 성공: {image_path}")
        return product_name, image
    except Exception as e:
        print(f"❌ 이미지 열기 실패: {e}")
        return None, None


# 4. 프롬프트 템플릿 정의 (이미지는 텍스트가 아니라 별도 전달할 것이므로 텍스트에 포함하지 않음)
prompt_template = PromptTemplate.from_template(
    template="""
당신은 광고 기획자입니다. 
광고 기획자로써 "{product_name}"에 가장 적절한 광고 문구를 3가지 말해주세요. 
광고 문구는 최대 8글자입니다. 
그리고 어떤 배경 이미지와 잘 어울릴까요? 
배경에 사용될 사물이나 질감, 컬러 등을 포함해서 말해주세요.
"""
)

# 2. Gemini 모델 초기화
llm = genai.GenerativeModel("gemini-2.5-flash-image-preview")



if __name__ == "__main__":
    product_name, image = get_product_info()

    # 프롬프트 텍스트 생성
    prompt_text = prompt_template.format(product_name=product_name)

    # Gemini에 이미지 + 텍스트 프롬프트 함께 전달
    print("🧠 Gemini 모델에 질의 중...")
    response = llm.generate_content([image, prompt_text])

    # 결과 출력
    print("\n📝 생성된 광고 문구 및 배경 제안:\n")
    print(response.text)

