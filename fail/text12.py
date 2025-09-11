import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from google import genai
from google.genai import types
import google.generativeai as genai2

# 1. 환경 변수 로드 및 API 키 설정
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai2.configure(api_key=api_key)

# 2. Gemini 모델 초기화
model = genai2.GenerativeModel("gemini-2.5-flash-image-preview")

def get_product_info():
    """상품명과 이미지 경로를 입력받아 이미지 객체 반환"""
    product_name = input("상품명을 입력하세요: ").strip()
    image_path = input("광고 이미지 파일 경로를 입력하세요 (예: ./image.jpg): ").strip()

    try:
        image = Image.open(image_path)
        print(f"✅ 이미지 불러오기 성공: {image_path}")
        return product_name, image
    except Exception as e:
        print(f"❌ 이미지 열기 실패: {e}")
        return None, None

def generate_ad_text(product_name, image):
    """상품명과 이미지로 광고 문구 + 배경 설명 생성"""
    prompt_template = PromptTemplate.from_template(
        """
        당신은 광고 기획자입니다. "{product_name}"에 어울리는 광고 문구 3가지와 적절한 배경 스타일을 설명해 주세요. 
        문구는 최대 8글자. 배경에 어울리는 사물, 질감, 색상 등을 포함해서 설명해 주세요.
        """
    )
    prompt_text = prompt_template.format(product_name=product_name)

    # 이미지 객체를 JPEG 바이트로 변환
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    # 이미지와 텍스트를 types.Content 형태로 감싸서 전달
    contents = [
        types.Content(type="image", image=types.Image(data=image_bytes)),
        types.Content(type="text", text=prompt_text)
    ]

    response = model.generate_content(contents)
    ad_text = response.text.strip()

    print("\n📝 광고 문구 및 배경 설명:\n")
    print(ad_text)

    return ad_text

if __name__ == "__main__":
    product_name, image = get_product_info()

    if product_name is None or image is None:
        print("상품명 또는 이미지가 올바르지 않습니다. 프로그램을 종료합니다.")
        exit()

    # 광고 문구 생성
    ad_text = generate_ad_text(product_name, image)

    # 이미지 생성용 클라이언트 초기화 (다른 클라이언트 객체)
    client = genai.Client()

    prompt = f"{ad_text}의 지시사항을 따라서 3개의 이미지를 만들어줘"

    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[prompt],
    )

    image_count = 0
    for part in response.candidates[0].content.parts:
        if part.text:
            print(part.text)
        elif part.inline_data:
            generated_image = Image.open(BytesIO(part.inline_data.data))
            filename = f"generated_image_{image_count}.png"
            generated_image.save(filename)
            print(f"✅ 이미지 저장 완료: {filename}")
            image_count += 1

    if image_count == 0:
        print("❌ 이미지가 생성되지 않았습니다.")
