import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from io import BytesIO

# 1. 환경 변수 로드 및 설정
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# 2. Gemini 모델 초기화
llm = genai.GenerativeModel("gemini-2.5-flash-image-preview")


def get_product_info():
    """상품명과 이미지 경로를 입력받고 이미지 객체 반환"""
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
    """상품명과 이미지로부터 광고 문구 + 배경 설명 생성"""
    prompt_template = PromptTemplate.from_template(
        """
        당신은 광고 기획자입니다. "{product_name}"에 어울리는 광고 문구 3가지와 적절한 배경 스타일을 설명해 주세요. 
        문구는 최대 8글자. 배경에 어울리는 사물, 질감, 색상 등을 포함해서 설명해 주세요.
        """
    )
    prompt_text = prompt_template.format(product_name=product_name)

    response = llm.generate_content([image, prompt_text])
    ad_text = response.text.strip()

    print("\n📝 광고 문구 및 배경 설명:\n")
    print(ad_text)

    return ad_text


def generate_image_from_prompt(prompt: str, save_path: str = "generated_ad_image.png"):
    """텍스트 프롬프트로부터 이미지 생성"""
    client = genai.Client()

    response = model.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[prompt],
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            image.save(save_path)
            print(f"✅ 이미지 저장 완료: {save_path}")
            return save_path
        elif part.text:
            print("💬 텍스트 응답:", part.text)

    print("❌ 이미지 생성 실패")
    return None


def generate_ad_image_from_product(product_name, image, save_path="generated_ad_image.png"):
    """전체 광고 이미지 생성 프로세스"""
    ad_text = generate_ad_text(product_name, image)

    image_prompt = f"{ad_text}\n위 설명을 바탕으로 광고 이미지를 만들어줘."
    generated_image_path = generate_image_from_prompt(image_prompt, save_path)

    return ad_text, generated_image_path


# 5. 메인 실행 흐름
if __name__ == "__main__":
    product_name, image = get_product_info()

    if product_name is None or image is None:
        print("❌ 입력 오류로 인해 종료합니다.")
    else:
        ad_text, image_file = generate_ad_image_from_product(product_name, image)

        if image_file:
            img = Image.open(image_file)
            img.show()
