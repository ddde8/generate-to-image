import argparse
import mimetypes
import os
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai

# 1. 환경 변수에서 API 키 로드 및 설정
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("환경 변수 GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")
genai.configure(api_key=api_key)

# 2. 사용할 Gemini 모델 이름 지정
MODEL_NAME = "gemini-2.5-flash-image-preview"

# --- 보조 함수들 ---

def _get_mime_type(file_path: str) -> str:
    """
    파일 경로에서 MIME 타입을 추측.
    예: .jpeg -> image/jpeg
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"'{file_path}'의 MIME 타입을 결정할 수 없습니다.")
    return mime_type

def _save_binary_file(file_name: str, data: bytes):
    """
    바이너리 데이터를 받아서 파일로 저장.
    저장할 디렉토리가 없으면 생성.
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"✅ 파일이 저장되었습니다: {file_name}")

# --- AI 작업 함수 ---

def generate_ad_prompt(product_name: str, image_path: str) -> str:
    """
    제품명과 이미지를 AI에 주고,
    제품에 어울리는 광고 배경에 대한 텍스트 프롬프트 생성 요청 후 반환.
    """
    with open(image_path, "rb") as f:
        image_data = f.read()  # 이미지 바이너리 읽기
    image_mime_type = _get_mime_type(image_path)  # 이미지 MIME 타입 얻기

    # AI 모델에 입력할 데이터 리스트
    contents = [
        {"text": f"""
당신은 세계적인 광고 디자이너입니다. 당신의 목표는 사용자가 제공한 **제품 사진**과 **제품명({product_name})**을 보고, 이 제품의 특징을 가장 잘 살릴 수 있는 **광고 배경 아이디어**를 구체적인 텍스트 프롬프트로 만들어내는 것입니다.

생성할 프롬프트는 다음과 같은 조건을 충족해야 합니다:
1. **시각적으로 매력적**이고 제품의 가치를 높일 수 있어야 합니다.
2. Gemini 모델이 이미지를 생성할 때 바로 사용할 수 있도록 **구체적이고 상세한 묘사**를 포함해야 합니다.
3. 50단어 이내의 **단일 문장**으로 작성해주세요.
""" },
        {
            "mime_type": image_mime_type,  # 이미지 MIME 타입
            "data": image_data,  # 이미지 바이너리 데이터
        },
    ]

    print("🧠 AI가 제품 분석 및 최적의 배경 프롬프트를 생성 중입니다...")
    response = genai.GenerativeModel(MODEL_NAME).generate_content(
        contents=contents,
        generation_config={"temperature": 0.4},  # 생성 온도 낮게 설정 (덜 무작위)
        # 텍스트만 필요하므로 response_modalities 인자 제거
    )

    generated_prompt = response.text.strip()  # 결과 텍스트 정리
    print(f"✅ AI가 생성한 프롬프트:\n{generated_prompt}\n")
    return generated_prompt

async def synthesize_ad_image(input_image_path: str, text_prompt: str):
    """
    입력 이미지와 AI가 생성한 광고 배경 프롬프트를 사용해
    Gemini AI에 광고 배경이 합성된 이미지를 생성 요청.
    결과 이미지를 입력 이미지와 같은 위치에 저장.
    """
    with open(input_image_path, "rb") as f:
        image_data = f.read()
    image_mime_type = _get_mime_type(input_image_path)

    contents = [
        {"text": "당신은 세계적인 광고 디자이너입니다. 제공된 이미지의 피사체를 다음 배경에 가장 자연스럽고 멋지게 합성해주세요."},
        {
            "mime_type": image_mime_type,
            "data": image_data,
        },
        {"text": text_prompt},
    ]

    print("🎨 광고 배경을 디자인하고 최종 이미지를 합성 중입니다...")

    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
        },
    )

    # generate_content_stream 사용
    # response_modalities는 generation_config 딕셔너리에 포함
    response_stream = await model.generate_content_stream(
        contents=contents,
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "response_modalities": ["TEXT", "IMAGE"]
        }
    )

    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    output_dir = os.path.dirname(input_image_path)
    
    # 스트림 응답 처리
    async for chunk in response_stream:
        for part in chunk.content.parts:
            if part.text:
                print("📄 모델 응답 텍스트:", part.text)
            if part.inline_data:
                ext = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
                file_name = os.path.join(output_dir, f"{base_name}_ad{ext}")
                _save_binary_file(file_name, part.inline_data.data)


# --- 프로그램 진입점 ---

async def main():
    parser = argparse.ArgumentParser(
        description="Gemini AI를 활용해 제품 이미지에 어울리는 광고 배경을 생성하고 합성하는 에이전트."
    )
    parser.add_argument(
        "-i", "--image", type=str, required=True, help="제품 이미지 경로"
    )
    parser.add_argument(
        "-n", "--name", type=str, required=True, help="제품 이름 또는 설명"
    )

    args = parser.parse_args()

    try:
        prompt = generate_ad_prompt(args.name, args.image)
        await synthesize_ad_image(args.image, prompt)
        print("\n🎉 광고 이미지 생성이 완료되었습니다!")
    except Exception as e:
        print(f"❌ 예기치 않은 오류가 발생했습니다: {e}")

# 실행부
if __name__ == "__main__":
    asyncio.run(main())