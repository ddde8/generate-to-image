import argparse  # CLI 인자 처리용
import mimetypes  # 파일 MIME 타입 추측용
import os  # 파일 경로 및 디렉토리 조작용
from dotenv import load_dotenv  # .env 파일에서 환경변수 로드용
import google.generativeai as genai  # 구글 Gemini AI 라이브러리

# 1. 환경 변수에서 API 키 로드 및 설정
load_dotenv()  # 현재 디렉토리의 .env 파일 읽어서 환경변수 등록
api_key = os.getenv('GOOGLE_API_KEY')  # API 키 가져오기
if not api_key:
    raise ValueError("환경 변수 GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")
genai.configure(api_key=api_key)  # genai 클라이언트에 API 키 설정

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
    os.makedirs(os.path.dirname(file_name), exist_ok=True)  # 폴더 없으면 생성
    with open(file_name, "wb") as f:  # 바이너리 쓰기 모드로 파일 열기
        f.write(data)  # 데이터 저장
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
        response_modalities=["TEXT"],  # 텍스트 응답 요청
    )

    generated_prompt = response.text.strip()  # 결과 텍스트 정리
    print(f"✅ AI가 생성한 프롬프트:\n{generated_prompt}\n")
    return generated_prompt

def synthesize_ad_image(input_image_path: str, text_prompt: str):
    """
    입력 이미지와 AI가 생성한 광고 배경 프롬프트를 사용해
    Gemini AI에 광고 배경이 합성된 이미지를 생성 요청.
    결과 이미지를 입력 이미지와 같은 위치에 저장.
    """
    with open(input_image_path, "rb") as f:
        image_data = f.read()  # 원본 이미지 바이너리 읽기
    image_mime_type = _get_mime_type(input_image_path)  # MIME 타입 확인

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

    response = model.generate_content(
        contents=contents,
        response_modalities=["IMAGE", "TEXT"],
    )

    # 받은 응답에서 이미지와 텍스트 데이터 추출
    if response.candidates and response.candidates[0].content:
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        output_dir = os.path.dirname(input_image_path)

        # content.parts 리스트의 각 원소 순회
        for i, part in enumerate(response.candidates[0].content.parts):
            # 텍스트 응답 처리
            if part.text:
                print("📄 모델 응답 텍스트:", part.text)
            # 이미지 응답 처리
            if part.inline_data:
                # 확장자 추출, 못 찾으면 png 기본값
                ext = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
                file_name = os.path.join(output_dir, f"{base_name}_ad{i}{ext}")
                _save_binary_file(file_name, part.inline_data.data)
    else:
        print("❌ 이미지 생성에 실패했습니다.")

# --- 프로그램 진입점 ---

def main():
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
        # 1) 제품 이름, 이미지로 광고 배경 문구 생성 요청
        prompt = generate_ad_prompt(args.name, args.image)
        # 2) 생성된 프롬프트와 원본 이미지로 광고 배경 합성 및 이미지 저장
        synthesize_ad_image(args.image, prompt)
        print("\n🎉 광고 이미지 생성이 완료되었습니다!")
    except Exception as e:
        print(f"❌ 예기치 않은 오류가 발생했습니다: {e}")

# 실행부
if __name__ == "__main__":
    main()
