import argparse
import mimetypes
import os
import time
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash-image-preview"

def _get_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {file_path}")
    return mime_type

def _load_image_parts(image_paths: list[str]) -> list[types.Part]:
    parts = []
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_data = f.read()
        mime_type = _get_mime_type(image_path)
        parts.append(
            types.Part(inline_data=types.Blob(data=image_data, mime_type=mime_type))
        )
    return parts

def _save_binary_file(file_name: str, data: bytes):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"✅ 파일 저장 완료: {file_name}")

def _process_api_stream_response(stream, output_dir: str):
    file_index = 0
    accumulated_text = []  # 텍스트 누적용 리스트

    for chunk in stream:
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue

        for part in chunk.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                timestamp = int(time.time())
                file_extension = mimetypes.guess_extension(part.inline_data.mime_type)
                file_name = os.path.join(
                    output_dir,
                    f"generated_image_{timestamp}_{file_index}{file_extension}",
                )
                _save_binary_file(file_name, part.inline_data.data)
                file_index += 1
            elif part.text:
                accumulated_text.append(part.text)

    # 모든 chunk를 다 받은 후에 텍스트 한꺼번에 출력
    if accumulated_text:
        full_text = "".join(accumulated_text).strip()
        print("📝 생성된 텍스트:\n", full_text)



def main():
    parser = argparse.ArgumentParser(description="Generate advertising style images using Gemini.")
    parser.add_argument(
        "-i",
        "--image",
        required=True,
        help="Path to the input image."
    )
    parser.add_argument(
        "--product-name",
        required=True,
        help="Product name to use in the advertising prompt."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save generated images."
    )

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
        exit(1)

    client = genai.Client(api_key=api_key)

    # 이미지 파일 확인
    if not os.path.isfile(args.image):
        print(f"❌ 입력 이미지 파일이 존재하지 않습니다: {args.image}")
        exit(1)

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 프롬프트 생성 (광고 스타일 배경 요청)
    prompt = (
        f"Create an advertising style background and slogan for the product named '{args.product_name}'. "
        "Generate a visually appealing commercial image that suits the product."
    )

    # 이미지 파일을 바이너리로 로드하고 Part 리스트로 만듦
    contents = _load_image_parts([args.image])
    contents.append(types.Part.from_text(text=prompt))

    print(f"✨ '{args.image}' 이미지와 다음 프롬프트로 이미지 생성 시작:\n{prompt}")

    # 스트리밍 방식으로 API 호출
    stream = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )

    # 스트림 응답 처리
    _process_api_stream_response(stream, args.output_dir)

if __name__ == "__main__":
    main()
