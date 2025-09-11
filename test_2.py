import base64
import mimetypes
import os
from google import genai
from google.genai import types

def load_image_as_part(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return types.Part(
        inline_data=types.Blob(
            mime_type=mime_type,
            data=image_bytes,
        )
    )

def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-image-preview"

    # ⬇️ 이미지와 텍스트 함께 입력
    image_path = "/Users/jieunchoi/Documents/GitHub/generate-to-image/123.jpeg"  # 여기에 입력할 이미지 경로
    image_part = load_image_as_part(image_path)

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""이미지와 어울리는 배경으로 바꿔줘, 광고 스타일"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
    )

    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "TEXT",  # 이미지 분석 결과를 텍스트로 받기
        ],
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:
            print(chunk.text)

if __name__ == "__main__":
    generate()
