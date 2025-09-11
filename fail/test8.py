import mimetypes
import os
from google import genai
from google.genai import types


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to: {file_name}")

def get_image_part(image_path, mime_type):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_data))


def generate_ad_image(product_name, product_image_path):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-image-preview"

    mime_type = mimetypes.guess_type(product_image_path)[0]

    if not mime_type or not mime_type.startswith('image/'):
        print(f"Error: Could not determine valid image MIME type for {product_image_path}")
        return

    # --- 디버깅을 위한 추가 코드 ---
    prompt_text_part1 = f"당신은 전문 광고 제작자입니다. 다음 '{product_name}' 제품의 광고 이미지를 만들어주세요. 제품에 가장 잘 어울리는 매력적인 배경을 생성하여, 제품이 돋보이는 광고 이미지를 완성해주세요. 여기 제품 이미지가 있습니다:"
    prompt_text_part2 = f"'{product_name}' 제품이 잘 드러나도록 멋진 광고 배경을 생성해주세요."

    print(f"--- Debug Info ---")
    print(f"product_name: '{product_name}' (type: {type(product_name)})")
    print(f"prompt_text_part1: '{prompt_text_part1}' (type: {type(prompt_text_part1)}, length: {len(prompt_text_part1)})")
    print(f"prompt_text_part2: '{prompt_text_part2}' (type: {type(prompt_text_part2)}, length: {len(prompt_text_part2)})")
    # --- 디버깅을 위한 추가 코드 끝 ---

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(prompt_text_part1), # 단순화된 변수 사용
                get_image_part(product_image_path, mime_type),
                types.Part.from_text(prompt_text_part2), # 단순화된 변수 사용
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
        ],
    )

    file_index = 0
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
            file_name = f"ad_image_{product_name.replace(' ', '_')}_{file_index}"
            file_index += 1
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            save_binary_file(f"{file_name}{file_extension}", data_buffer)
        else:
            print(f"Failed to generate ad image. Received text response: {chunk.text}")

if __name__ == "__main__":
    product_name = input("광고를 제작할 제품명을 입력하세요: ")
    product_image_path = input("상품 이미지 파일 경로를 입력하세요 (예: my_product.png): ")

    if not os.path.exists(product_image_path):
        print(f"오류: 지정된 파일 '{product_image_path}'이(가) 존재하지 않습니다. 정확한 경로를 입력해주세요.")
    else:
        generate_ad_image(product_name, product_image_path)