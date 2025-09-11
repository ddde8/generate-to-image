from google import genai
import os
import base64
import mimetypes
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")


def get_ad_copies(brand_name):
    prompt = f"제품명: {brand_name}\n이 제품에 어울리는 광고 문구 3개 만들어줘."

    try:
        # 최신 공식 문서 기준
        response = client.chat.completions.create(
            model="models/chat-bison-001",
            messages=[
                {"role": "system", "content": "당신은 광고 카피라이터입니다."},
                {"role": "user", "content": prompt},
            ],
        )
    except AttributeError:
        # AttributeError 발생 시 대안
        response = client.chats.create(
            model="models/chat-bison-001",
            messages=[
                {"role": "system", "content": "당신은 광고 카피라이터입니다."},
                {"role": "user", "content": prompt},
            ],
        )

    return response.choices[0].message.content



def generate_image_with_prompt(prompt):
    response = client.images.generate(
        model="gemini-2.5-flash-image-preview",
        prompt=prompt,
        image_count=1,
        size="1024x1024",
    )
    return response.data[0].url


def main():
    print("📢 Gemini 광고 생성기")
    brand_name = input("제품명 또는 브랜드명을 입력하세요: ").strip()
    image_path = input("제품 이미지 경로를 입력하세요 (없으면 엔터): ").strip()

    # 광고 문구 3개 생성
    ad_copies_raw = get_ad_copies(brand_name)
    print("\n생성된 광고 문구:\n", ad_copies_raw)

    # 광고 문구 3개를 줄바꿈이나 번호 기준으로 분리 (간단히 줄바꿈 기준)
    ad_copies = [line.strip() for line in ad_copies_raw.split('\n') if line.strip()]
    for idx, copy in enumerate(ad_copies, 1):
        print(f"{idx}. {copy}")

    selected_idx = input(f"사용할 광고 문구 번호를 선택하세요 (1~{len(ad_copies)}): ").strip()
    try:
        selected_idx = int(selected_idx)
        selected_copy = ad_copies[selected_idx - 1]
    except (ValueError, IndexError):
        print("잘못된 입력입니다. 기본 첫번째 문구를 사용합니다.")
        selected_copy = ad_copies[0]

    # 이미지 프롬프트 생성
    image_prompt = (
        f"제품명: {brand_name}\n"
        f"광고 문구: {selected_copy}\n"
        "사람 모델은 제품에 어울리는 인종과 성별로, 얼굴은 보이지 않고 제품 위주로."
    )

    # 이미지 경로가 있으면 base64 인코딩 (필요시 API 맞게 적용)
    if image_path:
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            image_mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            # 실제 API가 이미지 첨부 지원할 경우 여기에 넣기
            image_prompt += f"\n[image data: {image_b64}]"
        except FileNotFoundError:
            print("이미지 파일을 찾을 수 없습니다. 이미지 없이 진행합니다.")

    # 이미지 생성
    image_url = generate_image_with_prompt(image_prompt)
    print("\n생성된 이미지 URL:", image_url)


if __name__ == "__main__":
    main()
