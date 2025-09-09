# 기존 코드의 import 문을 아래와 같이 수정
import google.generativeai as genai
from google.generativeai import types
import json

# MIME 타입 추론을 위한 import는 그대로 유지
import base64
import mimetypes
from dotenv import load_dotenv
import os

MODEL_NAME = "gemini-2.5-flash-image-preview"

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

# 이미지 파일을 base64로 인코딩
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# --- 멀티 AI 에이전트 시스템 시작 ---

# 1단계: 제품 분석 에이전트
def analyze_product(product_name: str, base64_image: str, image_path: str) -> dict:
    """
    제품 이미지와 이름을 분석해 주요 특징 및 소비자 페르소나를 추론합니다.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    
    # MIME 타입 추론
    mime_type = mimetypes.guess_type(image_path)[0] if image_path else "image/jpeg"
    
    prompt = f"""
    제품명: {product_name}
    
    이 이미지는 {product_name} 제품입니다. 
    1. 이 제품의 주요 특징(소재, 색상, 디자인 스타일 등)을 분석해주세요.
    2. 이 제품을 구매할 것으로 예상되는 타겟 소비자의 페르소나를 추론해주세요. (성별, 연령대, 취향, 라이프스타일 등)
    3. 이 제품의 주요 강점(핵심 가치 제안)은 무엇인가요?

    모든 분석 결과를 다음 JSON 형식으로 출력해주세요:
    {{
      "product_features": {{
        "material": "소재 분석",
        "color": "색상 분석",
        "style": "스타일 분석"
      }},
      "target_persona": {{
        "demographics": "성별, 연령대 등",
        "lifestyle": "라이프스타일",
        "preferences": "취향"
      }},
      "core_value_proposition": "제품의 핵심 가치"
    }}
    """
    
    try:
        # types.content.Part 대신 genai.types.Content.Part를 사용하도록 수정
        # 또는 from google.generativeai import Part, Blob 으로 임포트 후 Part, Blob 사용 가능
        content = [
            genai.types.Content.Part(text=prompt),
            genai.types.Content.Part(inline_data=genai.types.Blob(mime_type=mime_type, data=base64.b64decode(base64_image)))
        ]
        response = model.generate_content(content)
        json_output = response.text.replace('```json', '').replace('```', '').strip()
        print("\n---1단계: 제품 분석 결과 및 페르소나 추론---")
        print(json_output)
        return json.loads(json_output)
    except Exception as e:
        print(f"❌ 1단계 분석 실패: {e}")
        return {}
        
# 2단계와 3단계 함수는 이전과 동일합니다.
# main() 함수도 동일합니다.
# 수정된 analyze_product 함수만 사용하면 됩니다.

# ... (2단계, 3단계, main 함수 코드는 이전과 동일)

# 2단계: 카피라이팅 에이전트
def generate_ad_copies(analysis_data: dict) -> list:
    """
    분석된 페르소나에 맞춰 3가지 광고 문구(카피)를 제안합니다.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    
    persona = analysis_data.get("target_persona", {})
    product_features = analysis_data.get("product_features", {})
    
    prompt = f"""
    제품 특징: {json.dumps(product_features, ensure_ascii=False)}
    타겟 페르소나: {json.dumps(persona, ensure_ascii=False)}

    위 정보를 바탕으로 타겟 페르소나에게 어필할 수 있는 광고 문구(카피) 3개를 생성해주세요.
    각 문구는 간결하고 매력적으로 작성하며, JSON 배열 형태로 출력해주세요.
    예시: ["문구1", "문구2", "문구3"]
    """

    try:
        response = model.generate_content(prompt)
        json_output = response.text.replace('```json', '').replace('```', '').strip()
        print("\n---2단계: 광고 문구 제안---")
        print(json.loads(json_output))
        return json.loads(json_output)
    except Exception as e:
        print(f"❌ 2단계 문구 생성 실패: {e}")
        return []

# 3단계: 상세 페이지 콘텐츠 생성 에이전트
def generate_detail_page_content(analysis_data: dict, selected_copy: str) -> dict:
    """
    제품 분석 정보와 선택된 문구를 결합하여 상세 페이지 콘텐츠를 생성합니다.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    
    core_value = analysis_data.get("core_value_proposition", "")
    features = analysis_data.get("product_features", {})
    persona_desc = json.dumps(analysis_data.get("target_persona", {}), ensure_ascii=False)
    
    prompt = f"""
    핵심 문구: {selected_copy}
    제품 분석 정보: {json.dumps(analysis_data, ensure_ascii=False)}
    
    위 정보를 바탕으로 쇼핑몰 상세 페이지에 들어갈 주요 콘텐츠를 생성해주세요.
    
    내용에 포함되어야 할 항목:
    1. {selected_copy} 를 활용한 헤드라인
    2. 제품의 핵심 가치를 설명하는 도입부
    3. 제품 특징(소재, 색상, 디자인 등)을 자세히 설명하는 본문
    4. 이 제품이 타겟 페르소나에게 왜 완벽한 선택인지 설명하는 문구
    5. 제품명과 주요 특징을 해시태그 형식으로 요약
    
    모든 콘텐츠를 다음 JSON 형식으로 출력해주세요:
    {{
      "headline": "생성된 헤드라인",
      "introduction": "생성된 도입부",
      "features_and_details": "본문",
      "call_to_persona": "페르소나에게 어필하는 문구",
      "hashtags": ["#해시태그1", "#해시태그2"]
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        json_output = response.text.replace('```json', '').replace('```', '').strip()
        print("\n---3단계: 최종 상세 페이지 콘텐츠 생성---")
        print(json_output)
        return json.loads(json_output)
    except Exception as e:
        print(f"❌ 3단계 콘텐츠 생성 실패: {e}")
        return {}

def main():
    """
    전체 멀티 AI 에이전트 파이프라인을 순차적으로 실행합니다.
    """
    
    # 사용자 입력 받기 및 이미지 인코딩
    product_name = input("제품 이름을 입력하세요: ")
    image_path = input("제품 이미지 파일 경로를 입력하세요 (예: './image.jpg'): ")
    
    if not os.path.exists(image_path):
        print(f"❌ 오류: 파일 경로를 찾을 수 없습니다: {image_path}")
        return
        
    base64_image = encode_image_to_base64(image_path)
    print("✅ 이미지 인코딩 완료")
    
    # 1단계: 제품 분석 에이전트 실행
    # 'image_path' 매개변수를 추가했습니다.
    analysis_data = analyze_product(product_name, base64_image, image_path)
    if not analysis_data:
        print("❌ 파이프라인 종료: 1단계 실패")
        return

    # 2단계: 카피라이팅 에이전트 실행
    ad_copies = generate_ad_copies(analysis_data)
    if not ad_copies:
        print("❌ 파이프라인 종료: 2단계 실패")
        return

    # 사용자 선택 시뮬레이션
    print("\n--- 광고 문구 선택 ---")
    for i, copy in enumerate(ad_copies):
        print(f"[{i+1}] {copy}")
    
    try:
        choice = int(input("마음에 드는 문구의 번호를 입력하세요: ")) - 1
        selected_copy = ad_copies[choice]
        print(f"✅ 선택된 문구: {selected_copy}")
    except (ValueError, IndexError):
        print("❌ 잘못된 입력입니다. 첫 번째 문구를 기본값으로 사용합니다.")
        selected_copy = ad_copies[0]

    # 3단계: 상세 페이지 콘텐츠 생성 에이전트 실행
    final_content = generate_detail_page_content(analysis_data, selected_copy)
    
    if final_content:
        print("\n🎉 최종 상세 페이지 콘텐츠 생성이 완료되었습니다. 🎉")

if __name__ == "__main__":
    main()