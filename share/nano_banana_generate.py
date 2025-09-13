# nano_banana_generate.py
# Stage 3: Gemini 2.5 Flash Image(= Nano Banana)로 "제품 보존 + 배경 합성"
# Google Gen AI SDK + Vertex AI 백엔드 사용

import os
import sys
import io
import json
import argparse
from PIL import Image
from typing import List

# Google Gen AI SDK (Vertex 사용은 환경변수로 전환)
from google import genai
from google.genai.types import GenerateContentConfig, Modality

# ----------------------------
# 이미지 리사이즈 (최대 변 기준, 비율 유지)
# ----------------------------
def resize_max_side(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    if w >= h:
        nh = int(h * max_side / w)
        return img.resize((max_side, nh), Image.LANCZOS)
    else:
        nw = int(w * max_side / h)
        return img.resize((nw, max_side), Image.LANCZOS)

# ----------------------------
# 프롬프트 구성 (제품만 남기고 배경 합성, 텍스트/로고 금지)
# ----------------------------
def build_prompt(meta: dict) -> str:
    product = meta.get("product", {})
    background = meta.get("background", {}) or {}
    layout = meta.get("layout", {}) or {}
    subj = layout.get("subject_layout", {}) or {}
    ng = layout.get("nongraphic_layout", []) or []
    gg = layout.get("graphic_layout", []) or []
    bg_objs = meta.get("background_objects", []) or []

    # 1) 예약/금지 영역 수집 (문구/로고/언더레이 모두 '비워둘 영역')
    negative_rects = []
    for t in ng:
        b = t.get("bbox")
        if isinstance(b, list) and len(b) == 4:
            negative_rects.append({
                "type": t.get("type", "text"),
                "bbox": b
            })
    for g in gg:
        b = g.get("bbox")
        if isinstance(b, list) and len(b) == 4:
            # underlay도 Stage3에서는 그리지 않도록 'reserved' 처리
            negative_rects.append({
                "type": g.get("type", "graphic"),
                "bbox": b,
                "for": g.get("for", None)
            })

    # 2) 배경 세부 스펙 정리
    ideal_color   = background.get("ideal_color")
    texture       = background.get("texture")
    lighting      = background.get("lighting", {})
    lighting_type = lighting.get("type")
    lighting_dir  = lighting.get("direction")
    style         = background.get("style")
    bg_prompt     = background.get("prompt")
    neg_prompt    = background.get("negative_prompt")
    camera        = background.get("camera", {})
    cam_angle     = camera.get("angle")
    cam_distance  = camera.get("distance")
    palette       = background.get("palette", [])

    # 3) 시스템 지시(핵심 규칙)
    system_text = (
        "You are an advertising image compositor.\n"
        "Strictly REPLACE the entire background with the requested style while preserving the product pixels.\n"
        "Do NOT render any text, logos, or underlay shapes. Keep all reserved boxes CLEAN.\n"
        "Do NOT letterbox, pad, or add borders."
    )

    # 4) 룰/제약
    rules = [
        "- Preserve the foreground product EXACTLY (pixel-preserve). No redraw/smoothing.",
        "- The original table/surface must disappear (full background replacement).",
        "- Keep all reserved boxes EMPTY (negative space) for later typography/graphics.",
        "- Follow subject_layout center/ratio for framing and composition.",
        "- No vignettes, borders, or drop shadows unless explicitly asked.",
        "- Output a single photorealistic image; same or higher resolution than input."
    ]

    # 5) 배경 스펙/카메라/팔레트/네거티브 프롬프트 반영
    spec_lines = []
    if ideal_color:   spec_lines.append(f"- Ideal background color: {ideal_color}.")
    if texture:       spec_lines.append(f"- Texture: {texture}.")
    if style:         spec_lines.append(f"- Style: {style}.")
    if lighting_type: spec_lines.append(f"- Lighting: {lighting_type}.")
    if lighting_dir:  spec_lines.append(f"- Key light direction: {lighting_dir}.")
    if cam_angle:     spec_lines.append(f"- Camera angle: {cam_angle}.")
    if cam_distance:  spec_lines.append(f"- Camera distance: {cam_distance}.")
    if palette:       spec_lines.append(f"- Prefer palette: {', '.join(palette)}.")
    if bg_prompt:     spec_lines.append(f"- Positive prompt: {bg_prompt}")
    if neg_prompt:    spec_lines.append(f"- Negative prompt: {neg_prompt}")

    # 6) 배경 오브젝트 지시 (제품 뒤에만, 텍스트 박스/subject와 IoU 제한)
    obj_lines = []
    for obj in bg_objs:
        name = obj.get("name")
        style_o = obj.get("style")
        bbox_hint = obj.get("bbox_hint")
        depth = obj.get("depth")
        notes = obj.get("notes")
        line = f"- Optional background object: {name}"
        if style_o:   line += f" (style: {style_o})"
        if depth:     line += f", depth: {depth}"
        if bbox_hint: line += f", place within bbox_hint {bbox_hint}"
        if notes:     line += f", notes: {notes}"
        obj_lines.append(line + ".")

    # 7) 사용자 지시문(메타 정보 삽입)
    user_text = (
        "TASK: Replace the background completely while preserving the product.\n\n"
        f"PRODUCT:\n{json.dumps(product, ensure_ascii=False)}\n\n"
        "BACKGROUND SPEC:\n" + "\n".join(spec_lines) + "\n\n"
        f"SUBJECT LAYOUT (normalized 0~1):\n{json.dumps({'subject_layout': subj}, ensure_ascii=False)}\n\n"
        f"RESERVED NEGATIVE SPACES (keep empty):\n{json.dumps(negative_rects, ensure_ascii=False)}\n\n"
        "BACKGROUND OBJECTS (optional):\n" + ("\n".join(obj_lines) if obj_lines else "(none)") + "\n\n"
        "RULES:\n" + "\n".join(rules)
    )

    return system_text + "\n\n" + user_text


# ----------------------------
# 첫 이미지 파트 저장
# ----------------------------
def save_first_image_part(resp, out_path: str) -> bool:
    cand = None
    if getattr(resp, "candidates", None):
        cand = resp.candidates[0]
    if not cand or not getattr(cand, "content", None):
        return False

    parts = getattr(cand.content, "parts", []) or []
    for p in parts:
        # TEXT or IMAGE가 섞여서 나옵니다. (이미지 전용은 지원X)  :contentReference[oaicite:5]{index=5}
        if getattr(p, "inline_data", None):
            mime = getattr(p.inline_data, "mime_type", "")
            data = getattr(p.inline_data, "data", None)
            if mime and data:
                ext = mime.split("/")[-1].lower().replace("jpeg", "jpg")
                root, _ = os.path.splitext(out_path)
                out_file = f"{root}.{ext}"
                with open(out_file, "wb") as f:
                    f.write(data)
                print(f"✅ [저장 완료] {out_file}")
                return True
    return False

# ----------------------------
# 메인
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage 3 with Gemini 2.5 Flash Image (Vertex backend via google-genai).")
    ap.add_argument("--image", required=True, help="Path to the product foreground image.")
    ap.add_argument("--layout_json", required=True, help="Path to the layout JSON file.")
    ap.add_argument("--out", default="stage3_output.png", help="Output file path (extension adapts to returned MIME).")
    ap.add_argument("--max_side", type=int, default=1024, help="Max side length for resizing input image.")
    ap.add_argument("--model", default="gemini-2.5-flash-image-preview", help="Model id.")
    args = ap.parse_args()

    # 환경변수 점검 (Vertex 백엔드 사용 설정)  :contentReference[oaicite:6]{index=6}
    need_vars = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_GENAI_USE_VERTEXAI"]
    missing = [v for v in need_vars if not os.environ.get(v)]
    if missing:
        print(f"❌ 환경변수 누락: {', '.join(missing)}")
        print("   예) PowerShell:")
        print('   $env:GOOGLE_CLOUD_PROJECT="nano-471710"')
        print('   $env:GOOGLE_CLOUD_LOCATION="global"')
        print('   $env:GOOGLE_GENAI_USE_VERTEXAI="True"')
        sys.exit(1)

    # 인증(ADC) 점검은 SDK가 진행. 실패 시 예외 발생.
    client = genai.Client()

    # 입력 로드
    try:
        with open(args.layout_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"❌ 레이아웃 JSON 로드 실패: {e}")
        sys.exit(1)

    try:
        img = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"❌ 이미지 로드 실패({args.image}): {e}")
        sys.exit(1)

    img = resize_max_side(img, args.max_side)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    prompt_text = build_prompt(meta)

    # 중요: 응답 모달리티에 TEXT와 IMAGE를 모두 요청해야 함  :contentReference[oaicite:7]{index=7}
    cfg = GenerateContentConfig(
        response_modalities=[Modality.TEXT, Modality.IMAGE],
        candidate_count=1,
    )

    print(f"... Requesting '{args.model}' (Vertex backend) ...")
    try:
        response = client.models.generate_content(
            model=args.model,
            contents=[prompt_text, img],  # ← PIL.Image.Image 객체 그대로
            config=cfg,
            )
    except Exception as e:
        print(f"❌ [호출 실패] {e}")
        print("   - 모델/리전/인증/결제를 점검하세요.")
        print("   - 모델은 gemini-2.5-flash-image-preview, LOCATION은 global 권장.")
        sys.exit(1)

    saved = save_first_image_part(response, args.out)
    if not saved:
        print("⚠️ 이미지 파트를 받지 못했습니다. 모델이 텍스트만 반환했을 수 있습니다.")
        txt = getattr(response, "text", None)
        if txt:
            print("---- Response text (truncated) ----")
            print(txt[:800])
        sys.exit(2)

if __name__ == "__main__":
    main()