# qwen25_vl_layout_hybrid.py (adaptive background + layout v2)
# - Pass1: 제품·배경 요약 + 레이아웃(JSON)
# - Pass2 (옵션 --bg_prompt): Pass1 결과를 이용해 전경/배치/각도에 맞는 배경 프롬프트 + 소품 계획 JSON 생성
# - 픽셀 좌표 자동 정규화(0~1) + 규칙 정제 + 비었을 때 배너/로고/언더레이 자동 보강
# - 출력: product/background(+prompt…) + layout(subject/nongraphic/graphic) + background_objects JSON

import json, argparse, os, sys, math
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# ----------------------------
# 프롬프트 스키마 (confidence 포함, 다중 후보)
# ----------------------------
SCHEMA_TEXT = (
    '오직 다음 JSON만 출력해. 다른 설명/마크다운 금지.\n'
    '{\n'
    '  "product": { "type":"...", "material":"...", "design":"...", "features":"..." },\n'
    '  "background": { "ideal_color":"...", "texture":"...", "lighting":"...", "style":"..." },\n'
    '  "layout": {\n'
    '    "subject_layout": { "center":[cx,cy], "ratio":[rw,rh] },\n'
    '    "nongraphic_layout": [\n'
    '      {"type":"headline","bbox":[x,y,w,h],"confidence":c},\n'
    '      {"type":"subhead","bbox":[x,y,w,h],"confidence":c}\n'
    '    ],\n'
    '    "graphic_layout": [\n'
    '      {"type":"logo","content":"","bbox":[x,y,w,h],"confidence":c}\n'
    '    ]\n'
    '  }\n'
    '}\n'
    '\n'
    '# 제약 사항:\n'
    '- 모든 좌표/크기는 0~1 비율. bbox=[x,y,w,h].\n'
    '- subject와 텍스트의 IoU < 0.2 권장.\n'
    '- 텍스트끼리 겹침을 줄여(권장 IoU < 0.3). 겹치면 confidence 낮은 쪽.\n'
    '- 텍스트 박스는 가로형을 우선 추천(가로:세로 비 1.8 이상 권장).\n'
    '- headline 1+, logo 1+ 최소 추천(확신이 낮으면 confidence만 낮게).\n'
    '- 불확실하더라도 최소 후보는 제시.\n'
)

SYSTEM = (
  "너는 광고 상품 이미지를 분석해 제품·배경 요약과 레이아웃을 JSON으로 제안하는 VLM이야. "
  "이미지 맥락만으로 텍스트/로고 위치를 추천하고, confidence로 확신도를 제공해. "
  "JSON 외의 텍스트는 출력하지 마."
)

# ----------------------------
# (NEW) 배경 프롬프트 전용 스키마/시스템
# ----------------------------
BG_SYSTEM = (
  "너는 제품 전경과 배치를 고려해, 배경 생성 모델(예: SDXL/Imagen/Flux 등)에 줄 프롬프트를 설계하는 어시스턴트야. "
  "주어진 subject bbox와 텍스트 박스, 자유공간(safe zone), 색 팔레트, 카메라 각도를 반영해서 "
  "배경 소품과 조명/질감/스타일을 정하고, 배경 오브젝트는 subject와 텍스트를 가리지 않도록 배치 힌트를 제공해. "
  "반드시 JSON만 출력하고, 설명은 금지해."
)

BG_SCHEMA = (
  '오직 다음 JSON만 출력해. 다른 설명/마크다운 금지.\n'
  '{\n'
  '  "background_prompt": "...",\n'
  '  "negative_prompt": "...",\n'
  '  "camera": { "angle":"eye-level|top-down|low-angle|macro|oblique", "distance":"closeup|medium|wide" },\n'
  '  "lighting": { "type":"soft|hard|rim|ambient", "direction":"left|right|front|back|top|bottom" },\n'
  '  "palette": ["#RRGGBB", "..."],\n'
  '  "objects": [\n'
  '    {\n'
  '      "name": "flower petals",\n'
  '      "style": "bokeh|flat|painterly|realistic",\n'
  '      "bbox_hint": [x,y,w,h],\n'
  '      "depth": "behind_product|same_plane",\n'
  '      "avoid_iou_with": "subject",\n'
  '      "notes": "subject와 IoU<0.1, 텍스트 박스 피하기"\n'
  '    }\n'
  '  ]\n'
  '}\n'
)

# ----------------------------
# 유틸 / 후처리
# ----------------------------

def extract_json(text: str):
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"raw": text}


def clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def clip_bbox(b):
    x,y,w,h = b
    x = clip01(x); y = clip01(y)
    w = clip01(w); h = clip01(h)
    if x + w > 1: w = max(0.0, 1 - x)
    if y + h > 1: h = max(0.0, 1 - y)
    return [x,y,w,h]


def iou(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xa = max(x1, x2); ya = max(y1, y2)
    xb = min(x1+w1, x2+w2); yb = min(y1+h1, y2+h2)
    inter = max(0.0, xb-xa) * max(0.0, yb-ya)
    a1 = max(0.0, w1*h1); a2 = max(0.0, w2*h2)
    union = a1 + a2 - inter
    return inter/union if union > 0 else 0.0


def nms(boxes, iou_thr=0.3):
    # boxes: [{bbox:[x,y,w,h], confidence:0~1, ...}]
    bxs = [b for b in boxes if isinstance(b.get("bbox"), list) and len(b["bbox"])==4]
    bxs.sort(key=lambda b: float(b.get("confidence", 0.5)), reverse=True)
    kept = []
    for b in bxs:
        if all(iou(b["bbox"], k["bbox"]) < iou_thr for k in kept):
            kept.append(b)
    return kept


def enforce_text_rules(items, subject_bbox,
                       min_ar=1.8,      # 가로형 권장
                       min_margin=0.03, # 가장자리 여백
                       max_iou=0.2):    # subject와 겹침 제한
    out = []
    for it in items:
        b = it.get("bbox", [0,0,0,0])
        if not (isinstance(b, list) and len(b)==4):
            continue
        b = clip_bbox(b)

        # subject 겹침 제한
        if subject_bbox and iou(b, subject_bbox) >= max_iou:
            continue

        # 가장자리 여백
        x,y,w,h = b
        if x < min_margin or y < min_margin or x+w > 1-min_margin or y+h > 1-min_margin:
            continue

        # 가로형 체크
        ar = (w / h) if h > 0 else 999
        if ar < min_ar:
            continue

        it["confidence"] = float(it.get("confidence", 0.5))
        it["bbox"] = b
        out.append(it)
    return out


def postprocess_layout(parsed,
                       text_iou_thr=0.3,
                       logo_iou_thr=0.3,
                       subj_text_iou_max=0.2):
    """VLM 결과 후처리: clip, 규칙 적용, NMS + 텍스트 id 부여"""
    if not isinstance(parsed, dict) or "layout" not in parsed:
        return parsed

    layout = parsed["layout"]
    subj = layout.get("subject_layout", {})
    # subject bbox 계산
    try:
        cx,cy = subj.get("center", [0.5,0.5])
        rw,rh = subj.get("ratio", [0.3,0.3])
        subject_bbox = clip_bbox([cx - rw/2, cy - rh/2, rw, rh])
    except Exception:
        subject_bbox = [0.4, 0.4, 0.2, 0.2]  # 안전 기본값

    # 1) 텍스트 정제
    texts = layout.get("nongraphic_layout", [])
    if not isinstance(texts, list): texts = []
    texts = [ {**t, "confidence": float(t.get("confidence", 0.5))} for t in texts ]
    texts = enforce_text_rules(texts, subject_bbox, max_iou=subj_text_iou_max)
    texts = nms(texts, iou_thr=text_iou_thr)
    for i, t in enumerate(texts):
        t.setdefault("id", f"text#{i}")

    # 2) 로고/그래픽 정제
    graphics = layout.get("graphic_layout", [])
    if not isinstance(graphics, list): graphics = []
    cleaned_g = []
    for g in graphics:
        b = g.get("bbox")
        if not (isinstance(b, list) and len(b)==4):
            continue
        b = clip_bbox(b)
        # subject를 과도하게 가리는 로고 제외
        if iou(b, subject_bbox) > 0.4:
            continue
        g["bbox"] = b
        g["confidence"] = float(g.get("confidence", 0.5))
        cleaned_g.append(g)
    graphics = nms(cleaned_g, iou_thr=logo_iou_thr)

    layout["nongraphic_layout"] = texts
    layout["graphic_layout"] = graphics
    return parsed


# ----------------------------
# (신규) 픽셀 → 0~1 정규화 보정
# ----------------------------

def normalize_if_pixels_layout(parsed, image_path):
    """subject_layout.center/ratio 및 모든 bbox를 0~1로 강제 정규화.
    값 중 1을 넘는 항목이 있으면 '픽셀'로 판단해 이미지 크기로 나눔."""
    try:
        W, H = Image.open(image_path).size
    except Exception:
        W, H = 1, 1  # 실패 시 no-op

    def norm_center_ratio(center, ratio):
        cx, cy = center
        rw, rh = ratio
        if max(cx, cy) > 1.0:
            cx = float(cx) / W
            cy = float(cy) / H
        if max(rw, rh) > 1.0:
            rw = float(rw) / W
            rh = float(rh) / H
        return [clip01(cx), clip01(cy)], [clip01(rw), clip01(rh)]

    def norm_bbox(b):
        x, y, w, h = b
        if max(x, y, w, h) > 1.0:
            x = float(x) / W
            y = float(y) / H
            w = float(w) / W
            h = float(h) / H
        return clip_bbox([x, y, w, h])

    if not isinstance(parsed, dict) or "layout" not in parsed:
        return parsed

    layout = parsed["layout"]

    # subject_layout
    subj = layout.get("subject_layout", {})
    c = subj.get("center", [0.5, 0.5])
    r = subj.get("ratio", [0.3, 0.3])
    c, r = norm_center_ratio(c, r)
    layout["subject_layout"] = {"center": c, "ratio": r}

    # nongraphic_layout
    if isinstance(layout.get("nongraphic_layout"), list):
        for t in layout["nongraphic_layout"]:
            if "bbox" in t and isinstance(t["bbox"], list) and len(t["bbox"]) == 4:
                t["bbox"] = norm_bbox(t["bbox"])

    # graphic_layout
    if isinstance(layout.get("graphic_layout"), list):
        for g in layout["graphic_layout"]:
            if "bbox" in g and isinstance(g["bbox"], list) and len(g["bbox"]) == 4:
                g["bbox"] = norm_bbox(g["bbox"])

    return parsed


# ----------------------------
# 규칙: 비면 자동 생성 Fallback(상·하 텍스트, 우상단 로고)
# ----------------------------

def inject_fallback_boxes(parsed,
                          headline_h=0.12,     # 상/하 배너 높이
                          margin=0.04,         # 테두리 여백
                          logo_box=(0.25,0.10) # 로고 w,h
                          ):
    if "layout" not in parsed or not isinstance(parsed["layout"], dict):
        parsed["layout"] = {}
    layout = parsed["layout"]

    # subject bbox
    s = layout.get("subject_layout", {"center":[0.5,0.5], "ratio":[0.3,0.3]})
    cx, cy = s.get("center", [0.5, 0.5])
    rw, rh = s.get("ratio", [0.3, 0.3])
    subj = clip_bbox([cx - rw/2, cy - rh/2, rw, rh])

    # 텍스트가 비면 상/하 배너형 두 개 제안
    ng = layout.get("nongraphic_layout")
    if not isinstance(ng, list) or len(ng) == 0:
        top = [margin, margin, 1 - 2*margin, headline_h]
        bot = [margin, 1 - margin - headline_h, 1 - 2*margin, headline_h]

        def shrink_if_overlap(b):
            if iou(b, subj) >= 0.1:
                x,y,w,h = b
                return [x, y, w, max(0.05, h*0.5)]
            return b

        top = shrink_if_overlap(clip_bbox(top))
        bot = shrink_if_overlap(clip_bbox(bot))

        layout["nongraphic_layout"] = [
            {"type": "headline", "bbox": top, "confidence": 0.5},
            {"type": "headline", "bbox": bot, "confidence": 0.5},
        ]

    # 그래픽(로고) 비면 우상단에 배치
    gg = layout.get("graphic_layout")
    if not isinstance(gg, list) or len(gg) == 0:
        lw, lh = logo_box
        gx = 1 - margin - lw
        gy = margin
        logo = clip_bbox([gx, gy, lw, lh])

        if iou(logo, subj) >= 0.3:
            logo = clip_bbox([margin, margin, lw, lh])

        layout["graphic_layout"] = [
            {"type": "logo", "content": "", "bbox": logo, "confidence": 0.5}
        ]

    return parsed


# ----------------------------
# (NEW) 텍스트 언더레이(반투명 라운드 박스) 자동 삽입
# ----------------------------

def add_text_underlays(parsed, pad=0.015, opacity=0.6, radius=0.08):
    if not isinstance(parsed, dict) or "layout" not in parsed:
        return parsed
    layout = parsed["layout"]
    texts = layout.get("nongraphic_layout", []) or []
    if not isinstance(layout.get("graphic_layout"), list):
        layout["graphic_layout"] = []

    def expand(b, p):
        x,y,w,h = b
        return clip_bbox([x - p, y - p, w + 2*p, h + 2*p])

    # 각 텍스트에 매칭되는 언더레이 추가
    for idx, t in enumerate(texts):
        b = t.get("bbox")
        if not (isinstance(b, list) and len(b)==4):
            continue
        under = {
            "type": "underlay",
            "for": t.get("type", "text") + f"#{idx}",
            "bbox": expand(b, pad),
            "style": {"shape":"rounded", "radius": radius, "opacity": opacity},
            "confidence": min(0.9, float(t.get("confidence", 0.5)) + 0.1)
        }
        layout["graphic_layout"].append(under)
    return parsed


# ----------------------------
# (NEW) 이미지 팔레트 추출 (PIL 적응형 팔레트 사용)
# ----------------------------

def extract_palette_hex(image_path, k=5):
    try:
        im = Image.open(image_path).convert("RGB")
        im_thumb = im.copy()
        im_thumb.thumbnail((256, 256))
        pal = im_thumb.convert("P", palette=Image.ADAPTIVE, colors=k).convert("RGB")
        colors = pal.getcolors(256*256) or []
        colors.sort(key=lambda x: x[0], reverse=True)
        hexes = []
        for _, rgb in colors[:k]:
            hexes.append('#%02x%02x%02x' % rgb)
        dedup = []
        for h in hexes:
            if h not in dedup:
                dedup.append(h)
        return dedup[:k] or ["#ffffff", "#000000"]
    except Exception:
        return ["#ffffff", "#000000"]


# ----------------------------
# (NEW) 레이아웃 요약: subject/텍스트/자유 공간 힌트
# ----------------------------

def summarize_layout_for_bg(parsed):
    """배경 프롬프트에 들어갈 컨텍스트 텍스트(요약) 생성."""
    if not isinstance(parsed, dict) or "layout" not in parsed:
        return "no layout"
    layout = parsed["layout"]
    subj = layout.get("subject_layout", {"center":[0.5,0.5], "ratio":[0.3,0.3]})
    cx, cy = subj.get("center", [0.5,0.5])
    rw, rh = subj.get("ratio", [0.3,0.3])
    subj_bbox = [cx - rw/2, cy - rh/2, rw, rh]

    texts = layout.get("nongraphic_layout", []) or []
    logos = layout.get("graphic_layout", []) or []

    # 자유 공간 힌트: 상단/하단/좌/우 중 넓은 영역을 서술적으로
    free_hints = []
    top_free = cy - rh/2
    bottom_free = 1 - (cy + rh/2)
    left_free = cx - rw/2
    right_free = 1 - (cx + rw/2)
    if top_free > 0.25: free_hints.append("top has ample negative space")
    if bottom_free > 0.25: free_hints.append("bottom has ample negative space")
    if left_free > 0.25: free_hints.append("left side has ample negative space")
    if right_free > 0.25: free_hints.append("right side has ample negative space")

    return json.dumps({
        "subject_bbox": subj_bbox,
        "text_boxes": [t.get("bbox") for t in texts if isinstance(t.get("bbox"), list)],
        "logo_boxes": [g.get("bbox") for g in logos if isinstance(g.get("bbox"), list)],
        "free_space_hints": free_hints
    }, ensure_ascii=False)


# ----------------------------
# (NEW) 배경 프롬프트/소품 계획 생성 (2nd pass VLM 호출)
# ----------------------------

def generate_bg_plan(model, processor, image_path, product_name, parsed, palette):
    context = summarize_layout_for_bg(parsed)
    user_text = (
        f"[제품명 힌트] {product_name or ''}\n"
        f"[레이아웃 컨텍스트] {context}\n"
        f"[권장 팔레트] {palette}\n"
        f"{BG_SCHEMA}"
    )
    messages = [
        {"role": "system", "content": [{"type":"text","text": BG_SYSTEM}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": user_text}
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.9, temperature=0.7)
    gen = processor.batch_decode(out_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    try:
        start = gen.index("{"); end = gen.rindex("}") + 1
        return json.loads(gen[start:end])
    except Exception:
        return {"background_prompt": gen.strip()[:800], "negative_prompt": "", "palette": palette}


# ----------------------------
# 메인
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="분석할 이미지 경로", required=False)
    ap.add_argument("--product_name", help="제품 이름(힌트)", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=640)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--save", help="결과를 저장할 파일 경로(json)", default=None)
    # (NEW) 옵션: 2패스 배경 프롬프트 생성 on/off
    ap.add_argument("--bg_prompt", action="store_true", help="배경 프롬프트/소품 계획 생성 활성화")
    args = ap.parse_args()

    product_name = args.product_name or input("제품 이름을 입력하세요: ").strip()
    image_path = args.image or input("제품 이미지 파일 경로를 입력하세요 (예: './image.jpg'): ").strip()
    if not os.path.exists(image_path):
        print(f"[에러] 이미지 경로를 찾을 수 없습니다: {image_path}", file=sys.stderr)
        sys.exit(1)

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    user_text = f"[제품명 힌트] {product_name}\n{SCHEMA_TEXT}"

    messages = [
      {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
      {"role": "user", "content": [
          {"type": "image", "image": f"file://{image_path}"},
          {"type": "text",  "text": user_text}
      ]}
    ]

    # 전처리
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    # 생성 (1st pass)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_p=args.top_p,
            temperature=args.temperature
        )

    gen_text = processor.batch_decode(
        out_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]

    # JSON 추출 + 보정/후처리/폴백 + 언더레이
    parsed = extract_json(gen_text)
    parsed = normalize_if_pixels_layout(parsed, image_path)  # (1) 픽셀→정규화
    parsed = postprocess_layout(parsed)                      # (2) 규칙/NMS 정제 + id
    parsed = inject_fallback_boxes(parsed)                   # (3) 비면 자동 보강
    parsed = add_text_underlays(parsed)                      # (4) 가독성 언더레이 추가

    # (NEW) 2패스: 배경 프롬프트/소품 계획 생성
    if args.bg_prompt:
        palette = extract_palette_hex(image_path, k=5)
        bg_plan = generate_bg_plan(model, processor, image_path, product_name, parsed, palette)
        # background 필드에 결합
        if "background" not in parsed or not isinstance(parsed["background"], dict):
            parsed["background"] = {}
        parsed["background"]["prompt"] = bg_plan.get("background_prompt", "")
        parsed["background"]["negative_prompt"] = bg_plan.get("negative_prompt", "")
        parsed["background"]["camera"] = bg_plan.get("camera", {})
        parsed["background"]["lighting"] = bg_plan.get("lighting", {})
        parsed["background"]["palette"] = bg_plan.get("palette", palette)
        # 소품은 레이아웃의 별도 섹션으로도 보존
        parsed.setdefault("background_objects", bg_plan.get("objects", []))

    # 출력/저장
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        print(f"[저장 완료] {args.save}")
    else:
        print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()