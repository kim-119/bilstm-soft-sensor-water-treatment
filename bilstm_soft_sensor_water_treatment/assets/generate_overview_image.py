import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "images")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "project_overview_ko.png")

FONT_REG = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD = r"C:\Windows\Fonts\malgunbd.ttf"


def font(bold, size):
    path = FONT_BOLD if bold else FONT_REG
    return ImageFont.truetype(path, size)


BG = (247, 249, 252)
CARD_BG = (255, 255, 255)
CARD_BORDER = (223, 228, 234)
TITLE_DARK = (23, 37, 58)
TEXT_DARK = (45, 55, 72)
TEXT_MUTED = (90, 101, 117)

ACCENTS = [
    (37, 99, 235),
    (13, 148, 136),
    (124, 58, 237),
    (217, 119, 6),
    (190, 24, 93),
    (5, 150, 105),
]

CARDS = [
    {
        "title": "프로젝트 목적",
        "sub": "Soft Sensor 설계 목표",
        "items": [
            "실시간 센서 데이터 기반 수질 예측",
            "측정 지연 항목(NH3-N, COD 등) 간접 추정",
            "운영자 의사결정 보조 (자동제어 아님)",
        ],
    },
    {
        "title": "입력 데이터",
        "sub": "공정 센서 변수",
        "items": [
            "DO (용존산소) · pH · ORP",
            "수온 · MLSS · 유입 유량",
            "폭기량 · 반송률 · 유입 부하",
            "시간 정보 (시간/요일/계절)",
        ],
    },
    {
        "title": "모델 흐름",
        "sub": "데이터 → 예측 파이프라인",
        "items": [
            "센서 데이터 수집",
            "결측치 · 이상치 처리",
            "정규화 (스케일링)",
            "슬라이딩 윈도우 구성",
            "Bi-LSTM 학습",
            "수질 예측값 출력",
        ],
    },
    {
        "title": "예측 대상",
        "sub": "출력 수질 항목",
        "items": [
            "NH3-N (암모니아성 질소)",
            "COD (화학적 산소요구량)",
            "BOD (생물학적 산소요구량)",
            "TN (총질소)",
            "방류수질 기준 초과 가능성",
        ],
    },
    {
        "title": "하드웨어 환경",
        "sub": "학습 GPU 조건",
        "items": [
            "NVIDIA GeForce RTX 3050",
            "CUDA 기반 GPU 학습",
            "경량 Bi-LSTM 구조",
            "배치 크기 16 기준",
        ],
    },
    {
        "title": "기대 효과",
        "sub": "운영 보조 가치",
        "items": [
            "실시간 수질 추정 보조",
            "측정 지연 보완",
            "기준 초과 조기 경고 지원",
            "공정 운영 안정성 향상 지원",
        ],
    },
]

W, H = 1600, 1120
MARGIN = 50
COLS, ROWS = 3, 2
COL_GAP, ROW_GAP = 36, 36

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def rounded(box, radius, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_center(cx, y, s, fnt, fill):
    w = d.textlength(s, font=fnt)
    d.text((cx - w / 2, y), s, font=fnt, fill=fill)


title_h = 150
rounded((MARGIN, MARGIN, W - MARGIN, MARGIN + title_h), 18,
        fill=(23, 37, 58))
text_center(W / 2, MARGIN + 28,
            "Bi-LSTM 기반 수처리 공정 Soft Sensor 모델 설계",
            font(True, 44), (255, 255, 255))
text_center(W / 2, MARGIN + 92,
            "실시간 센서 데이터로 수질 항목을 간접 예측하는 운영자 의사결정 보조 모델",
            font(False, 24), (183, 197, 219))

grid_top = MARGIN + title_h + 34
grid_bottom = H - MARGIN - 28
avail_w = W - 2 * MARGIN - (COLS - 1) * COL_GAP
avail_h = grid_bottom - grid_top - (ROWS - 1) * ROW_GAP
card_w = avail_w / COLS
card_h = avail_h / ROWS

for idx, card in enumerate(CARDS):
    r = idx // COLS
    c = idx % COLS
    x0 = MARGIN + c * (card_w + COL_GAP)
    y0 = grid_top + r * (card_h + ROW_GAP)
    x1 = x0 + card_w
    y1 = y0 + card_h
    accent = ACCENTS[idx]

    rounded((x0, y0, x1, y1), 16, fill=CARD_BG, outline=CARD_BORDER, width=2)
    rounded((x0, y0, x1, y0 + 70), 16, fill=accent)
    d.rectangle((x0, y0 + 40, x1, y0 + 70), fill=accent)

    cnum = str(idx + 1)
    cy = y0 + 35
    cx = x0 + 38
    d.ellipse((cx - 20, cy - 20, cx + 20, cy + 20), fill=(255, 255, 255))
    nf = font(True, 26)
    nw = d.textlength(cnum, font=nf)
    d.text((cx - nw / 2, cy - 17), cnum, font=nf, fill=accent)

    d.text((x0 + 70, y0 + 12), card["title"], font=font(True, 28),
           fill=(255, 255, 255))
    d.text((x0 + 70, y0 + 44), card["sub"], font=font(False, 16),
           fill=(235, 240, 248))

    ty = y0 + 92
    item_font = font(False, 23)
    for it in card["items"]:
        d.ellipse((x0 + 28, ty + 9, x0 + 38, ty + 19), fill=accent)
        d.text((x0 + 50, ty), it, font=item_font, fill=TEXT_DARK)
        ty += 40

note = "※ 실제 성능 수치(MAE/RMSE/R²)는 현장 데이터 확보·학습 후 산출 / 본 모델은 자동제어가 아닌 운영 보조용"
text_center(W / 2, H - 34, note, font(False, 18), TEXT_MUTED)

img.save(OUT_PNG, "PNG")
print("Saved:", OUT_PNG, img.size)


OUT_SVG = os.path.join(OUT_DIR, "project_overview_ko.svg")


def hx(rgb):
    return "#%02x%02x%02x" % rgb


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


FONT_FAMILY = "'Malgun Gothic','맑은 고딕','Apple SD Gothic Neo','Noto Sans KR',sans-serif"
svg = []
svg.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" font-family="{FONT_FAMILY}">')
svg.append(f'<rect width="{W}" height="{H}" fill="{hx(BG)}"/>')

svg.append(
    f'<rect x="{MARGIN}" y="{MARGIN}" width="{W-2*MARGIN}" height="{title_h}" '
    f'rx="18" fill="{hx((23,37,58))}"/>')
svg.append(
    f'<text x="{W/2}" y="{MARGIN+78}" text-anchor="middle" fill="#ffffff" '
    f'font-size="44" font-weight="700">'
    f'Bi-LSTM 기반 수처리 공정 Soft Sensor 모델 설계</text>')
svg.append(
    f'<text x="{W/2}" y="{MARGIN+118}" text-anchor="middle" fill="{hx((183,197,219))}" '
    f'font-size="24">실시간 센서 데이터로 수질 항목을 간접 예측하는 운영자 의사결정 보조 모델</text>')

for idx, card in enumerate(CARDS):
    r = idx // COLS
    c = idx % COLS
    x0 = MARGIN + c * (card_w + COL_GAP)
    y0 = grid_top + r * (card_h + ROW_GAP)
    accent = ACCENTS[idx]
    svg.append(
        f'<rect x="{x0}" y="{y0}" width="{card_w}" height="{card_h}" rx="16" '
        f'fill="{hx(CARD_BG)}" stroke="{hx(CARD_BORDER)}" stroke-width="2"/>')
    svg.append(f'<path d="M{x0},{y0+16} a16,16 0 0 1 16,-16 h{card_w-32} '
               f'a16,16 0 0 1 16,16 v54 h{-card_w} z" fill="{hx(accent)}"/>')
    svg.append(f'<circle cx="{x0+38}" cy="{y0+35}" r="20" fill="#ffffff"/>')
    svg.append(f'<text x="{x0+38}" y="{y0+44}" text-anchor="middle" '
               f'fill="{hx(accent)}" font-size="26" font-weight="700">{idx+1}</text>')
    svg.append(f'<text x="{x0+70}" y="{y0+38}" fill="#ffffff" font-size="28" '
               f'font-weight="700">{esc(card["title"])}</text>')
    svg.append(f'<text x="{x0+70}" y="{y0+60}" fill="{hx((235,240,248))}" '
               f'font-size="16">{esc(card["sub"])}</text>')
    ty = y0 + 110
    for it in card["items"]:
        svg.append(f'<circle cx="{x0+33}" cy="{ty-7}" r="5" fill="{hx(accent)}"/>')
        svg.append(f'<text x="{x0+50}" y="{ty}" fill="{hx(TEXT_DARK)}" '
                   f'font-size="23">{esc(it)}</text>')
        ty += 40

svg.append(f'<text x="{W/2}" y="{H-26}" text-anchor="middle" '
           f'fill="{hx(TEXT_MUTED)}" font-size="18">'
           f'※ 실제 성능 수치(MAE/RMSE/R²)는 현장 데이터 확보·학습 후 산출 / '
           f'본 모델은 자동제어가 아닌 운영 보조용</text>')
svg.append('</svg>')

with open(OUT_SVG, "w", encoding="utf-8") as f:
    f.write("\n".join(svg))
print("Saved:", OUT_SVG)
