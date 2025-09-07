from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import random, os, math, itertools
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

# ------------------------------ утилиты текста ------------------------------

EN_WORDS = [
    "example","system","network","kernel","device","window","library","number","random","vector",
    "matrix","function","object","string","module","control","feature","signal","memory","thread",
    "policy","engine","layout","format","buffer","parser","packet","printer","window","screen"
]
RU_WORDS = [
    "пример","система","модуль","память","окно","устройство","библиотека","число","случайный","вектор",
    "матрица","функция","объект","строка","ядро","контроль","фича","сигнал","поток","политика",
    "движок","разметка","формат","буфер","парсер","пакет","принтер","экран","сеть","указатель"
]

def make_neologisms(lang="en", seed=123) -> List[str]:
    random.seed(seed)
    if lang.startswith("ru"):
        syll = ["зу", "ри", "фа", "кло", "ша", "мпи", "тру", "джа", "ско", "вэ", "йо", "пли"]
    else:
        syll = ["zo", "ri", "fa", "klo", "sha", "mpi", "tru", "dja", "sko", "ve", "yo", "pli"]
    neos = set()
    while len(neos) < 50:
        w = "".join(random.choices(syll, k=random.randint(2,4)))
        neos.add(w)
    return list(neos)

def letters_for_lang(alphabet:str) -> List[str]:
    return list("абвгдеёжзийклмнопрстуфхцчшщьыъэюя") if alphabet.lower().startswith("ru") else list("abcdefghijklmnopqrstuvwxyz")

def inject_typos(word:str, rate:float, alphabet:str) -> str:
    if rate <= 0: return word
    ops = []
    if len(word) >= 2: ops.append("swap")
    ops += ["sub","del","ins"]
    chars = letters_for_lang(alphabet)
    s = list(word)
    i = 0
    while i < len(s):
        if random.random() < rate:
            op = random.choice(ops)
            if op == "swap" and i < len(s)-1:
                s[i], s[i+1] = s[i+1], s[i]; i += 1
            elif op == "sub":
                s[i] = random.choice(chars)
            elif op == "del":
                del s[i]; i -= 1
            elif op == "ins":
                s.insert(i, random.choice(chars)); i += 1
        i += 1
    return "".join(s)

def make_text_tokens(
    mode:str="real_words",         # real_words | alphabet_mix
    alphabet:str="en",
    words_range:Tuple[int,int]=(5,30),
    letter_repeats:Tuple[int,int]=(1,6),
    use_typos:bool=True, typo_rate:float=0.05,
    use_neologisms:bool=True, neologism_rate:float=0.1,
    seed:Optional[int]=None
) -> List[str]:
    if seed is not None: random.seed(seed)
    tokens: List[str] = []
    if mode == "real_words":
        base = RU_WORDS if alphabet.startswith("ru") else EN_WORDS
        pool = base.copy()
        # расширим пул неологизмами
        neos = make_neologisms("ru" if alphabet.startswith("ru") else "en", seed=999)
        n_words = random.randint(*words_range)
        for _ in range(n_words):
            if use_neologisms and random.random() < neologism_rate:
                w = random.choice(neos)
            else:
                w = random.choice(pool)
            if use_typos:
                w = inject_typos(w, typo_rate, alphabet)
            tokens.append(w)
    else:
        # alphabet_mix: каждая буква появляется 1..6 раз, потом нарезаем на слова
        letters = letters_for_lang(alphabet)
        bag = []
        lo, hi = letter_repeats
        for ch in letters:
            bag += [ch] * random.randint(lo, hi)
        random.shuffle(bag)
        # разрезаем на слова длиной 2..8
        while bag:
            n = random.randint(2, 8)
            chunk = bag[:n]; bag = bag[n:]
            tokens.append("".join(chunk))
    return tokens

# ------------------------------ шум и фоны ------------------------------

def apply_noise(img: Image.Image, noise_type: str = "gaussian", noise_level: float = 0.02) -> Image.Image:
    if not noise_type or noise_type.lower() == "none" or noise_level <= 0:
        return img

    # приводим к массиву
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]  # H×W×1
    H, W, C = arr.shape
    nt = noise_type.lower()

    if nt == "gaussian":
        sigma = float(noise_level) * 255.0
        arr = np.clip(arr + np.random.normal(0.0, sigma, arr.shape), 0, 255)

    elif nt == "saltpepper":
        p = float(noise_level)
        p = max(0.0, min(p, 1.0))
        m = np.random.rand(H, W)  # H×W
        salt = m > 1.0 - p / 2.0
        pepper = m < p / 2.0
        arr[pepper, :] = 0.0
        arr[salt, :] = 255.0

    elif nt == "speckle":
        n = np.random.normal(0.0, float(noise_level), arr.shape)
        arr = np.clip(arr * (1.0 + n), 0, 255)

    elif nt == "poisson":
        a = np.clip(arr / 255.0, 0.0, 1.0)
        p = np.random.poisson(a * 255.0) / 255.0
        alpha = float(noise_level)
        arr = np.clip((1.0 - alpha) * a + alpha * p, 0.0, 1.0) * 255.0

    arr = arr.astype(np.uint8)
    if C == 1:
        arr = arr[..., 0]
        return Image.fromarray(arr, mode="L").convert("RGB")  # возвращаем RGB для консистентности
    elif C == 3:
        return Image.fromarray(arr, mode="RGB")
    elif C == 4:
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    else:
        # на всякий — схлопываем в RGB
        return Image.fromarray(arr[..., :3].astype(np.uint8), mode="RGB")


def paste_background_full(canvas:Image.Image, path:str):
    try:
        bg = Image.open(path).convert("RGB")
        bg = bg.resize(canvas.size, Image.BICUBIC)
        canvas.paste(bg)
    except Exception:
        pass

def paste_background_patches(canvas:Image.Image, paths:List[str], count:int=3, alpha:float=0.35, seed:Optional[int]=None):
    if seed is not None: random.seed(seed)
    if not paths: return
    W,H = canvas.size
    for _ in range(count):
        p = random.choice(paths)
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        # случайный размер патча
        scale = random.uniform(0.2, 0.7)
        w = int(W*scale*random.uniform(0.6,1.2))
        h = int(H*scale*random.uniform(0.6,1.2))
        im = im.resize((max(1,w), max(1,h)), Image.BICUBIC)
        x = random.randint(-w//4, W - 3*w//4)
        y = random.randint(-h//4, H - 3*h//4)
        overlay = Image.new("RGBA", im.size, (0,0,0,0))
        overlay.paste(im, (0,0))
        overlay.putalpha(int(alpha*255))
        canvas.paste(overlay, (x,y), overlay)

from PIL import Image, ImageDraw
import numpy as np, random, math

# --- NEW: смешанный фон ---
def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i]-a[i])*t) for i in range(3))

def paint_mixed_background(
    canvas: Image.Image,
    mode: str = "stripes",                # stripes|grid|patches|linear|radial
    colors=None,                          # список RGB, напр. [(255,255,255),(245,245,255),(255,245,245)]
    stripes: int = 5,
    orient: str = "v",                    # h|v для stripes
    grid=(3, 4),                          # rows, cols
    patch_count: int = 6,
    patch_scale=(0.2, 0.5),               # min,max доли от ширины/высоты
    gradient_stops: int = 3,              # для linear/radial — кол-во цветов в градиенте
    seed=None
):
    if not colors:
        colors = [(255,255,255), (245,245,245), (235,235,250), (250,235,235)]
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    W,H = canvas.size
    draw = ImageDraw.Draw(canvas)

    if mode == "stripes":
        n = max(1, stripes)
        if orient.lower().startswith("h"):
            h = math.ceil(H / n)
            for i in range(n):
                c = colors[i % len(colors)]
                y0, y1 = i*h, min(H, (i+1)*h)
                draw.rectangle([0, y0, W, y1], fill=c)
        else:
            w = math.ceil(W / n)
            for i in range(n):
                c = colors[i % len(colors)]
                x0, x1 = i*w, min(W, (i+1)*w)
                draw.rectangle([x0, 0, x1, H], fill=c)

    elif mode == "grid":
        rows, cols = grid
        rows = max(1, int(rows)); cols = max(1, int(cols))
        cell_w = math.ceil(W / cols)
        cell_h = math.ceil(H / rows)
        k = 0
        for r in range(rows):
            for c in range(cols):
                x0, y0 = c*cell_w, r*cell_h
                x1, y1 = min(W, (c+1)*cell_w), min(H, (r+1)*cell_h)
                draw.rectangle([x0,y0,x1,y1], fill=colors[k % len(colors)])
                k += 1

    elif mode == "patches":
        for _ in range(max(1, patch_count)):
            c = random.choice(colors)
            sw = random.uniform(*patch_scale)
            sh = random.uniform(*patch_scale)
            w = max(1, int(W*sw)); h = max(1, int(H*sh))
            x = random.randint(-w//6, max(0, W - 5*w//6))
            y = random.randint(-h//6, max(0, H - 5*h//6))
            draw.rectangle([x,y,x+w,y+h], fill=c)

    elif mode in ("linear","radial"):
        # готовим массив и мульти-стоп градиент
        arr = np.zeros((H, W, 3), dtype=np.float32)
        # выберем последовательность стопов
        stops = [colors[i % len(colors)] for i in range(max(2, gradient_stops))]
        if mode == "linear":
            # случайный угол
            angle = random.uniform(0, math.pi)
            # нормализованный вектор направления
            dx, dy = math.cos(angle), math.sin(angle)
            # координаты нормируем в [0,1]
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            nx = (xx - W/2) / max(W,1)
            ny = (yy - H/2) / max(H,1)
            t = (nx*dx + ny*dy)
            t = (t - t.min()) / (np.ptp(t) + 1e-6)
        else:  # radial
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            nx = (xx - W/2) / max(W/2,1)
            ny = (yy - H/2) / max(H/2,1)
            t = np.sqrt(nx*nx + ny*ny)
            t = np.clip(t, 0, 1)

        # многостопная интерполяция
        nseg = len(stops) - 1
        seg = np.clip((t * nseg).astype(int), 0, nseg-1)
        local_t = (t * nseg) - seg
        stops_np = np.array(stops, dtype=np.float32)
        c0 = stops_np[seg]
        c1 = stops_np[seg+1]
        arr = c0 + (c1 - c0) * local_t[..., None]
        canvas.paste(Image.fromarray(arr.astype(np.uint8), mode="RGB"))

    else:
        # неизвестный режим — ничего не делаем
        pass
    return canvas

# ------------------------------ разметка/стили ------------------------------

@dataclass
class SegmentStyle:
    font_path: Optional[str]
    font_size: int
    fill: Tuple[int,int,int]
    bg_fill: Optional[Tuple[int,int,int]] = None  # фон под текстом (лента)

@dataclass
class DrawnToken:
    token: str
    bbox: Tuple[int,int,int,int]   # x0,y0,x1,y1
    style: SegmentStyle
    line_idx: int
    hyphenated: bool = False

# ------------------------------ основной генератор ------------------------------

def generate_ocr_image(
    # канвас
    resolution:Tuple[int,int]=(1400, 1000),
    bg_color:Tuple[int,int,int]=(255,255,255),
    # базовые цвета
    font_color:Tuple[int,int,int]=(0,0,0),
    uniform_colors:bool=True,   # если False — можем красить сегменты по-разному
    # текст
    alphabet:str="en",          # en|ru
    text_mode:str="real_words", # real_words|alphabet_mix
    words_range:Tuple[int,int]=(5,30),
    letter_repeats:Tuple[int,int]=(1,6),
    hyphen_breaks:bool=True,
    hyphen_prob:float=0.2,      # вероятность переносить слово с дефисом
    use_typos:bool=True, typo_rate:float=0.04,
    use_neologisms:bool=True, neologism_rate:float=0.1,
    # шрифты/размеры
    fonts:Optional[List[str]]=None,  # пути к ttf/otf
    base_font_path:Optional[str]=None, # запасной
    base_font_size:int=48,
    allow_mixed_fonts:bool=False,
    allow_mixed_sizes:bool=False,
    sizes_pool:Optional[List[int]]=None,  # если None, возьму окрестности base_font_size
    # фоновые «ленты» под отдельные части текста
    allow_segment_backgrounds:bool=False,
    segment_bg_prob:float=0.25,
    segment_bg_colors:Optional[List[Tuple[int,int,int]]]=None,
    # фоновые картинки
    background_images:Optional[List[str]]=None,
    background_mode:str="none",  # none|full|patches|both
    bg_patch_count:int=3,
    bg_patch_alpha:float=0.35,
    # шум
    noise_type:str="gaussian",
    noise_level:float=0.02,
    # прочее
    margin:int=40,
    line_spacing:float=1.25,
    seed:Optional[int]=None,
    return_meta:bool=True,
    colors_pool:Optional[List[Tuple[int,int,int]]]=None,
    mixed_background=False,
    mixed_bg_mode="stripes",         # stripes|grid|patches|linear|radial
    mixed_bg_colors=None,
    mixed_bg_stripes=5,
    mixed_bg_orient="v",
    mixed_bg_grid=(3,4),
    mixed_bg_patch_count=6,
    mixed_bg_patch_scale=(0.2,0.5),
    mixed_bg_gradient_stops=3,
):
    """
    Генерирует изображение и разметку.
    Возвращает (image, text, tokens, metadata) где metadata — список DrawnToken (as dict).
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    W,H = resolution
    canvas = Image.new("RGB", (W,H), bg_color)

    if mixed_background:
        canvas = paint_mixed_background(
            canvas,
            mode=mixed_bg_mode,
            colors=mixed_bg_colors,
            stripes=mixed_bg_stripes,
            orient=mixed_bg_orient,
            grid=mixed_bg_grid,
            patch_count=mixed_bg_patch_count,
            patch_scale=mixed_bg_patch_scale,
            gradient_stops=mixed_bg_gradient_stops,
            seed=seed
        )

    # потом — фоновые картинки, если просили
    if background_images and background_mode in ("full","both"):
        paste_background_full(canvas, random.choice(background_images))
    if background_images and background_mode in ("patches","both"):
        paste_background_patches(canvas, background_images, count=bg_patch_count, alpha=bg_patch_alpha, seed=seed)

    # выбор шрифтов/размеров
    if not fonts or len(fonts)==0:
        fonts = []
    font_candidates = fonts + [base_font_path, "DejaVuSans.ttf", "NotoSans-Regular.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]
    font_candidates = [p for p in font_candidates if p]  # убрать None
    def load_font(sz:int) -> ImageFont.FreeTypeFont:
        for p in font_candidates:
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                continue
        return ImageFont.load_default()

    if sizes_pool is None:
        sizes_pool = [max(12, int(base_font_size * s)) for s in (0.8, 1.0, 1.2, 1.5)]
    if not colors_pool:
        colors_pool = [
            font_color,
            (20,20,20), (40,40,40), (60,60,60),
            (10,10,10)
        ] if not uniform_colors else [font_color]
    seg_bg_pool = segment_bg_colors or [(240,240,240), (230,230,255), (255,235,235), (235,255,235)]

    # готовим текстовые токены
    tokens = make_text_tokens(
        mode=text_mode, alphabet=alphabet, words_range=words_range, letter_repeats=letter_repeats,
        use_typos=use_typos, typo_rate=typo_rate,
        use_neologisms=use_neologisms, neologism_rate=neologism_rate, seed=seed
    )

    # раскладываем в строки с учётом шрифтов/размеров/фоновых лент
    draw = ImageDraw.Draw(canvas)
    x, y = margin, margin
    max_w = W - margin*2
    line_idx = 0
    line_tokens: List[DrawnToken] = []
    all_tokens: List[DrawnToken] = []

    def pick_style() -> SegmentStyle:
        size = random.choice(sizes_pool) if allow_mixed_sizes else base_font_size
        fp = random.choice(font_candidates) if allow_mixed_fonts else (font_candidates[0] if font_candidates else None)
        fill = random.choice(colors_pool)
        bgf = (random.choice(seg_bg_pool) if (allow_segment_backgrounds and random.random()<segment_bg_prob) else None)
        return SegmentStyle(font_path=fp, font_size=size, fill=fill, bg_fill=bgf)

    # предварительная функция измерения
    def text_size(t:str, font:ImageFont.FreeTypeFont):
        b = draw.textbbox((0,0), t, font=font)
        return b[2]-b[0], b[3]-b[1]

    curr_style = pick_style()
    curr_font = load_font(curr_style.font_size)
    space_w, space_h = text_size(" ", curr_font)
    line_h = int(curr_style.font_size * line_spacing)

    def new_line():
        nonlocal x, y, line_idx, line_h, curr_style, curr_font, space_w, space_h
        x = margin
        y += line_h
        line_idx += 1
        curr_style = pick_style()
        curr_font = load_font(curr_style.font_size)
        space_w, space_h = text_size(" ", curr_font)
        line_h = int(max(line_h, curr_style.font_size * line_spacing))

    for tok in tokens:
        # возможно сменить стиль на токене
        if allow_mixed_fonts or allow_mixed_sizes or allow_segment_backgrounds or (not uniform_colors):
            curr_style = pick_style()
            curr_font = load_font(curr_style.font_size)
            space_w, space_h = text_size(" ", curr_font)
            line_h = max(line_h, int(curr_style.font_size * line_spacing))

        # проверка влезания
        tw, th = text_size(tok, curr_font)
        need_w = (0 if x == margin else space_w) + tw
        # переносы
        hyphenated = False
        if x + need_w > margin + max_w:
            if hyphen_breaks and len(tok) > 4 and random.random() < hyphen_prob:
                # перенос с дефисом
                cut = random.randint(2, len(tok)-2)
                left, right = tok[:cut] + "-", tok[cut:]
                # рисуем левую часть на этой строке, правую — на следующей
                # левая часть
                ltw,_ = text_size(left, curr_font)
                if x != margin:
                    x += space_w
                # фон под сегмент
                if curr_style.bg_fill:
                    draw.rectangle([x-2, y-2, x+ltw+2, y+th+2], fill=curr_style.bg_fill)
                draw.text((x, y), left, font=curr_font, fill=curr_style.fill)
                dt = DrawnToken(token=left, bbox=(x,y,x+ltw,y+th), style=curr_style, line_idx=line_idx, hyphenated=True)
                all_tokens.append(dt)
                # новая строка для правой части
                new_line()
                tw2, th2 = text_size(right, curr_font)
                if curr_style.bg_fill:
                    draw.rectangle([x-2, y-2, x+tw2+2, y+th2+2], fill=curr_style.bg_fill)
                draw.text((x, y), right, font=curr_font, fill=curr_style.fill)
                dt2 = DrawnToken(token=right, bbox=(x,y,x+tw2,y+th2), style=curr_style, line_idx=line_idx, hyphenated=True)
                all_tokens.append(dt2)
                x += tw2
                continue
            # перенос без дефиса
            new_line()
            tw, th = text_size(tok, curr_font)
            need_w = tw

        if x != margin:
            x += space_w

        # фон-лента под токен при необходимости
        if curr_style.bg_fill:
            draw.rectangle([x-2, y-2, x+tw+2, y+th+2], fill=curr_style.bg_fill)
        draw.text((x, y), tok, font=curr_font, fill=curr_style.fill)
        all_tokens.append(DrawnToken(token=tok, bbox=(x,y,x+tw,y+th), style=curr_style, line_idx=line_idx))
        x += tw

        # выход за нижнюю границу — досрочно стоп
        if y + line_h > H - margin:
            break

    # итоговый текст (с восстановлением переносов с дефисом как есть на картинке)
    # Соберём строки по line_idx и позициям
    lines_map: Dict[int, List[DrawnToken]] = {}
    for dt in all_tokens:
        lines_map.setdefault(dt.line_idx, []).append(dt)
    ordered_lines = []
    for idx in sorted(lines_map.keys()):
        line = sorted(lines_map[idx], key=lambda d: d.bbox[0])
        ordered_lines.append(" ".join(d.token for d in line).replace(" -", "-"))
    final_text = "\n".join(ordered_lines)

    # добавляем шум в самом конце
    noisy = apply_noise(canvas, noise_type=noise_type, noise_level=noise_level)

    meta = [ {**asdict(t), "style": asdict(t.style)} for t in all_tokens ]
    return (noisy, final_text, tokens, meta) if return_meta else (noisy, final_text)

# ------------------------------ пример использования ------------------------------
# img, final_text, tokens, meta = generate_ocr_image(
#     resolution=(1600, 900),
#     bg_color=(250,250,250),
#     font_color=(15,15,15),
#     uniform_colors=False,
#     alphabet="ru",
#     text_mode="real_words",     # или "alphabet_mix"
#     words_range=(12, 24),
#     hyphen_breaks=True, hyphen_prob=0.35,
#     use_typos=True, typo_rate=0.06,
#     use_neologisms=True, neologism_rate=0.12,
#     fonts=["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
#            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
#     base_font_size=48,
#     allow_mixed_fonts=True,
#     allow_mixed_sizes=True,
#     sizes_pool=[32, 40, 48, 56, 64],
#     allow_segment_backgrounds=True,
#     segment_bg_colors=[(240,240,255),(255,240,240),(240,255,240)],
#     background_images=["/path/to/bg1.jpg","/path/to/bg2.jpg"],
#     background_mode="both",  # none|full|patches|both
#     bg_patch_count=4, bg_patch_alpha=0.30,
#     noise_type="speckle", noise_level=0.03,
#     seed=42,
# )
# img.save("sample_ocr.png")
# print(final_text)
# print(tokens[:10], "...")
# print(meta[0])
