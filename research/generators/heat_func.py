import random
from math import ceil

from generators.gen_pics_complex import generate_ocr_image

def _clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def _lerp(a, b, t): 
    return a + (b - a) * t

def _choice(seq, default=None):
    return random.choice(seq) if seq else default

def _heat_prob(h, lo=0.0, hi=1.0):
    """Вероятность как линейная функция от heat в [0..100]."""
    t = _clamp(h/100.0, 0.0, 1.0)
    return _lerp(lo, hi, t)

def _mix_palette(gray_base=(0,0,0)):
    # наборы более заметных цветов для текста/фона
    neon = [(255,60,0), (0,180,255), (255,0,200), (20,220,120), (255,220,0), (120,0,200)]
    vivid = [(0,0,0), (20,20,20), (200,0,0), (0,140,0), (0,0,200), (180,120,0)]
    mild  = [(15,15,15), (40,40,40), (70,70,70)]
    return gray_base, mild, vivid, neon

def generate_ocr_image_by_heat(
    heat:int,
    resolution=(1600, 900),
    alphabet="en",
    fonts=None,                # список путей к ttf/otf (опционально)
    background_images=None,    # опц. картинки для бэкграунда
    base_font_size=48,
    seed=None
):
    """
    heat ∈ [0..100]:
      0   — белый фон, чёрный текст, один шрифт/размер, без шума
      50  — умеренно пёстро (цвета, небольшой шум, пары шрифтов/размеров)
      100 — кислотный фон, разноцветные буквы, куча шрифтов/размеров, шум повсюду
    """
    heat = int(_clamp(heat, 0, 100))
    if seed is not None:
        random.seed(seed)

    # Базовые величины от heat
    t = heat / 100.0
    # шум
    noise_level = round(_lerp(0.0, 0.06, t), 4)
    noise_types = ["none","gaussian","speckle","saltpepper","poisson"]
    if heat == 0:
        noise_type = "none"
    else:
        # чем больше heat — тем «злее» шум с большей вероятностью
        pool = (["gaussian"]*3 + ["speckle"]*2 + ["saltpepper"]*2 + ["poisson"])
        noise_type = _choice(pool)

    # фон: смешанный включаем примерно с 20% на heat=20 и ~90% на heat=100
    use_mixed_bg = random.random() < _heat_prob(heat, lo=0.0, hi=0.9)
    mixed_modes = ["stripes","grid","patches","linear","radial"]
    mixed_bg_mode = _choice(mixed_modes) if use_mixed_bg else "stripes"

    # контрастность фона: от «почти белый» к «вырви глаз»
    def rand_color(bright=True):
        if bright:
            return (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        # пастельней
        base = random.randint(200,255)
        r = _clamp(base + random.randint(-40, 40), 120, 255)
        g = _clamp(base + random.randint(-40, 40), 120, 255)
        b = _clamp(base + random.randint(-40, 40), 120, 255)
        return (int(r),int(g),int(b))

    # палитры фона по heat
    if heat <= 10:
        mixed_bg_colors = [(255,255,255),(250,250,250)]
    elif heat <= 40:
        mixed_bg_colors = [
            (255,255,255),
            (230,240,255),
            (255,235,235),
            (235,255,240)
        ]
    elif heat <= 75:
        mixed_bg_colors = [
            (255,255,255),
            (220,240,255),
            (255,220,235),
            (220,255,235),
            (245,245,245)
        ]
    else:
        # кислотно и смело
        mixed_bg_colors = [rand_color(bright=True) for _ in range(5)]

    # параметры смешанного фона
    mixed_bg_stripes = random.randint(4, 9)
    mixed_bg_orient  = _choice(["h","v"])
    mixed_bg_grid    = (random.randint(3,5), random.randint(4,7))
    mixed_bg_patch_count = random.randint(5, 10)
    mixed_bg_patch_scale = (0.15, 0.45)
    mixed_bg_gradient_stops = random.randint(3, 5)

    # фоновые картинки
    bg_mode = "none"
    if background_images:
        # шанс использовать изображения как патчи/фулл растёт с heat
        r = random.random()
        if r < _heat_prob(heat, 0.0, 0.2):
            bg_mode = "full"
        elif r < _heat_prob(heat, 0.2, 0.6):
            bg_mode = "patches"
        elif r < _heat_prob(heat, 0.3, 0.8):
            bg_mode = "both"

    bg_patch_count = random.randint(2, 5)
    bg_patch_alpha = round(_lerp(0.18, 0.35, t), 2)

    # цвета текста: от монохрома к неону
    base_col, mild, vivid, neon = _mix_palette()
    if heat == 0:
        colors_pool = [(0,0,0)]
        uniform_colors = True
    elif heat < 30:
        colors_pool = mild
        uniform_colors = False
    elif heat < 70:
        colors_pool = vivid
        uniform_colors = False
    else:
        colors_pool = neon + vivid
        uniform_colors = False

    # шрифты/размеры/ленты
    allow_mixed_fonts  = random.random() < _heat_prob(heat, 0.0, 0.9)
    allow_mixed_sizes  = random.random() < _heat_prob(heat, 0.2, 1.0)
    allow_seg_bg       = random.random() < _heat_prob(heat, 0.0, 0.7)
    seg_bg_colors = None
    if allow_seg_bg:
        if heat < 50:
            seg_bg_colors = [(240,240,255),(255,240,240),(240,255,240)]
        else:
            # более контрастные ленты
            seg_bg_colors = [(255,245,180),(210,235,255),(255,210,230),(210,255,225)]

    # масштаб размеров
    spread = _lerp(0.2, 1.2, t)  # насколько сильнее отклоняемся от базового
    sizes_pool = sorted(set([
        max(12, int(base_font_size * s)) 
        for s in [1.0 - 0.3*spread, 1.0, 1.0 + 0.3*spread, 1.0 + 0.6*spread, 1.0 + spread]
    ]))

    # переносы/дефисы — чаще при большом heat
    hyphen_breaks = random.random() < _heat_prob(heat, 0.2, 0.95)
    hyphen_prob   = round(_lerp(0.15, 0.6, t), 2)

    # опечатки/неологизмы — больше при большем heat
    use_typos       = random.random() < _heat_prob(heat, 0.0, 1.0)
    typo_rate       = round(_lerp(0.00, 0.09, t), 3)
    use_neologisms  = random.random() < _heat_prob(heat, 0.1, 1.0)
    neologism_rate  = round(_lerp(0.05, 0.2, t), 3)

    # режим текста: на высоком heat периодически переключаемся на alphabet_mix
    text_mode = "real_words"
    if random.random() < _heat_prob(heat, 0.0, 0.45):
        text_mode = "alphabet_mix"
    words_range = (5, 30) if text_mode == "real_words" else (14, 28)
    letter_repeats = (1, 6)

    # базовые цвета и фон
    bg_color = (255,255,255) if heat < 5 else _choice([(255,255,255),(250,250,250)])
    font_color = (0,0,0)

    params = dict(
        resolution=resolution,
        bg_color=bg_color,
        font_color=font_color,
        uniform_colors=uniform_colors,
        alphabet=alphabet,
        text_mode=text_mode,
        words_range=words_range,
        letter_repeats=letter_repeats,
        hyphen_breaks=hyphen_breaks,
        hyphen_prob=hyphen_prob,
        use_typos=use_typos, typo_rate=typo_rate,
        use_neologisms=use_neologisms, neologism_rate=neologism_rate,
        fonts=fonts or [],
        base_font_path=(fonts[0] if fonts else None),
        base_font_size=base_font_size,
        allow_mixed_fonts=allow_mixed_fonts,
        allow_mixed_sizes=allow_mixed_sizes,
        sizes_pool=sizes_pool,
        allow_segment_backgrounds=allow_seg_bg,
        segment_bg_prob=0.35,
        segment_bg_colors=seg_bg_colors,
        background_images=background_images or [],
        background_mode=bg_mode,
        bg_patch_count=bg_patch_count,
        bg_patch_alpha=bg_patch_alpha,
        noise_type=noise_type,
        noise_level=noise_level,
        margin=40,
        line_spacing=1.25,
        seed=seed,
        # смешанный фон
        mixed_background=use_mixed_bg,
        mixed_bg_mode=mixed_bg_mode,
        mixed_bg_colors=colors_pool if not use_mixed_bg else mixed_bg_colors,  # если mixed off — не важно
        mixed_bg_stripes=mixed_bg_stripes,
        mixed_bg_orient=mixed_bg_orient,
        mixed_bg_grid=mixed_bg_grid,
        mixed_bg_patch_count=mixed_bg_patch_count,
        mixed_bg_patch_scale=mixed_bg_patch_scale,
        mixed_bg_gradient_stops=mixed_bg_gradient_stops,
        # цвета текста
        colors_pool=colors_pool,
    )

    img, final_text, tokens, meta = generate_ocr_image(**params)
    return img, final_text, tokens, meta, params

