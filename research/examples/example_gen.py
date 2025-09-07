import os, json, random
from PIL import ImageFont
import datetime
os.makedirs("demo_out", exist_ok=True)

def pick_existing_fonts(candidates):
    out=[]
    for p in candidates or []:
        try:
            ImageFont.truetype(p, 24); out.append(p)
        except:
            continue
    return out

def generate_demo_images(
    outdir="demo_out",
    base_font_size=48,
    fonts=None,
    background_images=None,
    seed=7
):
    os.makedirs(outdir, exist_ok=True)
    random.seed(seed)

    fonts = pick_existing_fonts(fonts or [
        "DejaVuSans.ttf","DejaVuSerif.ttf","NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ])
    bg_images = [p for p in (background_images or []) if os.path.exists(p)]

    cases = [
        dict(name="uniform_clean",
             resolution=(1400,900), bg_color=(250,250,250), font_color=(15,15,15),
             uniform_colors=True, allow_mixed_fonts=False, allow_mixed_sizes=False,
             allow_segment_backgrounds=False, colors_pool=[(15,15,15)],
             alphabet="en", text_mode="real_words", words_range=(12,20),
             noise_type="none", noise_level=0.0, background_mode="none"),

        dict(name="colors_only",
             resolution=(1400,900), bg_color=(255,255,255),
             uniform_colors=False, allow_mixed_fonts=False, allow_mixed_sizes=False,
             allow_segment_backgrounds=False, colors_pool=[(0,0,0),(220,0,0),(0,140,0),(0,0,200),(180,120,0)],
             alphabet="ru", text_mode="real_words", words_range=(10,18),
             noise_type="gaussian", noise_level=0.02, background_mode="none"),

        dict(name="mixed_fonts_sizes",
             resolution=(1600,900), bg_color=(245,245,250), font_color=(10,10,10),
             uniform_colors=False, allow_mixed_fonts=True, allow_mixed_sizes=True,
             allow_segment_backgrounds=True, colors_pool=[(0,0,0),(10,10,10),(40,40,40),(0,90,160)],
             alphabet="en", text_mode="real_words", words_range=(16,26),
             sizes_pool=[32,40,48,56,64],
             noise_type="speckle", noise_level=0.03, background_mode="none"),

        dict(name="bg_stripes",
             resolution=(1600,900), bg_color=(255,255,255),
             mixed_background=True, mixed_bg_mode="stripes", mixed_bg_orient="v", mixed_bg_stripes=7,
             mixed_bg_colors=[(255,255,255),(245,245,255),(255,245,245),(245,255,245)],
             uniform_colors=False, allow_mixed_sizes=True, sizes_pool=[32,40,48,56],
             alphabet="en", text_mode="real_words", words_range=(14,24),
             noise_type="none", noise_level=0.0, background_mode="none"),

        dict(name="bg_grid",
             resolution=(1600,900), bg_color=(255,255,255),
             mixed_background=True, mixed_bg_mode="grid", mixed_bg_grid=(4,6),
             mixed_bg_colors=[(252,252,252),(240,248,255),(255,240,245),(245,255,250)],
             uniform_colors=False, allow_mixed_fonts=True, allow_mixed_sizes=True,
             fonts=None, sizes_pool=[28,36,44,52,60],
             alphabet="ru", text_mode="real_words", words_range=(16,26),
             noise_type="gaussian", noise_level=0.02, background_mode="none"),

        dict(name="bg_patches",
             resolution=(1600,900), bg_color=(255,255,255),
             mixed_background=True, mixed_bg_mode="patches", mixed_bg_patch_count=8, mixed_bg_patch_scale=(0.15,0.4),
             mixed_bg_colors=[(255,255,255),(245,245,255),(255,245,230),(235,255,235),(240,240,240)],
             uniform_colors=False, allow_mixed_fonts=False, allow_mixed_sizes=True,
             sizes_pool=[30,38,46,54],
             alphabet="en", text_mode="alphabet_mix", letter_repeats=(1,6),
             hyphen_breaks=True, hyphen_prob=0.5,
             noise_type="saltpepper", noise_level=0.015, background_mode="none"),

        dict(name="bg_linear_gradient",
             resolution=(1600,1000), bg_color=(255,255,255),
             mixed_background=True, mixed_bg_mode="linear", mixed_bg_gradient_stops=4,
             mixed_bg_colors=[(255,255,255),(246,248,255),(238,242,255),(232,238,255)],
             uniform_colors=False, allow_mixed_fonts=True, allow_mixed_sizes=True,
             sizes_pool=[28,36,44,52,60],
             alphabet="en", text_mode="real_words", words_range=(20,30),
             noise_type="speckle", noise_level=0.02, background_mode="none"),

        dict(name="bg_radial_gradient_with_img_patches",
             resolution=(1600,1000), bg_color=(255,255,255),
             mixed_background=True, mixed_bg_mode="radial", mixed_bg_gradient_stops=3,
             mixed_bg_colors=[(255,255,255),(210,230,255),(180,205,255)],
             uniform_colors=False, allow_mixed_fonts=True, allow_mixed_sizes=True,
             sizes_pool=[30,38,46,54,62],
             alphabet="ru", text_mode="real_words", words_range=(18,28),
             noise_type="poisson", noise_level=0.04,
             background_mode=("patches" if bg_images else "none"), bg_patch_count=3, bg_patch_alpha=0.28),

        dict(name="bg_stripes_fullimg_combo",
             resolution=(1600,1000), bg_color=(255,255,255),
             mixed_background=True, mixed_bg_mode="stripes", mixed_bg_orient="h", mixed_bg_stripes=6,
             mixed_bg_colors=[(255,255,255),(255,220,0),(0,200,255)],
             uniform_colors=False, allow_mixed_fonts=True, allow_mixed_sizes=True,
             sizes_pool=[30,38,46,54,62],
             alphabet="en", text_mode="real_words", words_range=(18,28),
             noise_type="gaussian", noise_level=0.015,
             background_mode=("both" if bg_images else "none"), bg_patch_count=2, bg_patch_alpha=0.22),

        # прежние примеры с фонкартинками для полноты
        dict(name="with_background_patches",
            resolution=(1600,1000), bg_color=(255,255,255), font_color=(0,0,0),
            # смешанный фон включаем
            mixed_background=True,
            mixed_bg_mode="patches",  # stripes|grid|patches|linear|radial
            mixed_bg_patch_count=8,
            mixed_bg_patch_scale=(0.15,0.4),
            mixed_bg_colors=[(245,245,245),(210,230,255),(255,220,200),(200,255,210),(230,230,230)],
            # остальное как было
            uniform_colors=False, allow_mixed_fonts=True, allow_mixed_sizes=True,
            allow_segment_backgrounds=True, colors_pool=[(0,0,0),(40,40,40)],
            alphabet="en", text_mode="real_words", words_range=(20,30),
            sizes_pool=[28,36,44,52,60],
            noise_type="poisson", noise_level=0.06,
            background_mode=("patches" if bg_images else "none"),
            bg_patch_count=4, bg_patch_alpha=0.35),


        dict(name="full_bg_combo",
             resolution=(1600,1000), bg_color=(255,255,255), font_color=(10,10,10),
             uniform_colors=False, allow_mixed_fonts=True, allow_mixed_sizes=True,
             allow_segment_backgrounds=True, colors_pool=[(0,0,0),(0,90,160),(160,60,0)],
             alphabet="ru", text_mode="real_words", words_range=(18,28),
             mixed_background=True,
             mixed_bg_colors=[(99, 255, 141),(253, 107, 255),(0,200,255)],
             sizes_pool=[30,38,46,54,62],
             noise_type="speckle", noise_level=0.025,
             background_mode=("both" if bg_images else "none"),
             bg_patch_count=3, bg_patch_alpha=0.28),
    ]

    made=[]
    for i, cfg in enumerate(cases, 1):
        params = dict(
            resolution=cfg.get("resolution"),
            bg_color=cfg.get("bg_color"),
            font_color=cfg.get("font_color", (0,0,0)),
            uniform_colors=cfg.get("uniform_colors", True),
            alphabet=cfg.get("alphabet","en"),
            text_mode=cfg.get("text_mode","real_words"),
            words_range=cfg.get("words_range", (5,30)),
            letter_repeats=cfg.get("letter_repeats", (1,6)),
            hyphen_breaks=cfg.get("hyphen_breaks", True),
            hyphen_prob=cfg.get("hyphen_prob", 0.3),
            use_typos=True, typo_rate=0.05,
            use_neologisms=True, neologism_rate=0.12,
            fonts=fonts,
            base_font_path=fonts[0] if fonts else None,
            base_font_size=base_font_size,
            allow_mixed_fonts=cfg.get("allow_mixed_fonts", False),
            allow_mixed_sizes=cfg.get("allow_mixed_sizes", False),
            sizes_pool=cfg.get("sizes_pool"),
            allow_segment_backgrounds=cfg.get("allow_segment_backgrounds", False),
            segment_bg_prob=0.35,
            segment_bg_colors=cfg.get("segment_bg_colors"),
            background_images=bg_images,
            background_mode=cfg.get("background_mode","none"),
            bg_patch_count=cfg.get("bg_patch_count",3),
            bg_patch_alpha=cfg.get("bg_patch_alpha",0.35),
            noise_type=cfg.get("noise_type","gaussian"),
            noise_level=cfg.get("noise_level",0.02),
            margin=40, line_spacing=1.25,
            seed=seed+i,
            # mixed background params
            mixed_background=cfg.get("mixed_background", False),
            mixed_bg_mode=cfg.get("mixed_bg_mode","stripes"),
            mixed_bg_colors=cfg.get("mixed_bg_colors"),
            mixed_bg_stripes=cfg.get("mixed_bg_stripes",5),
            mixed_bg_orient=cfg.get("mixed_bg_orient","v"),
            mixed_bg_grid=cfg.get("mixed_bg_grid",(3,4)),
            mixed_bg_patch_count=cfg.get("mixed_bg_patch_count",6),
            mixed_bg_patch_scale=cfg.get("mixed_bg_patch_scale",(0.2,0.5)),
            mixed_bg_gradient_stops=cfg.get("mixed_bg_gradient_stops",3),
        )
        if "colors_pool" in cfg: params["colors_pool"]=cfg["colors_pool"]
        
        import sys, pathlib
        sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
        from generators import generate_ocr_image

        img, final_text, tokens, meta = generate_ocr_image(**params)
        stem = f"{i:02d}_{cfg['name']}"
        img.save(os.path.join(outdir, f"{stem}.png"))
        with open(os.path.join(outdir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(final_text)
        with open(os.path.join(outdir, f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump({"params":params,"tokens":tokens,"meta":meta}, f, ensure_ascii=False, indent=2)
        made.append(stem)

    return made

# пример
variants = generate_demo_images(
    outdir="demo_out",
    base_font_size=48,
    fonts=[
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ],
    background_images=["/home/sergei-scv/temp/meme-translator/fon.jpg"],
    seed=datetime.datetime.now().microsecond % 100,
)
print(variants)
 