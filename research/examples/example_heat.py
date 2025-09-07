import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from generators import generate_ocr_image_by_heat

import os

os.makedirs("demo_out", exist_ok=True)
# -------- Примеры использования --------


# 1) стерильная картинка
img, text, tokens, meta, p = generate_ocr_image_by_heat(
    heat=0,
    resolution=(1200, 700),
    alphabet="ru",
    fonts=["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    seed=1
)
img.save("demo_out/heat_00.png")

# 2) средний «жар»
img, text, tokens, meta, p = generate_ocr_image_by_heat(
    heat=50,
    resolution=(1400, 800),
    alphabet="en",
    fonts=["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
    seed=2
)
img.save("demo_out/heat_50.png")

# 3) адская кислотность
img, text, tokens, meta, p = generate_ocr_image_by_heat(
    heat=100,
    resolution=(1600, 900),
    alphabet="en",
    fonts=[
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ],
    background_images=["/path/bg1.jpg","/path/bg2.jpg"],
    seed=3
)
img.save("demo_out/heat_100.png")