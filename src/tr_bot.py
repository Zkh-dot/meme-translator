import os, asyncio, tempfile, re
import numpy as np, cv2
from typing import List, Tuple
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from paddleocr import PaddleOCR
import requests
import json
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("set TELEGRAM_BOT_TOKEN")

MIN_CONF = float(os.getenv("MIN_CONF", "0.0"))
MIN_CYR  = float(os.getenv("MIN_CYR",  "0.20"))
EPS_PCT  = float(os.getenv("EPS_PCT",  "0.08"))
CAP_LIMIT = 1000

YA_TR_TOKEN = os.getenv("YA_TR_TOKEN")
YA_TR_FOLDER = os.getenv("YA_TR_FOLDER")
YA_TR_LINK = "https://translate.api.cloud.yandex.net/translate/v2/translate"

with open(os.path.join(os.path.dirname(__file__), 'admins.json')) as admin_list:
    TG_ADMIN_CHANNEL = json.load(admin_list)

CYR = re.compile(r"[А-Яа-яЁё]")

def translate(texts: list[str]):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YA_TR_TOKEN}",
    }
    body = {
        "targetLanguageCode": "en",
        "texts": texts,
        "folderId": YA_TR_FOLDER,
    }
    response = json.loads(requests.post(
        YA_TR_LINK,
        json=body,
        headers=headers,
    ).text)
    if "translations" in response:
        return [x['text'] for x in response['translations']]
    else:
        a = [""] * len(texts)
        a[0] = str(response)   

CONFUSABLES_LAT2CYR = str.maketrans({
    "A":"А","B":"В","C":"С","E":"Е","H":"Н","K":"К","M":"М","O":"О","P":"Р","T":"Т","X":"Х","Y":"У",
    "a":"а","c":"с","e":"е","o":"о","p":"р","x":"х","y":"у","k":"к","m":"м","t":"т","h":"һ",
})

def cyr_ratio(s: str) -> float:
    s2 = "".join(ch for ch in s if not ch.isspace())
    if not s2: return 0.0
    return len(CYR.findall(s2)) / len(s2)

def normalize_confusables(s: str) -> str:
    if cyr_ratio(s) > 0.0 or any(ch.isdigit() for ch in s):
        return s.translate(CONFUSABLES_LAT2CYR)
    return s

def parse_items(res_obj):
    items=[]
    if hasattr(res_obj, "json"):
        d = res_obj.json
        if isinstance(d, dict) and "res" in d and isinstance(d["res"], dict):
            d = d["res"]
        texts  = d.get("rec_texts", []) or []
        scores = d.get("rec_scores", []) or []
        polys  = d.get("rec_polys") or d.get("dt_polys")
        boxes  = d.get("rec_boxes", None)
        n = max(len(texts), len(scores), len(polys or []), len(boxes or []))
        for i in range(n):
            txt = (texts[i] if i < len(texts) else "") or ""
            txt = normalize_confusables(txt.strip())
            sc  = float(scores[i]) if i < len(scores) else 0.0
            if polys is not None and i < len(polys):
                pts = np.array(polys[i], dtype=np.float32)
            elif boxes is not None and i < len(boxes):
                x1,y1,x2,y2 = boxes[i]
                pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
            else:
                pts = np.zeros((4,2), dtype=np.float32)
            if not txt: continue
            x1,y1 = float(np.min(pts[:,0])), float(np.min(pts[:,1]))
            x2,y2 = float(np.max(pts[:,0])), float(np.max(pts[:,1]))
            cx,cy = (x1+x2)/2.0, (y1+y2)/2.0
            items.append({"text":txt, "conf":sc, "box":pts, "bbox":[x1,y1,x2,y2], "center":(cx,cy)})
    elif isinstance(res_obj, list):
        for it in res_obj:
            try:
                box = np.array(it[0], dtype=np.float32)
                txt, sc = it[1][0], float(it[1][1])
                txt = normalize_confusables((txt or "").strip())
                if not txt: continue
                x1,y1 = float(np.min(box[:,0])), float(np.min(box[:,1]))
                x2,y2 = float(np.max(box[:,0])), float(np.max(box[:,1]))
                cx,cy = (x1+x2)/2.0, (y1+y2)/2.0
                items.append({"text":txt,"conf":sc,"box":box,"bbox":[x1,y1,x2,y2],"center":(cx,cy)})
            except: pass
    return items

def sort_reading_order(items):
    def key(l):
        ys = l["box"][:,1]; xs = l["box"][:,0]
        return (float(np.min(ys)//12), float(np.min(xs)))
    return sorted(items, key=key)

def union_find(n):
    p=list(range(n)); r=[0]*n
    def f(x):
        while p[x]!=x:
            p[x]=p[p[x]]; x=p[x]
        return x
    def u(a,b):
        ra,rb=f(a),f(b)
        if ra==rb: return
        if r[ra]<r[rb]: p[ra]=rb
        elif r[ra]>r[rb]: p[rb]=ra
        else: p[rb]=ra; r[ra]+=1
    return f,u

def cluster(items, hw, eps_pct=0.08):
    if not items: return []
    H,W = hw
    diag = (H**2+W**2)**0.5
    eps = max(24.0, eps_pct*diag)
    f,u = union_find(len(items))
    centers = np.array([i["center"] for i in items], dtype=np.float32)
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            if float(np.linalg.norm(centers[i]-centers[j])) <= eps:
                u(i,j)
    groups={}
    for i in range(len(items)):
        r=f(i); groups.setdefault(r,[]).append(i)
    clusters=[]
    for _, idxs in groups.items():
        sub=[items[i] for i in idxs]
        sub = sort_reading_order(sub)
        x1=min(x["bbox"][0] for x in sub); y1=min(x["bbox"][1] for x in sub)
        x2=max(x["bbox"][2] for x in sub); y2=max(x["bbox"][3] for x in sub)
        clusters.append({"indices":idxs, "items":sub, "bbox":[x1,y1,x2,y2]})
    clusters.sort(key=lambda c:(c["bbox"][1], c["bbox"][0]))
    return clusters

bot = Bot(TOKEN)
dp = Dispatcher()
ocr = PaddleOCR(
    lang="ru",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

def process_image_bytes(data: bytes) -> Tuple[str, list[str]]:
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return "no image", []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imencode(".png", img)[1].tofile(tmp.name)
        tmp_path = tmp.name
    try:
        res_list = ocr.predict(
            tmp_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=0.0,
        )
    finally:
        try: os.remove(tmp_path)
        except: pass
    if not res_list:
        return "no russian text", []
    items = parse_items(res_list[0])
    items = [it for it in items if it["conf"] >= MIN_CONF and cyr_ratio(it["text"]) >= MIN_CYR]
    if not items:
        return "no russian text", []
    H,W = img.shape[:2]
    clusters_list = cluster(items, (H,W), EPS_PCT)
    if not clusters_list:
        return "no russian text", []
    texts_ru = ["".join(i["text"] for i in c["items"]) for c in clusters_list]
    texts_en = translate(texts_ru) or []
    parts = []
    for i, t in enumerate(texts_en, 1):
        parts.append(f"{i}. {t}")
    full = "\n".join(parts) if parts else "no translation"
    if len(full) <= CAP_LIMIT:
        return full, []
    extra = []
    cur = []
    cur_len = 0
    for para in parts:
        block = (para)
        if cur_len + len(block) > CAP_LIMIT and cur:
            extra.append("".join(cur).strip())
            cur = [block]
            cur_len = len(block)
        else:
            cur.append(block)
            cur_len += len(block)
    if cur:
        extra.append("".join(cur).strip())
    caption = extra.pop(0)
    return caption, extra

from typing import Optional
from aiogram.types import Message, Chat

def get_forward_channel_link(m: Message) -> Optional[str]:
    ch: Optional[Chat] = None
    if m.forward_from_chat and m.forward_from_chat.type == "channel":
        ch = m.forward_from_chat
    if not ch:
        return None
    if ch.username:
        return f'@{ch.username}'
    return None


@dp.message(CommandStart())
async def start(m: Message):
    await m.reply("шлёпни картинку — верну ту же с подписью по блокам")

async def handle_image_message(message: Message, file_id: str):
    if str(message.chat.id) not in TG_ADMIN_CHANNEL:
        await message.answer('sorry, you are not in my admin list. to use this bot please message @Lunitarik')
        return
    file = await bot.get_file(file_id)
    stream = await bot.download_file(file.file_path)
    data = stream.read()
    caption, extras = await asyncio.to_thread(process_image_bytes, data)
    forward = get_forward_channel_link(message)
    cap = caption if caption else "no russian text"
    cap += f'\n------\nfrom {forward}' if forward else ''
    await message.answer_photo(file_id, caption=caption if caption else "no russian text")
    await bot.send_photo(TG_ADMIN_CHANNEL[str(message.chat.id)], file_id, caption=cap)
    for chunk in extras:
        await message.answer(chunk)

@dp.message(F.photo)
async def on_photo(m: Message):
    await handle_image_message(m, m.photo[-1].file_id)

@dp.message(F.document)
async def on_doc(m: Message):
    d = m.document
    if not d.mime_type or not d.mime_type.startswith("image/"):
        await m.reply("мне нужны картинки")
        return
    await handle_image_message(m, d.file_id)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
