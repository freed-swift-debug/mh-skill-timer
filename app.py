import os
import re
import tempfile
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import cv2
import pandas as pd
import pytesseract
import streamlit as st

@dataclass
class SkillState:
    name: str
    active: bool = False
    start_time: float | None = None
    intervals: List[Tuple[float, float]] = field(default_factory=list)
    last_start: float = -9999.0
    last_end: float = -9999.0


def sec_to_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"


def normalize(text: str) -> str:
    text = str(text)
    fixes = {
        " ": "", "　": "", "\n": "", "\r": "", "\t": "",
        "。": "", "、": "", ",": "", ".": "", "・": "",
        "發動": "発動", "発働": "発動", "発勁": "発動",
        "効果か切れました": "効果が切れました",
        "効果が切れましだ": "効果が切れました",
        "切れまし": "切れました",
    }
    for k, v in fixes.items():
        text = text.replace(k, v)
    return text


def split_skills(raw: str) -> List[str]:
    out = []
    for x in re.split(r"[\n,、]+", raw):
        x = x.strip()
        if x and x not in out:
            out.append(x)
    return out


def fuzzy_contains(text: str, word: str, threshold: float) -> bool:
    text = normalize(text)
    word = normalize(word)
    if word in text:
        return True
    if len(word) <= 2:
        return False
    best = 0.0
    for size in range(max(1, len(word) - 2), len(word) + 3):
        if size > len(text):
            continue
        for i in range(len(text) - size + 1):
            score = SequenceMatcher(None, word, text[i:i+size]).ratio()
            if score > best:
                best = score
            if score >= threshold:
                return True
    return False


def read_text(crop, scale: float, threshold_on: bool) -> str:
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if threshold_on:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 日本語＋英語。psm 6は「まとまった文字ブロック」向け。
    return pytesseract.image_to_string(gray, lang="jpn+eng", config="--psm 6")


def analyze(video_path, skill_names, sample_sec, crop_box, fuzzy, cooldown, scale, threshold_on):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした。")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames else 0
    step = max(1, int(fps * sample_sec))

    states: Dict[str, SkillState] = {s: SkillState(s) for s in skill_names}
    logs = []
    progress = st.progress(0)

    frame_no = 0
    while frame_no < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        x1, x2, y1, y2 = crop_box
        crop = frame[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]
        text_raw = read_text(crop, scale, threshold_on)
        text = normalize(text_raw)
        now = frame_no / fps

        events = []
        for name, stt in states.items():
            skill_hit = fuzzy_contains(text, name, fuzzy)
            start_hit = skill_hit and fuzzy_contains(text, "発動", fuzzy) and "切れ" not in text
            end_hit = skill_hit and ("切れ" in text or fuzzy_contains(text, "効果が切れました", fuzzy))

            if start_hit and not stt.active and now - stt.last_start >= cooldown:
                stt.active = True
                stt.start_time = now
                stt.last_start = now
                events.append(f"{name}: START")

            if end_hit and stt.active and now - stt.last_end >= cooldown:
                stt.intervals.append((stt.start_time, now))
                stt.active = False
                stt.start_time = None
                stt.last_end = now
                events.append(f"{name}: END")

        if text or events:
            logs.append({"時刻": sec_to_time(now), "判定": " / ".join(events), "OCR文字": text})

        frame_no += step
        progress.progress(min(1.0, frame_no / max(1, total_frames)))

    cap.release()
    progress.empty()

    for name, stt in states.items():
        if stt.active and stt.start_time is not None:
            stt.intervals.append((stt.start_time, duration))
            logs.append({"時刻": sec_to_time(duration), "判定": f"{name}: END_BY_VIDEO_END", "OCR文字": "動画終了まで継続扱い"})

    return states, duration, pd.DataFrame(logs)


st.set_page_config(page_title="スキル発動時間チェッカー", layout="wide")
st.title("モンハン 複数スキル発動時間チェッカー 軽量版")
st.write("動画右端中央の通知欄をOCRで読み取り、スキルごとの発動時間を集計します。")

with st.sidebar:
    skill_raw = st.text_area("解析したいスキル名（1行に1つ）", "逆恨み\n逆襲\n挑戦者\n力の解放\n災禍転福\n連撃", height=170)
    sample_sec = st.slider("何秒ごとに読むか", 0.1, 1.0, 0.3, 0.1)
    fuzzy = st.slider("OCR誤字許容", 0.60, 1.00, 0.82, 0.01)
    cooldown = st.slider("重複カウント防止 秒", 0.0, 5.0, 1.0, 0.5)
    scale = st.slider("文字拡大", 1.0, 4.0, 2.0, 0.5)
    threshold_on = st.checkbox("白黒補正", value=False)

    st.subheader("読み取り範囲")
    x1 = st.slider("左端 X", 0.0, 1.0, 0.55, 0.01)
    x2 = st.slider("右端 X", 0.0, 1.0, 0.98, 0.01)
    y1 = st.slider("上端 Y", 0.0, 1.0, 0.30, 0.01)
    y2 = st.slider("下端 Y", 0.0, 1.0, 0.70, 0.01)

uploaded = st.file_uploader("ここに動画をアップロード", type=["mp4", "mov", "mkv", "avi"])

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name

    st.video(path)

    if st.button("解析開始", type="primary"):
        skills = split_skills(skill_raw)
        if not skills:
            st.error("スキル名を入力してください。")
            st.stop()

        with st.spinner("解析中…"):
            states, duration, logs = analyze(path, skills, sample_sec, (x1, x2, y1, y2), fuzzy, cooldown, scale, threshold_on)

        summary = []
        details = []
        for name, stt in states.items():
            total = 0.0
            for i, (start, end) in enumerate(stt.intervals, 1):
                d = max(0.0, end - start)
                total += d
                details.append({"スキル": name, "回数": i, "開始": sec_to_time(start), "終了": sec_to_time(end), "発動時間_秒": round(d, 2)})
            summary.append({"スキル": name, "発動回数": len(stt.intervals), "合計発動時間_秒": round(total, 2), "合計発動時間": sec_to_time(total), "発動率_%": round(total / duration * 100, 1) if duration else 0})

        summary_df = pd.DataFrame(summary).sort_values("合計発動時間_秒", ascending=False)
        detail_df = pd.DataFrame(details)

        st.subheader("スキルごとの合計")
        st.dataframe(summary_df, use_container_width=True)
        st.download_button("合計CSV", summary_df.to_csv(index=False).encode("utf-8-sig"), "skill_summary.csv", "text/csv")

        st.subheader("発動区間の詳細")
        st.dataframe(detail_df, use_container_width=True)
        st.download_button("詳細CSV", detail_df.to_csv(index=False).encode("utf-8-sig"), "skill_detail.csv", "text/csv")

        with st.expander("OCRログ"):
            st.dataframe(logs, use_container_width=True)
else:
    st.info("動画をアップロードしてください。")