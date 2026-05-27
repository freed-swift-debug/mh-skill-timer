import os
import tempfile
from difflib import SequenceMatcher

import cv2
import pandas as pd
import streamlit as st

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

st.set_page_config(page_title="スキル発動時間チェッカー", layout="wide")

st.title("モンハン スキル発動時間チェッカー")
st.caption("動画をアップロードして、右端中央の通知からスキル発動時間を集計します。")

with st.sidebar:
    st.header("解析設定")
    skill_text = st.text_area(
        "スキル名（1行に1つ）",
        value="逆恨み\n逆襲\n挑戦者\n力の解放\n災禍転福\n連撃",
        height=180,
    )
    sample_interval = st.slider("解析間隔（秒）", 0.1, 1.0, 0.3, 0.1)
    fuzzy_threshold = st.slider("誤字許容の強さ", 0.50, 0.95, 0.72, 0.01)
    st.divider()
    st.write("通知欄の切り抜き範囲")
    x1 = st.slider("左端 X", 0.0, 1.0, 0.52, 0.01)
    x2 = st.slider("右端 X", 0.0, 1.0, 0.99, 0.01)
    y1 = st.slider("上端 Y", 0.0, 1.0, 0.25, 0.01)
    y2 = st.slider("下端 Y", 0.0, 1.0, 0.75, 0.01)
    st.caption("うまく読めない時は、右端中央の通知が入るように範囲を調整してください。")

uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "mkv", "avi"])


def sec_to_time(sec: float) -> str:
    sec = max(0, float(sec))
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


def normalize_text(text: str) -> str:
    return (
        text.replace(" ", "")
        .replace("　", "")
        .replace("\n", "")
        .replace("。", "")
        .replace("､", "")
        .replace(",", "")
    )


def fuzzy_contains(text: str, target: str, threshold: float) -> bool:
    text = normalize_text(text)
    target = normalize_text(target)
    if target in text:
        return True
    if len(text) < 2 or len(target) < 2:
        return False
    # Sliding similarity for OCR mistakes.
    win = max(len(target), 3)
    best = SequenceMatcher(None, text, target).ratio()
    for i in range(0, max(1, len(text) - win + 1)):
        part = text[i : i + win]
        best = max(best, SequenceMatcher(None, part, target).ratio())
    return best >= threshold


@st.cache_resource(show_spinner=False)
def load_ocr():
    if PaddleOCR is None:
        return None
    return PaddleOCR(use_angle_cls=True, lang="japan", show_log=False)


def crop_notice_area(frame):
    h, w = frame.shape[:2]
    xa = int(w * min(x1, x2))
    xb = int(w * max(x1, x2))
    ya = int(h * min(y1, y2))
    yb = int(h * max(y1, y2))
    return frame[ya:yb, xa:xb]


def ocr_text(ocr, image) -> str:
    # Upscale to help small game UI text.
    image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    result = ocr.ocr(image, cls=True)
    texts = []
    if result:
        for block in result:
            if not block:
                continue
            for line in block:
                try:
                    texts.append(line[1][0])
                except Exception:
                    pass
    return normalize_text("".join(texts))


def detect_event(text: str, skills: list[str], threshold: float):
    events = []
    for skill in skills:
        start_patterns = [f"{skill}発動", f"{skill}が発動", f"{skill}の効果が発動"]
        end_patterns = [f"{skill}の効果が切れました", f"{skill}効果が切れました", f"{skill}が切れました"]
        if any(fuzzy_contains(text, p, threshold) for p in start_patterns):
            events.append((skill, "start"))
        if any(fuzzy_contains(text, p, threshold) for p in end_patterns):
            events.append((skill, "end"))
    return events


def analyze_video(video_path: str, skills: list[str]):
    ocr = load_ocr()
    if ocr is None:
        raise RuntimeError("PaddleOCRの読み込みに失敗しました。requirements.txtを確認してください。")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames else 0
    step = max(1, int(fps * sample_interval))

    active = {s: False for s in skills}
    starts = {s: None for s in skills}
    last_event_time = {(s, "start"): -9999 for s in skills}
    last_event_time.update({(s, "end"): -9999 for s in skills})
    intervals = []
    detections = []

    progress = st.progress(0)
    status = st.empty()

    frame_no = 0
    while frame_no < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_no / fps
        crop = crop_notice_area(frame)
        text = ocr_text(ocr, crop)
        events = detect_event(text, skills, fuzzy_threshold)

        for skill, kind in events:
            # Prevent duplicate count while the same notification remains on screen.
            if t - last_event_time[(skill, kind)] < 1.2:
                continue
            last_event_time[(skill, kind)] = t
            detections.append({"時刻": t, "スキル": skill, "判定": kind, "OCR文字": text})
            if kind == "start" and not active[skill]:
                active[skill] = True
                starts[skill] = t
            elif kind == "end" and active[skill]:
                intervals.append({"スキル": skill, "開始": starts[skill], "終了": t, "発動時間": t - starts[skill]})
                active[skill] = False
                starts[skill] = None

        if total_frames:
            progress.progress(min(1.0, frame_no / total_frames))
        status.text(f"解析中... {sec_to_time(t)} / {sec_to_time(duration)}")
        frame_no += step

    cap.release()
    progress.progress(1.0)
    status.empty()

    # If an effect is still active at the end, close it at video end.
    for skill in skills:
        if active[skill] and starts[skill] is not None:
            intervals.append({"スキル": skill, "開始": starts[skill], "終了": duration, "発動時間": duration - starts[skill]})

    return intervals, detections, duration


if uploaded_file:
    st.video(uploaded_file)
    skills = [s.strip() for s in skill_text.splitlines() if s.strip()]
    st.write("解析対象スキル：", "、".join(skills) if skills else "未入力")

    if st.button("解析開始", type="primary"):
        if not skills:
            st.error("スキル名を1つ以上入力してください。")
        else:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                video_path = tmp.name

            try:
                with st.spinner("OCR解析中です。動画が長いほど時間がかかります。"):
                    intervals, detections, duration = analyze_video(video_path, skills)

                st.subheader("スキルごとの合計")
                if intervals:
                    df = pd.DataFrame(intervals)
                    summary = df.groupby("スキル", as_index=False)["発動時間"].sum()
                    summary["発動率"] = summary["発動時間"] / duration * 100 if duration else 0
                    summary["合計発動時間"] = summary["発動時間"].apply(sec_to_time)
                    summary["発動時間"] = summary["発動時間"].round(2)
                    summary["発動率"] = summary["発動率"].round(1)
                    st.dataframe(summary[["スキル", "合計発動時間", "発動時間", "発動率"]], use_container_width=True)

                    st.subheader("発動区間")
                    detail = df.copy()
                    detail["開始"] = detail["開始"].apply(sec_to_time)
                    detail["終了"] = detail["終了"].apply(sec_to_time)
                    detail["発動時間"] = detail["発動時間"].round(2)
                    st.dataframe(detail, use_container_width=True)

                    csv = detail.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("発動区間CSVをダウンロード", csv, "skill_intervals.csv", "text/csv")
                else:
                    st.warning("発動区間を検出できませんでした。切り抜き範囲や解析間隔を調整してください。")

                with st.expander("検出ログを見る"):
                    if detections:
                        log = pd.DataFrame(detections)
                        log["時刻"] = log["時刻"].apply(sec_to_time)
                        st.dataframe(log, use_container_width=True)
                    else:
                        st.write("検出ログはありません。")
            finally:
                try:
                    os.remove(video_path)
                except Exception:
                    pass
else:
    st.info("動画をアップロードすると解析できます。")