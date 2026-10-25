"""Wake phrase, labelling policy and prompt lists for recording / TTS.

Label policy (from the proposal): a sample is positive iff the product SHOULD wake up.
    label =  1  -> positive
    label =  0  -> negative (incl. hard negatives)
    label = -1  -> challenge set (ambiguous by policy; excluded from training, reported separately)
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

WAKE_PHRASE = "Alo GPT à em"

# Spellings fed to TTS engines so that "GPT" is pronounced in the ways real users say it.
# The team must decide which pronunciations are part of the product (policy!).
WAKE_TTS_SPELLINGS = [
    "Alo GPT à em",
    "A lô, Gi Pi Ti à em",
    "A lô Chi Pi Ti à em",
    "A lô gờ pê tê à em",
]

CATEGORY_LABEL = {
    # positives
    "positive": 1,
    "positive_command": 1,
    "positive_pause": 1,
    # hard negatives
    "hard_partial": 0,
    "hard_substitution": 0,
    "hard_phonetic": 0,
    "hard_reorder": 0,
    "hard_common": 0,
    "hard_long_pause": 0,
    "hard_mined": 0,
    # ordinary negatives
    "neg_speech": 0,
    "neg_length_matched": 0,
    "neg_noise": 0,
    "neg_silence": 0,
    # challenge sets
    "challenge_narrative": -1,
    "challenge_mid_pause": -1,
}

HARD_CATEGORIES = [c for c in CATEGORY_LABEL if c.startswith("hard_")]


@dataclass
class Prompt:
    prompt_id: str
    text: str
    category: str
    hint: str = ""
    repeats: int = 1

    @property
    def label(self) -> int:
        return CATEGORY_LABEL[self.category]


POSITIVE_STYLES = [
    ("binh_thuong", "Nói bình thường", 8),
    ("nhanh", "Nói nhanh", 5),
    ("cham", "Nói chậm, rõ từng chữ", 5),
    ("nho", "Nói nhỏ / thì thầm nhẹ", 4),
    ("to", "Nói to", 3),
    ("xa", "Đứng xa micro 1.5–3 m", 5),
    ("vui", "Ngữ điệu vui / gọi to như gọi người", 3),
    ("met", "Ngữ điệu mệt / uể oải", 3),
    ("ngap_ngung", "Ngập ngừng NGẮN (≤ 0.5 s) giữa các từ", 4),
]

COMMANDS = [
    "xem lịch ngày mai", "hôm nay thời tiết thế nào", "bật nhạc lên", "đặt báo thức sáu giờ sáng",
    "kể chuyện cười đi", "dịch câu này sang tiếng Anh", "tóm tắt tin tức hôm nay",
]

HARD_NEGATIVES = {
    "hard_partial": ["Alo", "Alo GPT", "GPT à em", "à em", "GPT", "Alo GPT à", "GPT à"],
    "hard_substitution": [
        "Alo GPT à anh", "Alo GPT ơi em", "Alo chị à em", "Alo GPT à chị", "Alo GPT à con",
        "Alo bé à em", "Alo GPT nhé em", "Alo GPT đây em", "Alo Gemini à em", "Alo Siri à em",
        "Hello GPT à em", "Ê GPT à em",
    ],
    "hard_phonetic": [
        "Alo GPS à em", "Alo GDP à em", "Alo CPU à em", "Alo ghi bi ti à em", "A lô dì Bi à em",
        "Alo ghê thế à em", "A lô chị Ti à em", "Alo TPT à em", "A lô bi pi ti à em",
    ],
    "hard_reorder": ["GPT à em alo", "À em alo GPT", "Em à GPT alo", "Alo à em GPT", "GPT alo à em"],
    "hard_common": [
        "Alo em à, anh đây", "Alo, mẹ à, con đây", "Alo anh à", "Alo à, em nghe đây",
        "À em ơi, lại đây", "Alo, ai đấy ạ", "Alo alo, nghe rõ không", "Em à, GPT là gì vậy",
        "Hỏi GPT đi em", "Dùng GPT à em",
    ],
}

NEG_LENGTH_MATCHED = [
    "Hôm nay trời đẹp quá", "Mở cửa sổ ra đi", "Anh ơi cho em hỏi", "Tắt đèn đi em",
    "Ăn cơm chưa em", "Mai mình đi học nhé", "Cái này bao nhiêu tiền", "Lát nữa gọi lại nhé",
    "Bật quạt lên đi", "Đi chợ mua rau", "Em đang làm bài tập", "Chờ anh một chút",
]

CHALLENGE_NARRATIVE = [
    "Hôm qua tôi cứ nói Alo GPT à em mà nó không chạy",
    "Bạn thử nói Alo GPT à em xem nó có lên không",
    "Câu kích hoạt là Alo GPT à em đấy",
    "Đừng nói Alo GPT à em lúc đang họp nhé",
]


def default_prompts(n_positive_scale: float = 1.0) -> list[Prompt]:
    prompts: list[Prompt] = []
    for sid, hint, reps in POSITIVE_STYLES:
        cat = "positive_pause" if sid == "ngap_ngung" else "positive"
        prompts.append(Prompt(f"pos_{sid}", WAKE_PHRASE, cat, hint, max(1, round(reps * n_positive_scale))))
    for i, cmd in enumerate(COMMANDS):
        prompts.append(Prompt(f"poscmd_{i:02d}", f"{WAKE_PHRASE}, {cmd}", "positive_command",
                              "Nói liền wake word và câu lệnh"))
    for cat, texts in HARD_NEGATIVES.items():
        for i, t in enumerate(texts):
            prompts.append(Prompt(f"{cat}_{i:02d}", t, cat, "", 2))
    prompts.append(Prompt("hard_long_pause_00", "Alo GPT ... (dừng ≥ 1.5 s) ... à em", "hard_long_pause",
                          "Dừng LÂU (đếm thầm 2 giây) giữa 'Alo GPT' và 'à em'", 3))
    prompts.append(Prompt("challenge_mid_pause_00", "Alo GPT ... (dừng ~1 s) ... à em", "challenge_mid_pause",
                          "Dừng khoảng 1 giây giữa 'Alo GPT' và 'à em'", 2))
    for i, t in enumerate(NEG_LENGTH_MATCHED):
        prompts.append(Prompt(f"neglen_{i:02d}", t, "neg_length_matched"))
    for i, t in enumerate(CHALLENGE_NARRATIVE):
        prompts.append(Prompt(f"challenge_narr_{i:02d}", t, "challenge_narrative"))
    return prompts


def tts_prompts() -> list[Prompt]:
    """Texts for synthetic speech: positives (several spellings) + hard negatives + length-matched."""
    out = [Prompt(f"pos_tts_{i}", t, "positive") for i, t in enumerate(WAKE_TTS_SPELLINGS)]
    out += [Prompt(f"poscmd_tts_{i:02d}", f"{WAKE_PHRASE}, {c}", "positive_command") for i, c in enumerate(COMMANDS)]
    for cat, texts in HARD_NEGATIVES.items():
        out += [Prompt(f"{cat}_{i:02d}", t, cat) for i, t in enumerate(texts)]
    out += [Prompt(f"neglen_{i:02d}", t, "neg_length_matched") for i, t in enumerate(NEG_LENGTH_MATCHED)]
    out += [Prompt(f"challenge_narr_{i:02d}", t, "challenge_narrative") for i, t in enumerate(CHALLENGE_NARRATIVE)]
    # building blocks for pause composition (see scripts/compose_pauses.py)
    out += [Prompt("part_head", "Alo GPT", "hard_partial"), Prompt("part_tail", "à em", "hard_partial")]
    return out


def write_prompts_csv(prompts: list[Prompt], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "text", "category", "label", "hint", "repeats"])
        w.writeheader()
        for p in prompts:
            w.writerow({**asdict(p), "label": p.label})


def read_prompts_csv(path: str | Path) -> list[Prompt]:
    with open(path, encoding="utf-8") as f:
        return [Prompt(r["prompt_id"], r["text"], r["category"], r.get("hint", ""), int(r.get("repeats", 1) or 1))
                for r in csv.DictReader(f)]
