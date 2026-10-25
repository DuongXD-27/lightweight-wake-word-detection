"""Manifest building (speaker-disjoint splits) and the training/eval Dataset.

Expected layout under ``data/raw`` (see README):

    real/<speaker_id>/<category>/*.wav      recorded people (team + external "ext_*" speakers)
    tts/<voice_id>/<category>/*.wav         synthetic speech
    speech/<corpus>/<speaker_id>/**/*.wav   public Vietnamese speech -> neg_speech
    noise/**/*.wav                          non-speech audio        -> neg_noise
    long/<train|val|test>/**/*.wav          long continuous negative audio (FA/hour evaluation)
plus ``data/mined/<run>/*.wav`` (hard_mined, always train).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .audio import audio_duration, fit_to_length, list_audio_files, load_audio, trim_silence
from .augment import AudioBank, WaveAugment
from .phrases import CATEGORY_LABEL

MANIFEST_COLUMNS = ["path", "label", "category", "source", "speaker", "duration", "split"]
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------------------- split
def _unit_hash(key: str, salt: str = "") -> float:
    h = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def assign_split(key: str, val_ratio: float, test_ratio: float, salt: str = "") -> str:
    u = _unit_hash(key, salt)
    if u < test_ratio:
        return "test"
    if u < test_ratio + val_ratio:
        return "val"
    return "train"


def speaker_split(speaker: str, scfg) -> str:
    if speaker in (scfg.get("test_speakers") or []):
        return "test"
    if speaker in (scfg.get("val_speakers") or []):
        return "val"
    if speaker in (scfg.get("train_speakers") or []):
        return "train"
    if scfg.get("external_prefix") and speaker.startswith(scfg.external_prefix):
        return "test"
    return assign_split(speaker, scfg.val_ratio, scfg.test_ratio, scfg.get("salt", ""))


# ------------------------------------------------------------------------------------ manifest
def _category_from_dir(name: str) -> str:
    if name not in CATEGORY_LABEL:
        raise ValueError(f"Unknown category folder '{name}'. Valid: {sorted(CATEGORY_LABEL)}")
    return name


def scan_raw(raw_root: str | Path, mined_root: str | Path | None, scfg, with_duration: bool = True) -> pd.DataFrame:
    raw_root = Path(raw_root)
    rows = []

    def add(p: Path, category: str, source: str, speaker: str, split: str):
        rows.append(dict(path=str(p.as_posix()), label=CATEGORY_LABEL[category], category=category,
                         source=source, speaker=speaker, duration=audio_duration(p) if with_duration else np.nan,
                         split=split))

    for source in ("real", "tts"):
        for spk_dir in sorted((raw_root / source).glob("*")) if (raw_root / source).exists() else []:
            if not spk_dir.is_dir():
                continue
            speaker = f"{source}:{spk_dir.name}"
            if source == "real":
                split = speaker_split(spk_dir.name, scfg)
            else:
                mode = scfg.get("tts_split", "train")
                split = "train" if mode == "train" else speaker_split(spk_dir.name, scfg)
            for cat_dir in sorted(spk_dir.glob("*")):
                if cat_dir.is_dir():
                    cat = _category_from_dir(cat_dir.name)
                    for p in list_audio_files(cat_dir):
                        add(p, cat, source, speaker, split)

    speech_root = raw_root / "speech"
    for corpus in sorted(speech_root.glob("*")) if speech_root.exists() else []:
        for spk_dir in sorted(corpus.glob("*")):
            files = list_audio_files(spk_dir) if spk_dir.is_dir() else ([spk_dir] if spk_dir.is_file() else [])
            speaker = f"{corpus.name}:{spk_dir.stem}"
            split = assign_split(speaker, scfg.val_ratio, scfg.test_ratio, "speech")
            for p in files:
                add(p, "neg_speech", "speech", speaker, split)

    for p in list_audio_files(raw_root / "noise"):
        add(p, "neg_noise", "noise", "noise", assign_split(p.as_posix(), scfg.val_ratio, scfg.test_ratio, "noise"))

    for split in SPLITS:
        for p in list_audio_files(raw_root / "long" / split):
            add(p, "neg_speech", "long", f"long:{p.stem}", split)

    if mined_root and Path(mined_root).exists():
        for p in list_audio_files(mined_root):
            add(p, "hard_mined", "mined", "mined", "train")

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return df


def check_leakage(df: pd.DataFrame) -> list[str]:
    """Return speakers that appear in more than one split (should be empty)."""
    sub = df[df.source.isin(["real", "tts", "speech"])]
    multi = sub.groupby("speaker").split.nunique()
    return sorted(multi[multi > 1].index.tolist())


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="category", columns="split", values="path", aggfunc="count", fill_value=0)


def read_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ------------------------------------------------------------------------------------- dataset
def item_group(category: str, source: str) -> str:
    if CATEGORY_LABEL.get(category) == 1:
        return "pos"
    if category.startswith("hard_"):
        return "hard"
    if category in ("neg_noise", "neg_silence"):
        return "noise"
    if source == "long":
        return "long"
    return "neg"


class WakeWordDataset(Dataset):
    """Returns (features, label, index). Labels: 1 wake, 0 not-wake (challenge items excluded unless asked)."""

    VIRTUAL_SILENCE = "<silence>"

    def __init__(self, manifest: pd.DataFrame, cfg, split: str, frontend, train: bool,
                 include_challenge: bool = False, seed: int = 0):
        df = manifest[manifest.split == split].copy()
        if not include_challenge:
            df = df[df.label >= 0]
        if train and cfg.data.get("drop_sources"):
            df = df[~df.source.isin(cfg.data.drop_sources)]
        if train and cfg.data.get("drop_categories"):
            df = df[~df.category.isin(cfg.data.drop_categories)]
        n_sil = int(cfg.data.get("virtual_silence", 0)) if train else 0
        if n_sil:
            sil = pd.DataFrame([dict(path=self.VIRTUAL_SILENCE, label=0, category="neg_silence", source="virtual",
                                     speaker="virtual", duration=cfg.audio.clip_seconds, split=split)] * n_sil)
            df = pd.concat([df, sil], ignore_index=True)
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.sr = cfg.audio.sample_rate
        self.n = int(cfg.audio.clip_seconds * self.sr)
        self.frontend = frontend
        self.train = train
        self.seed = seed
        self._rng = None
        self._cache: dict[str, np.ndarray] = {}
        self.groups = [item_group(c, s) for c, s in zip(self.df.category, self.df.source)]
        self.augment = None
        if train and cfg.aug.enabled:
            babble_files = [Path(p) for p in manifest[(manifest.split == "train") &
                                                       (manifest.category == "neg_speech") &
                                                       (manifest.source == "speech")].path]
            noise_files = [Path(p) for p in manifest[(manifest.split == "train") &
                                                      (manifest.category == "neg_noise")].path]
            noise_bank = AudioBank(noise_files + [Path(p) for d in cfg.aug.get("extra_noise_dirs", []) or []
                                                  for p in list_audio_files(d)], self.sr)
            rir_bank = AudioBank.from_dirs(cfg.aug.get("rir_dirs", []), sr=self.sr, max_seconds=2.0)
            self.augment = WaveAugment(cfg.aug, noise_bank, rir_bank, AudioBank(babble_files, self.sr), self.sr)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            base = torch.initial_seed() if self.train else self.seed
            self._rng = np.random.default_rng((base + 7919 * wid) % (2 ** 32))
        return self._rng

    # -- loading
    def _load(self, row) -> np.ndarray:
        if row.path == self.VIRTUAL_SILENCE:
            amp = 10 ** (self.rng.uniform(-90, -50) / 20)
            return (self.rng.standard_normal(self.n) * amp).astype(np.float32)
        long_file = row.duration > max(4.0, 2.5 * self.cfg.audio.clip_seconds) and row.label == 0
        if long_file:
            span = self.cfg.audio.clip_seconds * 1.25
            if self.train:
                off = float(self.rng.uniform(0, max(0.0, row.duration - span)))
            else:
                off = max(0.0, (row.duration - span) / 2)
            return load_audio(row.path, self.sr, offset_s=off, duration_s=span)
        x = self._cache.get(row.path)
        if x is None:
            x = load_audio(row.path, self.sr)
            if row.label == 1 or row.category.startswith("hard_") or row.category.startswith("challenge"):
                x = trim_silence(x, self.sr)
            if len(self._cache) < self.cfg.data.get("cache_items", 20000):
                self._cache[row.path] = x
        return x

    def _place(self, x: np.ndarray, row) -> np.ndarray:
        rng = self.rng
        speechlike = row.label == 1 or row.category.startswith("hard_") or row.category.startswith("challenge")
        if not speechlike:
            return fit_to_length(x, self.n, "random" if self.train else "center", rng)
        if len(x) > self.n:
            # keep the end of the phrase (the discriminative "à em") when too long
            return fit_to_length(x, self.n, "end", rng, keep="end")
        if not self.train:
            # eval: phrase ends ~0.25 s before the window end (as at streaming detection time)
            out = np.zeros(self.n, dtype=np.float32)
            end = max(len(x), self.n - int(0.25 * self.sr))
            out[end - len(x):end] = x
            return out
        out = fit_to_length(x, self.n, "random", rng)
        if (row.label == 1 and self.augment is not None and self.augment.babble is not None
                and rng.random() < self.cfg.aug.get("trailing_speech_prob", 0.0)):
            # "wake word + command": fill the area after the phrase with unrelated speech
            nz = np.nonzero(np.abs(out) > 1e-4)[0]
            end = int(nz[-1]) + int(0.05 * self.sr) if len(nz) else self.n
            if end < self.n - int(0.1 * self.sr):
                tail = self.augment.babble.sample(rng, self.n - end)
                tail *= (np.sqrt(np.mean(x ** 2)) + 1e-6) / (np.sqrt(np.mean(tail ** 2)) + 1e-6)
                out[end:] += tail.astype(np.float32)
        return out

    def get_waveform(self, i: int) -> np.ndarray:
        row = self.df.iloc[i]
        x = self._load(row)
        if self.augment is not None:
            x = self.augment.pre(x, self.rng)
        x = self._place(x, row)
        if self.augment is not None:
            x = self.augment.post(x, self.rng)
        return x

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.get_waveform(i)).unsqueeze(0)
        feats = self.frontend(x)[0]
        label = float(self.df.label.iloc[i] == 1)
        return feats, torch.tensor(label, dtype=torch.float32), i

    # -- sampling
    def make_sampler(self, num_samples: int) -> WeightedRandomSampler:
        gw = dict(self.cfg.data.group_weights)
        counts = Counter(self.groups)
        present = {g: w for g, w in gw.items() if counts.get(g, 0) > 0}
        total = sum(present.values()) or 1.0
        weights = [present.get(g, 0.0) / total / counts[g] for g in self.groups]
        return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples, replacement=True)

    def group_counts(self) -> dict:
        return dict(Counter(self.groups))
