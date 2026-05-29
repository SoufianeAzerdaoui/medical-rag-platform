from __future__ import annotations

import tempfile
import threading
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_MODEL_LOCK = threading.Lock()
_MODEL: Any | None = None
_MODEL_NAME = "medium"
_MIN_TRANSCRIPT_CHARS = 2
_MIN_AVERAGE_WORD_CHARS = 2.0
# Seuils qualité STT (assouplis pour accepter les requêtes courtes type "bonjour")
_REJECT_NO_SPEECH_THRESHOLD = 0.90
_REJECT_NO_SPEECH_MIN_CHARS = 12
_REJECT_LOW_LOGPROB_THRESHOLD = -1.80
_REJECT_LOW_LOGPROB_MIN_CHARS = 10
_GENERIC_HALLUCINATION_PATTERNS = (
    re.compile(r"merci d[’']avoir regard[ée] cette vid[ée]o", re.IGNORECASE),
    re.compile(r"j[' ]esp[èe]re que vous avez appr[ée]ci[ée] la vid[ée]o", re.IGNORECASE),
)


def _detect_suffix(audio_bytes: bytes, provided_suffix: str) -> str:
    head = audio_bytes[:16]
    if head.startswith(b"OggS"):
        return ".ogg"
    if head.startswith(b"RIFF"):
        return ".wav"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return ".webm"
    suffix = (provided_suffix or "").strip().lower()
    if suffix in {".ogg", ".webm", ".wav", ".mp3", ".m4a"}:
        return suffix
    return ".webm"


def _normalize_text(value: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text


def _clip(value: str, max_chars: int = 140) -> str:
    text = _normalize_text(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}…"


def _normalize_for_match(value: str) -> str:
    lowered = str(value or "").lower()
    ascii_like = "".join(
        c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
    )
    return " ".join(ascii_like.split()).strip()


def _is_repetitive_sentence(text: str) -> bool:
    normalized = _normalize_for_match(text)
    if not normalized:
        return True
    sentences = [
        s.strip(" .,!?:;")
        for s in re.split(r"[.!?]+", normalized)
        if s.strip(" .,!?:;")
    ]
    if len(sentences) < 2:
        return False
    unique = set(sentences)
    # Beaucoup de répétitions d'une même phrase => hallucination probable.
    if len(unique) <= max(1, len(sentences) // 3):
        return True
    return any(sentences.count(item) >= 3 for item in unique)


def _collect_segment_stats(segments: list[Any]) -> tuple[float, float, int]:
    if not segments:
        return (1.0, 0.0, 0)
    no_speech_values: list[float] = []
    avg_logprob_values: list[float] = []
    voiced_segments = 0
    for seg in segments:
        no_speech = getattr(seg, "no_speech_prob", None)
        avg_logprob = getattr(seg, "avg_logprob", None)
        text = _normalize_text(getattr(seg, "text", ""))
        if text:
            voiced_segments += 1
        if isinstance(no_speech, (int, float)):
            no_speech_values.append(float(no_speech))
        if isinstance(avg_logprob, (int, float)):
            avg_logprob_values.append(float(avg_logprob))
    mean_no_speech = sum(no_speech_values) / len(no_speech_values) if no_speech_values else 1.0
    mean_avg_logprob = sum(avg_logprob_values) / len(avg_logprob_values) if avg_logprob_values else -9.0
    return mean_no_speech, mean_avg_logprob, voiced_segments


def _token_unique_ratio(text: str) -> float:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _average_word_length(text: str) -> float:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)


def _hallucination_reason(text: str) -> str | None:
    if not text:
        return "empty_text"
    if len(text.strip()) < _MIN_TRANSCRIPT_CHARS:
        return "too_short"
    if _is_repetitive_sentence(text):
        return "repetitive_sentences"
    lowered = text.lower()
    unique_ratio = _token_unique_ratio(text)
    if len(re.findall(r"\w+", lowered, flags=re.UNICODE)) >= 8 and unique_ratio < 0.38:
        return "low_token_diversity"
    if _average_word_length(text) < _MIN_AVERAGE_WORD_CHARS and len(text) >= 10:
        return "low_word_information"
    for pattern in _GENERIC_HALLUCINATION_PATTERNS:
        if pattern.search(lowered):
            return "generic_video_outro_pattern"
    return None


def _quality_score(*, text: str, mean_no_speech: float, mean_avg_logprob: float, voiced_segments: int) -> int:
    # Score 0..100: combine signal quality + linguistic plausibility.
    length_component = min(len(text), 60) / 60.0
    diversity_component = min(max(_token_unique_ratio(text), 0.0), 1.0)
    logprob_component = min(max((mean_avg_logprob + 2.0) / 2.0, 0.0), 1.0)
    no_speech_component = 1.0 - min(max(mean_no_speech, 0.0), 1.0)
    voiced_component = 1.0 if voiced_segments >= 1 else 0.0

    score = (
        0.28 * no_speech_component
        + 0.24 * logprob_component
        + 0.18 * length_component
        + 0.20 * diversity_component
        + 0.10 * voiced_component
    )
    return int(round(min(max(score, 0.0), 1.0) * 100))


def _load_model() -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "faster-whisper n'est pas installé. Installez-le pour activer la transcription locale."
            ) from exc
        _MODEL = WhisperModel(_MODEL_NAME, device="cpu", compute_type="int8")
        return _MODEL


@dataclass
class TranscriptionAttemptDebug:
    strategy: str
    language: str
    vad_filter: bool
    transcript_preview: str
    transcript_chars: int
    quality_score: int
    rejected_reason: str | None
    mean_no_speech: float
    mean_avg_logprob: float
    voiced_segments: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "language": self.language,
            "vad_filter": self.vad_filter,
            "transcript_preview": self.transcript_preview,
            "transcript_chars": self.transcript_chars,
            "quality_score": self.quality_score,
            "rejected_reason": self.rejected_reason,
            "mean_no_speech": round(self.mean_no_speech, 4),
            "mean_avg_logprob": round(self.mean_avg_logprob, 4),
            "voiced_segments": self.voiced_segments,
        }


@dataclass
class TranscriptionDebugResult:
    transcript: str
    quality_score: int
    rejected_reason: str | None
    accepted_strategy: str | None
    attempts: list[TranscriptionAttemptDebug]

    def as_dict(self) -> dict[str, Any]:
        return {
            "quality_score": self.quality_score,
            "rejected_reason": self.rejected_reason,
            "accepted_strategy": self.accepted_strategy,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
        }


def transcribe_audio_bytes_debug(audio_bytes: bytes, suffix: str = ".webm") -> TranscriptionDebugResult:
    if not audio_bytes:
        return TranscriptionDebugResult(
            transcript="",
            quality_score=0,
            rejected_reason="empty_audio",
            accepted_strategy=None,
            attempts=[],
        )
    model = _load_model()
    detected_suffix = _detect_suffix(audio_bytes, suffix)
    with tempfile.NamedTemporaryFile(suffix=detected_suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)
    try:
        attempts: list[TranscriptionAttemptDebug] = []

        def _run_once(*, strategy: str, language: str | None, vad_filter: bool) -> tuple[str, int, str | None]:
            iterator, _info = model.transcribe(
                str(tmp_path),
                language=language,
                beam_size=5,
                vad_filter=vad_filter,
                task="transcribe",
                temperature=0.0,
                condition_on_previous_text=False,
            )
            segments = list(iterator)
            text = _normalize_text(" ".join((seg.text or "").strip() for seg in segments))
            mean_no_speech, mean_avg_logprob, voiced_segments = _collect_segment_stats(segments)
            quality_score = _quality_score(
                text=text,
                mean_no_speech=mean_no_speech,
                mean_avg_logprob=mean_avg_logprob,
                voiced_segments=voiced_segments,
            )
            rejected_reason = _hallucination_reason(text)
            if rejected_reason is None and voiced_segments == 0:
                rejected_reason = "no_voiced_segment"
            if (
                rejected_reason is None
                and mean_no_speech > _REJECT_NO_SPEECH_THRESHOLD
                and len(text) < _REJECT_NO_SPEECH_MIN_CHARS
            ):
                rejected_reason = "high_no_speech_probability"
            if (
                rejected_reason is None
                and mean_avg_logprob < _REJECT_LOW_LOGPROB_THRESHOLD
                and len(text) < _REJECT_LOW_LOGPROB_MIN_CHARS
            ):
                rejected_reason = "low_confidence_logprob"

            attempts.append(
                TranscriptionAttemptDebug(
                    strategy=strategy,
                    language=language or "auto",
                    vad_filter=vad_filter,
                    transcript_preview=_clip(text),
                    transcript_chars=len(text),
                    quality_score=quality_score,
                    rejected_reason=rejected_reason,
                    mean_no_speech=mean_no_speech,
                    mean_avg_logprob=mean_avg_logprob,
                    voiced_segments=voiced_segments,
                )
            )

            return text, quality_score, rejected_reason

        # 1) Priorité FR + VAD (propre si le signal est bon)
        try:
            text, score, reason = _run_once(strategy="fr_vad", language="fr", vad_filter=True)
            if text and reason is None:
                return TranscriptionDebugResult(
                    transcript=text,
                    quality_score=score,
                    rejected_reason=None,
                    accepted_strategy="fr_vad",
                    attempts=attempts,
                )
        except Exception:
            attempts.append(
                TranscriptionAttemptDebug(
                    strategy="fr_vad",
                    language="fr",
                    vad_filter=True,
                    transcript_preview="",
                    transcript_chars=0,
                    quality_score=0,
                    rejected_reason="runtime_error",
                    mean_no_speech=1.0,
                    mean_avg_logprob=-9.0,
                    voiced_segments=0,
                )
            )

        # 2) FR sans VAD (évite pertes si la VAD coupe trop)
        try:
            text, score, reason = _run_once(strategy="fr_no_vad", language="fr", vad_filter=False)
            if text and reason is None:
                return TranscriptionDebugResult(
                    transcript=text,
                    quality_score=score,
                    rejected_reason=None,
                    accepted_strategy="fr_no_vad",
                    attempts=attempts,
                )
        except Exception:
            attempts.append(
                TranscriptionAttemptDebug(
                    strategy="fr_no_vad",
                    language="fr",
                    vad_filter=False,
                    transcript_preview="",
                    transcript_chars=0,
                    quality_score=0,
                    rejected_reason="runtime_error",
                    mean_no_speech=1.0,
                    mean_avg_logprob=-9.0,
                    voiced_segments=0,
                )
            )

        # 3) fallback auto sans VAD
        try:
            text, score, reason = _run_once(strategy="auto_no_vad", language=None, vad_filter=False)
            if text and reason is None:
                return TranscriptionDebugResult(
                    transcript=text,
                    quality_score=score,
                    rejected_reason=None,
                    accepted_strategy="auto_no_vad",
                    attempts=attempts,
                )
            return TranscriptionDebugResult(
                transcript="",
                quality_score=score,
                rejected_reason=reason or "unreliable_transcription",
                accepted_strategy=None,
                attempts=attempts,
            )
        except Exception:
            attempts.append(
                TranscriptionAttemptDebug(
                    strategy="auto_no_vad",
                    language="auto",
                    vad_filter=False,
                    transcript_preview="",
                    transcript_chars=0,
                    quality_score=0,
                    rejected_reason="runtime_error",
                    mean_no_speech=1.0,
                    mean_avg_logprob=-9.0,
                    voiced_segments=0,
                )
            )
            return TranscriptionDebugResult(
                transcript="",
                quality_score=0,
                rejected_reason="runtime_error",
                accepted_strategy=None,
                attempts=attempts,
            )

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def transcribe_audio_bytes(audio_bytes: bytes, suffix: str = ".webm") -> str:
    result = transcribe_audio_bytes_debug(audio_bytes, suffix=suffix)
    return result.transcript
