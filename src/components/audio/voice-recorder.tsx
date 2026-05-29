"use client";

import { Mic, Square, X } from "lucide-react";
import { useRef, useState } from "react";
import { ApiError, type TranscribeDebugInfo, transcribeAudioDetailed } from "@/services/rag-api";
import { useAuthStore } from "@/store/auth-store";

const MIN_RECORDING_MS = 850;
const MIN_AUDIO_BYTES = 2_400;
const SUSPICIOUS_PATTERNS = [/merci d['’]avoir regard[ée] cette vid[ée]o/i, /j['’ ]esp[èe]re que vous avez appr[ée]ci[ée] la vid[ée]o/i];

function isLikelyBadTranscript(value: string): boolean {
  const text = String(value || "").trim();
  if (!text) return true;
  const normalized = text.toLowerCase().replace(/\s+/g, " ");
  for (const pattern of SUSPICIOUS_PATTERNS) {
    if (pattern.test(normalized)) return true;
  }
  const sentences = normalized
    .split(/[.!?]+/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (sentences.length >= 2) {
    const unique = new Set(sentences);
    if (unique.size <= Math.ceil(sentences.length / 3)) return true;
  }
  return false;
}

export function VoiceRecorder({ onTranscript }: { onTranscript: (value: string) => void }) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState("");
  const [debugInfo, setDebugInfo] = useState<TranscribeDebugInfo | null>(null);
  const [debugOpen, setDebugOpen] = useState(false);
  const token = useAuthStore((s) => s.accessToken);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const startedAtRef = useRef<number>(0);

  const isDev = process.env.NODE_ENV !== "production";

  function formatAttemptLine(debug?: TranscribeDebugInfo): string {
    if (!debug) return "";
    return (Array.isArray(debug.attempts) ? debug.attempts : [])
      .map((attempt) => `${attempt.strategy}:${attempt.quality_score}${attempt.rejected_reason ? ` (${attempt.rejected_reason})` : ""}`)
      .join(" · ");
  }

  function scoreColor(score: number): string {
    if (score >= 70) return "bg-emerald-400";
    if (score >= 45) return "bg-amber-400";
    return "bg-rose-400";
  }

  async function start() {
    setError("");
    if (isDev) setDebugInfo(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Navigateur non compatible avec l'audio.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const preferredType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")
          ? "audio/ogg;codecs=opus"
          : undefined;
      const rec = preferredType ? new MediaRecorder(stream, { mimeType: preferredType }) : new MediaRecorder(stream);
      chunksRef.current = [];
      startedAtRef.current = Date.now();
      rec.ondataavailable = (event) => chunksRef.current.push(event.data);
      rec.onstop = async () => {
        const elapsedMs = Date.now() - startedAtRef.current;
        const streamToStop = streamRef.current;
        streamRef.current = null;
        streamToStop?.getTracks().forEach((track) => track.stop());
        try {
          const blobType = rec.mimeType || preferredType || "audio/webm";
          const blob = new Blob(chunksRef.current, { type: blobType });
          if (elapsedMs < MIN_RECORDING_MS || blob.size < MIN_AUDIO_BYTES) {
            setError("Aucun son détecté. Réessaie en parlant plus près du micro.");
            return;
          }
          const result = await transcribeAudioDetailed(blob, token || undefined);
          if (isDev) setDebugInfo(result.debug || null);
          if (result.transcript && !isLikelyBadTranscript(result.transcript)) {
            onTranscript(result.transcript);
            setError("");
          } else {
            setError("Transcription non fiable. Réessaie dans un environnement plus calme.");
          }
        } catch (err) {
          if (err instanceof ApiError) {
            if (err.status === 401) {
              setError("Session expirée. Reconnecte-toi puis réessaie.");
              return;
            }
            if (err.status === 503) {
              setError(err.detail || "Service de transcription indisponible.");
              return;
            }
            setError(err.detail || `Erreur transcription (${err.status}).`);
            return;
          }
          const message = err instanceof Error ? err.message : "Transcription échouée.";
          setError(message);
        }
      };
      rec.start();
      mediaRef.current = rec;
      setRecording(true);
    } catch {
      setError("Permission micro refusée.");
    }
  }

  function stop() {
    mediaRef.current?.stop();
    setRecording(false);
  }

  return (
    <div className="flex items-center gap-2">
      {!recording ? (
        <button aria-label="Démarrer enregistrement" className="icon-button" onClick={start}>
          <Mic size={16} />
        </button>
      ) : (
        <button aria-label="Arrêter enregistrement" className="icon-button text-red-400" onClick={stop}>
          <Square size={16} />
        </button>
      )}
      {recording && <div className="h-2 w-2 animate-pulse rounded-full bg-red-400" />}
      {error && (
        <span className="inline-flex items-center gap-1 text-xs text-red-400">
          <X size={12} /> {error}
        </span>
      )}
      {isDev && debugInfo ? (
        <div
          className="relative"
          onMouseEnter={() => setDebugOpen(true)}
          onMouseLeave={() => setDebugOpen(false)}
          onFocus={() => setDebugOpen(true)}
          onBlur={() => setDebugOpen(false)}
        >
          <button
            type="button"
            className="inline-flex items-center gap-1 rounded-full border border-white/15 bg-white/5 px-2 py-1 text-[10px] text-slate-300"
            aria-label="Afficher le debug STT"
          >
            <span className={`h-1.5 w-1.5 rounded-full ${scoreColor(debugInfo.quality_score)}`} />
            STT
          </button>
          {debugOpen ? (
            <div className="absolute bottom-[calc(100%+8px)] right-0 z-40 w-[320px] rounded-lg border border-white/10 bg-slate-900/95 p-2 text-[11px] text-slate-200 shadow-xl">
              <p className="font-medium text-slate-100">Debug transcription</p>
              <p className="mt-1 text-slate-300">
                Score: {debugInfo.quality_score}/100
                {debugInfo.accepted_strategy ? ` · ${debugInfo.accepted_strategy}` : ""}
              </p>
              <p className="mt-1 text-slate-400">Reason: {debugInfo.rejected_reason || "none"}</p>
              <p className="mt-1 text-slate-400">{formatAttemptLine(debugInfo)}</p>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
