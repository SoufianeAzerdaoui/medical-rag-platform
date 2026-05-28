"use client";

import { Mic, Square, X } from "lucide-react";
import { useRef, useState } from "react";
import { transcribeAudio } from "@/services/rag-api";

export function VoiceRecorder({ onTranscript }: { onTranscript: (value: string) => void }) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState("");
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  async function start() {
    setError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Navigateur non compatible avec l'audio.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      chunksRef.current = [];
      rec.ondataavailable = (event) => chunksRef.current.push(event.data);
      rec.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const transcript = await transcribeAudio(blob);
        if (transcript) onTranscript(transcript);
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
    </div>
  );
}
