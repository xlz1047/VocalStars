import { useEffect, useState } from "react";

export interface GtsingerPhrase {
  id: string;
  audio_path: string;
  audio_url: string;
  group: string;
  index: number | null;
  duration_s?: number | null;
}

export interface GtsingerSong {
  id: string;
  title: string;
  singer: string;
  technique: string;
  default_audio_path: string;
  default_audio_url: string;
  phrase_count: number;
  phrases: GtsingerPhrase[];
  warnings: string[];
}

export interface GtsingerCatalog {
  schema_version: string;
  root: string;
  songs: GtsingerSong[];
  warnings: string[];
}

// Module-level cache — fetch once per page load, shared across all hook instances.
let _cache: GtsingerCatalog | null = null;
let _promise: Promise<GtsingerCatalog | null> | null = null;

function fetchCatalog(apiBase: string): Promise<GtsingerCatalog | null> {
  if (_cache) return Promise.resolve(_cache);
  if (_promise) return _promise;
  _promise = fetch(`${apiBase}/api/audio/gtsinger-catalog`)
    .then((r) => (r.ok ? (r.json() as Promise<GtsingerCatalog>) : null))
    .then((data) => {
      if (data) {
        _cache = data;
      } else {
        // Non-ok response: clear the promise so the next mount can retry.
        _promise = null;
      }
      return data;
    })
    .catch(() => {
      // Network failure: clear the promise so the next mount can retry.
      _promise = null;
      return null;
    });
  return _promise;
}

/** Returns the GTSinger catalog. Fetches once and caches; returns null while loading or on error. */
export function useGtsingerCatalog(): GtsingerCatalog | null {
  const [catalog, setCatalog] = useState<GtsingerCatalog | null>(_cache);
  const apiBase = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";

  useEffect(() => {
    if (_cache) {
      setCatalog(_cache);
      return;
    }
    fetchCatalog(apiBase).then(setCatalog);
  }, [apiBase]);

  return catalog;
}

/** Look up the phrases for a specific song by its referenceAudioUrl (0001.wav path). */
export function findSongPhrases(catalog: GtsingerCatalog | null, defaultAudioPath: string): GtsingerPhrase[] {
  if (!catalog) return [];
  const song = catalog.songs.find((s) => s.default_audio_path === defaultAudioPath);
  return song?.phrases ?? [];
}
