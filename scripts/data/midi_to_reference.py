#!/usr/bin/env python3
"""Convert a Standard MIDI File into VocalStars reference-note JSON.

This is intentionally dependency-free so practice MIDI files can be turned into
reference targets without adding a music stack yet. It supports ordinary PPQ
SMF format 0/1 files, tempo changes, running status, note-on, and note-off.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class MidiNote:
    start_tick: int
    end_tick: int
    midi: int
    channel: int
    track: int
    velocity: int

    @property
    def duration_ticks(self) -> int:
        return max(0, self.end_tick - self.start_tick)


def read_u16(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack(">H", data[offset : offset + 2])[0], offset + 2


def read_u32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack(">I", data[offset : offset + 4])[0], offset + 4


def read_vlq(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def midi_to_note_name(midi: int) -> str:
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"


def parse_header(data: bytes) -> tuple[int, int, int, int]:
    if data[:4] != b"MThd":
        raise ValueError("Not a Standard MIDI File: missing MThd header")
    header_len, offset = read_u32(data, 4)
    if header_len < 6:
        raise ValueError(f"Unsupported MIDI header length: {header_len}")
    fmt, offset = read_u16(data, offset)
    track_count, offset = read_u16(data, offset)
    division, offset = read_u16(data, offset)
    if division & 0x8000:
        raise ValueError("SMPTE MIDI timing is not supported yet; use PPQ MIDI files")
    return fmt, track_count, division, 8 + header_len


def parse_track(track_data: bytes, track_index: int) -> tuple[list[MidiNote], list[tuple[int, int]]]:
    offset = 0
    tick = 0
    running_status: int | None = None
    open_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    notes: list[MidiNote] = []
    tempos: list[tuple[int, int]] = []

    while offset < len(track_data):
        delta, offset = read_vlq(track_data, offset)
        tick += delta
        status = track_data[offset]
        if status & 0x80:
            offset += 1
            if status < 0xF0:
                running_status = status
        elif running_status is not None:
            status = running_status
        else:
            raise ValueError(f"Running status used before status byte in track {track_index}")

        if status == 0xFF:
            meta_type = track_data[offset]
            offset += 1
            length, offset = read_vlq(track_data, offset)
            payload = track_data[offset : offset + length]
            offset += length
            if meta_type == 0x51 and len(payload) == 3:
                tempos.append((tick, int.from_bytes(payload, "big")))
            if meta_type == 0x2F:
                break
            continue

        if status in (0xF0, 0xF7):
            length, offset = read_vlq(track_data, offset)
            offset += length
            continue

        event_type = status & 0xF0
        channel = status & 0x0F
        data_len = 1 if event_type in (0xC0, 0xD0) else 2
        payload = track_data[offset : offset + data_len]
        offset += data_len

        if event_type == 0x90 and len(payload) == 2:
            midi, velocity = payload[0], payload[1]
            key = (channel, midi)
            if velocity == 0:
                if open_notes.get(key):
                    start_tick, start_velocity = open_notes[key].pop(0)
                    notes.append(MidiNote(start_tick, tick, midi, channel, track_index, start_velocity))
            else:
                open_notes.setdefault(key, []).append((tick, velocity))
        elif event_type == 0x80 and len(payload) == 2:
            midi = payload[0]
            key = (channel, midi)
            if open_notes.get(key):
                start_tick, start_velocity = open_notes[key].pop(0)
                notes.append(MidiNote(start_tick, tick, midi, channel, track_index, start_velocity))

    return notes, tempos


def parse_midi(path: Path) -> tuple[int, list[MidiNote], list[tuple[int, int]]]:
    data = path.read_bytes()
    _fmt, track_count, ticks_per_beat, offset = parse_header(data)
    notes: list[MidiNote] = []
    tempos: list[tuple[int, int]] = [(0, 500000)]
    for track_index in range(track_count):
        if data[offset : offset + 4] != b"MTrk":
            raise ValueError(f"Expected MTrk at offset {offset}")
        length, offset = read_u32(data, offset + 4)
        track_data = data[offset : offset + length]
        offset += length
        track_notes, track_tempos = parse_track(track_data, track_index)
        notes.extend(track_notes)
        tempos.extend(track_tempos)
    notes.sort(key=lambda note: (note.start_tick, note.end_tick, note.midi))
    tempos = sorted(set(tempos), key=lambda item: item[0])
    return ticks_per_beat, notes, tempos


def tick_to_seconds(tick: int, tempos: list[tuple[int, int]], ticks_per_beat: int) -> float:
    seconds = 0.0
    prev_tick = 0
    current_tempo = 500000
    for tempo_tick, tempo in tempos:
        if tempo_tick >= tick:
            break
        seconds += (tempo_tick - prev_tick) * current_tempo / 1_000_000.0 / ticks_per_beat
        prev_tick = tempo_tick
        current_tempo = tempo
    seconds += (tick - prev_tick) * current_tempo / 1_000_000.0 / ticks_per_beat
    return seconds


def reference_from_midi(path: Path, title: str | None = None) -> dict[str, Any]:
    ticks_per_beat, notes, tempos = parse_midi(path)
    note_items: list[dict[str, Any]] = []
    for note in notes:
        start_s = tick_to_seconds(note.start_tick, tempos, ticks_per_beat)
        end_s = tick_to_seconds(note.end_tick, tempos, ticks_per_beat)
        if end_s <= start_s:
            continue
        note_items.append(
            {
                "start_s": round(start_s, 6),
                "end_s": round(end_s, 6),
                "duration_s": round(end_s - start_s, 6),
                "midi": note.midi,
                "note": midi_to_note_name(note.midi),
                "f0_hz": round(midi_to_hz(note.midi), 6),
                "channel": note.channel,
                "track": note.track,
                "velocity": note.velocity,
            }
        )

    duration_s = max((item["end_s"] for item in note_items), default=0.0)
    return {
        "schema_version": "vocalstars.reference_melody.v1",
        "source_path": str(path),
        "title": title or path.stem.replace("_", " ").title(),
        "type": "midi_note_sequence",
        "ticks_per_beat": ticks_per_beat,
        "duration_s": round(duration_s, 6),
        "notes": note_items,
        "task_config_patch": {
            "reference": {
                "type": "midi_note_sequence",
                "title": title or path.stem.replace("_", " ").title(),
                "notes": [item["note"] for item in note_items],
                "f0_hz": [item["f0_hz"] for item in note_items],
                "durations_s": [item["duration_s"] for item in note_items],
                "note_events": note_items,
            }
        },
    }


def vlq(value: int) -> bytes:
    if value == 0:
        return b"\x00"
    parts = [value & 0x7F]
    value >>= 7
    while value:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(parts))


def write_demo_midi(path: Path) -> None:
    ticks_per_beat = 480
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    events = bytearray()
    events.extend(vlq(0) + b"\xFF\x51\x03" + (500000).to_bytes(3, "big"))
    for pitch in pitches:
        events.extend(vlq(0) + bytes([0x90, pitch, 90]))
        events.extend(vlq(ticks_per_beat) + bytes([0x80, pitch, 0]))
    events.extend(vlq(0) + b"\xFF\x2F\x00")
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat)
    track = b"MTrk" + struct.pack(">I", len(events)) + bytes(events)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + track)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi", type=Path, nargs="?", help="Path to a .mid/.midi file")
    parser.add_argument("--output", type=Path, help="Output reference JSON path")
    parser.add_argument("--title", help="Reference title")
    parser.add_argument("--write-demo-midi", type=Path, help="Write a C major scale MIDI fixture and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_demo_midi:
        write_demo_midi(args.write_demo_midi)
        print(f"Wrote demo MIDI to {args.write_demo_midi}")
        return 0
    if not args.midi:
        raise SystemExit("Provide a MIDI file or --write-demo-midi")
    reference = reference_from_midi(args.midi, args.title)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote reference JSON to {args.output}")
    else:
        print(json.dumps(reference, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

