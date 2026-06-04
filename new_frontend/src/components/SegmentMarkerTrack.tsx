import { useState } from "react";
import { ChevronDown, ChevronUp, Info } from "lucide-react";
import { UiReadyAnalysis, UiSegment } from "../types";
import { hzToNoteName } from "../utils/noteUtils";

interface SegmentMarkerTrackProps {
  analysis: UiReadyAnalysis;
  compact?: boolean;
  selectedSegmentId?: string | null;
  onSelectSegment?: (segment: UiSegment, groupKey: string) => void;
}

interface GroupMeta {
  label: string;
  description: string;
  color: string;
  advanced?: boolean;
}

const GROUP_META: Record<string, GroupMeta> = {
  notes: {
    label: "Notes",
    description: "Each detected sung note — tap to jump to that moment.",
    color: "#ffb1c0",
  },
  phrases: {
    label: "Phrases",
    description: "Continuous vocal phrases without a break.",
    color: "#3cddc7",
  },
  reference_pitch_error_regions: {
    label: "Off Target",
    description: "Regions where pitch was noticeably off from the reference target.",
    color: "#ff7a90",
  },
  unstable_pitch_regions: {
    label: "Shaky Pitch",
    description: "Moments where pitch was unstable or wavering.",
    color: "#ddb7ff",
  },
  dropouts: {
    label: "Gaps",
    description: "Moments of silence or signal loss during the phrase.",
    color: "#ffb4ab",
    advanced: true,
  },
  low_confidence_regions: {
    label: "Unclear Signal",
    description: "Parts of the recording where the signal was noisy or unclear.",
    color: "#ffd166",
    advanced: true,
  },
  breath_phrase_proxy_regions: {
    label: "Breath Support",
    description: "Proxy estimate of breath phrase quality (not a direct breath diagnosis).",
    color: "#77ddaa",
    advanced: true,
  },
  tone_consistency_proxy_regions: {
    label: "Tone Quality",
    description: "Proxy estimate of tonal consistency (not a timbre diagnosis).",
    color: "#8ec5ff",
    advanced: true,
  },
};

const SEGMENT_GROUP_ORDER = [
  "notes",
  "phrases",
  "reference_pitch_error_regions",
  "unstable_pitch_regions",
  "dropouts",
  "low_confidence_regions",
  "breath_phrase_proxy_regions",
  "tone_consistency_proxy_regions",
];

function frameDuration(analysis: UiReadyAnalysis): number {
  const frames = analysis.frames || [];
  const last = frames.reduce((max, frame) => Math.max(max, frame.time_s || 0), 0);
  return Math.max(analysis.audio?.duration_s || 0, last, 0.1);
}

function segmentBounds(segment: UiSegment) {
  const start = Math.max(0, Number(segment.start_s || 0));
  const explicitEnd = Number(segment.end_s);
  const duration = Number(segment.duration_s || 0.04);
  const end = Number.isFinite(explicitEnd) && explicitEnd > start ? explicitEnd : start + Math.max(duration, 0.04);
  return { start, end };
}

function segmentNoteLabel(segment: UiSegment, groupKey: string): string | null {
  if (groupKey !== "notes") return null;
  const hz = segment.median_f0_hz;
  if (typeof hz !== "number" || hz <= 0) return null;
  return hzToNoteName(hz);
}

function TooltipCard({ segment, groupKey }: { segment: UiSegment; groupKey: string }) {
  const meta = GROUP_META[groupKey];
  const { start, end } = segmentBounds(segment);
  const dur = end - start;
  const severityColors: Record<string, string> = {
    error:   "text-red-400",
    warning: "text-amber-300",
    info:    "text-tertiary",
  };
  const severityColor = severityColors[segment.ui_severity || "info"] ?? "text-tertiary";
  const noteHz = segment.median_f0_hz;
  const noteNameDisplay = groupKey === "notes" && noteHz != null && noteHz > 0
    ? `${hzToNoteName(noteHz)} (${noteHz.toFixed(0)} Hz)`
    : noteHz != null && noteHz > 0
    ? `${noteHz.toFixed(0)} Hz`
    : null;
  return (
    <div className="text-left space-y-1">
      <p className={`text-[10px] font-bold uppercase tracking-wider ${severityColor}`}>
        {meta?.label ?? groupKey}
        {groupKey === "notes" && noteHz != null && noteHz > 0 && (
          <span className="ml-1.5 normal-case font-mono text-white">{hzToNoteName(noteHz)}</span>
        )}
        {segment.ui_severity && segment.ui_severity !== "info" && (
          <span className="ml-1 normal-case">· {segment.ui_severity}</span>
        )}
      </p>
      {noteNameDisplay && (
        <p className="text-[11px] text-white/70 font-mono">{noteNameDisplay}</p>
      )}
      <p className="text-[11px] text-on-surface-variant">
        {start.toFixed(2)} s – {end.toFixed(2)} s ({dur.toFixed(2)} s)
      </p>
      {(segment.summary || segment.actionable_hint) && (
        <p className="text-[11px] text-on-surface-variant leading-snug max-w-[220px]">
          {segment.summary || segment.actionable_hint}
        </p>
      )}
    </div>
  );
}

export default function SegmentMarkerTrack({
  analysis,
  compact = false,
  selectedSegmentId,
  onSelectSegment,
}: SegmentMarkerTrackProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showLegend, setShowLegend] = useState(false);
  const [tooltipId, setTooltipId] = useState<string | null>(null);

  const duration = frameDuration(analysis);
  const segments = analysis.segments || {};

  const allGroups = SEGMENT_GROUP_ORDER
    .map((key) => ({ key, items: segments[key] || [], meta: GROUP_META[key] }))
    .filter(({ items }) => items.length > 0);

  const primaryGroups = allGroups.filter(({ meta }) => !meta?.advanced);
  const advancedGroups = allGroups.filter(({ meta }) => meta?.advanced);

  if (!allGroups.length) {
    return (
      <section className="glass-card rounded-2xl p-5 border border-white/5">
        <h3 className="font-display font-bold text-lg text-white">Replay Markers</h3>
        <p className="text-sm text-on-surface-variant mt-2">No replay markers were returned for this take.</p>
      </section>
    );
  }

  const renderGroup = (key: string, items: UiSegment[], meta: GroupMeta | undefined) => (
    <div key={key} className="space-y-1.5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-1.5">
          <span
            className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
            style={{ background: meta?.color ?? "#ffb1c0" }}
          />
          <p className="text-xs font-bold uppercase tracking-wider text-on-surface-variant">
            {meta?.label ?? key.replaceAll("_", " ")}
          </p>
          {meta?.description && (
            <span
              className="text-on-surface-variant/40 hover:text-on-surface-variant/80 cursor-default transition-colors"
              title={meta.description}
            >
              <Info className="w-3 h-3" />
            </span>
          )}
        </div>
        <span className="text-[11px] text-on-surface-variant">{items.length}</span>
      </div>
      <div className="relative h-8 rounded-lg bg-surface-container-lowest border border-white/5 overflow-visible">
        {items.map((segment, index) => {
          const { start, end } = segmentBounds(segment);
          const left = Math.max(0, Math.min(100, (start / duration) * 100));
          const markerWidth = Math.max(((end - start) / duration) * 100, 0.8);
          const segmentId = segment.id || `${key}-${index}`;
          const selected = selectedSegmentId === segmentId;
          const isHovered = tooltipId === segmentId;
          const noteLabel = segmentNoteLabel(segment, key);
          const barWidthPx = (markerWidth / 100) * 100;

          return (
            <div key={segmentId} className="absolute top-0 bottom-0" style={{ left: `${left}%`, width: `${Math.min(markerWidth, 100 - left)}%` }}>
              <button
                type="button"
                aria-label={`${meta?.label ?? key} from ${start.toFixed(2)} to ${end.toFixed(2)} seconds${segment.actionable_hint ? `: ${segment.actionable_hint}` : ""}`}
                onClick={() => onSelectSegment?.({ ...segment, id: segmentId }, key)}
                onMouseEnter={() => setTooltipId(segmentId)}
                onMouseLeave={() => setTooltipId(null)}
                onFocus={() => setTooltipId(segmentId)}
                onBlur={() => setTooltipId(null)}
                className="absolute inset-y-1 inset-x-0 rounded cursor-pointer hover:brightness-125 transition-all focus:outline-none focus:ring-2 focus:ring-white/80"
                style={{
                  background: meta?.color ?? "#ffb1c0",
                  opacity: selected ? 1 : segment.ui_severity === "warning" || segment.ui_severity === "error" ? 0.95 : 0.72,
                  boxShadow: selected ? "0 0 0 2px rgba(255,255,255,0.8), 0 0 16px rgba(255,177,192,0.38)" : undefined,
                }}
              >
                {noteLabel && barWidthPx > 28 && (
                  <span className="absolute inset-0 flex items-center justify-center text-[9px] font-bold text-black/70 leading-none overflow-hidden whitespace-nowrap px-1">
                    {noteLabel}
                  </span>
                )}
              </button>
              {isHovered && (
                <div className="absolute top-[110%] left-0 z-50 pointer-events-none bg-surface-container-high border border-white/15 rounded-xl p-3 shadow-xl min-w-[160px] max-w-[240px]">
                  <TooltipCard segment={segment} groupKey={key} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );

  return (
    <section className="glass-card rounded-2xl p-5 border border-white/5 space-y-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="font-display font-bold text-lg text-white">Replay Markers</h3>
          <p className="text-xs text-on-surface-variant">
            Tap a marker to jump to that moment. Hover for details.
          </p>
        </div>
        <button
          onClick={() => setShowLegend((v) => !v)}
          className="flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider text-on-surface-variant/60 hover:text-on-surface-variant border border-white/8 rounded-lg px-2.5 py-1.5 hover:bg-white/5 transition-all"
        >
          Legend {showLegend ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </button>
      </div>

      {showLegend && (
        <div className="grid sm:grid-cols-2 gap-2 text-[11px] bg-surface-container-lowest/60 border border-white/5 rounded-xl p-3">
          {allGroups.map(({ key, meta }) => (
            <div key={key} className="flex items-start gap-2">
              <span className="w-2.5 h-2.5 rounded-sm mt-0.5 flex-shrink-0" style={{ background: meta?.color ?? "#ffb1c0" }} />
              <div>
                <span className="font-bold text-white">{meta?.label ?? key}</span>
                {meta?.description && (
                  <p className="text-on-surface-variant/70 leading-snug mt-0.5">{meta.description}</p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className={compact ? "space-y-2" : "space-y-3"}>
        {primaryGroups.map(({ key, items, meta }) => renderGroup(key, items, meta))}
      </div>

      {advancedGroups.length > 0 && (
        <div className="space-y-2">
          <button
            onClick={() => setShowAdvanced((v) => !v)}
            className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-on-surface-variant/50 hover:text-on-surface-variant transition-colors"
          >
            {showAdvanced ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            Advanced ({advancedGroups.length} more)
          </button>
          {showAdvanced && (
            <div className={compact ? "space-y-2" : "space-y-3"}>
              {advancedGroups.map(({ key, items, meta }) => renderGroup(key, items, meta))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
