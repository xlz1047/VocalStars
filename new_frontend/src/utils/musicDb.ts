import { Song, Exercise } from "../types";

export const SONGS: Song[] = [
  {
    id: "midnight-resonance",
    title: "Midnight Resonance",
    artist: "VocalStars Resident",
    genre: "Synth Wave",
    difficulty: "MEDIUM",
    duration: "03:15",
    bpm: 120,
    imageUrl: "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=400&auto=format&fit=crop&q=80",
    featured: true,
    lyrics: [
      "In the shadows of the neon light",
      "We are dancing through the deeper night",
      "Unleash your soulful resonance inside",
      "With the synthesizer as our guide",
      "Hold the note and let the feelings flow",
      "In the resonance of the midnight glow"
    ],
    referencePitchSeq: [50, 60, 48, 55, 62, 70, 65, 50, 42, 60, 55, 65, 70, 75, 62, 50, 45, 52, 60, 55]
  },
  {
    id: "electric-dreams",
    title: "Electric Dreams",
    artist: "Luna Ray",
    genre: "Synth Pop",
    difficulty: "EASY",
    duration: "02:50",
    bpm: 110,
    imageUrl: "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=400&auto=format&fit=crop&q=80",
    lyrics: [
      "Electric pulses running through my mind",
      "A golden signal that is hard to find",
      "We build a dream inside a starry shell",
      "Under the power of a techno spell",
      "Come on and wake up now",
      "Let's make it real somehow"
    ],
    referencePitchSeq: [40, 40, 45, 50, 50, 45, 40, 55, 60, 60, 55, 50, 45, 45, 50, 55, 60, 50, 40, 40]
  },
  {
    id: "after-hours",
    title: "After Hours",
    artist: "The Night Sky",
    genre: "Retrowave",
    difficulty: "MEDIUM",
    duration: "03:40",
    bpm: 115,
    imageUrl: "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=400&auto=format&fit=crop&q=80",
    lyrics: [
      "The sun went down an hour ago",
      "But we still have that inner spark inside",
      "Cruising past the city skylines slow",
      "In a dream where we don't have to hide",
      "Hold onto me, don't let it fade away",
      "We'll sing until the breaking of the day"
    ],
    referencePitchSeq: [30, 35, 45, 50, 42, 38, 50, 55, 65, 70, 58, 52, 65, 72, 80, 75, 62, 50, 45, 45]
  },
  {
    id: "unstoppable",
    title: "Unstoppable",
    artist: "Sia (Cover)",
    genre: "Pop Anthem",
    difficulty: "HARD",
    duration: "03:12",
    bpm: 125,
    imageUrl: "https://lh3.googleusercontent.com/aida-public/AB6AXuDxpI7FgU9FNUy001yzMNQfZdraRCuA1_acXoAJ8NUrnXYmioVldCg8BK25azx2I8Abj5_esb9GMn-tNTR_5JfBMrrtVcZzxlf-RonFh4r-OZwiQIYD5Tf5L8trBPPZ6AXqcAjeMpJhgnLeudnSjwaU7auCcnQKpJT4SAvDFIqwJkBX5pMImSJwOIcfL8VhRsVMgPJlHYTr-FTh1dUL8SdNjoVFyxmVI0nOb8oAZgj-gwuggiajNWWJ0XYwpe1-zTE7OazAbfOFzIpo",
    lyrics: [
      "I'm unstoppable, I'm a Porsche with no brakes",
      "I'm invincible, yeah, I win every single game",
      "I'm so powerful, I don't need batteries to play",
      "I'm so confident, yeah, I'm unstoppable today",
      "Break down the walls that hold you back tonight",
      "And lift your vocal power to the light"
    ],
    referencePitchSeq: [60, 65, 75, 70, 80, 82, 70, 60, 80, 85, 90, 85, 75, 65, 80, 85, 92, 88, 70, 65]
  },
  {
    id: "modern-soul",
    title: "Modern Soul",
    artist: "Echo Park",
    genre: "Soul & R&B",
    difficulty: "EASY",
    duration: "03:05",
    bpm: 95,
    imageUrl: "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=400&auto=format&fit=crop&q=80",
    lyrics: [
      "Walking down the old avenues of town",
      "Looking for a voice that doesn't put me down",
      "A modern soul is crying in the street",
      "Waiting for a rhythm to align the beat",
      "We have the key, we have the heart to sing",
      "Let's see the joy that music has to bring"
    ],
    referencePitchSeq: [45, 48, 52, 48, 55, 50, 45, 52, 58, 62, 55, 50, 58, 64, 68, 60, 52, 48, 45, 45]
  },
  {
    id: "frequency",
    title: "Frequency",
    artist: "Vibe Check",
    genre: "Electronic",
    difficulty: "MEDIUM",
    duration: "03:22",
    bpm: 124,
    imageUrl: "https://images.unsplash.com/photo-1507838153414-b4b713384a76?w=400&auto=format&fit=crop&q=80",
    lyrics: [
      "Send your frequency into the empty wide",
      "There's no reason left for us to run and hide",
      "Can you feel the pulse inside your very shell?",
      "Every single wavelength weaving in a spell",
      "Hold the high notes, keep them clear and grand",
      "Across the echoes of this digital land"
    ],
    referencePitchSeq: [50, 55, 65, 58, 70, 72, 60, 50, 65, 70, 78, 72, 65, 55, 70, 75, 82, 78, 60, 55]
  },
  {
    id: "midnight-reverie",
    title: "Midnight Reverie",
    artist: "Luna Ray",
    genre: "Jazz Pop",
    difficulty: "MEDIUM",
    duration: "03:45",
    bpm: 124,
    imageUrl: "https://images.unsplash.com/photo-1487180142328-054b783fc471?w=400&auto=format&fit=crop&q=80",
    lyrics: [
      "I used to dream in shades of blue",
      "Before the stars aligned with you",
      "A symphony of quiet hearts",
      "Is where the magic really starts",
      "Hold your breath and count to three",
      "The world is ours, just you and me",
      "Singing our sweet, silent reverie"
    ],
    referencePitchSeq: [42, 45, 55, 60, 52, 48, 58, 62, 70, 75, 65, 55, 68, 72, 80, 85, 72, 60, 50, 48]
  },
  {
    id: "midnight-starlight",
    title: "Midnight Starlight",
    artist: "Nova Lyra",
    genre: "Dream Pop",
    difficulty: "EASY",
    duration: "02:55",
    bpm: 105,
    imageUrl: "https://images.unsplash.com/photo-1506157786151-b8491531f063?w=400&auto=format&fit=crop&q=80",
    lyrics: [
      "Can we dance under the midnight starlight glow?",
      "Forget the fears and let our heavy worries go",
      "A voice is whispering a melody so bright",
      "Guiding our spirits safely through the starry night",
      "Just hold the tone, let it sustain and shine",
      "A perfect frequency of your voice and mine"
    ],
    referencePitchSeq: [40, 42, 48, 52, 45, 42, 50, 55, 60, 65, 55, 48, 58, 62, 70, 72, 60, 50, 42, 40]
  }
];

export const EXERCISES: Exercise[] = [
  {
    id: "breath-support",
    title: "Breath Support",
    description: "Hiss Exercise: Timed sustained exhales for core control.",
    duration: "5 MINS",
    type: "breath",
    difficulty: "EASY",
    progress: 33
  },
  {
    id: "pitch-resonance",
    title: "Pitch & Resonance",
    description: "Humming Scales, Lip Trills, and Straw Phonation.",
    duration: "8 MINS",
    type: "pitch",
    difficulty: "MEDIUM",
    progress: 50
  },
  {
    id: "agility-range",
    title: "Agility & Range",
    description: "Vocal Sirens, Vocal Glides, and Yawn-Sigh technique.",
    duration: "10 MINS",
    type: "agility",
    difficulty: "HARD",
    progress: 75
  },
  {
    id: "range-assessment",
    title: "Range Assessment",
    description: "Update your vocal profile with a smart dynamic diagnostic.",
    duration: "3 MINS",
    type: "assessment",
    difficulty: "MEDIUM",
    progress: 100
  }
];
