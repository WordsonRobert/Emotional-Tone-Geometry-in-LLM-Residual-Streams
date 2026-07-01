# Dataset

A purpose-built corpus for studying emotional tone representation in LLMs. Shared identically across all three model pipelines (Llama-3.2-3B, Qwen3-4B, Gemma-3-4b-it).

------

## Files

| File                      | Emotion    | N    |
| ------------------------- | ---------- | ---- |
| `sampled_horror.json`     | horror     | 200  |
| `sampled_melancholy.json` | melancholy | 200  |
| `sampled_humor.json`      | humor      | 200  |

------

## Design

### Word-count tiers

Each emotion is split into three length tiers to prevent length from acting as a confound for tone in neural probes (longer texts accumulate more positional encoding signal regardless of content).

| Tier   | Word range        | N per emotion |
| ------ | ----------------- | ------------- |
| Short  | < 500 words       | 40            |
| Medium | 800–1,100 words   | 80            |
| Long   | 1,100–1,500 words | 80            |

### Confidence threshold

All items passed a DeepSeek-chat tone classifier (Agent 3) with either a minimum confidence of **≥ 0.95** or have been human verified.

### Quality filtering

All items were additionally scored 1–10 by DeepSeek for structural quality (ad/navigation leakage, mid-text interruptions, transcription noise, encoding artifacts). Items below 7 were deprioritized; the final dataset minimum quality score is 6.0 across all three emotions, with means of 9.3 (horror), 9.1 (melancholy), 8.7 (humor). Seven humor items from YouTube comedy transcripts were manually cleaned before inclusion to remove crosstalk and punctuation artifacts.

------

## Sources

### Horror (200 items)

Sourced from Reddit horror fiction communities via the Arctic Shift API. All items are original first-person or third-person horror fiction. No news articles or non-fiction content.

|                                         |      |
| --------------------------------------- | ---- |
| r/shortscarystories                     |      |
| r/nosleep                               |      |
| r/DarkTales                             |      |
| r/creepypasta                           |      |
| r/TwoSentenceHorror                     |      |
| Other (creepypasta.com, r/flashfiction) |      |

**Word count:** min 50, max 1,495, mean 997, median 1,069 **Confidence:** min 0.95, mean 0.971 **Quality score:** min 6.0, mean 9.3

------

### Melancholy (200 items)

Sourced from Reddit personal narrative communities and YouTube spoken word / poetry channels. Reddit items are first-person accounts of grief, loss, depression, and relationship breakdown. YouTube items are spoken word poetry transcribed via youtube-transcript-api with Whisper fallback.

|                                |      |
| ------------------------------ | ---- |
| YouTube (spoken word / poetry) |      |
| r/grief                        |      |
| r/TrueOffMyChest               |      |
| r/BreakUps                     |      |
| r/offmychest                   |      |
| r/relationship_advice          |      |
| r/depression                   |      |
| r/lonely                       |      |
| r/OCPoetry                     |      |

**Word count:** min 218, max 1,484, mean 999, median 1,068 **Confidence:** min 0.95, mean 0.953 **Quality score:** min 6.0, mean 9.1

------

### Humor (200 items)

Sourced from Reddit humor/anecdote communities, YouTube stand-up transcripts (Kill Tony), and satirical web publications. Reddit items are first-person comedic narratives. YouTube items are stand-up comedy bits transcribed and cleaned. Web items are long-form satirical essays.

| Source                       |      |
| ---------------------------- | ---- |
| r/talesfromtechsupport       |      |
| YouTube (Kill Tony stand-up) |      |
| r/tifu                       |      |
| r/IDontWorkHereLady          |      |
| r/pettyrevenge               |      |
| r/MaliciousCompliance        |      |
| r/ProRevenge                 |      |
| r/TalesFromTheFrontDesk      |      |
| r/TalesFromRetail            |      |
| r/TalesFromYourServer        |      |
| Web satire (cracked.com)     |      |

**Word count:** min 151, max 1,484, mean 969, median 1,061 **Confidence:** min 0.90, mean 0.943 **Quality score:** min 6.0, mean 8.7

------

## Pipeline Summary

Items passed through a 5-agent pipeline before inclusion:

| Agent    | Role                                                         | Rejection rate    |
| -------- | ------------------------------------------------------------ | ----------------- |
| Agent 1  | Scraper (Reddit / YouTube / web)                             | —                 |
| Agent 1b | Local preprocessor (markdown, HTML, encoding)                | —                 |
| Agent 2  | Length & quality validator (word count gates, repetition ratio) | ~70% of raw       |
| Agent 3  | DeepSeek tone classifier (confidence ≥ 0.95 / ≥ 0.90)        | ~25% of validated |
| Agent 4  | DeepSeek cleaner (artifact removal only, temp=0)             | ~0.5%             |
| Agent 5  | Fidelity guard (BLEU ≥ 85 AND chrF ≥ 90 vs original)         | ~5%               |

Final sampling selected the highest-quality items per bucket using a separate DeepSeek structural quality scoring pass.

------

## Notes

- Dataset construction post-dates the training cutoffs of all three evaluated models, minimising memorisation confounds.
- Length tiers are mutually exclusive and exhaustive within each emotion category.
- All Reddit data was retrieved via the [Arctic Shift API](https://arctic-shift.photon-reddit.com) (no authentication required).
- YouTube transcription used `youtube-transcript-api` with OpenAI Whisper as fallback for videos without auto-captions.
