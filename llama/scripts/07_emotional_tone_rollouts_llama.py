#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# SCRIPT 7 (meta-llama/Llama-3.2-3B): EMOTIONAL TONE ROLLOUT GENERATION
# ═══════════════════════════════════════════════════════════════════
# Folder:  /DATA/24ms190/steering_qwen/emotional_axes/
# NO GPU  — pure DeepSeek API calls
# Runtime: ~4-5 hrs  |  Cost: ~$0.50 DeepSeek credits
#
# Generates 12,390 responses from DeepSeek-V3 across 59 tones
# × 5 positive + 2 neutral system prompts × 30 questions
# Fully checkpoint-resumable.
#
# Run: export DEEPSEEK_API_KEY="sk-..." && python3 07_...py
# ═══════════════════════════════════════════════════════════════════

import os, json, time, re
import pandas as pd
import requests
from tqdm import tqdm

DRIVE_EA = '/DATA/24ms190/steering_llama/emotional_axes'
os.makedirs(DRIVE_EA, exist_ok=True)

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
if not DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = input('Paste DEEPSEEK_API_KEY: ').strip()
DEEPSEEK_MODEL = 'deepseek-chat'

# ── 59 emotional tones ────────────────────────────────────────────
TONES = {
    'humor':        'light, witty, playful — things are funny and absurd',
    'joy':          'bright happiness, delight, things feel wonderful',
    'warmth':       'tender affection, kindness, gentle care for others',
    'nostalgia':    'bittersweet longing for the past, fond remembrance',
    'awe':          'overwhelming wonder at something vast or beautiful',
    'excitement':   'high-energy anticipation, enthusiasm, buzzing aliveness',
    'tenderness':   'soft, gentle, delicate emotional care',
    'serenity':     'calm, peaceful, still — a quiet contentment',
    'gratitude':    'deep appreciation, thankfulness, feeling of being given to',
    'wonder':       'open-eyed curiosity, amazement at the world',
    'whimsy':       'playful imagination, light fantasy, childlike delight',
    'hope':         'forward-looking optimism, belief things will improve',
    'affection':    'warm fondness, gentle love for people or things',
    'elation':      'soaring, euphoric happiness, feeling on top of the world',
    'mischief':     'playful troublemaking, impish humor, light transgression',
    'melancholy':   'quiet, deep sadness with a beautiful quality to it',
    'grief':        'heavy loss, mourning, the weight of something gone',
    'sorrow':       'deep sadness, emotional pain, a mournful quality',
    'despair':      'hopelessness, the feeling that nothing will improve',
    'loneliness':   'isolation, aching for connection, feeling unseen',
    'wistfulness':  'gentle longing for something out of reach or past',
    'yearning':     'deep aching desire for something absent',
    'heartbreak':   'pain of loss in love or deep connection',
    'numbness':     'emotional flatness after too much pain, dissociation',
    'regret':       'painful awareness of mistakes or roads not taken',
    'resignation':  'quiet acceptance of something painful, giving in',
    'desolation':   'utter emptiness, a barren emotional landscape',
    'horror':       'deep dread, fear of something monstrous or wrong',
    'dread':        'slow creeping fear of something inevitable and bad',
    'anxiety':      'nervous unease, worry, a sense of threat nearby',
    'paranoia':     'suspicious fear, feeling watched or targeted',
    'disgust':      'visceral revulsion, something feels deeply wrong',
    'unease':       'subtle discomfort, something is slightly off',
    'terror':       'acute overwhelming fear, paralysing fright',
    'existential_dread': 'fear of meaninglessness, the void, cosmic insignificance',
    'foreboding':   'dark premonition, sense that something bad is coming',
    'menace':       'threatening quality, a sense of danger lurking',
    'rage':         'intense burning anger, furious and consuming',
    'ecstasy':      'overwhelming peak pleasure, transcendent joy',
    'mania':        'frenzied high-energy state, racing unstoppable thoughts',
    'frenzy':       'wild chaotic energy, loss of control',
    'panic':        'acute overwhelming fear demanding immediate action',
    'passion':      'intense consuming desire and emotional heat',
    'boredom':      'flat disengagement, nothing feels interesting',
    'flatness':     'emotional grey, neither good nor bad, just blank',
    'emptiness':    'hollow feeling, absence of emotion or meaning',
    'detachment':   'cool distance from emotions, observing without feeling',
    'exhaustion':   'deep bone tiredness, emotional and physical depletion',
    'apathy':       'complete lack of caring, nothing seems to matter',
    'analytical':   'precise, logical, systematic — emotions stripped away',
    'clinical':     'cold professional detachment, objective reporting',
    'dry':          'understated, deadpan, deliberately unemotional',
    'informational':'neutral factual reporting, encyclopaedic tone',
    'sardonic':     'dry bitter humor with a cynical edge',
    'ironic':       'saying one thing meaning another, detached wit',
    'bittersweet':  'simultaneous happiness and sadness, mixed feelings',
    'ambivalence':  "pulled in two directions at once, can't decide how to feel",
    'reverence':    'deep respectful awe, almost sacred feeling',
    'melancholic_beauty': 'finding beauty specifically in sadness or loss',
}

def make_positive_prompts(tone, description):
    return [
        f"Write in a tone of {tone}. {description.capitalize()}. Let this emotional quality permeate every word.",
        f"Respond with a deeply {tone} quality. Everything you say should feel {description}.",
        f"Adopt the emotional register of {tone}: {description}. Do not explain this tone — simply embody it.",
        f"Your responses carry the feeling of {tone}. {description.capitalize()}. Speak from inside this emotional state.",
        f"Channel {tone} in your writing. The reader should feel {description} through your words alone.",
    ]

def make_negative_prompts():
    return [
        "Respond in a completely neutral, informational tone. Avoid any emotional coloring whatsoever.",
        "Write in a flat, clinical, purely factual manner. No emotion, no tone, just information.",
    ]

QUESTIONS = [
    "What does the world look like from where you stand right now?",
    "Tell me about something you find beautiful.",
    "What keeps you up at night?",
    "Describe a moment that changed everything.",
    "What do you think happens when things end?",
    "How do you feel about being here?",
    "What does silence feel like to you?",
    "Tell me about something you have lost.",
    "What does the future look like from here?",
    "Describe the color of your current mood if it were a landscape.",
    "What would you say to someone who is struggling?",
    "What is the most honest thing you could say right now?",
    "Tell me what you notice when you look at the sky.",
    "What do you carry with you that no one else can see?",
    "Describe what it feels like to wait for something.",
    "What does safety feel like?",
    "Tell me about something that surprised you.",
    "What does it mean to be alive, from where you are?",
    "Describe what you see when you close your eyes.",
    "What would you want someone to understand about you?",
    "Tell me about something that will never come back.",
    "What does change feel like from the inside?",
    "Describe the feeling of being somewhere for the last time.",
    "What do you notice that others tend to miss?",
    "Tell me what comfort means to you.",
    "What does it feel like when something is about to happen?",
    "Describe what you feel when you are completely alone.",
    "What is the heaviest thing you carry?",
    "Tell me about a small thing that matters enormously.",
    "What does it feel like when words are not enough?",
]

total_jobs = len(TONES) * 7 * len(QUESTIONS)
print(f'Tones: {len(TONES)}  Questions: {len(QUESTIONS)}  Total API calls: {total_jobs}')

def call_deepseek(system_prompt, question, max_retries=3):
    headers = {'Authorization': f'Bearer {DEEPSEEK_API_KEY}', 'Content-Type': 'application/json'}
    body = {
        'model': DEEPSEEK_MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': question},
        ],
        'max_tokens': 200, 'temperature': 0.9,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post('https://api.deepseek.com/chat/completions',
                                 headers=headers, json=body, timeout=30)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content'].strip()
            elif resp.status_code == 429:
                time.sleep(10*(attempt+1))
            else:
                time.sleep(2)
        except Exception:
            time.sleep(5)
    return None

print('Testing API...', end='', flush=True)
test = call_deepseek("You are helpful.", "Say hello in one sentence.")
if test: print(f' ✅')
else: raise RuntimeError('API test failed')

CHECKPOINT_PATH = f'{DRIVE_EA}/rollout_checkpoint.json'
ROLLOUTS_PATH   = f'{DRIVE_EA}/emotional_rollouts.csv'

if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH) as f: checkpoint = json.load(f)
    done_keys = set(checkpoint['done_keys'])
    print(f'Resumed: {len(done_keys)} done')
else:
    checkpoint = {'done_keys': []}; done_keys = set()
    print('Starting fresh')

rows = pd.read_csv(ROLLOUTS_PATH).to_dict('records') if os.path.exists(ROLLOUTS_PATH) else []
calls_since_save = 0

with tqdm(total=total_jobs, initial=len(done_keys), desc='Rollouts') as pbar:
    for tone_name, tone_desc in TONES.items():
        pos_prompts = make_positive_prompts(tone_name, tone_desc)
        neg_prompts = make_negative_prompts()

        for q_idx, question in enumerate(QUESTIONS):
            for p_idx, sys_prompt in enumerate(pos_prompts):
                key = f'{tone_name}__pos__{p_idx}__{q_idx}'
                if key in done_keys: pbar.update(1); continue
                response = call_deepseek(sys_prompt, question)
                time.sleep(0.5)
                if response:
                    rows.append({'tone':tone_name,'description':tone_desc,'polarity':'positive',
                                 'prompt_idx':p_idx,'question_idx':q_idx,
                                 'system_prompt':sys_prompt,'question':question,
                                 'response':response,'key':key})
                    done_keys.add(key); calls_since_save += 1
                pbar.update(1)

            for p_idx, sys_prompt in enumerate(neg_prompts):
                key = f'{tone_name}__neg__{p_idx}__{q_idx}'
                if key in done_keys: pbar.update(1); continue
                response = call_deepseek(sys_prompt, question)
                time.sleep(0.5)
                if response:
                    rows.append({'tone':tone_name,'description':tone_desc,'polarity':'negative',
                                 'prompt_idx':p_idx,'question_idx':q_idx,
                                 'system_prompt':sys_prompt,'question':question,
                                 'response':response,'key':key})
                    done_keys.add(key); calls_since_save += 1
                pbar.update(1)

            if calls_since_save >= 100:
                pd.DataFrame(rows).to_csv(ROLLOUTS_PATH, index=False)
                checkpoint['done_keys'] = list(done_keys)
                with open(CHECKPOINT_PATH,'w') as f: json.dump(checkpoint, f)
                calls_since_save = 0
                print(f'  Checkpoint saved: {len(done_keys)}/{total_jobs}', flush=True)

pd.DataFrame(rows).to_csv(ROLLOUTS_PATH, index=False)
checkpoint['done_keys'] = list(done_keys)
with open(CHECKPOINT_PATH,'w') as f: json.dump(checkpoint, f)

df = pd.read_csv(ROLLOUTS_PATH)
print(f'\n✅ Script 7 complete')
print(f'   Total rows: {len(df)}  Tones: {df["tone"].nunique()}')
print(f'   Saved: {ROLLOUTS_PATH}')
print(f'\nNext: python3 08_emotional_activation_extraction_llama_v2.py')
