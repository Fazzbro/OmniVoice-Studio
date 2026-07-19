"""
Long-Form Audiobook & Multi-Sentence Speech Engine for OmniVoice Studio.

Splits long chapters or articles into natural sentence chunks, synthesizes each chunk
with high-fidelity zero-shot voice cloning, and stitches the audio seamlessly with
configurable sentence and paragraph pauses.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf
from omnivoice import OmniVoiceGenerationConfig, VoiceClonePrompt


def split_into_chunks(text: str, max_chars: int = 180) -> List[Tuple[str, bool]]:
    """
    Splits long text into manageable chunks (under max_chars) at natural boundaries.
    
    Returns:
        List of tuples: (chunk_text, is_end_of_paragraph)
    """
    if not text or not text.strip():
        return []

    # First split by paragraphs (double or more newlines)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks_with_meta = []

    for p_idx, paragraph in enumerate(paragraphs):
        # Clean internal single newlines inside the paragraph
        clean_para = " ".join(paragraph.splitlines()).strip()
        
        # Split paragraph into sentences on period, exclamation, question mark, or semicolon
        # retaining the punctuation mark with the sentence.
        sentences = [s.strip() for s in re.split(r'(?<=[.!?;\n])\s+', clean_para) if s.strip()]
        
        para_chunks = []
        for sent in sentences:
            if len(sent) <= max_chars:
                para_chunks.append(sent)
            else:
                subclauses = [sub.strip() for sub in re.split(r'(?<=[,:])\s+', sent) if sub.strip()]
                current_sub = ""
                for clause in subclauses:
                    if len(current_sub) + len(clause) + 1 <= max_chars:
                        current_sub = f"{current_sub} {clause}".strip() if current_sub else clause
                    else:
                        if current_sub:
                            para_chunks.append(current_sub)
                            current_sub = ""
                        # If a single clause is STILL longer than max_chars, hard split on words
                        if len(clause) > max_chars:
                            words = clause.split()
                            word_chunk = ""
                            for w in words:
                                if len(word_chunk) + len(w) + 1 <= max_chars:
                                    word_chunk = f"{word_chunk} {w}".strip() if word_chunk else w
                                else:
                                    if word_chunk:
                                        para_chunks.append(word_chunk)
                                    word_chunk = w
                            if word_chunk:
                                para_chunks.append(word_chunk)
                        else:
                            current_sub = clause
                if current_sub:
                    para_chunks.append(current_sub)
                    current_sub = ""

        # Mark only the last chunk of the paragraph as `is_end_of_paragraph=True`
        for c_idx, c_text in enumerate(para_chunks):
            is_para_end = (c_idx == len(para_chunks) - 1) and (p_idx < len(paragraphs) - 1)
            chunks_with_meta.append((c_text, is_para_end))

    return chunks_with_meta


def synthesize_longform_audiobook(
    model: Any,
    text: str,
    language: Optional[str] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    prompt_file: Optional[str] = None,
    instruct: Optional[str] = None,
    speed: float = 0.9,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    denoise: bool = True,
    preprocess_prompt: bool = True,
    postprocess_output: bool = True,
    sentence_pause_ms: int = 350,
    paragraph_pause_ms: int = 750,
    progress_callback: Optional[Any] = None,
) -> Tuple[Optional[Tuple[int, np.ndarray]], Optional[str], str]:
    """
    Main long-form synthesis pipeline.
    
    Returns:
        ((sampling_rate, combined_waveform), filepath, status_report)
    """
    if not text or not text.strip():
        return None, None, "❌ Please enter text or paste a script to synthesize."

    chunks_with_meta = split_into_chunks(text, max_chars=180)
    if not chunks_with_meta:
        return None, None, "❌ Could not extract text chunks."

    total_chunks = len(chunks_with_meta)
    total_words = len(text.split())
    sampling_rate = getattr(model, "sampling_rate", 24000)

    start_time = time.time()

    # Prepare common generation configuration
    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None

    # Base kwargs for generation
    base_kw: Dict[str, Any] = dict(
        language=lang,
        generation_config=gen_config,
    )
    if speed is not None and float(speed) != 1.0:
        base_kw["speed"] = float(speed)
    if instruct and instruct.strip():
        base_kw["instruct"] = instruct.strip()

    # 1. Extract or load VoiceClonePrompt once right at the start
    if progress_callback:
        progress_callback(0, desc="Extracting/Loading Voice Clone Prompt...")

    voice_prompt = None
    if prompt_file and os.path.exists(prompt_file):
        try:
            voice_prompt = VoiceClonePrompt.load(prompt_file)
        except Exception as e:
            return None, None, f"❌ Error loading prompt file: {e}"
    elif ref_audio and os.path.exists(ref_audio):
        try:
            voice_prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text or None,
            )
        except Exception as e:
            return None, None, f"❌ Error extracting voice prompt from audio: {e}"
    else:
        return None, None, "❌ Please upload a reference audio or select a saved .pt prompt file for your audiobook narrator voice."

    base_kw["voice_clone_prompt"] = voice_prompt

    # 2. Iterate through chunks and synthesize each
    waveforms = []
    sentence_pause_samples = np.zeros(int(sampling_rate * (sentence_pause_ms / 1000.0)), dtype=np.int16)
    paragraph_pause_samples = np.zeros(int(sampling_rate * (paragraph_pause_ms / 1000.0)), dtype=np.int16)

    for idx, (chunk_text, is_para_end) in enumerate(chunks_with_meta):
        if progress_callback:
            progress_callback(
                (idx + 1) / total_chunks,
                desc=f"Synthesizing chunk {idx + 1}/{total_chunks}: '{chunk_text[:35]}...'"
            )

        kw = dict(base_kw)
        kw["text"] = chunk_text

        try:
            audio = model.generate(**kw)
            chunk_wave = (audio[0] * 32767).astype(np.int16)
            waveforms.append(chunk_wave)

            # Add appropriate pause after chunk (unless it's the very last chunk)
            if idx < total_chunks - 1:
                if is_para_end and paragraph_pause_ms > 0:
                    waveforms.append(paragraph_pause_samples)
                elif sentence_pause_ms > 0:
                    waveforms.append(sentence_pause_samples)

        except Exception as e:
            return None, None, f"❌ Error synthesizing chunk {idx + 1} ('{chunk_text}'): {e}"

    if not waveforms:
        return None, None, "❌ No audio generated."

    # 3. Combine waveforms
    combined_waveform = np.concatenate(waveforms)
    total_duration_sec = len(combined_waveform) / sampling_rate
    elapsed_sec = time.time() - start_time
    rtf = elapsed_sec / max(total_duration_sec, 0.1)  # Real-Time Factor

    # 4. Save to generated_audiobooks/
    save_dir = os.path.join(os.getcwd(), "generated_audiobooks")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"audiobook_{int(time.time())}.wav"
    out_path = os.path.join(save_dir, filename)

    try:
        sf.write(out_path, combined_waveform, sampling_rate)
    except Exception as e:
        pass

    mins = int(total_duration_sec // 60)
    secs = int(total_duration_sec % 60)
    duration_str = f"{mins}m {secs}s" if mins > 0 else f"{total_duration_sec:.1f}s"

    report = (
        f"✅ Long-Form Audiobook Successfully Generated!\n"
        f"• Total Words: {total_words} | Processed Chunks: {total_chunks}\n"
        f"• Audio Duration: {duration_str} ({sampling_rate} Hz)\n"
        f"• Processing Time: {elapsed_sec:.1f}s (Speed: {rtf:.2f}x RTF)\n"
        f"• Saved to: {filename}"
    )

    return (sampling_rate, combined_waveform), out_path, report
