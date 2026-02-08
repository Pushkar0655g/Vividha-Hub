import logging
import os
import shutil
import subprocess
import glob
import json
import argparse
import time
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.config import change_settings
from deep_translator import GoogleTranslator, MyMemoryTranslator
from pydub import AudioSegment
from TTS.api import TTS
import torch
import soundfile as sf
import numpy as np
from demucs.separate import main as demucs_separate
import whisper
import librosa
from pyannote.audio import Pipeline

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='conversion.log')

# Set ImageMagick binary path
imagemagick_path = os.path.join(os.path.dirname(__file__), "magick.exe")
if not os.path.exists(imagemagick_path):
    logging.error(f"ImageMagick not found at {imagemagick_path}")
    raise FileNotFoundError(f"ImageMagick not found at {imagemagick_path}")
os.environ["IMAGEMAGICK_BINARY"] = imagemagick_path
change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})

# Set FFmpeg path to bundled executable
ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
if not os.path.exists(ffmpeg_path):
    logging.error("Bundled FFmpeg not found.")
    raise RuntimeError("Bundled FFmpeg not found.")
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# Check FFmpeg availability
if shutil.which("ffmpeg") is None:
    logging.error("FFmpeg not found in PATH even with bundled version.")
    raise RuntimeError("FFmpeg not found in PATH even with bundled version.")

# Check GPU availability
if not torch.cuda.is_available():
    logging.warning("GPU not available. Falling back to CPU. This may significantly slow down processing.")
else:
    logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")

# LANGUAGE_MAP limited to supported languages
LANGUAGE_MAP = {
    "arabic": "ar",
    "chinese": "zh-cn",
    "czech": "cs",
    "dutch": "nl",
    "english": "en",
    "french": "fr",
    "german": "de",
    "hindi": "hi",
    "hungarian": "hu",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "turkish": "tr"
}

# Font mapping for subtitles
FONT_MAP = {
    "hi": "Mangal",
    "en": "Arial", "es": "Arial", "fr": "Arial", "de": "Arial", "it": "Arial",
    "pt": "Arial", "pl": "Arial", "tr": "Arial", "ru": "Arial", "nl": "Arial",
    "cs": "Arial", "ar": "Traditional Arabic", "zh-cn": "SimSun", "ja": "MS Gothic",
    "hu": "Arial", "ko": "Malgun Gothic"
}

# Languages supported by Coqui TTS
AUDIO_SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]

# Supported video extensions
SUPPORTED_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".wmv"]

def extract_audio(video_path, audio_path="temp_audio.wav"):
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video = VideoFileClip(video_path)
        if not video.audio:
            raise ValueError("No audio found in video.")
        video.audio.write_audiofile(audio_path, codec="pcm_s16le")
        duration = video.duration
        logging.info(f"Audio extracted to {audio_path}, duration: {duration}s")
        return audio_path, duration
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")
        return None, None
    finally:
        if 'video' in locals():
            video.close()

def extract_background_music(audio_path, output_path="background_music.wav"):
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        demucs_separate(["--two-stems", "vocals", "-o", "demucs_output", "--device", "cuda", audio_path])
        instrumental_path = os.path.join("demucs_output", "htdemucs", os.path.basename(audio_path).replace(".wav", ""), "no_vocals.wav")
        if os.path.exists(instrumental_path):
            shutil.move(instrumental_path, output_path)
            logging.info(f"Background music extracted to {output_path}")
            return output_path
        else:
            logging.error("Failed to extract background music")
            return None
    except Exception as e:
        logging.error(f"Error extracting background music: {e}")
        return None

def detect_speakers(audio_path):
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="hf_XXXXXXX")
        pipeline.to(device)
        diarization = pipeline(audio_path)
        speaker_segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.setdefault(speaker, []).append((turn.start, turn.end))
        logging.info("Speakers detected successfully")
        return speaker_segments
    except Exception as e:
        logging.error(f"Error in speaker detection: {e}")
        return None

def extract_speaker_audio(audio_path, speaker_segments):
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio, sample_rate = sf.read(audio_path)
        speaker_clips = {}
        for speaker, segments in speaker_segments.items():
            clip = AudioSegment.silent()
            for start, end in segments:
                start_ms = int(start * 1000)
                end_ms = int(end * 1000)
                segment = AudioSegment.from_wav(audio_path)[start_ms:end_ms]
                clip += segment
            if len(clip) < 100:
                logging.warning(f"Speaker {speaker} clip is too short: {len(clip)}ms. Skipping.")
                continue
            speaker_clip_path = f"speaker_{speaker}.wav"
            clip.export(speaker_clip_path, format="wav")
            if os.path.getsize(speaker_clip_path) < 1000:
                logging.warning(f"Speaker {speaker} clip file too small: {speaker_clip_path}. Skipping.")
                os.remove(speaker_clip_path)
                continue
            speaker_clips[speaker] = speaker_clip_path
        logging.info(f"Speaker audio clips extracted: {list(speaker_clips.keys())}")
        return speaker_clips
    except Exception as e:
        logging.error(f"Error extracting speaker audio: {e}")
        return None

def transcribe_audio(audio_path, input_lang):
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        model = whisper.load_model("medium", device="cuda")
        result = model.transcribe(audio_path, language=input_lang)
        logging.info(f"Transcribed audio in {input_lang}")
        return result["segments"]
    except Exception as e:
        logging.error(f"Error in transcription: {e}")
        return None

def assign_speakers_to_segments(segments, speaker_segments):
    try:
        if not speaker_segments:
            logging.warning("No speaker segments provided. Assigning default speaker.")
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"
            return segments
        for seg in segments:
            if not isinstance(seg, dict) or "start" not in seg or "end" not in seg:
                raise ValueError("Transcription segments must be a list of dictionaries with 'start' and 'end' keys.")
            seg_start, seg_end = seg["start"], seg["end"]
            assigned_speaker = None
            for speaker, times in speaker_segments.items():
                for start, end in times:
                    if (start <= seg_start <= end) or (start <= seg_end <= end) or (seg_start <= start <= seg_end):
                        assigned_speaker = speaker
                        break
                if assigned_speaker:
                    break
            seg["speaker"] = assigned_speaker if assigned_speaker else list(speaker_segments.keys())[0]
        logging.info("Speakers assigned to segments")
        return segments
    except Exception as e:
        logging.error(f"Error assigning speakers: {e}")
        return None

def translate_segments(segments, source_lang, target_lang):
    try:
        if source_lang == target_lang:
            logging.info(f"Source ({source_lang}) matches target ({target_lang}). Skipping translation.")
            return segments
        if target_lang not in LANGUAGE_MAP.values():
            raise ValueError(f"Unsupported language: {target_lang}. Supported languages: {list(LANGUAGE_MAP.values())}")
        logging.info(f"Translating from {source_lang} to {target_lang}")
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        for seg in segments:
            try:
                translated = translator.translate(seg["text"])
                seg["text"] = translated if translated else seg["text"]
                logging.debug(f"Translated: {seg['text']}")
            except Exception as e:
                logging.warning(f"GoogleTranslator failed for '{seg['text']}': {e}")
                fallback = MyMemoryTranslator(source=source_lang, target=target_lang)
                translated = fallback.translate(seg["text"])
                seg["text"] = translated if translated else seg["text"]
                logging.debug(f"Fallback translated: {seg['text']}")
        logging.info(f"Segments translated to {target_lang}")
        return segments
    except Exception as e:
        logging.error(f"Error translating segments: {e}")
        raise

def generate_tts_coqui(text, speaker_wav, output_path, language):
    try:
        if not os.path.exists(speaker_wav):
            logging.error(f"Speaker WAV file not found: {speaker_wav}")
            return None
        if not text.strip():
            logging.warning("Empty text provided for TTS. Skipping.")
            return None
        if language not in AUDIO_SUPPORTED_LANGUAGES:
            logging.error(f"Language {language} not supported for TTS. Supported: {AUDIO_SUPPORTED_LANGUAGES}")
            return None

        from TTS.tts.configs.xtts_config import XttsConfig
        try:
            from TTS.tts.configs.xtts_config import XttsAudioConfig
        except ImportError:
            from TTS.tts.models.xtts import XttsAudioConfig
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.models.xtts import XttsArgs
        torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        tts.tts_to_file(text=text, file_path=output_path, speaker_wav=speaker_wav, language=language, speed=0.9)
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            logging.error(f"TTS output file invalid or empty: {output_path}")
            return None
        logging.info(f"TTS generated at {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error generating TTS for text '{text}': {e}")
        return None

def create_dubbed_audio(translated_segments, speaker_clips, background_music_path, target_language, video_duration):
    try:
        if not speaker_clips:
            logging.error("No speaker clips available for TTS generation.")
            return None
        final_audio = AudioSegment.silent(duration=int(video_duration * 1000))
        for seg in translated_segments:
            speaker = seg["speaker"]
            if speaker not in speaker_clips:
                logging.warning(f"Speaker {speaker} not found in clips. Skipping segment: {seg['text']}")
                continue
            tts_path = f"temp_tts_{seg['start']}.wav"
            tts_result = generate_tts_coqui(seg["text"], speaker_clips[speaker], tts_path, target_language)
            if tts_result:
                tts_audio = AudioSegment.from_wav(tts_path)
                D_tts = len(tts_audio) / 1000.0
                D_seg = seg["end"] - seg["start"]
                if abs(D_tts - D_seg) > 0.1:
                    try:
                        y, sr = librosa.load(tts_path, sr=None)
                        rate = D_tts / D_seg
                        if 0.5 <= rate <= 2.0:
                            y_stretched = librosa.effects.time_stretch(y.astype(np.float32), rate=rate)
                            sf.write(tts_path, y_stretched, sr)
                            tts_audio = AudioSegment.from_wav(tts_path)
                        else:
                            logging.warning(f"Rate {rate} out of bounds for segment at {seg['start']}s.")
                    except Exception as e:
                        logging.error(f"Time stretching failed for segment at {seg['start']}s: {e}. Using original TTS audio.")
                start_ms = int(seg["start"] * 1000)
                final_audio = final_audio.overlay(tts_audio, position=start_ms)
                os.remove(tts_path)
                logging.info(f"Overlayed TTS for segment at {start_ms}ms")
            else:
                logging.warning(f"Failed to generate TTS for segment: {seg['text']}. Adding silence.")
                duration_ms = int((seg["end"] - seg["start"]) * 1000)
                final_audio = final_audio.overlay(AudioSegment.silent(duration_ms), position=int(seg["start"] * 1000))
        if background_music_path and os.path.exists(background_music_path):
            background_music = AudioSegment.from_wav(background_music_path)
            if len(background_music) > len(final_audio):
                background_music = background_music[:len(final_audio)]
            else:
                background_music = background_music.fade_out(1000) + AudioSegment.silent(duration=len(final_audio) - len(background_music))
            final_audio = final_audio.overlay(background_music - 10)
            logging.info("Background music overlayed")
        final_audio_path = "dubbed_audio.wav"
        final_audio.export(final_audio_path, format="wav")
        if os.path.getsize(final_audio_path) < 1000:
            logging.error(f"Dubbed audio file too small: {final_audio_path}")
            return None
        final_audio_check = AudioSegment.from_wav(final_audio_path)
        if final_audio_check.dBFS == -float('inf'):
            logging.error("Dubbed audio is silent")
            raise RuntimeError("Dubbed audio is silent")
        logging.info(f"Dubbed audio created at {final_audio_path}, duration: {len(final_audio)/1000}s")
        return final_audio_path
    except Exception as e:
        logging.error(f"Error creating dubbed audio: {e}")
        return None

def generate_srt(segments, output_path):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = int(seg["start"] * 1000)
                end = int(seg["end"] * 1000)
                f.write(f"{i}\n{start // 3600000:02d}:{(start % 3600000) // 60000:02d}:{(start % 60000) // 1000:02d},{start % 1000:03d} --> "
                        f"{end // 3600000:02d}:{(end % 3600000) // 60000:02d}:{(end % 60000) // 1000:02d},{end % 1000:03d}\n{seg['text']}\n\n")
        logging.info(f"SRT file generated at {output_path}")
    except Exception as e:
        logging.error(f"Error generating SRT: {e}")
        raise

def combine_audio_with_video(video_path, audio_path, translated_segments, subtitle_lang_code, output_path, use_original_audio=False):
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        srt_path = "subtitles.srt"
        generate_srt(translated_segments, srt_path)
        font = FONT_MAP.get(subtitle_lang_code, "Arial")
        if use_original_audio:
            cmd = [
                "ffmpeg", "-hwaccel", "cuda", "-i", video_path,
                "-vf", f"subtitles={srt_path}:force_style='FontName={font},FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,Outline=1'",
                "-c:v", "h264_nvenc", "-c:a", "copy", "-y", output_path
            ]
        else:
            if not audio_path or not os.path.exists(audio_path):
                raise ValueError(f"Dubbed audio path invalid or file does not exist: {audio_path}")
            cmd = [
                "ffmpeg", "-hwaccel", "cuda", "-i", video_path, "-i", audio_path,
                "-vf", f"subtitles={srt_path}:force_style='FontName={font},FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,Outline=1'",
                "-c:v", "h264_nvenc", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-y", output_path
            ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"FFmpeg output: {result.stdout}")
        logging.info(f"Final video created at {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"Error combining audio with video: {e}")
        raise
    finally:
        if os.path.exists(srt_path):
            os.remove(srt_path)

def cleanup_temp_files():
    temp_files = ["temp_audio.wav", "background_music.wav", "dubbed_audio.wav", "subtitles.srt"]
    for file in temp_files + glob.glob("speaker_*") + glob.glob("temp_tts_*"):
        if os.path.exists(file):
            for attempt in range(5):
                try:
                    os.remove(file)
                    logging.info(f"Cleaned up {file}")
                    break
                except PermissionError as e:
                    logging.warning(f"Attempt {attempt + 1} failed to clean up {file}: {e}. Retrying in 2 seconds...")
                    time.sleep(2)
                except Exception as e:
                    logging.error(f"Error cleaning up {file}: {e}")
                    break
    if os.path.exists("demucs_output"):
        try:
            shutil.rmtree("demucs_output")
            logging.info("Cleaned up demucs_output")
        except Exception as e:
            logging.error(f"Error cleaning up demucs_output: {e}")
    logging.info("Temporary files cleanup completed")

def process_video(video_path, input_lang_code, audio_lang_code, subtitle_lang_code, output_path, progress_callback):
    try:
        if audio_lang_code not in AUDIO_SUPPORTED_LANGUAGES:
            logging.warning(f"Audio language '{audio_lang_code}' is not supported for TTS. Using original audio instead.")
            use_original_audio = True
        else:
            use_original_audio = (input_lang_code == audio_lang_code)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not any(video_path.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            raise ValueError(f"Unsupported video format: {video_path}")

        progress_callback(0, "Starting processing...")
        
        audio_path, video_duration = extract_audio(video_path)
        if not audio_path:
            raise RuntimeError("Failed to extract audio")
        progress_callback(10, "Audio extracted")

        model = whisper.load_model("medium", device="cuda")
        transcription_result = model.transcribe(audio_path, language=input_lang_code)
        segments = transcription_result["segments"]
        if not segments:
            raise RuntimeError("Failed to transcribe audio")
        progress_callback(20, "Audio transcribed")

        if use_original_audio:
            logging.info("Using original audio; skipping dubbing.")
            dubbed_audio_path = None
            translated_audio_segments = segments
            progress_callback(70, "Using original audio")
        else:
            demucs_separate(["--two-stems", "vocals", "-o", "demucs_output", "--device", "cuda", audio_path])
            background_music_path = os.path.join("demucs_output", "htdemucs", os.path.basename(audio_path).replace(".wav", ""), "no_vocals.wav")
            if not os.path.exists(background_music_path):
                raise RuntimeError("Failed to extract background music")
            progress_callback(30, "Background music extracted")
            
            speaker_segments = detect_speakers(audio_path)
            if not speaker_segments:
                raise RuntimeError("Failed to detect speakers")
            progress_callback(40, "Speakers detected")
            
            speaker_clips = extract_speaker_audio(audio_path, speaker_segments)
            if not speaker_clips:
                raise RuntimeError("Failed to extract speaker audio")
            progress_callback(50, "Speaker audio extracted")
            
            assigned_segments = assign_speakers_to_segments(segments, speaker_segments)
            if not assigned_segments:
                raise RuntimeError("Failed to assign speakers")
            translated_audio_segments = translate_segments(assigned_segments, input_lang_code, audio_lang_code)
            if not translated_audio_segments:
                raise RuntimeError("Failed to translate audio segments")
            progress_callback(60, "Audio segments translated")
            
            dubbed_audio_path = create_dubbed_audio(translated_audio_segments, speaker_clips, background_music_path, audio_lang_code, video_duration)
            if not dubbed_audio_path:
                raise RuntimeError("Failed to create dubbed audio")
            progress_callback(80, "Dubbed audio created")

        translated_subtitle_segments = [dict(seg) for seg in segments]
        translated_subtitle_segments = translate_segments(translated_subtitle_segments, input_lang_code, subtitle_lang_code)
        if not translated_subtitle_segments:
            raise RuntimeError("Failed to translate subtitle segments")
        progress_callback(90, "Subtitles translated")

        combine_audio_with_video(video_path, dubbed_audio_path, translated_subtitle_segments, subtitle_lang_code, output_path, use_original_audio)
        progress_callback(100, f"Final video created at {output_path}")
        
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        progress_callback(-1, f"Error: {str(e)}")
        raise
    finally:
        cleanup_temp_files()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Dubbing Backend")
    parser.add_argument("--video", type=str, help="Path to the video file")
    parser.add_argument("--input_lang", type=str, default="english", help="Input language")
    parser.add_argument("--audio_lang", type=str, default="hindi", help="Audio language")
    parser.add_argument("--subtitle_lang", type=str, default="spanish", help="Subtitle language")
    parser.add_argument("--output", type=str, default="output_dubbed.mp4", help="Output video path")
    args = parser.parse_args()

    if args.video:
        video_path = args.video
        input_lang = args.input_lang
        audio_lang = args.audio_lang
        subtitle_lang = args.subtitle_lang
        output_path = args.output
    else:
        with open("input.json", "r") as f:
            data = json.load(f)
        video_path = data.get("file")
        input_lang = data.get("input_lang", "english")
        audio_lang = data.get("audio_lang", "hindi")
        subtitle_lang = data.get("subtitle_lang", "spanish")
        output_path = data.get("output", "output_dubbed.mp4")

    input_lang_code = LANGUAGE_MAP.get(input_lang.lower(), "en")
    audio_lang_code = LANGUAGE_MAP.get(audio_lang.lower(), "hi")
    subtitle_lang_code = LANGUAGE_MAP.get(subtitle_lang.lower(), "es")

    def progress_callback(percent, message):
        with open("progress.json", "w") as f:
            json.dump({"percent": percent, "message": message}, f)
        logging.info(f"Progress: {percent}% - {message}")

    process_video(video_path, input_lang_code, audio_lang_code, subtitle_lang_code, output_path, progress_callback)
    with open("result.json", "w") as f:
        json.dump({"status": "complete", "output": output_path}, f)