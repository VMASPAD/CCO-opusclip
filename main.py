# main.py

import os
import re
import subprocess
import random
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import pysrt
import argparse
import whisper
import torch

# Verificar si CUDA está disponible
print(torch.cuda.is_available())

# Funciones para manipulación de video y audio
def get_video_duration(video_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        print(f"Error al obtener la duración: {result.stderr}")
        return None
    return float(result.stdout.strip())

def generate_random_clips(total_duration, clip_duration, num_clips):
    start_times = []
    while len(start_times) < num_clips:
        start_time = random.randint(0, int(total_duration - clip_duration))
        if not any(abs(time - start_time) < clip_duration for time in start_times):
            start_times.append(start_time)
    return start_times

def format_time(time_in_seconds):
    hours = time_in_seconds // 3600
    minutes = (time_in_seconds % 3600) // 60
    seconds = time_in_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def cut_video_clip(start_time, clip_duration, clip_name, video_path):
    clip_folder = f'{clip_name}'
    os.makedirs(clip_folder, exist_ok=True)
    video_output_path = f"{clip_folder}/{clip_name}.mp4"
    audio_output_path = f"{clip_folder}/{clip_name}.wav"
    start_time_str = format_time(start_time)
    
    video_command = [
        'ffmpeg', '-ss', start_time_str, '-i', video_path, '-t', str(clip_duration),
        '-c:v', 'libx264', '-vf', 'crop=ih*9/16:ih', '-aspect', '9:16', video_output_path
    ]
    
    audio_command = [
        'ffmpeg', '-ss', start_time_str, '-i', video_path, '-t', str(clip_duration),
        '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_output_path
    ]
    
    subprocess.run(video_command)
    subprocess.run(audio_command)
    
    generate_subtitles(audio_output_path, clip_folder)

def generate_subtitles(audio_path, clip_folder):
    model = whisper.load_model("small", device="cuda" if torch.cuda.is_available() else "cpu")
    result = model.transcribe(audio_path)
    subtitle_srt_path = f'{clip_folder}/subtitulos.srt'
    
    with open(subtitle_srt_path, 'w', encoding='utf-8') as srt_file:
        for i, segment in enumerate(result['segments']):
            start = segment['start']
            end = segment['end']
            text = segment['text']
            srt_file.write(f"{i + 1}\n")
            srt_file.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            srt_file.write(f"{text}\n\n")

def format_timestamp(seconds):
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def extract_audio(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)

def split_text(text, n):
    words = text.split()
    return [' '.join(words[i:i + n]) for i in range(0, len(words), n)]

def srt_to_clips(subs, video, y_pos):
    text_clips = []
    for sub in subs:
        start_time = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000
        end_time = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000
        duration = end_time - start_time

        segments = split_text(sub.text, 5)
        segment_duration = duration / len(segments) if segments else duration

        for i, segment in enumerate(segments):
            text_clip = TextClip(segment, fontsize=34, color='white', bg_color='Purple', align='center',
                                 font=r'C:\Users\tomas\AppData\Local\Microsoft\Windows\Fonts\Montserrat-Black.ttf')
            text_clip = text_clip.set_duration(segment_duration).set_start(start_time + i * segment_duration).set_pos(("center", y_pos))
            fade_duration = 0.5
            text_clip = text_clip.crossfadein(fade_duration).crossfadeout(fade_duration)
            text_clips.append(text_clip)
    return text_clips

def add_subtitles_to_video(video_path, srt_path, y_pos):
    video = VideoFileClip(video_path)
    subs = pysrt.open(srt_path, encoding='utf-8')
    text_clips = srt_to_clips(subs, video, y_pos)
    video_final = CompositeVideoClip([video] + text_clips)
    video_final.write_videofile(video_path.replace(".mp4", "_subtitled.mp4"), codec="libx264")

def main(video_path, y_pos, clip_duration, num_clips):
    total_duration = get_video_duration(video_path)
    if total_duration is None:
        return

    start_times = generate_random_clips(total_duration, clip_duration, num_clips)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(cut_video_clip, start_time, clip_duration, f'clip_{i}', video_path) for i, start_time in enumerate(start_times)]
        for future in futures:
            future.result()

    # Añadir subtítulos a cada clip
    clip_folders = [f'clip_{i}' for i in range(num_clips)]
    for clip_folder in clip_folders:
        clip_video_path = f"{clip_folder}/clip_{clip_folders.index(clip_folder)}.mp4"
        clip_srt_path = f"{clip_folder}/subtitulos.srt"
        add_subtitles_to_video(clip_video_path, clip_srt_path, y_pos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar clips de video con subtítulos y posición personalizable en el eje y.")
    parser.add_argument("--video_path", help="Ruta del archivo de video")
    parser.add_argument("--y_pos", type=float, default=0.9, help="Posición del texto en el eje y (valor relativo entre 0 y 1)")
    parser.add_argument("--clip_duration", type=int, default=30, help="Duración de cada clip en segundos")
    parser.add_argument("--num_clips", type=int, default=20, help="Número de clips a generar")

    args = parser.parse_args()
    main(args.video_path, args.y_pos, args.clip_duration, args.num_clips)
