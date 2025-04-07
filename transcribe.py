import whisper
import time
import sys
from tqdm import tqdm
from pydub.utils import mediainfo

# === 1. Вибір моделі ===
print("Завантаження моделі Whisper (small)...")
model = whisper.load_model("small")

# === 2. Вибір файлу ===
audio_file = "Poznanska 28.mp3"  # заміни на назву свого аудіофайлу

# === 3. Отримуємо тривалість аудіофайлу ===
audio_info = mediainfo(audio_file)
duration = float(audio_info['duration'])

# === 4. Таймер і транскрипція з прогресом ===
print(f"\nПочинається транскрипція файлу: {audio_file}")
start_time = time.time()

# Використовуємо tqdm для створення прогрес-бара
progress_bar = tqdm(total=100, desc="Транскрибування", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} - {elapsed} s")

# Транскрибування аудіо
try:
    result = model.transcribe(audio_file)
    progress_bar.n = 100  # Коли завершиться транскрипція, прогрес буде 100%
    progress_bar.last_print_n = 100
    progress_bar.update(0)
except Exception as e:
    print(f"Помилка при транскрипції: {e}")
    sys.exit(1)

end_time = time.time()
elapsed_time = end_time - start_time
progress_bar.close()
print(f"\nТранскрипція завершена за {elapsed_time:.2f} секунд.\n")

# === 5. Збереження в .txt ===
output_file = audio_file.rsplit('.', 1)[0] + "_transcript.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"Транскрипцію збережено у файл: {output_file}")