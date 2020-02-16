#!/usr/bin/env python3
# encoding: utf-8
import argparse
import logging
import wave
from collections import defaultdict
from pathlib import Path


def save_wav(wav_path, wav_data):
  scd_wav = wave.open(str(wav_path), 'wb')
  scd_wav.setnchannels(1)
  scd_wav.setsampwidth(2)
  scd_wav.setframerate(16000)
  scd_wav.writeframes(wav_data)
  scd_wav.close()


def __cmd():
  """增强数据连接复原.
  """
  parser = argparse.ArgumentParser(description="增强数据连接复原.")
  parser.add_argument("src_dir", type=Path, help="分片增强的数据路径.")
  parser.add_argument("dst_dir", type=Path, help="保存路径.")
  args = parser.parse_args()

  args.dst_dir.mkdir(parents=True, exist_ok=True)

  logging.info("开始连接数据.")
  sess_to_wavs = defaultdict(list)
  for w in args.src_dir.glob("*.wav"):
    sess_id = w.stem.split('-')[0]
    sess_to_wavs[sess_id].append(w)

  for sess_id, wavs in sess_to_wavs.items():
    sorted_wavs = sorted(wavs, key=lambda x: int(x.split("-")[1]))
    wav_data = b""
    for w in sorted_wavs:
      with wave.open(str(w), 'rb')as win:
        params = win.getparams()
        wav_data += win.readframes(params[3])

    save_path = args.dst_dir / f"{sess_id}.wav"
    save_wav(save_path, wav_data)
  logging.info("完成数据增强.")


if __name__ == "__main__":
  logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s",
                      level=logging.INFO)
  __cmd()
