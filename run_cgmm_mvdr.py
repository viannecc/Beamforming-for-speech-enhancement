#!/usr/bin/env python3
# encoding: utf-8
import argparse
import logging
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from base_utils.utils import get_chunk_data

from beamformer import complexGMM_mvdr as cgmm

SAMPLING_FREQUENCY = 16000
FFT_LENGTH = 512
FFT_SHIFT = 128
NUMBER_EM_ITERATION = 5
MIN_SEGMENT_DUR = 2

WPE_DIR = Path(f"{os.environ['HOME']}") / "gpu_platform/wpe"
WIN_LEN = 20
WAV_MIN_DUR = {"S02": 8904.52, "S09": 7161.68, "S01": 9544.14, "S21": 9200.34}


def cgmm_mvdr(wav_multi, save_path):
  cgmm_beamformer = cgmm.complexGMM_mvdr(SAMPLING_FREQUENCY, FFT_LENGTH,
                                         FFT_SHIFT, NUMBER_EM_ITERATION,
                                         MIN_SEGMENT_DUR)

  complex_spectrum, R_x, R_n, noise_mask, speech_mask = cgmm_beamformer.get_spatial_correlation_matrix(
    wav_multi)

  beamformer, steering_vector = cgmm_beamformer.get_mvdr_beamformer(R_x, R_n)

  enhanced_speech = cgmm_beamformer.apply_beamformer(beamformer,
                                                     complex_spectrum)

  sf.write(str(save_path),
           enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65,
           SAMPLING_FREQUENCY)


def get_wavs_tps(sess_ids, task, array=None):
  sess_wavs_tps = dict()
  for sess_id in sess_ids:
    wavs = list((WPE_DIR / task).glob(f"{sess_id}_*.wav"))
    if array:
      wavs = [w for w in wavs if w.stem.split('.')[0].split('_')[1] == array]
    else:
      wavs = [w for w in wavs if w.stem.split('.')[1] in ["CH1", "CH4"]]

    tps_list = list()
    bt = 0
    while bt + WIN_LEN < WAV_MIN_DUR[sess_id]:
      tps_list.append((bt, bt + WIN_LEN))
      bt += WIN_LEN
    if bt < WAV_MIN_DUR[sess_id]:
      tps_list.append((bt, WAV_MIN_DUR[sess_id]))

    sess_wavs_tps[sess_id] = (wavs, tps_list)
  return sess_wavs_tps


def read_multi_channel(wavs, tp):
  bt, et = tp
  b_idx = int(bt * 16000)
  e_idx = int(et * 16000)
  wav, _ = sf.read(str(wavs[0]), dtype='float32')
  wav_data = wav[b_idx: e_idx]
  wav_multi = np.zeros((len(wav_data), len(wavs)), dtype=np.float32)
  wav_multi[:, 0] = wav_data

  for i in range(1, len(wavs)):
    wav_multi[:, i] = sf.read(str(wavs[i]), dtype='float32')[0][b_idx:e_idx]
  return wav_multi


def enhance(wavs, tp, save_dir):
  wav_multi = read_multi_channel(wavs, tp)
  sess_id = wavs[0].stem.split('_')[0]
  save_path = save_dir / f"{sess_id}-{tp[0]:07d}-{tp[1]:07d}.wav"
  cgmm_mvdr(wav_multi, save_path)


def __cmd():
  """采用cgmm-mvdr进行增强.
  """
  parser = argparse.ArgumentParser(description="采用cgmm-mvdr进行增强.")
  parser.add_argument("task", type=str, help="数据集.")
  parser.add_argument("save_dir", type=Path, help="保存路径.")
  parser.add_argument("--nj", type=int, default=16, help="并行数量.")
  parser.add_argument("--array", help="指定麦克风, 如U06, 否则用outer arrays.")
  args = parser.parse_args()

  if args.task == "dev":
    sess_ids = ["S02", "S09"]
  else:
    sess_ids = ["S01", "S21"]

  logging.info("获取各会话音频和分段的时间.")
  sess_wavs_tps = get_wavs_tps(sess_ids, args.task, args.array)

  args.save_dir.mkdir(parents=True, exist_ok=True)
  temp_dir = args.save_dir / "temp"
  temp_dir.mkdir(exist_ok=True)

  logging.info("开始增强数据.")
  for sess_id, (wavs, tps) in sess_wavs_tps.items():
    logging.info(f"处理{sess_id}")
    pool_args = list()
    for bt, et in tps:
      pool_args.append((wavs, (bt, et), temp_dir))

    chunks = get_chunk_data(pool_args)
    logging.info(f"一共{len(chunks)}个chunks")
    for i, chunk in enumerate(chunks, 1):
      logging.info(f"处理第{i}个chunk.")
      if args.nj == 1:
        for items in chunk:
          enhance(*items)
      else:
        with Pool(args.nj) as pool:
          pool.starmap(enhance, chunk)
  logging.info("完成数据增强.")


if __name__ == "__main__":
  logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s",
                      level=logging.INFO)
  __cmd()
