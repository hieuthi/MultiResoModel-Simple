import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchaudio

import torch.nn.functional as F

from torch.utils.data import Dataset


def get_basename(filepath):
    path, fname = os.path.split(filepath)
    bname, ext  = os.path.splitext(fname)
    return bname

# Using np.array due to python object cloen multithread problem
#   https://github.com/pytorch/pytorch/issues/13246
def scp_to_array(scp):
    ret = []
    with open(scp) as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                ret.append(line.strip())
    return np.array(ret, dtype=np.bytes_)

def seek_waveform_to(waveform, start, pad_value=0):
    if start >= 0:
        if start >= waveform.size(-1):
            return waveform
        else:
            return waveform[...,start:]
    else:
        return F.pad(waveform, (-start, 0), value=pad_value)

def get_truncated_segment(segment, size, pad_value=0):
  if segment.size(-1) < size:
    segment = F.pad(segment, (0, size - segment.size(-1)), value=pad_value)
  else:
    segment = segment[..., :size]
  return segment

def readlab(labtext):
    items     = labtext.split()
    labid, dur, labs = items[0], float(items[1]), []
    for item in items[3:]:
        args = item.split('-')
        if args[2] == "spoof":
            start, end = float(args[0]), float(args[1])
            labs.append([start,end])
    return labid, dur, labs

def intersect(a, b):
    area = min(a[1], b[1]) - max(a[0], b[0])
    return max(0, area)

def vad2seg(labs, dur, unit=0.02, sensitivity=0.0):
    segs = []
    n    = int(dur / unit)
    start, end = 0.0, 0.0
    lidx = 0
    for i in range(n):
        area = 0.0
        start, end = end, end + unit

        for lidx in range(lidx, len(labs)):
            if lidx >= len(labs):
                break
            if end < labs[lidx][0]:
                break
            else:
                area = area + intersect([start,end], labs[lidx])
                if end < labs[lidx][1]:
                    break

        label = 1 if area/unit > sensitivity else 0
        segs.append(label)

    return segs

def load_segmented_label(labs, ldur, units=[0.02], dur=0.0, seek=0.0, sensitivity=0.0):
    ldur = ldur-seek

    mlabs = []
    for lab in labs:
        if lab[1] < seek:
            continue
        mlabs.append([max(0,lab[0]-seek), lab[1]-seek])
    labs = mlabs
    dur = ldur if dur <=0 else dur

    segs_bag = []
    for unit in units:
        segs = vad2seg(labs, dur, unit=unit, sensitivity=sensitivity)
        segs_bag.append( torch.LongTensor(segs) )
    uttlab = max(segs)

    segs_bag.append(uttlab)

    return segs_bag


class PartialSpoofDataset(Dataset):
  def __init__(
    self, 
    uttscp, 
    labscp,
    units=[0.02,0.04,0.08,0.16,0.32,0.64],
    sample_rate=16000,
    segment_duration=6.0, 
    trailing_duration=0,
    random_seek=False
  ):
    self.uttpaths = scp_to_array(uttscp)
    self.labtexts = scp_to_array(labscp)
    assert len(self.uttpaths) == len(self.labtexts), f"ERROR: number of utterances (len(self.uttpaths)) and number of label texts (len(self.labtexts)) do not equals "

    self.units = units
    self.sample_rate  = sample_rate
    self.segment_duration = segment_duration
    self.segment_size = int(sample_rate*segment_duration)
    self.trailing_size = int(trailing_duration*sample_rate)
    self.random_seek  = random_seek

  def __len__(self):
    return len(self.uttpaths)

  def __getitem__(self, index):
    uttpath = self.uttpaths[index].decode('utf-8')
    waveform, sample_rate = torchaudio.load(uttpath)
    assert sample_rate == self.sample_rate, f"ERROR: {uttpath} should be at {self.sample_rate} Hz"

    # Draw a random segment from waveform and alignment
    start = 0
    if self.random_seek:
        most_left  = -waveform.size(1) // 4
        most_right = waveform.size(1) * 3 // 4
        start      = random.randrange(most_left, most_right)
        # print(f"Start {start} {waveform.size(1)} {self.segment_size}")
        waveform   = seek_waveform_to(waveform, start)
    waveform = get_truncated_segment(waveform, self.segment_size+self.trailing_size)

    labtext    = self.labtexts[index].decode('utf-8')
    labid, ldur, labs = readlab(labtext)
    assert labid == get_basename(uttpath), f"ERROR: Waveform {uttpath} and Label {labid} does not match"

    segs_bag = load_segmented_label(labs, ldur, units=self.units, dur=self.segment_duration, 
                                    seek=start/self.sample_rate, sensitivity=0)

    return ( waveform, *segs_bag )
