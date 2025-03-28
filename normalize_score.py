import os.path
import argparse
import numpy as np
import time
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Normalize score')
    parser.add_argument('input', type=str, help='path to a sorted input score')
    parser.add_argument('output', type=str, help='path to the normalized output score')
    parser.add_argument('--utt', action='store_true', help='if utterance-based')
    args = parser.parse_args()

    if args.utt:
        utts = {}
        with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
            for line in fin:
                vargs = line.strip().split()
                if 'PART' in vargs[0][-8:]:
                    name = vargs[0][:-8]
                    if name not in utts:
                        utts[name] = []
                    utts[name].append([float(vargs[1]), float(vargs[2])])
                else:
                    fout.write(line)
            for name in utts:
                data = np.array(utts[name])
                predict0 = np.min(data[:,0])
                predict1 = np.max(data[:,1])
                fout.write(f"{name} {predict0:.05f} {predict1:.05f}\n")
        sys.exit()

    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        for line in fin:
            vargs = line.strip().split()
            if 'PART001' in vargs[0][-8:]:
                if int(vargs[1]) == 0:
                    count = 0
                else:
                    count = count + 1
                name = vargs[0][:-8]
                fout.write(f"{name} {count} {vargs[2]} {vargs[3]}\n")
            elif 'PART' in vargs[0][-8:]:
                name = vargs[0][:-8]
                count = count + 1
                fout.write(f"{name} {count} {vargs[2]} {vargs[3]}\n")
            else:
                fout.write(line)