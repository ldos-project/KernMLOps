#!/usr/bin/python3

import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

parser = argparse.ArgumentParser()
parser.add_argument('-t', action='store', dest="gtitle", help="Graph Title", required=True)
parser.add_argument('-x', action='store', dest="xlabel", help="XAxis Label", required=True)
parser.add_argument('-y', action='store', dest="ylabel", help="YAxis Label", required=True)
parser.add_argument('-g', action='store', dest="groups", help="Group ARows", default=1, type=int)
parser.add_argument('--horizon', action='store_true', dest="horizon")
parser.add_argument('--no-horizon', dest='horizon', action='store_false')
parser.set_defaults(horizon=False)
parser.add_argument('-o', action='store', dest="ouputf", help="Output FileN", required=True)
parser.add_argument('-i', action='store', dest="inputf", help="Input FileN", default="")

new_color_cycle = cycler(color=['cyan', 'green', 'orange', 'purple', 'magenta'])

plt.rcParams['axes.prop_cycle'] = new_color_cycle
SMALL_SIZE = 16
MEDIUM_SIZE = 20
LARGE_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

"""
This function takes in from the inputf an input that looks like with groups:
    g_name0 g_name1 g_name2
    g0y0 g1y0 g2y0 g0x0 g0e0 g1e0 g2e0
    g0y1 g1y1 g2y1 g0x1 g0e1 g1e1 g2e1
This function takes in from inputf an input that looks like:
    y0 x0 e0
    y1 x1 e0
    y2 x2 e0
"""
def graph(gtitle, xlabel, ylabel, inputf, ouputf, groups, horizon):
  _ = plt.figure(figsize=(10,7))
  #ax = fig.add_axes([0,0,1,1])
  xaxis = []
  yaxis = []
  eaxis = []
  for i in range (0,groups):
    yaxis.append([])
    eaxis.append([])

  group_label = []
  for line in inputf.readlines():
    broken_line = re.split(r'\s+', line.rstrip())
    print(broken_line)
    if groups > 1 and group_label == []:
      group_label = broken_line
      continue
    for i in range(0,groups) :
      yaxis[i].append(float(broken_line[i]))
      if len(broken_line) > groups + 1 +i:
       eaxis[i].append(float(broken_line[groups + 1 + i]))
      else :
        eaxis[i].append(0)
    xaxis.append(broken_line[groups])

  width_orig = 0.8
  width = width_orig/groups
  if group_label == [] :
    group_label.append("")
  ind = np.arange(len(yaxis[0]))
  print(yaxis)
  print(xaxis)
  print(eaxis)
  for i in range(0, groups):
    pos = ind + (-(groups -1)/2.0 +i)*width
    print(pos)
    if horizon:
      plt.barh(pos, yaxis[i], width, yerr=eaxis[i], label=group_label[i])
    else:
      plt.bar(pos, yaxis[i], width, yerr=eaxis[i], label=group_label[i])

  if horizon:
    plt.yticks(ind, xaxis)
    plt.title(gtitle)
    plt.xlabel(ylabel)
    plt.ylabel(xlabel)
  else:
    plt.xticks(ind, xaxis)
    plt.title(gtitle)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
  if groups > 1:
    plt.legend(bbox_to_anchor=(1,.5),loc='center left')
  plt.savefig(ouputf, bbox_inches='tight')


if __name__ == "__main__":
  args = parser.parse_args()
  inputf = sys.stdin
  if args.inputf != "" or args.inputf is None:
    inputf = open(args.inputf)
  graph(args.gtitle,args.xlabel,args.ylabel,inputf,args.ouputf,args.groups,args.horizon)
