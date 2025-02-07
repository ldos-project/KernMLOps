#!/usr/bin/python3

import argparse
import re
import sys
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


class GraphType(Enum):
    bar = "bar"
    box = "box"

    def __self__(self):
        return self.value

    @staticmethod
    def from_string(s):
        try:
            return GraphType[s]
        except KeyError:
            raise ValueError()

parser = argparse.ArgumentParser()
parser.add_argument('-t', action='store', dest="gtitle", help="Graph Title", required=True)
parser.add_argument('-x', action='store', dest="xlabel", help="XAxis Label", required=True)
parser.add_argument('-y', action='store', dest="ylabel", help="YAxis Label", required=True)
parser.add_argument('-g', action='store', dest="groups", help="Group ARows", default=1, type=int)
parser.add_argument('--horizon', action='store_true', dest="horizon")
parser.add_argument('-v', action='store_true', dest="verbose")
parser.add_argument('--no-horizon', dest='horizon', action='store_false')
parser.set_defaults(horizon=False)
parser.add_argument('-o', action='store', dest="ouputf", help="Output FileN", required=True)
parser.add_argument('-i', action='store', dest="inputf", help="Input FileN", default="")
parser.add_argument('-p', dest='gtype', help="Graph Type", type=lambda val: GraphType[val], choices=list(GraphType), default = "bar")
parser.add_argument('-l', dest='log', help="Data Access Log Scale", action='store_true')

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

verbose = False

def parse_bar_input(inputf, groups: int):
  xaxis = []
  yaxis = []
  eaxis = []
  group_label = []

  for i in range (0,groups):
    yaxis.append([])
    eaxis.append([])

  for line in inputf.readlines():
    broken_line = re.split(r'\s+', line.rstrip())
    if verbose:
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
    return (xaxis, yaxis, eaxis, group_label)

def open_data_file(valFile: Path) -> [float]:
    data = []
    with open(valFile) as f:
        for line in f.readlines():
            data.append(float(line.rstrip()))
    return data

def parse_box_input(inputf, groups: int):
    xaxis = []
    yaxis = []
    eaxis = []
    group_label = []
    for i in range (0,groups):
        yaxis.append([])
        eaxis.append([])

    for line in inputf.readlines():
        broken_line = re.split(r'\s+', line.rstrip())
        if groups > 1 and group_label == []:
            group_label = broken_line
            continue
        for i in range(0,groups) :
            yaxis[i].append(open_data_file(Path(broken_line[i])))
            eaxis[i].append(0)
        xaxis.append(broken_line[groups])
    return (xaxis, yaxis, eaxis, group_label)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

"""
bar type graphs:
This function takes in from the inputf an input that looks like with groups:
    g_name0 g_name1 g_name2
    g0y0 g1y0 g2y0 g0x0 g0e0 g1e0 g2e0
    g0y1 g1y1 g2y1 g0x1 g0e1 g1e1 g2e1
This function takes in from inputf an input that looks like w/o groups:
    y0 x0 e0
    y1 x1 e1
    y2 x2 e2

box type graphs:
    g_name0 g_name1 g_name2
    pathG0Y0s pathG1Y0s pathG2Y0s
    pathG0Y1s pathG1Y1s pathG2Y1s
"""
def graph(gtitle, xlabel, ylabel, inputf, ouputf, groups, horizon, gtype: GraphType, log: bool):
    _ = plt.figure(figsize=(10,7))
    #ax = fig.add_axes([0,0,1,1])
    xaxis, yaxis, eaxis, group_label = ([], [], [], [])

    if verbose:
        print(gtype)

    if gtype is GraphType.bar:
        xaxis, yaxis, eaxis, group_label = parse_bar_input(inputf, groups)
    elif gtype is GraphType.box:
        xaxis, yaxis, eaxis, group_label = parse_box_input(inputf, groups)
    else:
        print("FAILURE")

    if verbose:
        print(xaxis)

    width_orig = 0.8
    width = width_orig/groups
    if group_label == [] :
        group_label.append("")
    ind = np.arange(len(yaxis[0]))
    for i,color in zip(range(0, groups), new_color_cycle):
        pos = ind + (-(groups -1)/2.0 +i)*width
        if verbose:
            print(pos)
        if gtype is GraphType.bar:
            if horizon:
              plt.barh(pos, yaxis[i], width, yerr=eaxis[i], label=group_label[i])
            else:
              plt.bar(pos, yaxis[i], width, yerr=eaxis[i], label=group_label[i])
        elif gtype is GraphType.box:
            bp = plt.boxplot(yaxis[i], positions=pos, widths=width, label=group_label[i])
            set_box_color(bp, color["color"])

    if horizon:
        plt.yticks(ind, xaxis)
        plt.title(gtitle)
        plt.xlabel(ylabel)
        plt.ylabel(xlabel)
        if log:
            plt.xscale("log")
    else:
        plt.xticks(ind, xaxis)
        plt.title(gtitle)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if log:
            plt.yscale("log")
    if groups > 1:
        plt.legend(bbox_to_anchor=(1,.5),loc='center left')
    plt.savefig(ouputf, bbox_inches='tight')


if __name__ == "__main__":
  args = parser.parse_args()
  inputf = sys.stdin
  verbose = args.verbose
  if args.inputf != "" or args.inputf is None:
    inputf = open(args.inputf)
  graph(args.gtitle,args.xlabel,args.ylabel,inputf,args.ouputf,args.groups,args.horizon,args.gtype, args.log)
