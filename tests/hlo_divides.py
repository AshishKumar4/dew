"""Scratch plugin: the largest optimized HLO modules the run dumped, with their divides and the
divides by a constant, so two programs' arithmetic can be told apart."""
import glob
import os
import re


def pytest_terminal_summary(terminalreporter):
    rows = []
    for path in glob.glob("/tmp/dew-hlo/*after_optimizations.txt"):
        text = open(path).read()
        lines = text.count("\n")
        divides = len(re.findall(r" divide\(", text))
        constant = len(re.findall(r" divide\([^)]*constant", text))
        multiplies = len(re.findall(r" multiply\(", text))
        reduces = len(re.findall(r" reduce\(", text))
        rows.append((lines, os.path.basename(path)[:90], divides, constant, multiplies, reduces))
    for lines, name, divides, constant, multiplies, reduces in sorted(rows, reverse=True)[:12]:
        terminalreporter.write_line(f"HLO {name} lines {lines} divide {divides} by-constant {constant} "
                                    f"multiply {multiplies} reduce {reduces}")
