@echo off
if not exist ..\test_output mkdir ..\test_output

for /L %%i in (1,1,56) do (
  setlocal enabledelayedexpansion
  set "n=0%%i"
  set "n=!n:~-2!"
  python clonalTree.py -i ..\LLC\LLC_dataset!n!_1_200_sequences.aln.fa -o ..\test_output\classical_!n!.nk -a 1 -r 1 -t 1
  endlocal
)
