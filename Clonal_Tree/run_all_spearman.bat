@echo off
setlocal enabledelayedexpansion

REM create output folder if it doesn't exist
if not exist ..\test_output mkdir ..\test_output

for /L %%i in (1,1,57) do (
  set "n=0%%i"
  set "n=!n:~-2!"

  set "IN=..\spearman_similarity\!n!_spearman_matrix.csv"
  set "OUT=..\test_output\spearman_!n!.nk"

  if exist "!IN!" (
    echo [RUN] !IN!
    python clonalTree_spearman.py -i "!IN!" -o "!OUT!" -a 1 -r 1 -t 1
  ) else (
    echo [SKIP missing] !IN!
  )
)

echo Done.
endlocal
