@echo off
cd /d C:\Users\matt6\projects\inside-trader-detector

set PYTHONFAULTHANDLER=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false

echo ===== finish-existing started %date% %time% ===== >> "C:\Users\matt6\projects\inside-trader-detector\data\gamma_verified_v2_finish.log"

"C:\Users\matt6\projects\inside-trader-detector\.venv311\Scripts\python.exe" -X faulthandler -m informed_flow --db "C:\Users\matt6\projects\inside-trader-detector\data\gamma_verified_v2.sqlite3" finish-existing --cohort balanced-tags-v2 --market-workers 2 --enrichment-workers 1 --resolution-workers 1 --label-workers 2 >> "C:\Users\matt6\projects\inside-trader-detector\data\gamma_verified_v2_finish.log" 2>&1

set EXIT_CODE=%ERRORLEVEL%
echo ===== finish-existing ended %date% %time% exit=%EXIT_CODE% ===== >> "C:\Users\matt6\projects\inside-trader-detector\data\gamma_verified_v2_finish.log"
exit /b %EXIT_CODE%
