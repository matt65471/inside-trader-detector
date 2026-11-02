@echo off
set PYTHONFAULTHANDLER=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false
echo ===== diagnostic restart %date% %time% ===== >> "C:\Users\matt6\projects\inside-trader-detector\data\gamma_verified_v2.log"
"C:\Users\matt6\projects\inside-trader-detector\.venv311\Scripts\python.exe" -X faulthandler -m informed_flow --db "data\gamma_verified_v2.sqlite3" run --cohort balanced-tags-v2 --markets-per-category 1000 --admission-rate 0.50 --seed 0 --similarity-threshold 0.90 --market-workers 1 --enrichment-workers 1 --resolution-workers 1 --label-workers 1 --embedding-device auto >> "C:\Users\matt6\projects\inside-trader-detector\data\gamma_verified_v2.log" 2>&1
exit /b %ERRORLEVEL%
