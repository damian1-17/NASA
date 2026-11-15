@echo off
title Desplegar AstroSalud en Vercel
echo ========================================================
echo   Desplegando AstroSalud en Vercel
echo ========================================================
cd /d "%~dp0"
call npx vercel
echo.
echo ========================================================
echo   Para desplegar a produccion definitiva ejecuta:
echo   npx vercel --prod
echo ========================================================
pause
