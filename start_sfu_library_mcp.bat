@echo off
REM SFU Library MCP Server Launcher
REM This script ensures the Docker container is running and starts the MCP server

setlocal enabledelayedexpansion

REM Configuration
set CONTAINER_NAME=claudebox-sfu-library-mcp-app
set PYTHON_PATH=/usr/bin/python3
set SCRIPT_PATH=/workspaces/sfu-library-mcp/src/sfu_library_mcp_server.py
set COMPOSE_FILE=C:\Users\gordo\OneDrive\Desktop\random ass scripts\sfu_library_mcp\.devcontainer\docker-compose.yml

REM Function to check if container is running
docker ps --format "{{.Names}}" | findstr /c:"%CONTAINER_NAME%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    REM Container is running, execute the MCP server
    docker exec -i %CONTAINER_NAME% %PYTHON_PATH% %SCRIPT_PATH% 2>nul
    exit /b %ERRORLEVEL%
)

REM Container is not running, try to start it
REM Check if docker-compose file exists
if exist "%COMPOSE_FILE%" (
    REM Use docker-compose to start the container
    cd /d "%~dp0"
    cd .devcontainer
    docker-compose up -d app 2>nul
    timeout /t 3 /nobreak >nul

    REM Try again to execute
    docker exec -i %CONTAINER_NAME% %PYTHON_PATH% %SCRIPT_PATH% 2>nul
    exit /b %ERRORLEVEL%
) else (
    REM Fallback: try docker start
    docker start %CONTAINER_NAME% 2>nul
    timeout /t 2 /nobreak >nul
    docker exec -i %CONTAINER_NAME% %PYTHON_PATH% %SCRIPT_PATH% 2>nul
    exit /b %ERRORLEVEL%
)
