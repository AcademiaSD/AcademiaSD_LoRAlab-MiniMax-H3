@echo off
setlocal EnableExtensions EnableDelayedExpansion

title AcademiaSD - MiniMax-H3 LoRA Trainer Updater
color 0B

cd /d "%~dp0"

set "REPO_URL=https://github.com/AcademiaSD/AcademiaSD_LoRAlab-MiniMax-H3.git"
set "REPO_DIR=AcademiaSD_LoRAlab-MiniMax-H3"
set "BACKUP_DIR=_settings_backup"

rem Ficheros que edita el usuario y que NO se deben perder al actualizar.
rem Files the user edits, which must survive the update.
set "USER_FILES=pre_cache_settings.json train_settings.json caption_settings.json HF_token.json"

echo ================================================================
echo   ACADEMIASD - MINIMAX-H3 LORA TRAINER UPDATER
echo   [EN] Repository Update Utility
echo   [ES] Utilidad de Actualizacion del Repositorio
echo ================================================================
echo.

rem ---------------------------------------------------------------- 1. Git
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not available in PATH.
    echo [ERROR] Git no esta instalado o no esta disponible en el PATH.
    echo.
    echo [EN] Install it with:  winget install --id Git.Git -e
    echo [ES] Instalalo con:    winget install --id Git.Git -e
    echo.
    echo [EN] Or download it from https://git-scm.com/download/win
    echo [ES] O descargalo de   https://git-scm.com/download/win
    echo.
    echo [EN] Close and reopen this window after installing.
    echo [ES] Cierra y vuelve a abrir esta ventana despues de instalarlo.
    echo.
    pause
    exit /b 1
)

rem ------------------------------------------------- 2. Localizar el repositorio
if exist ".git" goto :HAVE_REPO

if exist "%~dp0%REPO_DIR%\.git" (
    cd /d "%~dp0%REPO_DIR%"
    goto :HAVE_REPO
)

goto :NO_REPO


rem ================================================================
rem   NO HAY .git  ->  la carpeta viene de un ZIP de GitHub
rem   No .git folder -> the folder came from a GitHub ZIP
rem ================================================================
:NO_REPO
echo [WARNING] No '.git' folder found in this directory.
echo [ADVERTENCIA] No se encontro la carpeta '.git' en este directorio.
echo.
echo [EN] Cause: You likely downloaded the repository as a ZIP file.
echo [ES] Causa: Es probable que hayas descargado el proyecto en formato ZIP.
echo.
echo [EN] Converting it to a Git repository lets you update with one click
echo [EN] from now on. Your settings, models and trained LoRAs are NOT touched.
echo [ES] Convertirla en repositorio Git te permite actualizar con un clic
echo [ES] a partir de ahora. Tus ajustes, modelos y LoRAs entrenados NO se tocan.
echo.
echo [EN] BUT every script (.py, .html, .bat) WILL BE REPLACED by the version
echo [EN] on GitHub. If you edited any of them, those changes are lost. A copy
echo [EN] is saved in _backup_before_update_... before anything is overwritten.
echo [ES] PERO todos los scripts (.py, .html, .bat) SE SUSTITUIRAN por los de
echo [ES] GitHub. Si has editado alguno, esos cambios se pierden. Antes de
echo [ES] sobrescribir se guarda una copia en _backup_before_update_...
echo.

choice /C YN /M "[Y] Yes / Si  -  [N] No"
if errorlevel 2 exit /b 0

call :BACKUP_SETTINGS

echo.
echo [EN] Initializing Git repository and fetching latest code...
echo [ES] Inicializando repositorio Git y descargando codigo reciente...
echo.

git init
git remote add origin "%REPO_URL%"
git fetch origin
if errorlevel 1 (
    echo.
    echo [ERROR] Could not reach GitHub. Check your connection.
    echo [ERROR] No se pudo conectar con GitHub. Revisa tu conexion.
    pause
    exit /b 1
)

call :RESOLVE_BRANCH
if "!BRANCH!"=="" (
    echo [ERROR] Could not find branch 'main' or 'master' on the remote.
    echo [ERROR] No se encontro la rama 'main' ni 'master' en el remoto.
    pause
    exit /b 1
)

call :SAFE_BACKUP
git reset --hard origin/!BRANCH!
if errorlevel 1 (
    echo [ERROR] Could not synchronize with GitHub.
    echo [ERROR] No se pudo sincronizar con GitHub.
    call :RESTORE_SETTINGS
    pause
    exit /b 1
)

call :RESTORE_SETTINGS

echo.
echo [OK] Repository converted and synchronized with GitHub!
echo [OK] Repositorio convertido y sincronizado con GitHub!
goto :FINISH


rem ================================================================
rem   HAY .git  ->  comprobar si hay novedades antes de tocar nada
rem   Repo present -> check for updates before touching anything
rem ================================================================
:HAVE_REPO
echo [EN] Checking GitHub for updates...
echo [ES] Comprobando si hay actualizaciones en GitHub...
echo.

git fetch origin
if errorlevel 1 (
    echo [ERROR] Could not reach GitHub. Check your connection.
    echo [ERROR] No se pudo conectar con GitHub. Revisa tu conexion.
    pause
    exit /b 1
)

call :RESOLVE_BRANCH
if "!BRANCH!"=="" (
    echo [ERROR] Could not find branch 'main' or 'master' on the remote.
    echo [ERROR] No se encontro la rama 'main' ni 'master' en el remoto.
    pause
    exit /b 1
)

rem Cuantos commits nos faltan / how many commits we are behind
set "BEHIND=0"
for /f %%C in ('git rev-list --count HEAD..origin/!BRANCH! 2^>nul') do set "BEHIND=%%C"

if "!BEHIND!"=="0" (
    echo ================================================================
    echo [OK] Already up to date. Nothing to download.
    echo [OK] Ya estas al dia. No hay nada que descargar.
    echo ================================================================
    goto :FINISH
)

echo ================================================================
echo [EN] !BEHIND! new update(s) available:
echo [ES] !BEHIND! actualizacion(es) disponible(s):
echo ================================================================
echo.
git --no-pager log --oneline --decorate -n 15 HEAD..origin/!BRANCH!
echo.

choice /C YN /M "[EN] Update now?  [ES] Actualizar ahora?  [Y] Si  -  [N] No"
if errorlevel 2 (
    echo.
    echo [EN] Update cancelled. Nothing was changed.
    echo [ES] Actualizacion cancelada. No se ha cambiado nada.
    goto :FINISH
)

call :BACKUP_SETTINGS

echo.
echo [EN] Downloading updates...
echo [ES] Descargando actualizaciones...
echo.

git pull
if errorlevel 1 goto :PULL_FAILED

call :RESTORE_SETTINGS

echo.
echo ================================================================
echo [OK] Repository updated successfully! / Repositorio actualizado!
echo ================================================================
goto :FINISH


rem ---------------------------------------------------------------- pull fallido
:PULL_FAILED
echo.
echo [WARNING] 'git pull' could not merge automatically.
echo [ADVERTENCIA] 'git pull' no pudo fusionar automaticamente.
echo.
echo [EN] This happens when you have edited files that the update also changes.
echo [ES] Ocurre cuando has editado ficheros que la actualizacion tambien cambia.
echo.
echo [EN] Forcing will DISCARD your local edits to the project files and leave
echo [EN] the official version. Your settings JSON files, your datasets, your
echo [EN] models and your trained LoRAs are NOT affected.
echo [ES] Forzar DESCARTARA tus cambios locales en los ficheros del proyecto y
echo [ES] dejara la version oficial. Tus JSON de ajustes, tus datasets, tus
echo [ES] modelos y tus LoRAs entrenados NO se ven afectados.
echo.

choice /C YN /M "[EN] Force update?  [ES] Forzar actualizacion?  [Y] Si  -  [N] No"
if errorlevel 2 (
    echo.
    echo [EN] Cancelled. Your files are untouched.
    echo [ES] Cancelado. Tus ficheros siguen intactos.
    call :RESTORE_SETTINGS
    goto :FINISH
)

call :SAFE_BACKUP
git reset --hard origin/!BRANCH!
if errorlevel 1 (
    echo.
    echo [ERROR] Forced update failed.
    echo [ERROR] La actualizacion forzada fallo.
    call :RESTORE_SETTINGS
    pause
    exit /b 1
)

call :RESTORE_SETTINGS

echo.
echo ================================================================
echo [OK] Forced update completed. / Actualizacion forzada completada.
echo ================================================================
goto :FINISH


rem ================================================================
rem   SUBRUTINAS / SUBROUTINES
rem ================================================================

rem Averigua si la rama principal se llama main o master.
rem Works out whether the default branch is main or master.
:RESOLVE_BRANCH
set "BRANCH="
git show-ref --verify --quiet refs/remotes/origin/main
if not errorlevel 1 (
    set "BRANCH=main"
    exit /b 0
)
git show-ref --verify --quiet refs/remotes/origin/master
if not errorlevel 1 set "BRANCH=master"
exit /b 0

rem Copia los ajustes del usuario antes de tocar el repositorio. Sin esto un
rem 'git pull' o un 'reset --hard' devuelve los JSON a los valores de fabrica y
rem se pierden el proyecto, el trigger, la ruta del dataset y el perfil de VRAM.
rem Saves the user's settings before touching the repo. Without this a pull or a
rem hard reset returns the JSONs to factory values, losing the project name, the
rem trigger word, the dataset path and the VRAM profile.
:BACKUP_SETTINGS
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%" >nul 2>&1
set "SAVED="
for %%F in (%USER_FILES%) do (
    if exist "%%F" (
        copy /Y "%%F" "%BACKUP_DIR%\%%F" >nul 2>&1
        set "SAVED=!SAVED! %%F"
    )
)
if not "!SAVED!"=="" (
    echo [EN] Settings saved:!SAVED!
    echo [ES] Ajustes guardados:!SAVED!
)
exit /b 0

rem Devuelve los ajustes a su sitio despues de actualizar.
rem Puts the settings back after updating.
:RESTORE_SETTINGS
set "BACK="
for %%F in (%USER_FILES%) do (
    if exist "%BACKUP_DIR%\%%F" (
        copy /Y "%BACKUP_DIR%\%%F" "%%F" >nul 2>&1
        set "BACK=!BACK! %%F"
    )
)
if not "!BACK!"=="" (
    echo.
    echo [EN] Settings restored:!BACK!
    echo [ES] Ajustes restaurados:!BACK!
)
exit /b 0


:FINISH
call :CHECK_DEPS
echo.
echo [EN] Update process completed.
echo [ES] Proceso de actualizacion completado.
echo.
pause
exit /b 0


:CHECK_DEPS
rem   Actualizar traia el codigo pero no los paquetes: una version que anade una
rem   funcion dejaba el boton fallando con un ImportError en una instalacion por
rem   lo demas correcta. Se comprueba e instala SOLO lo que falte; rehacer el
rem   venv por dos paquetes de unos KB serian treinta minutos por nada.
rem   Para anadir una dependencia, tocala en las DOS lineas de abajo.
rem   Updating brought code but not packages. Only what is missing is installed;
rem   rebuilding the venv over two tiny packages would be half an hour for
rem   nothing. To add a dependency, edit BOTH lines below.
set "VENV_PYTHON=%~dp0venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" exit /b 0

"%VENV_PYTHON%" -c "import einops, rotary_embedding_torch" >nul 2>&1
if not errorlevel 1 (
    echo [EN] Dependencies up to date.
    echo [ES] Dependencias al dia.
    exit /b 0
)
echo.
echo [EN] This update needs packages that are not in the venv. Installing...
echo [ES] Esta actualizacion necesita paquetes que no estan en el venv. Instalando...
"%VENV_PYTHON%" -m pip install --upgrade einops rotary_embedding_torch
exit /b 0


:SAFE_BACKUP
rem   Copia los scripts antes de sobrescribirlos. Cuesta menos de un segundo y se
rem   hace siempre: comprobar si hacia falta costaria mas que la copia. Los pesos,
rem   las caches y los LoRA entrenados NO se copian -- son gigabytes y el reset no
rem   los toca, por no estar versionados.
rem   Copies the scripts before overwriting: under a second, done unconditionally.
rem   Weights, caches and trained LoRAs are NOT copied, being gigabytes and
rem   untracked.
for /f "tokens=1-5 delims=/:. " %%a in ("%DATE% %TIME%") do set "STAMP=%%c%%b%%a_%%d%%e"
set "SAFE=%~dp0_backup_before_update_%STAMP%"
md "%SAFE%" >nul 2>&1
copy /Y "%~dp0*.py"   "%SAFE%\" >nul 2>&1
copy /Y "%~dp0*.html" "%SAFE%\" >nul 2>&1
copy /Y "%~dp0*.bat"  "%SAFE%\" >nul 2>&1
copy /Y "%~dp0*.json" "%SAFE%\" >nul 2>&1
echo.
echo [EN] Current scripts copied to: %SAFE%
echo [ES] Scripts actuales copiados a: %SAFE%
echo.
exit /b 0
