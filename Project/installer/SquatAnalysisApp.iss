#define MyAppName "Squat Analysis"
#define MyAppVersion "1.0.0"
#define MyAppExeName "main.exe"

[Setup]
SetupIconFile={#SourcePath}\squat.ico
AppId={{C6E5F02A-2D4B-4D1C-9C9F-0A9E52F11A23}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=.
OutputBaseFilename=SquatAnalysisInstaller
Compression=lzma
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
PrivilegesRequired=admin

[Languages]
Name: "german"; MessagesFile: "compiler:Languages\German.isl"

[Files]
; === Your PyInstaller build (ENTIRE onedir folder) ===
Source: "..\dist\main\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; === Microsoft Visual C++ Runtime (Python 3.12 needs this) ===
Source: "vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Run]
; Install VC++ Runtime silently
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; \
  StatusMsg: "Installing Microsoft Visual C++ Runtime..."; Flags: waituntilterminated

; Launch app after install
Filename: "{app}\{#MyAppExeName}"; Description: "Start {#MyAppName}"; \
  Flags: nowait postinstall skipifsilent

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"

