"""Trusted startup code shared by local and scheduled execution.

These strings are fixed host-side code. Scene source is always a command argument.
"""

ENVIRONMENT_PROBE = """import importlib.metadata as md,json,os,shutil,subprocess,sys
from pathlib import Path
def version(command,flag):
    path=shutil.which(command)
    if not path: return ''
    try:
        result=subprocess.run([path,flag],capture_output=True,text=True,timeout=5,check=False)
        return ((result.stdout or result.stderr).splitlines() or [''])[0][:300] if result.returncode==0 else ''
    except (OSError,subprocess.TimeoutExpired): return ''
try: manim=md.version('manim')
except md.PackageNotFoundError: manim=''
payload={'python':sys.version.split()[0],'manim':manim,'ffmpeg':version('ffmpeg','-version'),'xelatex':version('xelatex','--version'),'renderer':os.environ.get('MANIM_RENDERER',''),'pyopengl_platform':os.environ.get('PYOPENGL_PLATFORM','')}
payload.update(json.loads(sys.argv[1]))
"""

LOCAL_LAUNCH = (
    ENVIRONMENT_PROBE
    + """
marker=Path(sys.argv[2])
marker.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as handle:
    json.dump(payload,handle,ensure_ascii=False)
    handle.flush()
    os.fsync(handle.fileno())
os.execvpe(sys.argv[3],sys.argv[3:],os.environ)
"""
)

VERIFY_IMAGE_LAUNCH = """import hashlib,os,sys
with open(sys.argv[1],'rb') as handle:
    digest=hashlib.file_digest(handle,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else None
    if digest is None:
        value=hashlib.sha256()
        for chunk in iter(lambda:handle.read(1048576),b''): value.update(chunk)
        digest=value.hexdigest()
if digest!=sys.argv[2]: raise SystemExit(73)
os.execvpe(sys.argv[3],sys.argv[3:],os.environ)
"""
