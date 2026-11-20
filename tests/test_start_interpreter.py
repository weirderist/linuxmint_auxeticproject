import os as _os, pathlib as _pl
ROOT_STR = str(_pl.Path(__file__).resolve().parent.parent)          # repository root: the tests run from any folder
import os, sys, pathlib
P = pathlib.Path(_os.path.join(ROOT_STR)); SM = P / "smoke_test"
os.environ["CAMPAIGN_DIR"] = str(SM / "campaigns"); os.environ["AUX_AUTORESUME"] = "0"
sys.path.insert(0, str(P / "webui"))
import main, campaign_api as ca
from fastapi.testclient import TestClient
cmds = []
class FP: returncode = 0; stdout = "runner started (own session, SIGHUP ignored): pid 1, log x\n"; stderr = ""
ca.subprocess.run = lambda cmd, **k: (cmds.append((cmd, k)), FP())[1]
ca._cm().alive = lambda att: False
fake_py = SM / "fake_venv_python"; fake_py.write_text("")
c = TestClient(main.app)
ca.PROJECT_PY = fake_py; ca._last_start.clear()
r = c.post("/api/campaigns/smoke1/start")
ok1 = r.status_code == 200 and cmds[-1][0][0] == str(fake_py) and cmds[-1][0][2:] == ["run", "smoke1", "--detach"] and cmds[-1][1].get("timeout") == 60
print(("PASS " if ok1 else "FAIL ") + "Start uses the project interpreter and the detach command:", cmds[-1][0])
ca.PROJECT_PY = SM / "missing"; ca._last_start.clear(); r = c.post("/api/campaigns/smoke1/start")
ok2 = r.status_code == 200 and cmds[-1][0][0] == sys.executable
print(("PASS " if ok2 else "FAIL ") + "without the project venv it falls back to the service's own interpreter")
print("ALL OK" if ok1 and ok2 else "FAILURES")
