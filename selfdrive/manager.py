#!/usr/bin/env python3
import os
import time
import sys
import fcntl
import errno
import signal
import shutil
import subprocess
import datetime
import textwrap
from typing import Dict, List
from selfdrive.swaglog import cloudlog, add_logentries_handler


from common.basedir import BASEDIR, PARAMS
from common.android import ANDROID
from common.op_params import opParams
WEBCAM = os.getenv("WEBCAM") is not None
sys.path.append(os.path.join(BASEDIR, "pyextra"))
os.environ['BASEDIR'] = BASEDIR

TOTAL_SCONS_NODES = 1020
prebuilt = os.path.exists(os.path.join(BASEDIR, 'prebuilt'))
kill_updated = opParams().get('update_behavior').lower().strip() == 'off' or os.path.exists('/data/no_ota_updates')

# Create folders needed for msgq
try:
  os.mkdir("/dev/shm")
except FileExistsError:
  pass
except PermissionError:
  print("WARNING: failed to make /dev/shm")

if ANDROID:
  os.chmod("/dev/shm", 0o777)

def unblock_stdout():
  # get a non-blocking stdout
  child_pid, child_pty = os.forkpty()
  if child_pid != 0:  # parent

    # child is in its own process group, manually pass kill signals
    signal.signal(signal.SIGINT, lambda signum, frame: os.kill(child_pid, signal.SIGINT))
    signal.signal(signal.SIGTERM, lambda signum, frame: os.kill(child_pid, signal.SIGTERM))

    fcntl.fcntl(sys.stdout, fcntl.F_SETFL,
       fcntl.fcntl(sys.stdout, fcntl.F_GETFL) | os.O_NONBLOCK)

    while True:
      try:
        dat = os.read(child_pty, 4096)
      except OSError as e:
        if e.errno == errno.EIO:
          break
        continue

      if not dat:
        break

      try:
        sys.stdout.write(dat.decode('utf8'))
      except (OSError, IOError, UnicodeDecodeError):
        pass

    # os.wait() returns a tuple with the pid and a 16 bit value
    # whose low byte is the signal number and whose high byte is the exit satus
    exit_status = os.wait()[1] >> 8
    os._exit(exit_status)


if __name__ == "__main__":
  unblock_stdout()

if __name__ == "__main__" and ANDROID:
  from common.spinner import Spinner
  from common.text_window import TextWindow
else:
  from common.spinner import FakeSpinner as Spinner
  from common.text_window import FakeTextWindow as TextWindow

import importlib
import traceback
from multiprocessing import Process

# Run scons
spinner = Spinner()
spinner.update("0")

if not prebuilt:
  for retry in [True, False]:
    # run scons
    env = os.environ.copy()
    env['SCONS_PROGRESS'] = "1"
    env['SCONS_CACHE'] = "1"

    nproc = os.cpu_count()
    j_flag = "" if nproc is None else "-j%d" % (nproc - 1)
    scons = subprocess.Popen(["scons", "-j8"], cwd=BASEDIR, env=env, stderr=subprocess.PIPE)

    compile_output = []

    # Read progress from stderr and update spinner
    while scons.poll() is None:
      try:
        line = scons.stderr.readline()  # type: ignore
        if line is None:
          continue
        line = line.rstrip()

        prefix = b'progress: '
        if line.startswith(prefix):
          i = int(line[len(prefix):])
          if spinner is not None:
            spinner.update("%d" % (70.0 * (i / TOTAL_SCONS_NODES)))
        elif len(line):
          compile_output.append(line)
          print(line.decode('utf8', 'replace'))
      except Exception:
        pass

    if scons.returncode != 0:
      # Read remaining output
      r = scons.stderr.read().split(b'\n')   # type: ignore
      compile_output += r

      if retry:
        if not os.getenv("CI"):
          print("scons build failed, cleaning in")
          for i in range(3, -1, -1):
            print("....%d" % i)
            time.sleep(1)
          subprocess.check_call(["scons", "-c"], cwd=BASEDIR, env=env)
          shutil.rmtree("/tmp/scons_cache")
        else:
          print("scons build failed after retry")
          sys.exit(1)
      else:
        # Build failed log errors
        errors = [line.decode('utf8', 'replace') for line in compile_output
                  if any([err in line for err in [b'error: ', b'not found, needed by target']])]
        error_s = "\n".join(errors)
        add_logentries_handler(cloudlog)
        cloudlog.error("scons build failed\n" + error_s)

        # Show TextWindow
        error_s = "\n \n".join(["\n".join(textwrap.wrap(e, 65)) for e in errors])
        with TextWindow("openpilot failed to build\n \n" + error_s) as t:
          t.wait_for_exit()

        exit(1)
    else:
      break

import cereal
import cereal.messaging as messaging

from common.params import Params
import selfdrive.crash as crash
from selfdrive.registration import register
from selfdrive.version import version, dirty
from selfdrive.loggerd.config import ROOT
from selfdrive.launcher import launcher
from common import android
from common.apk import update_apks, pm_apply_packages, start_offroad

ThermalStatus = cereal.log.ThermalData.ThermalStatus

# comment out anything you don't want to run
managed_processes = {
  "thermald": "selfdrive.thermald.thermald",
  "uploader": "selfdrive.loggerd.uploader",
  "deleter": "selfdrive.loggerd.deleter",
  "controlsd": "selfdrive.controls.controlsd",
  "plannerd": "selfdrive.controls.plannerd",
  "radard": "selfdrive.controls.radard",
  "dmonitoringd": "selfdrive.monitoring.dmonitoringd",
  "ubloxd": ("selfdrive/locationd", ["./ubloxd"]),
  "loggerd": ("selfdrive/loggerd", ["./loggerd"]),
  "logmessaged": "selfdrive.logmessaged",
  "locationd": "selfdrive.locationd.locationd",
  "tombstoned": "selfdrive.tombstoned",
  "logcatd": ("selfdrive/logcatd", ["./logcatd"]),
  "proclogd": ("selfdrive/proclogd", ["./proclogd"]),
  "boardd": ("selfdrive/boardd", ["./boardd"]),   # not used directly
  "pandad": "selfdrive.pandad",
  "ui": ("selfdrive/ui", ["./ui"]),
  "calibrationd": "selfdrive.locationd.calibrationd",
  "paramsd": "selfdrive.locationd.paramsd",
  "camerad": ("selfdrive/camerad", ["./camerad"]),
  "sensord": ("selfdrive/sensord", ["./sensord"]),
  "clocksd": ("selfdrive/clocksd", ["./clocksd"]),
  "gpsd": ("selfdrive/sensord", ["./gpsd"]),
  "updated": "selfdrive.updated",
  "dmonitoringmodeld": ("selfdrive/modeld", ["./dmonitoringmodeld"]),
  "modeld": ("selfdrive/modeld", ["./modeld"]),
  "driverview": "selfdrive.monitoring.driverview",

  "lanespeedd": "selfdrive.controls.lib.lane_speed",
}

daemon_processes = {
  "manage_athenad": ("selfdrive.athena.manage_athenad", "AthenadPid"),
}

running: Dict[str, Process] = {}
def get_running():
  return running

# due to qualcomm kernel bugs SIGKILLing camerad sometimes causes page table corruption
unkillable_processes = ['camerad']

# processes to end with SIGINT instead of SIGTERM
interrupt_processes: List[str] = []

# processes to end with SIGKILL instead of SIGTERM
kill_processes = ['sensord']

# processes to end if thermal conditions exceed Green parameters
green_temp_processes = ['uploader']

persistent_processes = [
  'thermald',
  'logmessaged',
  'ui',
  'uploader',
]

if ANDROID:
  persistent_processes += [
    'logcatd',
    'tombstoned',
    # 'updated',
    'deleter',
  ]
  if not kill_updated:
    persistent_processes.append('updated')

car_started_processes = [
  'controlsd',
  'plannerd',
  'loggerd',
  'radard',
  'dmonitoringd',
  'calibrationd',
  'paramsd',
  'camerad',
  'modeld',
  'proclogd',
  'ubloxd',
  'locationd',
  'lanespeedd',
]

if WEBCAM:
  car_started_processes += [
    'dmonitoringmodeld',
  ]

if ANDROID:
  car_started_processes += [
    'sensord',
    'clocksd',
    'gpsd',
    'dmonitoringmodeld',
  ]

def register_managed_process(name, desc, car_started=False):
  global managed_processes, car_started_processes, persistent_processes
  print("registering %s" % name)
  managed_processes[name] = desc
  if car_started:
    car_started_processes.append(name)
  else:
    persistent_processes.append(name)

# ****************** process management functions ******************
def nativelauncher(pargs, cwd):
  # exec the process
  os.chdir(cwd)

  # because when extracted from pex zips permissions get lost -_-
  os.chmod(pargs[0], 0o700)

  os.execvp(pargs[0], pargs)

def start_managed_process(name):
  if name in running or name not in managed_processes:
    return
  proc = managed_processes[name]
  if isinstance(proc, str):
    cloudlog.info("starting python %s" % proc)
    running[name] = Process(name=name, target=launcher, args=(proc,))
  else:
    pdir, pargs = proc
    cwd = os.path.join(BASEDIR, pdir)
    cloudlog.info("starting process %s" % name)
    running[name] = Process(name=name, target=nativelauncher, args=(pargs, cwd))
  running[name].start()

def start_daemon_process(name):
  params = Params()
  proc, pid_param = daemon_processes[name]
  pid = params.get(pid_param, encoding='utf-8')

  if pid is not None:
    try:
      os.kill(int(pid), 0)
      with open(f'/proc/{pid}/cmdline') as f:
        if proc in f.read():
          # daemon is running
          return
    except (OSError, FileNotFoundError):
      # process is dead
      pass

  cloudlog.info("starting daemon %s" % name)
  proc = subprocess.Popen(['python', '-m', proc],  # pylint: disable=subprocess-popen-preexec-fn
                          stdin=open('/dev/null', 'r'),
                          stdout=open('/dev/null', 'w'),
                          stderr=open('/dev/null', 'w'),
                          preexec_fn=os.setpgrp)

  params.put(pid_param, str(proc.pid))

def prepare_managed_process(p):
  proc = managed_processes[p]
  if isinstance(proc, str):
    # import this python
    cloudlog.info("preimporting %s" % proc)
    importlib.import_module(proc)
  elif os.path.isfile(os.path.join(BASEDIR, proc[0], "Makefile")):
    # build this process
    cloudlog.info("building %s" % (proc,))
    try:
      subprocess.check_call(["make", "-j4"], cwd=os.path.join(BASEDIR, proc[0]))
    except subprocess.CalledProcessError:
      # make clean if the build failed
      cloudlog.warning("building %s failed, make clean" % (proc, ))
      subprocess.check_call(["make", "clean"], cwd=os.path.join(BASEDIR, proc[0]))
      subprocess.check_call(["make", "-j4"], cwd=os.path.join(BASEDIR, proc[0]))


def join_process(process, timeout):
  # Process().join(timeout) will hang due to a python 3 bug: https://bugs.python.org/issue28382
  # We have to poll the exitcode instead
  t = time.time()
  while time.time() - t < timeout and process.exitcode is None:
    time.sleep(0.001)


def kill_managed_process(name):
  if name not in running or name not in managed_processes:
    return
  cloudlog.info("killing %s" % name)

  if running[name].exitcode is None:
    if name in interrupt_processes:
      os.kill(running[name].pid, signal.SIGINT)
    elif name in kill_processes:
      os.kill(running[name].pid, signal.SIGKILL)
    else:
      running[name].terminate()

    join_process(running[name], 5)

    if running[name].exitcode is None:
      if name in unkillable_processes:
        cloudlog.critical("unkillable process %s failed to exit! rebooting in 15 if it doesn't die" % name)
        join_process(running[name], 15)
        if running[name].exitcode is None:
          cloudlog.critical("unkillable process %s failed to die!" % name)
          if ANDROID:
            cloudlog.critical("FORCE REBOOTING PHONE!")
            os.system("date >> /sdcard/unkillable_reboot")
            os.system("reboot")
          raise RuntimeError
      else:
        cloudlog.info("killing %s with SIGKILL" % name)
        os.kill(running[name].pid, signal.SIGKILL)
        running[name].join()

  cloudlog.info("%s is dead with %d" % (name, running[name].exitcode))
  del running[name]


def cleanup_all_processes(signal, frame):
  cloudlog.info("caught ctrl-c %s %s" % (signal, frame))

  if ANDROID:
    pm_apply_packages('disable')

  for name in list(running.keys()):
    kill_managed_process(name)
  cloudlog.info("everything is dead")

# ****************** run loop ******************

def manager_init(should_register=True):
  if should_register:
    reg_res = register()
    if reg_res:
      dongle_id = reg_res
    else:
      raise Exception("server registration failed")
  else:
    dongle_id = "c"*16

  # set dongle id
  cloudlog.info("dongle id is " + dongle_id)
  os.environ['DONGLE_ID'] = dongle_id

  cloudlog.info("dirty is %d" % dirty)
  if not dirty:
    os.environ['CLEAN'] = '1'

  cloudlog.bind_global(dongle_id=dongle_id, version=version, dirty=dirty, is_eon=True)
  crash.bind_user(id=dongle_id)
  crash.bind_extra(version=version, dirty=dirty, is_eon=True)

  os.umask(0)
  try:
    os.mkdir(ROOT, 0o777)
  except OSError:
    pass

  # ensure shared libraries are readable by apks
  if ANDROID:
    os.chmod(BASEDIR, 0o755)
    os.chmod(os.path.join(BASEDIR, "cereal"), 0o755)
    os.chmod(os.path.join(BASEDIR, "cereal", "libmessaging_shared.so"), 0o755)

def manager_thread():
  # now loop
  thermal_sock = messaging.sub_sock('thermal')

  cloudlog.info("manager start")
  cloudlog.info({"environ": os.environ})

  # save boot log
  subprocess.call(["./loggerd", "--bootlog"], cwd=os.path.join(BASEDIR, "selfdrive/loggerd"))

  params = Params()

  # start daemon processes
  for p in daemon_processes:
    start_daemon_process(p)

  # start persistent processes
  for p in persistent_processes:
    start_managed_process(p)

  # start offroad
  if ANDROID:
    pm_apply_packages('enable')
    start_offroad()

  if os.getenv("NOBOARD") is None:
    start_managed_process("pandad")

  if os.getenv("BLOCK") is not None:
    for k in os.getenv("BLOCK").split(","):
      del managed_processes[k]

  logger_dead = False

  while 1:
    msg = messaging.recv_sock(thermal_sock, wait=True)

    # heavyweight batch processes are gated on favorable thermal conditions
    if msg.thermal.thermalStatus >= ThermalStatus.yellow:
      for p in green_temp_processes:
        if p in persistent_processes:
          kill_managed_process(p)
    else:
      for p in green_temp_processes:
        if p in persistent_processes:
          start_managed_process(p)

    if msg.thermal.freeSpace < 0.05:
      logger_dead = True
    run_all = False
    if (msg.thermal.started and "driverview" not in running) or run_all:
      for p in car_started_processes:
        if p == "loggerd" and logger_dead:
          kill_managed_process(p)
        else:
          start_managed_process(p)
    else:
      logger_dead = False
      for p in reversed(car_started_processes):
        kill_managed_process(p)
      # this is ugly
      if "driverview" not in running and params.get("IsDriverViewEnabled") == b"1":
        start_managed_process("driverview")
      elif "driverview" in running and params.get("IsDriverViewEnabled") == b"0":
        kill_managed_process("driverview")

    # check the status of all processes, did any of them die?
    running_list = ["%s%s\u001b[0m" % ("\u001b[32m" if running[p].is_alive() else "\u001b[31m", p) for p in running]
    cloudlog.debug(' '.join(running_list))

    # Exit main loop when uninstall is needed
    if params.get("DoUninstall", encoding='utf8') == "1":
      break

def manager_prepare(spinner=None):
  # build all processes
  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  # Spinner has to start from 70 here
  total = 100.0 if prebuilt else 30.0

  for i, p in enumerate(managed_processes):
    if spinner is not None:
      spinner.update("%d" % ((100.0 - total) + total * (i + 1) / len(managed_processes),))
    prepare_managed_process(p)

def uninstall():
  cloudlog.warning("uninstalling")
  with open('/cache/recovery/command', 'w') as f:
    f.write('--wipe_data\n')
  # IPowerManager.reboot(confirm=false, reason="recovery", wait=true)
  android.reboot(reason="recovery")

def main():
  os.environ['PARAMS_PATH'] = PARAMS

  if ANDROID:
    # the flippening!
    os.system('LD_LIBRARY_PATH="" content insert --uri content://settings/system --bind name:s:user_rotation --bind value:i:1')

    # disable bluetooth
    os.system('service call bluetooth_manager 8')

  params = Params()
  params.manager_start()

  default_params = [
    ("CommunityFeaturesToggle", "0"),
    ("CompletedTrainingVersion", "0"),
    ("IsRHD", "0"),
    ("IsMetric", "0"),
    ("RecordFront", "0"),
    ("HasAcceptedTerms", "0"),
    ("HasCompletedSetup", "0"),
    ("IsUploadRawEnabled", "1"),
    ("IsLdwEnabled", "1"),
    ("IsGeofenceEnabled", "-1"),
    ("SpeedLimitOffset", "0"),
    ("LongitudinalControl", "0"),
    ("LimitSetSpeed", "0"),
    ("LimitSetSpeedNeural", "0"),
    ("LastUpdateTime", datetime.datetime.utcnow().isoformat().encode('utf8')),
    ("OpenpilotEnabledToggle", "1"),
    ("LaneChangeEnabled", "1"),
    ("IsDriverViewEnabled", "0"),
  ]

  # set unset params
  for k, v in default_params:
    if params.get(k) is None:
      params.put(k, v)

  # is this chffrplus?
  if os.getenv("PASSIVE") is not None:
    params.put("Passive", str(int(os.getenv("PASSIVE"))))

  if params.get("Passive") is None:
    raise Exception("Passive must be set to continue")

  if ANDROID:
    update_apks()
  manager_init()
  manager_prepare(spinner)
  spinner.close()

  if os.getenv("PREPAREONLY") is not None:
    return

  # SystemExit on sigterm
  signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))

  try:
    manager_thread()
  except Exception:
    traceback.print_exc()
    crash.capture_exception()
  finally:
    cleanup_all_processes(None, None)

  if params.get("DoUninstall", encoding='utf8') == "1":
    uninstall()


if __name__ == "__main__":
  try:
    main()
  except Exception:
    add_logentries_handler(cloudlog)
    cloudlog.exception("Manager failed to start")

    # Show last 3 lines of traceback
    error = traceback.format_exc(3)

    error = "Manager failed to start. Press Reset to pull and reset to origin!\n \n" + error
    with TextWindow(error) as t:
      exit_status = t.wait_for_exit()
    if exit_status == 'reset':
      for _ in range(2):
        try:
          subprocess.check_output(["git", "pull"], cwd=BASEDIR)
          subprocess.check_output(["git", "reset", "--hard", "@{u}"], cwd=BASEDIR)
          print('git reset successful!')
          break
        except subprocess.CalledProcessError as e:
          # print(e.output)
          if _ != 1:
            print('git reset failed, trying again')
            time.sleep(5)  # wait 5 seconds and try again

    time.sleep(1)
    subprocess.check_output(["am", "start", "-a", "android.intent.action.REBOOT"])
    raise

  # manual exit because we are forked
  sys.exit(0)
