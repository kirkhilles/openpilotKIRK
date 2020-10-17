"""Install exception handler for process crash."""
import os
import sys
import capnp
import threading
import traceback
from selfdrive.version import version, dirty
from datetime import datetime

from selfdrive.swaglog import cloudlog
from common.android import ANDROID

if os.getenv("NOLOG") or os.getenv("NOCRASH") or not ANDROID:
  def capture_exception(*args, **kwargs):
    pass

  def bind_user(**kwargs):
    pass

  def bind_extra(**kwargs):
    pass

  def install():
    pass
else:
  from raven import Client
  from raven.transport.http import HTTPTransport
  from selfdrive.version import origin, branch, commit
  from common.op_params import opParams

  COMMUNITY_DIR = '/data/community'
  CRASHES_DIR = '{}/crashes'.format(COMMUNITY_DIR)
  if not os.path.exists(COMMUNITY_DIR):
    os.mkdir(COMMUNITY_DIR)
  if not os.path.exists(CRASHES_DIR):
    os.mkdir(CRASHES_DIR)

  error_tags = {'dirty': dirty, 'origin': origin, 'branch': branch, 'commit': commit}
  username = opParams().get('username')
  if username is None or not isinstance(username, str):
    username = 'undefined'
  error_tags['username'] = username

  sentry_uri = 'https://1994756b5e6f41cf939a4c65de45f4f2:cefebaf3a8aa40d182609785f7189bd7@app.getsentry.com/77924'  # stock
  if 'github.com/shanesmiskol' in origin.lower():  # CHANGE TO YOUR remote and sentry key to receive errors if you fork this fork
    sentry_uri = 'https://7f66878806a948f9a8b52b0fe7781201@o237581.ingest.sentry.io/5252098'
  client = Client(sentry_uri, install_sys_hook=False, transport=HTTPTransport, release=version, tags=error_tags)

  def save_exception(exc_text):
    log_file = '{}/{}'.format(CRASHES_DIR, datetime.now().strftime('%d-%m-%Y--%I:%M.%S-%p.log'))
    with open(log_file, 'w') as f:
      f.write(exc_text)
    print('Logged current crash to {}'.format(log_file))

  def capture_exception(*args, **kwargs):
    save_exception(traceback.format_exc())
    exc_info = sys.exc_info()
    if not exc_info[0] is capnp.lib.capnp.KjException:
      client.captureException(*args, **kwargs)
    cloudlog.error("crash", exc_info=kwargs.get('exc_info', 1))

  def bind_user(**kwargs):
    client.user_context(kwargs)

  def bind_extra(**kwargs):
    client.extra_context(kwargs)

  def install():
    """
    Workaround for `sys.excepthook` thread bug from:
    http://bugs.python.org/issue1230540
    Call once from the main thread before creating any threads.
    Source: https://stackoverflow.com/a/31622038
    """
    # installs a sys.excepthook
    __excepthook__ = sys.excepthook

    def handle_exception(*exc_info):
      if exc_info[0] not in (KeyboardInterrupt, SystemExit):
        capture_exception()
      __excepthook__(*exc_info)
    sys.excepthook = handle_exception

    init_original = threading.Thread.__init__

    def init(self, *args, **kwargs):
      init_original(self, *args, **kwargs)
      run_original = self.run

      def run_with_except_hook(*args2, **kwargs2):
        try:
          run_original(*args2, **kwargs2)
        except Exception:
          sys.excepthook(*sys.exc_info())

      self.run = run_with_except_hook

    threading.Thread.__init__ = init
