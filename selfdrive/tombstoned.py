#!/usr/bin/env python3
import os
import time

from raven import Client
from raven.transport.http import HTTPTransport

from selfdrive.version import version, dirty
from selfdrive.swaglog import cloudlog

MAX_SIZE = 100000 * 10  # Normal size is 40-100k, allow up to 1M


def get_tombstones():
  """Returns list of (filename, ctime) for all tombstones in /data/tombstones"""
  DIR_DATA = "/data/tombstones/"
  return [(DIR_DATA + fn, int(os.stat(DIR_DATA + fn).st_ctime))
          for fn in os.listdir(DIR_DATA) if fn.startswith("tombstone")]


def report_tombstone(fn, client):
  f_size = os.path.getsize(fn)
  if f_size > MAX_SIZE:
    cloudlog.error(f"Tombstone {fn} too big, {f_size}. Skipping...")
    return

  with open(fn, encoding='ISO-8859-1') as f:
    contents = f.read()

  # Get summary for sentry title
  message = " ".join(contents.split('\n')[5:7])

  # Cut off pid/tid, since that varies per run
  name_idx = message.find('name')
  if name_idx >= 0:
    message = message[name_idx:]

  # Cut off fault addr
  fault_idx = message.find(', fault addr')
  if fault_idx >= 0:
    message = message[:fault_idx]

  cloudlog.error({'tombstone': message})
  client.captureMessage(
    message=message,
    sdk={'name': 'tombstoned', 'version': '0'},
    extra={
      'tombstone_fn': fn,
      'tombstone': contents
    },
  )


def main():
  initial_tombstones = set(get_tombstones())

  client = Client('https://d3b175702f62402c91ade04d1c547e68:b20d68c813c74f63a7cdf9c4039d8f56@sentry.io/157615',
                  install_sys_hook=False, transport=HTTPTransport, release=version, tags={'dirty': dirty}, string_max_length=10000)

  client.user_context({'id': os.environ.get('DONGLE_ID')})
  while True:
    now_tombstones = set(get_tombstones())

    for fn, _ in (now_tombstones - initial_tombstones):
      try:
        cloudlog.info(f"reporting new tombstone {fn}")
        report_tombstone(fn, client)
      except Exception:
        cloudlog.exception(f"Error reporting tombstone {fn}")

    initial_tombstones = now_tombstones
    time.sleep(5)


if __name__ == "__main__":
  main()
