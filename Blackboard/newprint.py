import time
import requests
from constants import Constants

class NewPrint:
    def __init__(self, id):
        self.id = id

    def newprint(self, *msgs, end="\n", event="log", skipconsole=False, level="info"):
        # timestamp in [] and id in [] then only message
        # allow multiple message parts (e.g. self.newprint("text", videoId))
        msg = " ".join(str(m) for m in msgs) if msgs else ""
        # ensure end is a string or None
        if end is not None and not isinstance(end, str):
            end = str(end)

        lokiline = f"[{self.id}] {msg}"
        logtime = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        line = logtime + lokiline
        print(line, end=end) if not skipconsole else None

        # Stream the same log line to a local Loki instance at port 3100
        try:
            loki_url = "http://localhost:3100/loki/api/v1/push"
            # Loki expects nanosecond epoch as a string
            timestamp_ns = str(int(time.time() * 1e9))
            # include the printed end (if any) in the log line sent to Loki
            send_line = lokiline + (end if end is not None else "")
            payload = {
                "streams": [
                    {
                        "stream": {
                            "job": "consumer",
                            "consumer_id": self.id,
                            "queue": getattr(self, "queuename", "flaskserver"),
                            "name": getattr(self, "name", "Flask Server"),
                            "event": event,
                            "level": level,
                            "session": Constants.SESSION,
                        },
                        "values": [[timestamp_ns, send_line]],
                    }
                ]
            }
            # short timeout and ignore failures so logging doesn't crash the app
            requests.post(loki_url, json=payload, timeout=1)
        except Exception:
            pass