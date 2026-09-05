"""The worker's URL on stdout, or nothing while the instance is still coming up."""
import json
import sys

d = json.load(sys.stdin)
port = ((d.get("ports") or {}).get("8732/tcp") or [{}])[0].get("HostPort")
if port and d.get("actual_status") == "running":
    print("http://{}:{}".format(d["public_ipaddr"], port))
