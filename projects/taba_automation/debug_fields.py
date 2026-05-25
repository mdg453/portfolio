import requests
import urllib3
import ssl
from requests.adapters import HTTPAdapter

class LegacyAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.options |= 0x4
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        context.set_ciphers("DEFAULT@SECLEVEL=1")
        kwargs["ssl_context"] = context
        return super(LegacyAdapter, self).init_poolmanager(*args, **kwargs)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

session = requests.Session()
session.mount("https://", LegacyAdapter())
session.verify = False

layers = [1, 4]
base_url = "https://ags.iplan.gov.il/arcgisiplan/rest/services/PlanningPublic/Xplan/MapServer"

for lid in layers:
    print(f"\n--- Layer {lid} ---")
    try:
        r = session.get(f"{base_url}/{lid}?f=json", timeout=15)
        if r.status_code == 200:
            data = r.json()
            fields = [f['name'] for f in data.get('fields', [])]
            print(f"Fields: {fields}")
        else:
            print(f"Error {r.status_code}")
    except Exception as e:
        print(f"Failed: {e}")
