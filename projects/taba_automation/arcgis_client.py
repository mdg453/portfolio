import requests
import urllib3
import logging
import ssl
from requests.adapters import HTTPAdapter

# Custom adapter to handle legacy SSL/TLS ciphers on some Israeli gov servers
class LegacyAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        # Disable hostname checking and certificate validation for legacy servers
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        # Allow lower security level for legacy certificates
        context.set_ciphers("DEFAULT@SECLEVEL=1")
        kwargs["ssl_context"] = context
        return super(LegacyAdapter, self).init_poolmanager(*args, **kwargs)

# Disable SSL warnings since iplan has legacy TLS
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ArcGISClient:
    def __init__(self):
        self.base_url = "https://ags.iplan.gov.il/arcgisiplan/rest/services/PlanningPublic/Xplan/MapServer"
        self.session = requests.Session()
        self.session.verify = False
        # Mount the legacy adapter
        self.session.mount("https://", LegacyAdapter())

    def query_layer(self, layer_id, where="1=1", out_fields="*", return_geometry=False):
        """
        Query a specific layer in the ArcGIS MapServer.
        Layer 1: Plan Boundaries
        Layer 4: Land Uses (Plots)
        """
        url = f"{self.base_url}/{layer_id}/query"
        
        params = {
            'f': 'json',
            'where': where,
            'outFields': out_fields,
            'returnGeometry': 'true' if return_geometry else 'false',
            'outSR': '2039', # Israel New Grid (ITM)
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'features' in data:
                return data['features']
            else:
                logging.warning(f"No features found or error in response for layer {layer_id}: {data}")
                return []
        except Exception as e:
            logging.error(f"Failed to query layer {layer_id}: {str(e)}")
            return []

    def get_plans(self, limit=100):
        """
        Fetch plans from Layer 1.
        """
        # pl_name: Plan Name, pl_number: Plan Number, station_desc: Status, last_update_date: Last Status Date
        fields = "objectid,pl_name,pl_number,station_desc,last_update_date"
        features = self.query_layer(layer_id=1, where="1=1", out_fields=fields)
        return features[:limit]

    def get_land_uses_for_plan(self, plan_number):
        """
        Fetch plots/land uses for a specific plan from Layer 4.
        """
        where_clause = f"pl_number = '{plan_number}'"
        
        # pl_number: Plan Num, num: Plot/Cell Num, mavat_name: Designation
        fields = "objectid,pl_number,num,mavat_name"
        
        features = self.query_layer(layer_id=4, where=where_clause, out_fields=fields, return_geometry=True)
        return features
        
    def get_plan_url(self, plan_number: str) -> str:
        """
        Queries the Plan Boundaries layer (Layer 1) to get the direct Mavat URL (pl_url) for a plan.
        """
        where_clause = f"pl_number='{plan_number}'"
        features = self.query_layer(layer_id=1, where=where_clause, out_fields="pl_url")
        if features:
            url = features[0].get('attributes', {}).get('pl_url')
            if url:
                return url
        return f"https://mavat.iplan.gov.il/SV4/1/{plan_number}"

if __name__ == '__main__':
    client = ArcGISClient()
    logging.basicConfig(level=logging.INFO)
    plans = client.get_plans(limit=5)
    print(f"Fetched {len(plans)} plans.")
    if plans:
        sample_plan = plans[0]['attributes']['NUMR']
        print(f"Fetching plots for plan: {sample_plan}")
        plots = client.get_land_uses_for_plan(sample_plan)
        print(f"Found {len(plots)} plots for this plan.")
