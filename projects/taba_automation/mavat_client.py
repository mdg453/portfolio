import logging
from datetime import datetime
from arcgis_client import ArcGISClient

logger = logging.getLogger("MavatClient")

class MavatClient:
    def __init__(self):
        self.arcgis = ArcGISClient()

    def get_plans_by_date_range(self, start_date: datetime, end_date: datetime):
        """
        Fetches plans from the Mavat / Planning Administration (Minhal HaTichnun)
        by filtering on the 'pl_date_advertise' (or 'last_update_date' as fallback) 
        within the given date range.
        """
        logger.info(f"Fetching Mavat plans from {start_date.date()} to {end_date.date()}")
        
        # Mavat (iplan) stores dates as epoch timestamps in milliseconds
        start_epoch = int(start_date.timestamp() * 1000)
        end_epoch = int(end_date.timestamp() * 1000)
        
        # We query the Plan Boundaries layer (Layer 1)
        # We use pl_date_advertise to represent "Plan Issuance / Publication Date" 
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        where_clause = f"pl_date_advertise >= date '{start_str}' AND pl_date_advertise <= date '{end_str}'"
        
        fields = "objectid,pl_name,pl_number,station_desc,last_update_date,pl_date_advertise"
        
        try:
            features = self.arcgis.query_layer(layer_id=1, where=where_clause, out_fields=fields)
            
            # Format the output to match RMIClient's structure so main.py can process them uniformly
            formatted_plans = []
            for feature in features:
                attrs = feature.get('attributes', {})
                # Convert epoch timestamp to readable date
                raw_date = attrs.get('last_update_date')
                if raw_date and isinstance(raw_date, (int, float)):
                    from datetime import datetime as dt
                    date_str = dt.fromtimestamp(raw_date / 1000).strftime('%d/%m/%y')
                else:
                    date_str = str(raw_date) if raw_date else 'Unknown'
                    
                formatted_plans.append({
                    'planNumber': attrs.get('pl_number'),
                    'mahut': attrs.get('pl_name'),
                    'status': attrs.get('station_desc'),
                    'statusDate': date_str,
                    'source': 'Mavat'
                })
                
            logger.info(f"Found {len(formatted_plans)} plans in Mavat.")
            return formatted_plans
            
        except Exception as e:
            logger.error(f"Failed to fetch plans from Mavat: {str(e)}")
            return []

if __name__ == '__main__':
    from datetime import timedelta
    logging.basicConfig(level=logging.INFO)
    client = MavatClient()
    end = datetime.now()
    start = end - timedelta(days=7)
    plans = client.get_plans_by_date_range(start, end)
    for p in plans[:5]:
        print(p)
