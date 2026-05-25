import requests
import json
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger("RMIClient")

class RMIClient:
    def __init__(self):
        self.url = "https://apps.land.gov.il/TabaSearch/api/SerachPlans/GetPlans"
        self.headers = {"Content-Type": "application/json"}

    def get_plans_by_date_range(self, start_date: datetime, end_date: datetime):
        """
        Fetches plans from the Israel Land Authority (RMI) API
        given a start date and an end date.
        """
        all_plans = []
        current_day = start_date

        while current_day <= end_date:
            from_date = current_day.strftime("%Y-%m-%dT22:00:00.000Z")
            to_date = (current_day + timedelta(days=1)).strftime("%Y-%m-%dT22:00:00.000Z")
            
            logger.info(f"Fetching RMI plans for date range: {current_day.date()} to {(current_day + timedelta(days=1)).date()}")

            payload = {
                "planNumber": "",
                "gush": "",
                "chelka": "",
                "statuses": None,
                "planTypes": [72, 21, 1, 8, 9, 10, 12, 20, 62, 31, 41, 25, 22, 2, 11, 13, 61, 32, 74, 78, 77, 73, 76, 75, 80, 79, 40, 60, 71, 70, 67, 68, 69, 30, 50, 3],
                "fromStatusDate": from_date,
                "toStatusDate": to_date,
                "planTypesUsed": False,
            }

            try:
                response = requests.post(self.url, json=payload, headers=self.headers, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    day_plans = data.get("plansSmall", [])
                    if day_plans:
                        all_plans.extend(day_plans)
                        logger.info(f"Found {len(day_plans)} plans.")
                else:
                    logger.error(f"Error {response.status_code} fetching plans from RMI.")
            except Exception as e:
                logger.error(f"Request failed: {e}")

            current_day += timedelta(days=1)
            time.sleep(1) # Be nice to the server

        return all_plans
