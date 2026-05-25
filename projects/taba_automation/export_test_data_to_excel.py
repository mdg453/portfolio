import os
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

from document_parser import DocumentParser
from database_builder import DatabaseBuilder
from arcgis_client import ArcGISClient

def extract_plan_num(filename):
    # Try to extract the plan number from strings like "plan_101-1253475_horhaot.pdf" or "504-1064039_הוראות.pdf"
    # Match patterns like XXX-XXXXXXX or similar
    match = re.search(r'(\d{3,4}[-_/]\d{4,7})', filename)
    if match:
        return match.group(1).replace('_', '-')
    
    # Try another pattern just in case
    match = re.search(r'plan_(.*?)_horhaot', filename)
    if match:
        return match.group(1).replace('_', '-')
        
    return "UNKNOWN"

def get_plan_metadata(plan_num, arcgis_client=None):
    """Get plan name, status, and date. Tries RMI first, then ArcGIS as fallback."""
    import requests
    
    # Try RMI API first
    url = "https://apps.land.gov.il/TabaSearch/api/SerachPlans/GetPlans"
    try:
        r = requests.post(url, json={"planNumber": plan_num}, headers={"Content-Type": "application/json"}, timeout=5)
        if r.status_code == 200:
            plans = r.json().get("plansSmall", [])
            if plans:
                p = plans[0]
                name = p.get('mahut') or ''
                status = p.get('status') or ''
                date = (p.get('statusDate') or '').strip()
                city = p.get('cityText') or ''
                if name and status:
                    return {'name': name, 'status': status, 'date': date or 'Unknown', 'city': city}
    except Exception:
        pass
    
    # Fallback: ArcGIS Layer 1
    if arcgis_client:
        try:
            features = arcgis_client.query_layer(
                layer_id=1, 
                where=f"pl_number='{plan_num}'", 
                out_fields="pl_name,station_desc,last_update_date,plan_area_name,pa_concat,jurstiction_area_name"
            )
            if features:
                attrs = features[0].get('attributes', {})
                raw_date = attrs.get('last_update_date')
                if raw_date and isinstance(raw_date, (int, float)):
                    from datetime import datetime as dt
                    date_str = dt.fromtimestamp(raw_date / 1000).strftime('%d/%m/%y')
                else:
                    date_str = 'Unknown'
                city = attrs.get('plan_area_name') or attrs.get('pa_concat') or attrs.get('jurstiction_area_name') or ''
                return {
                    'name': attrs.get('pl_name') or plan_num,
                    'status': attrs.get('station_desc') or 'Unknown',
                    'date': date_str,
                    'city': city
                }
        except Exception:
            pass
    
    return {'name': plan_num, 'status': "Unknown", 'date': "Unknown", 'city': ""}

def main():
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'test_data')
    test_data_dir = os.path.abspath(test_data_dir)
    
    if not os.path.exists(test_data_dir):
        print(f"ERROR: test_data directory not found at {test_data_dir}")
        return
        
    pdf_files = [f for f in os.listdir(test_data_dir) if f.lower().endswith('.pdf')]
    
    parser = DocumentParser()
    db = DatabaseBuilder()
    arcgis = ArcGISClient()
    
    # Clear existing db for this test run so we only see test data
    if os.path.exists('taba_tourism.db'):
        try:
            os.remove('taba_tourism.db')
        except:
            pass
    
    # Re-initialize DB
    db = DatabaseBuilder()
    
    processed_plans = set()
    
    for pdf_file in sorted(pdf_files):
        plan_num = extract_plan_num(pdf_file)
        if plan_num in processed_plans:
            continue
            
        pdf_path = os.path.join(test_data_dir, pdf_file)
        print(f"Processing {pdf_file} (Plan: {plan_num})")
        
        result = parser.parse_pdf(pdf_path)
        is_tourism = result.get('has_tourism_keywords') or result.get('has_hotel_in_table_5')
        
        if is_tourism and plan_num != "UNKNOWN":
            print(f"  -> Found Tourism. Fetching metadata and ArcGIS data for {plan_num}...")
            meta = get_plan_metadata(plan_num, arcgis_client=arcgis)
            plots = arcgis.get_land_uses_for_plan(plan_num)
            
            TOURISM_DESIGNATIONS = ['תיירות', 'מלונ', 'מלון', 'אכסניה', 'כפר נופש', 'אירוח', 'קייטנ', 'צימר']
            
            for plot in plots:
                p_attrs = plot.get('attributes', {})
                mavat_name = p_attrs.get('mavat_name', '') or ''
                
                if not any(kw in mavat_name for kw in TOURISM_DESIGNATIONS):
                    continue
                    
                geom = plot.get('geometry', {})
                
                # Try to find centroid
                x = geom.get('x') if 'x' in geom else None
                y = geom.get('y') if 'y' in geom else None
                
                if 'rings' in geom and geom['rings']:
                    pts = geom['rings'][0]
                    x = sum(p[0] for p in pts) / len(pts)
                    y = sum(p[1] for p in pts) / len(pts)
                
                # Determine exclusive or mixed
                is_mixed = 'מעורב' in mavat_name or 'משולב' in mavat_name or ('תיירות' in mavat_name and ('מסחר' in mavat_name or 'מגורים' in mavat_name or 'תעסוקה' in mavat_name))
                exclusive_or_mixed = 'מעורב' if is_mixed else 'בלעדי'
                
                plot_data = {
                    'plan_number': plan_num,
                    'plan_name': meta['name'],
                    'city': meta.get('city', ''),
                    'plan_status': meta['status'],
                    'last_status_date': meta['date'],
                    'total_plots_with_rights': len(plots),
                    'total_plots_for_tourism': len(plots),
                    
                    'plot_number': p_attrs.get('num'),
                    'tourism_use_type': mavat_name,
                    'is_exclusive_or_mixed': exclusive_or_mixed,
                    'hotel_and_commercial_mix_type': '',
                    
                    'rights_main_area': 0.0,
                    'rights_service_area': 0.0,
                    'total_rights_min': 0.0,
                    'total_rights_max': 0.0,
                    
                    'guest_units_min': 0,
                    'guest_units_max': 0,
                    
                    'building_height': 0.0,
                    'number_of_floors': 0,
                    
                    'center_x_itm': x,
                    'center_y_itm': y,
                    'plot_designation': p_attrs.get('mavat_name'),
                    'land_ownership': '',
                    'pdf_link': arcgis.get_plan_url(plan_num)
                }
                
                db.insert_tourism_plot(plot_data)
            processed_plans.add(plan_num)
            
    print("Exporting database to Excel...")
    db.export_to_excel()
    print("Done! Check tourism_database.xlsx")

if __name__ == '__main__':
    main()
