import os
import logging
from arcgis_client import ArcGISClient
from document_fetcher import DocumentFetcher
from document_parser import DocumentParser
from database_builder import DatabaseBuilder
from rmi_client import RMIClient
from mavat_client import MavatClient
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TabaAutomation')

def run_pipeline(start_date=None, end_date=None, days_back=30):
    logger.info("Initializing system modules...")
    arcgis = ArcGISClient()
    fetcher = DocumentFetcher()
    parser = DocumentParser()
    db = DatabaseBuilder()
    rmi = RMIClient()
    mavat = MavatClient()
    
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=days_back)
    
    logger.info(f"Fetching plans from {start_date.date()} to {end_date.date()} from RMI...")
    rmi_plans = rmi.get_plans_by_date_range(start_date, end_date)
    
    logger.info(f"Fetching plans from {start_date.date()} to {end_date.date()} from Mavat...")
    mavat_plans = mavat.get_plans_by_date_range(start_date, end_date)
    
    # Merge and deduplicate new plans
    all_plans = rmi_plans + mavat_plans
    seen = set()
    plans = []
    for plan in all_plans:
        pnum = plan.get('planNumber') or plan.get('attributes', {}).get('pl_number')
        if pnum and pnum not in seen:
            seen.add(pnum)
            plans.append(plan)
            
    # Fetch previously failed plans to retry
    failed_plans = db.get_failed_plans(max_retries=3)
    logger.info(f"Found {len(failed_plans)} previously failed plans to retry.")
    
    for failed_plan in failed_plans:
        pnum = failed_plan['plan_number']
        if pnum not in seen:
            seen.add(pnum)
            # Create a mock plan dict so it can be processed
            plans.append({'planNumber': pnum, 'planName': failed_plan['plan_name']})
            
    logger.info(f"Total unique plans to process: {len(plans)}")
    
    for plan in plans:
        plan_num = plan.get('planNumber') or plan.get('attributes', {}).get('pl_number')
        plan_name = plan.get('mahut') or plan.get('planName') or plan.get('attributes', {}).get('pl_name')
        status = plan.get('status') or plan.get('statusString') or plan.get('attributes', {}).get('station_desc')
        last_date = plan.get('statusDate') or plan.get('attributes', {}).get('last_update_date')
        city = plan.get('cityText') or ''
        
        if not plan_num:
            continue
        
        # Fallback: if name or status or city are missing, try ArcGIS Layer 1
        if not plan_name or not status or not city:
            try:
                feats = arcgis.query_layer(layer_id=1, where=f"pl_number='{plan_num}'", out_fields="pl_name,station_desc,last_update_date,plan_area_name,pa_concat,jurstiction_area_name")
                if feats:
                    a = feats[0].get('attributes', {})
                    plan_name = plan_name or a.get('pl_name') or plan_num
                    status = status or a.get('station_desc') or 'Unknown'
                    city = city or a.get('plan_area_name') or a.get('pa_concat') or a.get('jurstiction_area_name') or ''
                    if not last_date:
                        raw = a.get('last_update_date')
                        if raw and isinstance(raw, (int, float)):
                            last_date = datetime.fromtimestamp(raw / 1000).strftime('%d/%m/%y')
            except Exception:
                pass
            
        logger.info(f"Processing Plan: {plan_num} ({plan_name})")
        
        # 1. Get Land Uses/Plots for this plan
        plots = arcgis.get_land_uses_for_plan(plan_num)
        
        # 2. Fetch Document
        pdf_path = fetcher.fetch_plan_instructions(plan_num)
        
        is_tourism = False
        parsed_data = {}
        
        if pdf_path:
            # 3. Parse Document
            logger.info(f"Parsing document for {plan_num}")
            parsed_data = parser.parse_pdf(pdf_path)
            
            # Application Logic: If keywords found or hotel mentioned in Table 5
            if parsed_data.get('has_tourism_keywords') or parsed_data.get('has_hotel_in_table_5'):
                is_tourism = True
                
            db.upsert_plan_log(plan_num, plan_name, 'SUCCESS')
        else:
            logger.warning(f"No document available for {plan_num}. Skipping deep analysis.")
            db.upsert_plan_log(plan_num, plan_name, 'FAILED_DOCUMENT')
        
        # 4. Insert into DB if it's related to tourism
        if is_tourism:
            logger.info(f"*** Found Tourism Plan: {plan_num} ***")
            
            # Only include plots whose designation is relevant to the Ministry of Tourism.
            # The key rule: the designation must contain 'תיירות' OR a direct hospitality term.
            # This excludes pure residential/commercial/employment designations.
            TOURISM_DESIGNATIONS = ['תיירות','משולב', 'מלונ', 'מיוחד', 'מלון', 'אכסניה', 'כפר נופש', 'אירוח', 'קייטנ', 'צימר']
            
            for plot in plots:
                p_attrs = plot.get('attributes', {})
                mavat_name = p_attrs.get('mavat_name', '') or ''
                
                # Only insert plots whose designation explicitly involves tourism/hospitality
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

                # Prepare plot data to schema
                plot_data = {
                    'plan_number': plan_num,
                    'plan_name': plan_name,
                    'city': city,
                    'plan_status': status,
                    'last_status_date': last_date,
                    'total_plots_with_rights': len(plots),
                    'total_plots_for_tourism': len(plots),
                    
                    'plot_number': p_attrs.get('num'),
                    'tourism_use_type': p_attrs.get('mavat_name'),
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
                
    # 5. Finally export to Excel
    logger.info("Exporting database to Excel...")
    db.export_to_excel()
    
    # 6. Export Failed Plans Report
    logger.info("Exporting failed plans report...")
    db.export_failed_plans_report()
    
    logger.info("Pipeline complete.")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="TABA Tourism Automation Pipeline")
    parser.add_argument('--days', type=int, default=30, help="Number of days back to fetch plans from (default: 30)")
    parser.add_argument('--start', type=str, help="Start date in YYYY-MM-DD format (overrides --days)")
    parser.add_argument('--end', type=str, help="End date in YYYY-MM-DD format (default: today)")
    
    args = parser.parse_args()
    
    if args.start:
        try:
            start_date = datetime.strptime(args.start, '%Y-%m-%d')
            end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else datetime.now()
            run_pipeline(start_date=start_date, end_date=end_date)
        except ValueError:
            logger.error("Invalid date format. Please use YYYY-MM-DD.")
    else:
        run_pipeline(days_back=args.days)
