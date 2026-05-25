import sqlite3
import pandas as pd
from datetime import datetime
import os

class DatabaseBuilder:
    def __init__(self, db_path='taba_tourism.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tourism_plots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_number TEXT,
                plan_name TEXT,
                city TEXT,
                plan_status TEXT,
                last_status_date DATE,
                total_plots_with_rights INTEGER,
                total_plots_for_tourism INTEGER,
                
                plot_number TEXT,
                tourism_use_type TEXT,
                is_exclusive_or_mixed TEXT,
                hotel_and_commercial_mix_type TEXT,
                
                rights_main_area REAL,
                rights_service_area REAL,
                total_rights_min REAL,
                total_rights_max REAL,
                
                guest_units_min INTEGER,
                guest_units_max INTEGER,
                
                building_height REAL,
                number_of_floors INTEGER,
                
                center_x_itm REAL,
                center_y_itm REAL,
                plot_designation TEXT,
                land_ownership TEXT,
                pdf_link TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Migration: add pdf_link and city columns to existing databases
        cursor.execute("PRAGMA table_info(tourism_plots)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        if 'pdf_link' not in existing_columns:
            cursor.execute('ALTER TABLE tourism_plots ADD COLUMN pdf_link TEXT')
        if 'city' not in existing_columns:
            cursor.execute('ALTER TABLE tourism_plots ADD COLUMN city TEXT')
        
        
        # Migration: plan_processing_log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plan_processing_log (
                plan_number TEXT PRIMARY KEY,
                plan_name TEXT,
                status TEXT,
                retry_count INTEGER DEFAULT 0,
                last_attempt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

    def upsert_plan_log(self, plan_number: str, plan_name: str, status: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT retry_count FROM plan_processing_log WHERE plan_number = ?", (plan_number,))
        row = cursor.fetchone()
        
        if row:
            retry_count = row[0]
            if status != 'SUCCESS':
                retry_count += 1
            cursor.execute('''
                UPDATE plan_processing_log 
                SET plan_name = ?, status = ?, retry_count = ?, last_attempt = CURRENT_TIMESTAMP
                WHERE plan_number = ?
            ''', (plan_name, status, retry_count, plan_number))
        else:
            cursor.execute('''
                INSERT INTO plan_processing_log (plan_number, plan_name, status, retry_count)
                VALUES (?, ?, ?, 0)
            ''', (plan_number, plan_name, status))
            
        conn.commit()
        conn.close()

    def get_failed_plans(self, max_retries: int = 3):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT plan_number, plan_name FROM plan_processing_log 
            WHERE status != 'SUCCESS' AND retry_count < ?
        ''', (max_retries,))
        failed = [{'plan_number': row[0], 'plan_name': row[1]} for row in cursor.fetchall()]
        conn.close()
        return failed

    def export_failed_plans_report(self, output_path='failed_plans_report.xlsx'):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM plan_processing_log WHERE status != 'SUCCESS'", conn)
        conn.close()
        
        if not df.empty:
            df = df.rename(columns={
                'plan_number': 'Plan Number',
                'plan_name': 'Plan Name',
                'status': 'Error Status',
                'retry_count': 'Retry Count',
                'last_attempt': 'Last Attempt Date'
            })
            
            try:
                df.to_excel(output_path, index=False)
                print(f"Failed plans report exported to {output_path}")
            except PermissionError:
                alt_path = output_path.replace('.xlsx', '_new.xlsx')
                df.to_excel(alt_path, index=False)
                print(f"WARNING: '{output_path}' is open (in Excel?). Saved to '{alt_path}' instead.")
        else:
            print("No failed plans to export (100% coverage!).")

    def insert_tourism_plot(self, plot_data: dict):
        """
        Inserts a single plot record into the database.
        `plot_data` should be a dictionary matching the columns.
        """
        columns = ', '.join(plot_data.keys())
        placeholders = ':' + ', :'.join(plot_data.keys())
        
        query = f'INSERT INTO tourism_plots ({columns}) VALUES ({placeholders})'
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query, plot_data)
        conn.commit()
        conn.close()

    def export_to_excel(self, output_path='tourism_database.xlsx'):
        """
        Exports the SQLite database to an Excel file matching the required schema.
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM tourism_plots", conn)
        conn.close()
        
        # Rename columns to match the requested English Excel schema
        rename_map = {
            'plan_number': 'Plan Number',
            'plan_name': 'Plan Name',
            'city': 'City',
            'plan_status': 'Plan Status',
            'last_status_date': 'Last Status Date',
            'total_plots_with_rights': 'Total Plots With Building Rights',
            'total_plots_for_tourism': 'Total Tourism Plots With Rights',
            
            'plot_number': 'Plot/Cell Number for Tourism Use',
            'tourism_use_type': 'Tourism Use Type (Hotel, Hostel, etc.)',
            
            'center_x_itm': 'Center X ITM',
            'center_y_itm': 'Center Y ITM',
            'plot_designation': 'Plot Designation in Plan',
            'pdf_link': 'PDF Link',
            
            # --- EMPTY COLUMNS AT THE END ---
            'guest_units_min': 'Guest Units Min',
            'guest_units_max': 'Guest Units Max',
            'building_height': 'Building Height (m)',
            'number_of_floors': 'Number of Floors',
            'land_ownership': 'Land Ownership',
            'is_exclusive_or_mixed': 'Exclusive or Mixed Use',
            'hotel_and_commercial_mix_type': 'Hotel and Commercial Mix Type',
            'rights_main_area': 'Main Rights Area',
            'rights_service_area': 'Service Rights Area',
            'total_rights_min': 'Total Rights Min',
            'total_rights_max': 'Total Rights Max'
        }
        
        # Only include columns that exist in the dataframe
        available_cols = [c for c in rename_map.keys() if c in df.columns]
        export_df = df[available_cols].rename(columns=rename_map)
        
        try:
            export_df.to_excel(output_path, index=False)
            print(f"Database exported to {output_path}")
        except PermissionError:
            # If the file is open in Excel, save to an alternative path
            alt_path = output_path.replace('.xlsx', '_new.xlsx')
            export_df.to_excel(alt_path, index=False)
            print(f"WARNING: '{output_path}' is open (in Excel?). Saved to '{alt_path}' instead.")
            print(f"Close the original file and rename '{alt_path}' if needed.")

if __name__ == '__main__':
    db = DatabaseBuilder()
    print("Database initialized.")
