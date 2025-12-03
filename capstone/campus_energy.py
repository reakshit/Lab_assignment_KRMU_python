import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import logging
from datetime import datetime

# --- Configuration ---
DATA_DIR = Path('data')
OUTPUT_DIR = Path('output')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Task 3: Object-Oriented Modeling ---

class MeterReading:
    """Represents a single energy consumption reading."""
    def __init__(self, timestamp: datetime, kwh: float):
        self.timestamp = timestamp
        self.kwh = kwh

    def __repr__(self):
        return f"MeterReading({self.timestamp.strftime('%Y-%m-%d %H:%M')}, {self.kwh:.2f} kWh)"

class Building:
    """Represents a building with its meter readings and analysis methods."""
    def __init__(self, name: str):
        self.name = name
        self.meter_readings: list[MeterReading] = []
        self._total_consumption = 0.0

    def add_reading(self, timestamp: datetime, kwh: float):
        """Adds a single reading and updates total consumption."""
        reading = MeterReading(timestamp, kwh)
        self.meter_readings.append(reading)
        self._total_consumption += kwh

    def calculate_total_consumption(self) -> float:
        """Returns the total consumption calculated incrementally."""
        return self._total_consumption

    def generate_report(self) -> dict:
        """Generates a summary dictionary for the building."""
        if not self.meter_readings:
            return {'building': self.name, 'total_kwh': 0.0, 'num_readings': 0}

        kwh_list = [r.kwh for r in self.meter_readings]

        return {
            'building': self.name,
            'total_kwh': self.calculate_total_consumption(),
            'mean_kwh': sum(kwh_list) / len(kwh_list),
            'min_kwh': min(kwh_list),
            'max_kwh': max(kwh_list),
            'num_readings': len(kwh_list)
        }

class BuildingManager:
    """Manages all Building objects and provides high-level campus analysis."""
    def __init__(self):
        self.buildings: dict[str, Building] = {}
        self.df_combined: pd.DataFrame = pd.DataFrame()

    def add_building(self, name: str):
        """Creates and adds a new Building object if it doesn't exist."""
        if name not in self.buildings:
            self.buildings[name] = Building(name)

    # --- Task 1: Data Ingestion and Validation ---
    def ingest_data(self):
        """
        Reads and validates multiple CSV files from the data directory.
        Combines them into a single, clean DataFrame.
        """
        logging.info(f"Starting data ingestion from {DATA_DIR}...")
        all_data = []

        # Check if data directory exists
        if not DATA_DIR.is_dir():
            logging.error(f"Data directory '{DATA_DIR}' not found. Please create it and add CSV files.")
            return

        for file_path in DATA_DIR.glob('*.csv'):
            building_name = file_path.stem.split('_')[0].capitalize() # e.g., 'campus_a_jan' -> 'Campus'
            month_name = file_path.stem.split('_')[-1].capitalize() # e.g., 'campus_a_jan' -> 'Jan'

            try:
                # Attempt to read CSV, skipping bad lines
                df = pd.read_csv(file_path, on_bad_lines='skip', parse_dates=['Timestamp'])

                # Check for required columns and rename if necessary (assuming ['Timestamp', 'kwh'] are the keys)
                if 'Timestamp' not in df.columns or 'kwh' not in df.columns:
                    logging.warning(f"File {file_path.name} skipped: Missing 'Timestamp' or 'kwh' column.")
                    continue

                # Add metadata columns
                df['Building'] = building_name
                df['Month'] = month_name
                df['File'] = file_path.name

                # Data Cleaning: Convert kwh to numeric, coerce errors to NaN, then drop NaNs
                df['kwh'] = pd.to_numeric(df['kwh'], errors='coerce')
                df.dropna(subset=['kwh'], inplace=True)

                # Validation: Ensure Timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
                    logging.warning(f"File {file_path.name}: 'Timestamp' column not in datetime format after initial read.")
                    continue

                all_data.append(df)
                logging.info(f"Successfully ingested and cleaned {file_path.name} ({len(df)} rows).")

            except FileNotFoundError:
                # Should not happen if using glob, but good practice
                logging.error(f"Missing file: {file_path.name}")
            except pd.errors.EmptyDataError:
                logging.warning(f"Empty file: {file_path.name}. Skipping.")
            except Exception as e:
                logging.error(f"Corrupt or unreadable file {file_path.name}: {e}")

        if all_data:
            self.df_combined = pd.concat(all_data, ignore_index=True)
            self.df_combined.set_index('Timestamp', inplace=True)
            logging.info(f"Combined DataFrame created with {len(self.df_combined)} total rows.")
            self._populate_building_objects()
        else:
            logging.error("No valid data files were processed. Combined DataFrame is empty.")


    def _populate_building_objects(self):
        """Populates the OOP model from the combined DataFrame."""
        if self.df_combined.empty:
            return

        for name in self.df_combined['Building'].unique():
            self.add_building(name)

        # Iterating over the DataFrame rows to populate the Building objects
        for index, row in self.df_combined.iterrows():
            building_name = row['Building']
            timestamp = index.to_pydatetime()
            kwh = row['kwh']
            self.buildings[building_name].add_reading(timestamp, kwh)
        logging.info(f"Populated {len(self.buildings)} Building objects.")


    # --- Task 2: Core Aggregation Logic ---

    def calculate_daily_totals(self) -> pd.DataFrame:
        """Calculates daily total consumption for all buildings."""
        if self.df_combined.empty: return pd.DataFrame()
        # Resample daily ('D') and sum the kwh, then unstack to get buildings as columns
        daily_totals = self.df_combined.groupby('Building')['kwh'].resample('D').sum().unstack(level=0)
        return daily_totals

    def calculate_weekly_aggregates(self) -> pd.DataFrame:
        """Calculates weekly total consumption for all buildings."""
        if self.df_combined.empty: return pd.DataFrame()
        # Resample weekly ('W') and sum the kwh
        weekly_totals = self.df_combined.groupby('Building')['kwh'].resample('W').sum().unstack(level=0)
        return weekly_totals

    def building_wise_summary(self) -> pd.DataFrame:
        """Creates a summary table per building (mean, min, max, total)."""
        summary_data = [b.generate_report() for b in self.buildings.values()]

        if not summary_data:
            return pd.DataFrame()

        df_summary = pd.DataFrame(summary_data)
        df_summary.set_index('building', inplace=True)
        return df_summary


    # --- Task 4: Visual Output with Matplotlib ---

    def generate_dashboard_visualization(self, daily_data: pd.DataFrame, summary_data: pd.DataFrame):
        """Generates a multi-chart visualization and saves it as dashboard.png."""
        if daily_data.empty or summary_data.empty:
            logging.error("Cannot generate visualization: Aggregation data is empty.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        fig.suptitle('Campus Energy Consumption Dashboard', fontsize=20, y=0.95)

        # 1. Trend Line – daily consumption over time for all buildings
        axes[0].set_title('Daily Energy Consumption Trend', fontsize=14)
        daily_data.plot(ax=axes[0], legend=True)
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Total kWh')
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend(title='Building')

        # 2. Bar Chart – compare average daily usage across buildings
        avg_daily_usage = daily_data.mean()
        axes[1].set_title('Average Daily Usage Comparison', fontsize=14)
        avg_daily_usage.sort_values(ascending=False).plot(kind='bar', ax=axes[1], color=plt.cm.viridis.colors)
        axes[1].set_xlabel('Building')
        axes[1].set_ylabel('Average Daily kWh')
        axes[1].tick_params(axis='x', rotation=0)
        axes[1].grid(axis='y', linestyle='--', alpha=0.6)

        # 3. Scatter Plot – plot peak-hour consumption (max_kwh) vs. total_kwh
        # Note: Peak hour analysis is approximated by the max reading.
        axes[2].set_title('Building Peak-Load (Max kWh) vs. Total Consumption', fontsize=14)

        # Map building names to numerical category for color/marker separation
        buildings = summary_data.index.tolist()
        summary_data['Building_ID'] = [buildings.index(b) for b in buildings]

        scatter = axes[2].scatter(
            summary_data['total_kwh'],
            summary_data['max_kwh'],
            c=summary_data['Building_ID'],
            cmap='plasma',
            s=summary_data['mean_kwh'] * 5, # Size marker by mean usage for insight
            alpha=0.8
        )

        # Add labels to the scatter points
        for i, txt in enumerate(buildings):
            axes[2].annotate(txt, (summary_data['total_kwh'].iloc[i], summary_data['max_kwh'].iloc[i]),
                             fontsize=9, alpha=0.7, ha='center')

        axes[2].set_xlabel('Total Campus Consumption (kWh)')
        axes[2].set_ylabel('Peak Load (Max Single Reading kWh)')
        axes[2].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_path = OUTPUT_DIR / 'dashboard.png'
        plt.savefig(output_path)
        logging.info(f"Dashboard visualization saved to {output_path}")

    # --- Task 5: Persistence and Executive Summary ---

    def generate_executive_summary(self, df_summary: pd.DataFrame) -> str:
        """Generates a concise written report from the analysis."""
        if df_summary.empty:
            return "Analysis not completed due to empty data."

        # Calculate campus-wide metrics
        total_campus_consumption = df_summary['total_kwh'].sum()

        # Find highest consuming building
        highest_consumer_name = df_summary['total_kwh'].idxmax()
        highest_consumer_kwh = df_summary['total_kwh'].max()

        # Identify peak load time (approximated by the timestamp with the single highest kwh reading)
        peak_reading = self.df_combined['kwh'].max()
        peak_time_row = self.df_combined[self.df_combined['kwh'] == peak_reading].iloc[0]
        peak_time = peak_time_row.name # The Timestamp index
        peak_time_building = peak_time_row['Building']

        # Determine overall daily/weekly trends (simple average change)
        daily_trends = self.calculate_daily_totals().diff().mean().mean() # Avg daily change across all buildings
        weekly_trends = self.calculate_weekly_aggregates().diff().mean().mean() # Avg weekly change across all buildings

        summary = f"""
EXECUTIVE ENERGY CONSUMPTION SUMMARY REPORT

Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Buildings Analyzed: {len(self.buildings)}
Data Coverage: {self.df_combined.index.min().strftime('%Y-%m-%d')} to {self.df_combined.index.max().strftime('%Y-%m-%d')}

1. TOTAL CAMPUS CONSUMPTION:
   The entire campus consumed a total of {total_campus_consumption:,.2f} kWh during the reporting period.

2. HIGHEST CONSUMING BUILDING:
   - Building: {highest_consumer_name}
   - Total Consumption: {highest_consumer_kwh:,.2f} kWh

3. PEAK LOAD TIME:
   - Peak Single Reading: {peak_reading:,.2f} kWh
   - Occurred at: {peak_time.strftime('%Y-%m-%d %H:%M')} in {peak_time_building}.

4. USAGE TRENDS:
   - Average Daily Change (Trend): {'Increased' if daily_trends > 0 else 'Decreased'} by {abs(daily_trends):.2f} kWh/day on average.
   - Average Weekly Change (Trend): {'Increased' if weekly_trends > 0 else 'Decreased'} by {abs(weekly_trends):.2f} kWh/week on average.
"""
        return summary.strip()

    def export_data(self, df_summary: pd.DataFrame, summary_text: str):
        """Exports the final processed dataset, summary stats, and executive report."""

        # Export 1: Final processed dataset
        cleaned_path = OUTPUT_DIR / 'cleaned_energy_data.csv'
        self.df_combined.to_csv(cleaned_path)
        logging.info(f"Exported cleaned dataset to {cleaned_path}")

        # Export 2: Summary statistics
        summary_path = OUTPUT_DIR / 'building_summary.csv'
        df_summary.to_csv(summary_path)
        logging.info(f"Exported summary stats to {summary_path}")

        # Export 3: Summary report
        summary_text_path = OUTPUT_DIR / 'summary.txt'
        with open(summary_text_path, 'w') as f:
            f.write(summary_text)
        logging.info(f"Exported executive summary to {summary_text_path}")

        print("\n" + "="*50)
        print("EXECUTIVE SUMMARY PREVIEW")
        print("="*50)
        print(summary_text)
        print("="*50 + "\n")


def run_pipeline():
    """Main function to execute the entire capstone pipeline."""
    manager = BuildingManager()

    # Task 1: Data Ingestion and OOP Population
    manager.ingest_data()

    if manager.df_combined.empty:
        logging.error("Pipeline terminated: No data available for analysis.")
        return

    # Task 2: Core Aggregation Logic
    df_daily_totals = manager.calculate_daily_totals()
    df_building_summary = manager.building_wise_summary()

    # Task 4: Visual Output
    manager.generate_dashboard_visualization(df_daily_totals, df_building_summary)

    # Task 5: Persistence and Executive Summary
    executive_summary = manager.generate_executive_summary(df_building_summary)
    manager.export_data(df_building_summary, executive_summary)

    logging.info("Capstone Project Pipeline finished successfully.")


if __name__ == '__main__':
    run_pipeline()
