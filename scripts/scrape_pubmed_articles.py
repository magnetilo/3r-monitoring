#!/usr/bin/env python3
"""
Scrape PubMed articles by country for 3R monitoring and analysis.

This script downloads comprehensive metadata for publications from
specific countries to support research trend monitoring and prediction.
"""

import json
from pathlib import Path
import traceback
from datetime import datetime, timedelta
from tqdm import tqdm
from r3_monitoring.data.pubmed_fetcher import PubMedFetcher, CountryFilter, DatasetBuilder


# Configuration
OUTPUT_DIR = Path("data/pubmed_scraping")
EMAIL = "thilo@zosimolab.ch"  # REPLACE WITH YOUR EMAIL

# Countries to monitor
TARGET_COUNTRIES = ["switzerland"] #, "germany", "france"]

# Date ranges for analysis
ANALYSIS_PERIODS = [
    {"start": 2010, "end": 2025, "label": "recent"},
    # {"start": 2015, "end": 2019, "label": "historical"},
    # {"start": 2010, "end": 2014, "label": "baseline"}
]


def generate_periods(start_year: int, end_year: int, period_days: int = 30) -> list:
    """
    Generate time periods for PubMed queries.

    Args:
        start_year: Starting year
        end_year: Ending year (inclusive)  
        period_days: Number of days per period (default: 30 for ~monthly)
    
    Returns:
        List of period dictionaries with start_date, end_date, and label
    """
    periods = []

    # Start from beginning of start_year
    current_date = datetime(start_year, 1, 1)
    end_date_limit = datetime(end_year + 1, 1, 1)  # End of end_year
    now = datetime.now()

    period_number = 1

    while current_date < end_date_limit:
        # Calculate period end date
        period_end = current_date + timedelta(days=period_days - 1)

        # Don't go beyond the specified end year
        if period_end >= end_date_limit:
            period_end = end_date_limit - timedelta(days=1)

        # Don't go beyond current date
        if period_end > now:
            period_end = now

        # Skip if start date is in the future
        if current_date > now:
            break

        # Format dates for PubMed API
        start_date_str = current_date.strftime("%Y/%m/%d")
        end_date_str = period_end.strftime("%Y/%m/%d")

        # Create label based on period size
        if period_days <= 7:
            label = f"{current_date.strftime('%Y-W%U')}"  # Weekly
        elif period_days <= 31:
            label = f"{current_date.strftime('%Y-%m')}-{period_number:02d}"  # Monthly-like
        elif period_days <= 93:  # ~quarterly
            quarter = (current_date.month - 1) // 3 + 1
            label = f"{current_date.year}-Q{quarter}"
        else:
            label = f"{current_date.strftime('%Y')}-P{period_number:02d}"  # Yearly periods

        periods.append({
            "start_date": start_date_str,
            "end_date": end_date_str,
            "label": label,
            "year": current_date.year,
            "month": current_date.month,
            "period_days": period_days,
            "actual_days": (period_end - current_date).days + 1
        })

        # Move to next period
        current_date = period_end + timedelta(days=1)
        period_number += 1

        # Safety check to avoid infinite loops
        if period_number > 1000:
            print("Warning: Generated more than 1000 periods, stopping")
            break

    return periods


def search_pmids_by_periods(
        fetcher: PubMedFetcher, countries: list,
        start_year: int, end_year: int, period_days: int = 30
) -> dict:
    """Search PMIDs using time periods to avoid 9,999 limit."""

    # Generate time periods
    periods = generate_periods(start_year, end_year, period_days)
    all_pmids = []
    period_counts = {}

    period_type = "monthly" if period_days <= 31 else f"{period_days}-day"
    print(f"Searching {len(periods)} {period_type} periods from {start_year} to {end_year}")

    # Build base query without dates
    base_query_parts = []
    for country in countries:
        if country in CountryFilter.COUNTRIES:
            terms = CountryFilter.COUNTRIES[country]
            country_query = " OR ".join([f'"{term}"[Affiliation]' for term in terms])
            base_query_parts.append(f"({country_query})")

    base_query = " OR ".join(base_query_parts)

    # Search each period
    for period in tqdm(periods, desc="Searching periods", unit="period"):
        period_query = f"({base_query}) AND ({period['start_date']}[PDAT]:{period['end_date']}[PDAT])"

        try:
            period_pmids = fetcher.search_pmids(period_query, max_results=9000)
            period_counts[period['label']] = len(period_pmids)
            all_pmids.extend(period_pmids)

            if len(period_pmids) > 0:
                print(f"  {period['label']}: {len(period_pmids)} PMIDs ({period['actual_days']} days)")

        except Exception as e:
            print(f"  Error in {period['label']}: {e}")
            period_counts[period['label']] = 0

    # Remove duplicates while preserving order
    unique_pmids = list(dict.fromkeys(all_pmids))

    # Organize PMIDs by year-month for easier processing later
    pmids_by_month = {}
    pmids_by_year = {}  # Keep for backward compatibility
    for period in periods:
        if period['label'] in period_counts and period_counts[period['label']] > 0:
            year = period['year']
            month = f"{period['year']}-{period['month']:02d}"

            # Initialize containers
            if month not in pmids_by_month:
                pmids_by_month[month] = []
            if year not in pmids_by_year:
                pmids_by_year[year] = []

            # Get PMIDs for this specific period (approximate based on period order)
            start_idx = sum(period_counts.get(p['label'], 0) for p in periods if p['label'] < period['label'])
            end_idx = start_idx + period_counts[period['label']]
            period_pmids = all_pmids[start_idx:end_idx] if start_idx < len(all_pmids) else []

            pmids_by_month[month].extend(period_pmids)
            pmids_by_year[year].extend(period_pmids)

    # Remove duplicates within each month and year
    for month in pmids_by_month:
        pmids_by_month[month] = list(dict.fromkeys(pmids_by_month[month]))
    for year in pmids_by_year:
        pmids_by_year[year] = list(dict.fromkeys(pmids_by_year[year]))

    print("\nPeriod search complete:")
    print(f"  Total PMIDs found: {len(all_pmids)}")
    print(f"  Unique PMIDs: {len(unique_pmids)}")
    print(f"  Duplicates removed: {len(all_pmids) - len(unique_pmids)}")
    print(f"  Months covered: {sorted(pmids_by_month.keys())}")
    for month in sorted(pmids_by_month.keys()):
        print(f"    {month}: {len(pmids_by_month[month])} PMIDs")
    print(f"  Years covered: {sorted(pmids_by_year.keys())}")
    for year in sorted(pmids_by_year.keys()):
        print(f"    {year}: {len(pmids_by_year[year])} PMIDs")

    return {
        "pmids": unique_pmids,
        "pmids_by_month": pmids_by_month,
        "pmids_by_year": pmids_by_year,  # Keep for backward compatibility
        "period_counts": period_counts,
        "total_found": len(all_pmids),
        "unique_count": len(unique_pmids),
        "period_days": period_days
    }


def process_country_period(
        fetcher: PubMedFetcher,
        country: str,
        period: dict,
        max_articles: int = None
) -> dict:
    """
    Process publications for a specific country and time period using monthly search.

    Args:
        fetcher: PubMedFetcher instance
        country: Country key
        period: Dictionary with start, end, and label
        max_articles: Maximum articles to download (for testing)
    
    Returns:
        Summary dictionary
    """
    print(f"\n=== Processing {country.upper()} ({period['start']}-{period['end']}) ===")

    # Create output directory for country-specific search metadata
    country_dir = OUTPUT_DIR / f"search_metadata_{country}"
    country_dir.mkdir(parents=True, exist_ok=True)

    # Use shared metadata directory for all countries
    shared_metadata_dir = OUTPUT_DIR / "pubmed_metadata"
    shared_metadata_dir.mkdir(parents=True, exist_ok=True)

    # Check if PMIDs file already exists
    pmid_file = country_dir / f"pmids_{period['start']}_{period['end']}.json"
    if pmid_file.exists():
        print(f"Loading existing PMIDs from {pmid_file}")
        with open(pmid_file, 'r', encoding='utf-8') as f:
            pmid_data = json.load(f)
            pmids = pmid_data.get("pmids", [])
    else:
        # Search for PMIDs using period approach (30-day periods by default)
        search_results = search_pmids_by_periods(
            fetcher, [country], period['start'], period['end'], period_days=20
        )

        pmids = search_results["pmids"]

        # Limit results if specified
        if max_articles and len(pmids) > max_articles:
            pmids = pmids[:max_articles]
            print(f"Limited to {max_articles} PMIDs for testing")

        print(f"Processing {len(pmids)} PMIDs for metadata download")

        # Save PMID list with search details and monthly organization
        with open(pmid_file, 'w', encoding='utf-8') as f:
            json.dump({
                "search_method": "time_periods",
                "period_days": search_results["period_days"],
                "country": country,
                "year_range": f"{period['start']}-{period['end']}",
                "period_counts": search_results["period_counts"],
                "total_found": search_results["total_found"],
                "unique_count": search_results["unique_count"],
                "pmids_processed": len(pmids),
                "pmids": pmids,
                "pmids_by_month": search_results["pmids_by_month"],
                "pmids_by_year": search_results["pmids_by_year"],  # Keep for backward compatibility
                "search_timestamp": datetime.now().isoformat()
            }, f, indent=2)
    
        # Save individual month files for easier access
        for month, month_pmids in search_results["pmids_by_month"].items():
            month_file = country_dir / f"pmids_{month}.json"
            with open(month_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "country": country,
                    "month": month,
                    "pmid_count": len(month_pmids),
                    "pmids": month_pmids,
                    "search_timestamp": datetime.now().isoformat()
                }, f, indent=2)
            print(f"  Saved {len(month_pmids)} PMIDs for {month} to {month_file.name}")
    
        # Also save individual year files for backward compatibility
        for year, year_pmids in search_results["pmids_by_year"].items():
            year_file = country_dir / f"pmids_{year}.json"
            with open(year_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "country": country,
                    "year": year,
                    "pmid_count": len(year_pmids),
                    "pmids": year_pmids,
                    "search_timestamp": datetime.now().isoformat()
                }, f, indent=2)
            print(f"  Saved {len(year_pmids)} PMIDs for {year} to {year_file.name}")

    if not pmids:
        return {
            "country": country,
            "years": f"{period['start']}-{period['end']}",
            "found": 0,
            "total_existing_files": 0,
            "newly_downloaded": 0,
            "download_errors": 0
        }

    # Check for existing JSON metadata files and filter out already downloaded PMIDs
    # Look in both flat structure and yearly subdirectories
    existing_pmids = set()
    if shared_metadata_dir.exists():
        existing_files = list(shared_metadata_dir.rglob("*.json"))  # Recursive search
        existing_pmids = {f.stem for f in existing_files}

    # Filter out PMIDs that already have metadata files
    pmids_to_download = [pmid for pmid in pmids if pmid not in existing_pmids]

    if existing_pmids:
        print(f"Found {len(existing_pmids)} existing JSON metadata files")
        print(f"Downloading metadata for {len(pmids_to_download)}/{len(pmids)} articles (skipping {len(existing_pmids)} existing)")
    else:
        print(f"Downloading metadata for {len(pmids_to_download)} articles using batch API...")

    # Only download PMIDs that don't already exist
    if pmids_to_download:
        # Download and save JSON metadata files directly
        fetcher.download_batch_metadata(pmids_to_download, shared_metadata_dir, batch_size=200, show_progress=True)
        print(f"Downloaded JSON metadata for {len(pmids_to_download)} articles")
    else:
        print("All PMIDs already have JSON metadata files - skipping download")

    # Count existing JSON files (including yearly subdirectories)
    existing_json_files = list(shared_metadata_dir.rglob("*.json"))

    # Get year breakdown if available
    year_breakdown = {}
    if pmid_file.exists():
        try:
            with open(pmid_file, 'r', encoding='utf-8') as f:
                pmid_data = json.load(f)
                pmids_by_year = pmid_data.get("pmids_by_year", {})
                for year, year_pmids in pmids_by_year.items():
                    year_breakdown[year] = len(year_pmids)
        except:
            pass

    summary = {
        "country": country,
        "years": f"{period['start']}-{period['end']}",
        "found": len(pmids),
        "total_existing_files": len(existing_json_files),
        "newly_downloaded": len(pmids_to_download),
        "pmids_by_year": year_breakdown,
        "country_output_dir": str(country_dir),
        "shared_metadata_dir": str(shared_metadata_dir)
    }

    # Save summary
    summary_file = country_dir / f"summary_{period['start']}_{period['end']}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Completed {country} ({period['start']}-{period['end']}): {len(existing_json_files)} total JSON files ({len(existing_pmids)} existing, {len(pmids_to_download)} newly downloaded)")

    return summary


def main():
    """Main execution function."""
    print("=== PubMed Country Scraper ===")
    print(f"Target countries: {', '.join(TARGET_COUNTRIES)}")
    print(f"Analysis periods: {len(ANALYSIS_PERIODS)} periods")

    # Initialize fetcher
    fetcher = PubMedFetcher(email=EMAIL, tool_name="3R-Country-Monitoring")

    # Process all country-period combinations
    all_summaries = []

    for country in TARGET_COUNTRIES:
        for period in ANALYSIS_PERIODS:
            try:
                summary = process_country_period(
                    fetcher, country, period,
                    # max_articles=9000  # Limit for testing - remove for full run
                )
                all_summaries.append(summary)

            except Exception as e:
                traceback.print_exc()
                print(f"Failed to process {country} {period['label']}: {e}")
                continue

    # Save overall summary
    overall_summary_file = OUTPUT_DIR / "scraping_summary.json"
    with open(overall_summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "scraping_timestamp": datetime.now().isoformat(),
            "total_combinations": len(TARGET_COUNTRIES) * len(ANALYSIS_PERIODS),
            "completed": len(all_summaries),
            "summaries": all_summaries
        }, f, indent=2)

    # Print final summary
    print("\n=== Scraping Complete ===")
    print(f"Processed {len(all_summaries)} country-period combinations")

    total_found = sum(s["found"] for s in all_summaries)
    total_existing = sum(s.get("total_existing_files", 0) for s in all_summaries)
    total_newly_downloaded = sum(s.get("newly_downloaded", 0) for s in all_summaries)

    print(f"Total articles found: {total_found}")
    print(f"Total existing JSON files: {total_existing}")
    print(f"Total newly downloaded: {total_newly_downloaded}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
