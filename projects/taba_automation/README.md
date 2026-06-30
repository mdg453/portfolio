# Taba Automation

## Overview
An automated data pipeline designed to extract, parse, and analyze urban planning (TABA) data. The system automatically navigates complex government planning websites, retrieves relevant documents, and structures the unstructured PDF data into a queryable database.

## Technical Details
- **Tech Stack:** Python, Playwright, ArcGIS API
- **Core Functionality:** Automated web scraping, PDF text extraction, and geospatial data tagging.

## Interesting Concept
The integration of ArcGIS allows the pipeline to not just extract textual data about urban plans, but to map them geographically, enabling spatial analysis of urban development trends over time.

## Key Challenge
**Dynamic Content Loading:** The government website (Mavat) heavily relied on dynamic asynchronous loading and complex DOM structures, causing traditional scrapers to fail sporadically. 
*Solution:* Transitioned to Playwright to handle SPA behavior natively, utilizing robust waiting mechanisms and specific DOM element locators to ensure stable and reliable data extraction.
