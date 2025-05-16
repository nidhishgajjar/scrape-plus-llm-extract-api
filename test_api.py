import requests
import json
import os

# API endpoint (change if testing against deployed version)
API_URL = "https://scrape-plus-llm-extract-api.onrender.com/scrape/llm-extract"

# Sample request
payload = {
    "url": "https://emplois.legroupemaurice.com/fr/postes",  # Test URL
    "extraction_prompt": "Extract all job posting URLs from this page. If there are no job posting URLs, return null for job_posting_urls and explain why in extraction_notes.",
    "output_format": {
        "job_posting_urls": ["https://example.com/job1", "https://example.com/job2"],
        "url_patterns": "Describe any patterns in the job URLs",
        "extraction_notes": "Any additional notes about the extraction",
        "requires_javascript": False
    },
    "model": "gpt-4o-mini"
}

# Make the request
response = requests.post(API_URL, json=payload)

# Print the response
print(f"Status code: {response.status_code}")
print(json.dumps(response.json(), indent=2)) 