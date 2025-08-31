#!/bin/bash

# Test script for scrape extract endpoint with both OSS models

URL="https://workwithus.circlek.com/global/en/job/R521911/Customer-Service-Representative"
ENDPOINT="http://localhost:9000/scrape/llm-extract"

EXTRACTION_PROMPT="You are an expert job posting analyzer. Your task is to extract specific information from a job posting while preserving the original content and formatting. Please extract and structure the following information:

1. Job Title: Extract the complete, official job title.

2. Job Details: Extract the full job description and requirements. Preserve all original formatting using markdown. Include:
   - Main responsibilities
   - Required qualifications
   - Preferred qualifications
   - Any other relevant details
   Format this as a well-structured markdown document, maintaining the original hierarchy and emphasis.

3. Additional Information: Extract these specific data points:
   - Work Model: Identify if position is hybrid or remote
   - Employment Type: Categorize as one of: Full-time, Part-time, Contract, Internship, Volunteer
   - Location: Extract only city and state (e.g., \"San Francisco, CA\"). If remote, specify \"Remote\"
   - Country Code: Extract the country code from location (e.g., \"CA\" for Canada, \"US\" for United States). If remote, use the country code from the job posting or default to \"US\"
   - Seniority Level: Analyze job requirements and responsibilities to determine seniority level as either entry_level, mid_level, or senior_level. Use these guidelines:
     * entry_level: 0-2 years experience, basic skills, junior positions
     * mid_level: 3-5 years experience, intermediate skills, lead responsibilities
     * senior_level: 5+ years experience, advanced skills, strategic responsibilities
   - Salary Information:
     - Minimum salary (as integer)
     - Maximum salary (as integer)
     - Currency (only USD or CAD)
     - Payment period (hour, day, week, month, or year)
   - Benefits: List all mentioned benefits

Rules:
- Do not modify or paraphrase the original content
- Preserve all formatting using markdown
- For salary and benefits, only include explicitly stated information
- Return numeric values for salary without currency symbols or commas
- Categorize employment type using only the provided options
- Seniority level MUST be assigned based on the job description, even if not explicitly stated
- Format location as \"City, State\" or \"Remote\" only
- Country code should be either \"CA\" or \"US\" based on the location or job posting context
- Include ONLY benefits that are explicitly mentioned in the posting"

echo "Testing with OSS 20B model..."
echo "=============================="

curl -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "'"$URL"'",
    "extraction_prompt": "'"$(echo "$EXTRACTION_PROMPT" | sed 's/"/\\"/g' | tr '\n' ' ')"'",
    "output_format": {
      "job_title": "",
      "job_details": "",
      "hybrid": false,
      "remote": false,
      "employment_type": "",
      "location": "",
      "country_code": "",
      "seniority_level": "",
      "salary": {
        "min": 0,
        "max": 0,
        "currency": "CAD",
        "period": ""
      },
      "benefits": []
    },
    "model": "gpt-oss-20b"
  }' | jq '.'

echo -e "\n\nTesting with OSS 120B model..."
echo "=============================="

curl -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "'"$URL"'",
    "extraction_prompt": "'"$(echo "$EXTRACTION_PROMPT" | sed 's/"/\\"/g' | tr '\n' ' ')"'",
    "output_format": {
      "job_title": "",
      "job_details": "",
      "hybrid": false,
      "remote": false,
      "employment_type": "",
      "location": "",
      "country_code": "",
      "seniority_level": "",
      "salary": {
        "min": 0,
        "max": 0,
        "currency": "CAD",
        "period": ""
      },
      "benefits": []
    },
    "model": "gpt-oss-120b"
  }' | jq '.'