


# Example Usage for LLM Extraction endpoint

```bash
curl -X POST "http://localhost:8007/scrape/llm-extract" \
-H "Content-Type: application/json" \
-d '{
    "url": "https://openai.com/careers/search/",
    "extraction_prompt": "Extract individual job URLs for all open jobs in USA or Canada from this markdown content. Only include direct job posting URLs.",
    "output_format": {"job_urls": []},
    "model": "gpt-4o-mini"
}'
```