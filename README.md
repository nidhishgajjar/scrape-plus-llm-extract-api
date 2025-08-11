# Scrape-Plus-LLM-Extract-API

A FastAPI-based web service that combines web scraping with LLM-powered data extraction from multiple providers.

## Supported Models

- **OpenAI**: GPT-4o, GPT-4o-mini
- **Google**: Gemini 2.5 Flash, Gemini 2.5 Pro
- **TogetherAI**: GPT-OSS-20B, GPT-OSS-120B
- **Anthropic**: Claude Sonnet 4, Claude 3.7 Sonnet, Claude 3.5 Haiku
- **xAI**: Grok 4, Grok 4 Latest

## Example Usage for LLM Extraction endpoint

### Basic Usage
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

### Using TogetherAI Models
```bash
curl -X POST "http://localhost:8007/scrape/llm-extract" \
-H "Content-Type: application/json" \
-d '{
    "url": "https://example.com",
    "extraction_prompt": "Extract company information",
    "output_format": {"company_name": "string", "description": "string"},
    "model": "gpt-oss-120b",
    "use_inhouse_scraping": true,
    "delay_page_load": 3000
}'
```

### Using Anthropic Claude Models
```bash
curl -X POST "http://localhost:8007/scrape/llm-extract" \
-H "Content-Type: application/json" \
-d '{
    "url": "https://example.com/jobs",
    "extraction_prompt": "Extract all job posting URLs and their titles from this page.",
    "output_format": {
        "job_postings": [
            {
                "title": "Job Title",
                "url": "Job URL",
                "location": "Job Location"
            }
        ]
    },
    "model": "claude-sonnet-4-20250514"
}'
```

### Using Google Gemini Models
```bash
curl -X POST "http://localhost:8007/scrape/llm-extract" \
-H "Content-Type: application/json" \
-d '{
    "url": "https://example.com/articles",
    "extraction_prompt": "Extract article titles and publication dates.",
    "output_format": {
        "articles": [
            {
                "title": "Article Title",
                "date": "Publication Date"
            }
        ]
    },
    "model": "gemini-2.5-pro"
}'
```

### Using xAI Grok Models
```bash
curl -X POST "http://localhost:8007/scrape/llm-extract" \
-H "Content-Type: application/json" \
-d '{
    "url": "https://example.com/products",
    "extraction_prompt": "Extract product names, prices, and availability status.",
    "output_format": {
        "products": [
            {
                "name": "Product Name",
                "price": "Product Price",
                "available": true
            }
        ]
    },
    "model": "grok-4",
    "use_inhouse_scraping": true,
    "delay_page_load": 7000
}'
```

### Using In-house Scraping (Playwright) with Custom Delay
```bash
curl -X POST "http://localhost:8007/scrape/llm-extract" \
-H "Content-Type: application/json" \
-d '{
    "url": "https://example.com",
    "extraction_prompt": "Extract data...",
    "output_format": {...},
    "model": "gpt-4o-mini",
    "use_inhouse_scraping": true,
    "delay_page_load": 3000
}'
```

### Using Firecrawl with Custom Delay
```bash
curl -X POST "http://localhost:8007/scrape/llm-extract" \
-H "Content-Type: application/json" \
-d '{
    "url": "https://example.com",
    "extraction_prompt": "Extract data...",
    "output_format": {...},
    "model": "gpt-4o-mini",
    "use_inhouse_scraping": false,
    "delay_page_load": 3000
}'
```

## Environment Variables Required

- `OPENAI_API_KEY` - For OpenAI models
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` - For Gemini models
- `ANTHROPIC_API_KEY` - For Anthropic Claude models
- `XAI_API_KEY` - For xAI Grok models
- `FIRECRAWL_API_KEY` - For external scraping (optional)
- `DEBUG_MODE` - Enable file saving and verbose logging

## API Request Fields

- `url`: The webpage URL to scrape
- `extraction_prompt`: Instructions for what data to extract
- `output_format`: Expected JSON structure for extracted data
- `model`: LLM model to use for extraction
- `use_inhouse_scraping`: Whether to use Playwright (true) or Firecrawl (false)
- `delay_page_load`: Page load delay in milliseconds for both in-house scraping (Playwright) and Firecrawl (default: 3000ms)