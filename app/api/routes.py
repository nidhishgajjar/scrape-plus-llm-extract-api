from fastapi import APIRouter, HTTPException
from playwright.async_api import async_playwright
from firecrawl import FirecrawlApp
from app.services.scraper import scroll_to_bottom, save_markdown_to_file
from app.services.markdown_converter import convert_html_to_markdown
from app.services.llm_processor import LLMProcessor, ModelType
from typing import Dict, Any
from pydantic import BaseModel
import os
router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Hello World"}

@router.get("/scrape")
async def scrape_url(url: str):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            await page.goto(url)
            
            await scroll_to_bottom(page)
            
            html_content = await page.content()
            await browser.close()

        markdown_content = await convert_html_to_markdown(html_content)
        
        if markdown_content is None:
            raise HTTPException(status_code=500, detail="Failed to convert HTML to markdown")
        
        file_path = save_markdown_to_file(markdown_content)
        
        return {
            "status": "success",
            "markdown": markdown_content,
            "saved_to": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

class ExtractRequest(BaseModel):
    url: str
    extraction_prompt: str
    output_format: Dict[str, Any]
    model: ModelType = "gpt-4o-mini"

@router.post("/scrape/llm-extract")
async def scrape_and_extract(request: ExtractRequest):
    try:

        app = FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY'))

        response = app.scrape_url(url=request.url, params={
            'formats': [ 'markdown' ],
        })
        
        # save the markdown to a file
        file_path = save_markdown_to_file(response['markdown'])
        print(f"Markdown saved to {file_path}")
        
        # Process with LLM
        llm_processor = LLMProcessor(model=request.model)
        extracted_data, extraction_file_path = await llm_processor.extract_information(
            content=response['markdown'],
            extraction_prompt=request.extraction_prompt,
            output_format=request.output_format
        )
        
        return {
            "status": "success",
            "extracted_data": extracted_data,
            "extraction_file": extraction_file_path,
            "model_used": request.model
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 