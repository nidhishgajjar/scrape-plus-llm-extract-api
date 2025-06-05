from fastapi import APIRouter, HTTPException
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from firecrawl import FirecrawlApp
from app.services.scraper import scroll_to_bottom, save_markdown_to_file
from app.services.markdown_converter import convert_html_to_markdown
from app.services.llm_processor import LLMProcessor, ModelType
from typing import Dict, Any, Optional
from pydantic import BaseModel
import os
import asyncio
import httpx
from app.config import get_settings

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Health check", "status": "ok"}

@router.get("/health")
async def health():
    return {"status": "healthy", "message": "Service is running"}

@router.get("/scrape")
async def scrape_url(url: str):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": 1280, "height": 720})
            
            try:
                # Set a page navigation timeout
                await page.goto(url, timeout=5000)  # 5 second timeout
                
                # Scroll with timeout
                await asyncio.wait_for(
                    scroll_to_bottom(page),
                    timeout=15  # 15 second timeout for scrolling
                )
                
                html_content = await page.content()
            except PlaywrightTimeout:
                return {
                    "status": "error",
                    "error": "Page loading timed out",
                    "partial_content": await page.content() if page else None
                }
            finally:
                await browser.close()

        markdown_content = await convert_html_to_markdown(html_content)
        
        if markdown_content is None:
            raise HTTPException(status_code=500, detail="Failed to convert HTML to markdown")
        
        settings = get_settings()
        if settings.DEBUG_MODE:
            file_path = save_markdown_to_file(markdown_content)
            saved_to = file_path
        else:
            saved_to = "File saving disabled in production mode"
        
        return {
            "status": "success",
            "markdown": markdown_content,
            "saved_to": saved_to
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Operation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

class ExtractRequest(BaseModel):
    url: str
    extraction_prompt: str
    output_format: Dict[str, Any]
    model: ModelType = "gpt-4o-mini"
    use_inhouse_scraping: bool = False

@router.post("/scrape/llm-extract")
async def scrape_and_extract(request: ExtractRequest):
    try:
        settings = get_settings()
        
        if request.use_inhouse_scraping:
            # Use in-house Playwright scrolling approach
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(viewport={"width": 1280, "height": 720})
                
                try:
                    # Set a page navigation timeout
                    await page.goto(request.url, timeout=30000)  # 30 second timeout
                    
                    # Scroll with timeout
                    await asyncio.wait_for(
                        scroll_to_bottom(page),
                        timeout=15  # 15 second timeout for scrolling
                    )
                    
                    html_content = await page.content()
                except PlaywrightTimeout:
                    await browser.close()
                    return {
                        "status": "error",
                        "error": "Page loading timed out",
                        "partial_content": await page.content() if page else None
                    }
                finally:
                    await browser.close()

            markdown_content = await convert_html_to_markdown(html_content)
            
            if markdown_content is None:
                raise HTTPException(status_code=500, detail="Failed to convert HTML to markdown")
        else:
            # Use Firecrawl approach
            app = FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY'))
            response = app.scrape_url(
                url=request.url,
                params={'formats': ['markdown']}
            )
            markdown_content = response['markdown']
        
        # save the markdown to a file
        file_path = "File saving disabled in production mode"
        if settings.DEBUG_MODE:
            file_path = save_markdown_to_file(markdown_content)
            print(f"Markdown saved to {file_path}")
        else:
            print("File saving disabled in production mode")
        
        # Process with LLM
        llm_processor = LLMProcessor(model=request.model)
        try:
            extracted_data, extraction_file_path = await asyncio.wait_for(
                llm_processor.extract_information(
                    content=markdown_content,
                    extraction_prompt=request.extraction_prompt,
                    output_format=request.output_format
                ),
                timeout=150  # 2.5 minute timeout for LLM processing
            )
            
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "error": "LLM processing timed out after 150 seconds",
                "markdown_file": file_path,
                "scraping_method": "inhouse_playwright" if request.use_inhouse_scraping else "firecrawl"
            }
        
        return {
            "status": "success",
            "extracted_data": extracted_data,
            "extraction_file": extraction_file_path,
            "markdown_file": file_path,
            "model_used": request.model,
            "scraping_method": "inhouse_playwright" if request.use_inhouse_scraping else "firecrawl",
            "debug_mode": settings.DEBUG_MODE
        }
    except httpx.RequestError as e:
        return {
            "status": "error",
            "error": f"HTTP request error: {str(e)}"
        }    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        } 