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
import time
import traceback
from app.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

@router.get("/")
async def root():
    return {"message": "Health check", "status": "ok"}

@router.get("/health")
async def health():
    return {"status": "healthy", "message": "Service is running"}

@router.get("/scrape")
async def scrape_url(url: str):
    start_time = time.time()
    request_id = f"{int(time.time() * 1000)}"
    
    logger.info(f"[{request_id}] ===== SCRAPE REQUEST RECEIVED =====")
    logger.info(f"[{request_id}] URL: {url}")
    
    try:
        logger.info(f"[{request_id}] Starting Playwright scraping...")
        scrape_start = time.time()
        
        async with async_playwright() as p:
            logger.debug(f"[{request_id}] Launching browser...")
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(
                viewport={"width": 1280, "height": 720}
            )
            
            try:
                # Set a page navigation timeout
                logger.info(f"[{request_id}] Navigating to URL with 5s timeout...")
                await page.goto(url, timeout=5000)  # 5 second timeout
                logger.info(f"[{request_id}] Page loaded successfully")
                
                # Scroll with timeout
                logger.debug(f"[{request_id}] Starting page scrolling...")
                await asyncio.wait_for(
                    scroll_to_bottom(page),
                    timeout=15  # 15 second timeout for scrolling
                )
                logger.debug(f"[{request_id}] Scrolling completed")
                
                logger.info(f"[{request_id}] Extracting page content...")
                html_content = await page.content()
                logger.info(f"[{request_id}] HTML content extracted (size: {len(html_content)} chars)")
                
            except PlaywrightTimeout as e:
                logger.error(f"[{request_id}] Page loading timed out: {str(e)}")
                return {
                    "status": "error",
                    "error": "Page loading timed out",
                    "raw_error": str(e),
                    "partial_content": await page.content() if page else None
                }
            finally:
                await browser.close()
                logger.debug(f"[{request_id}] Browser closed")

        logger.info(f"[{request_id}] Converting HTML to Markdown...")
        markdown_content = await convert_html_to_markdown(html_content)
        
        if markdown_content is None:
            logger.error(f"[{request_id}] Failed to convert HTML to markdown")
            raise HTTPException(status_code=500, detail="Failed to convert HTML to markdown")
        
        scrape_time = time.time() - scrape_start
        logger.info(f"[{request_id}] Markdown conversion completed (size: {len(markdown_content)} chars)")
        
        settings = get_settings()
        if settings.DEBUG_MODE:
            file_path = save_markdown_to_file(markdown_content)
            saved_to = file_path
            logger.debug(f"[{request_id}] Markdown saved to: {file_path}")
        else:
            saved_to = "File saving disabled in production mode"
            logger.debug(f"[{request_id}] File saving disabled in production mode")
        
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] ===== SCRAPE REQUEST COMPLETED SUCCESSFULLY =====")
        logger.info(f"[{request_id}] Total time: {total_time:.2f}s, Scraping: {scrape_time:.2f}s")
        
        return {
            "status": "success",
            "request_id": request_id,
            "markdown": markdown_content,
            "saved_to": saved_to,
            "processing_time": f"{total_time:.2f}s"
        }
        
    except asyncio.TimeoutError as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] Operation timed out after {total_time:.2f}s")
        logger.error(f"[{request_id}] Timeout error: {str(e)}")
        raise HTTPException(status_code=504, detail="Operation timed out")
        
    except HTTPException:
        raise
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] ===== SCRAPE REQUEST FAILED =====")
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        logger.error(f"[{request_id}] Raw error traceback: {traceback.format_exc()}")
        logger.error(f"[{request_id}] Total time before failure: {total_time:.2f}s")
        raise HTTPException(status_code=500, detail=str(e)) 

class ExtractRequest(BaseModel):
    url: str
    extraction_prompt: str
    output_format: Dict[str, Any]
    model: ModelType = "gpt-4o-mini"
    use_inhouse_scraping: bool = False

@router.post("/scrape/llm-extract")
async def scrape_and_extract(request: ExtractRequest):
    start_time = time.time()
    request_id = f"{int(time.time() * 1000)}"  # Simple request ID based on timestamp
    
    logger.info(f"[{request_id}] ===== NEW REQUEST RECEIVED =====")
    logger.info(f"[{request_id}] URL: {request.url}")
    logger.info(f"[{request_id}] Model: {request.model}")
    logger.info(f"[{request_id}] Scraping Method: {'InHouse Playwright' if request.use_inhouse_scraping else 'Firecrawl'}")
    logger.info(f"[{request_id}] Extraction Prompt: {request.extraction_prompt}")
    logger.info(f"[{request_id}] Output Format: {request.output_format}")
    
    try:
        settings = get_settings()
        logger.debug(f"[{request_id}] Debug Mode: {settings.DEBUG_MODE}")
        
        if request.use_inhouse_scraping:
            # Use in-house Playwright scrolling approach
            logger.info(f"[{request_id}] Starting InHouse Playwright scraping...")
            scrape_start = time.time()
            
            async with async_playwright() as p:
                logger.debug(f"[{request_id}] Launching Chromium browser...")
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--no-first-run',
                        '--no-default-browser-check',
                        '--disable-extensions'
                    ]
                )
                logger.debug(f"[{request_id}] Browser launched successfully")
                
                page = await browser.new_page(
                    viewport={"width": 1280, "height": 720},
                    locale='en-CA',
                    timezone_id='America/Toronto',
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                logger.debug(f"[{request_id}] New page created with custom settings")
                
                # Hide automation indicators
                await page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined,
                    });
                """)
                
                # Set realistic headers
                await page.set_extra_http_headers({
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-CA,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br'
                })
                
                try:
                    # Set a page navigation timeout
                    timeout_ms = int(os.environ.get('INHOUSE_SCRAPING_TIMEOUT', '30000'))
                    logger.info(f"[{request_id}] Navigating to URL with timeout: {timeout_ms}ms")
                    await page.goto(request.url, timeout=timeout_ms)
                    logger.info(f"[{request_id}] Successfully loaded page")
                    
                    # Small delay to appear more human-like
                    delay_ms = int(os.environ.get('INHOUSE_SCRAPING_DELAY', '7000'))
                    logger.debug(f"[{request_id}] Waiting {delay_ms}ms for page to settle...")
                    await page.wait_for_timeout(delay_ms)
                    
                    # # Scroll with timeout
                    # await asyncio.wait_for(
                    #     scroll_to_bottom(page),
                    #     timeout=15  # 15 second timeout for scrolling
                    # )
                    
                    logger.info(f"[{request_id}] Extracting page content...")
                    html_content = await page.content()
                    logger.info(f"[{request_id}] Successfully extracted HTML content (size: {len(html_content)} chars)")
                    
                except PlaywrightTimeout as e:
                    logger.error(f"[{request_id}] Playwright timeout error: {str(e)}")
                    await browser.close()
                    return {
                        "status": "error",
                        "error": "Page loading timed out",
                        "raw_error": str(e),
                        "partial_content": await page.content() if page else None
                    }
                except Exception as e:
                    logger.error(f"[{request_id}] Unexpected error during page load: {str(e)}")
                    logger.error(f"[{request_id}] Raw error traceback: {traceback.format_exc()}")
                    await browser.close()
                    raise
                finally:
                    await browser.close()
                    logger.debug(f"[{request_id}] Browser closed")

            logger.info(f"[{request_id}] Converting HTML to Markdown...")
            markdown_content = await convert_html_to_markdown(html_content)
            
            if markdown_content is None:
                logger.error(f"[{request_id}] Failed to convert HTML to markdown")
                raise HTTPException(status_code=500, detail="Failed to convert HTML to markdown")
            
            scrape_time = time.time() - scrape_start
            logger.info(f"[{request_id}] Scraping completed in {scrape_time:.2f}s, markdown size: {len(markdown_content)} chars")
            
        else:
            # Use Firecrawl approach
            logger.info(f"[{request_id}] Starting Firecrawl scraping...")
            scrape_start = time.time()
            
            try:
                app = FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY'))
                logger.debug(f"[{request_id}] Firecrawl initialized, making request...")
                
                response = app.scrape_url(
                    url=request.url,
                    params={'formats': ['markdown']}
                )
                markdown_content = response['markdown']
                
                scrape_time = time.time() - scrape_start
                logger.info(f"[{request_id}] Firecrawl scraping completed in {scrape_time:.2f}s, markdown size: {len(markdown_content)} chars")
                
            except Exception as e:
                logger.error(f"[{request_id}] Firecrawl scraping failed: {str(e)}")
                logger.error(f"[{request_id}] Raw Firecrawl error: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Firecrawl scraping failed: {str(e)}")
        
        # save the markdown to a file
        file_path = "File saving disabled in production mode"
        if settings.DEBUG_MODE:
            logger.debug(f"[{request_id}] Saving markdown to file...")
            file_path = save_markdown_to_file(markdown_content)
            logger.info(f"[{request_id}] Markdown saved to {file_path}")
        else:
            logger.debug(f"[{request_id}] File saving disabled in production mode")
        
        # Process with LLM
        logger.info(f"[{request_id}] ===== STARTING LLM PROCESSING =====")
        logger.info(f"[{request_id}] Model: {request.model}")
        logger.info(f"[{request_id}] Initializing LLM processor...")
        
        llm_start = time.time()
        llm_processor = LLMProcessor(model=request.model, request_id=request_id)
        
        try:
            logger.info(f"[{request_id}] Sending request to LLM...")
            extracted_data, extraction_file_path = await asyncio.wait_for(
                llm_processor.extract_information(
                    content=markdown_content,
                    extraction_prompt=request.extraction_prompt,
                    output_format=request.output_format
                ),
                timeout=150  # 2.5 minute timeout for LLM processing
            )
            
            llm_time = time.time() - llm_start
            logger.info(f"[{request_id}] LLM processing completed in {llm_time:.2f}s")
            
            # Log extracted data summary (truncated)
            if "error" not in extracted_data:
                data_str = str(extracted_data)[:500] + "..." if len(str(extracted_data)) > 500 else str(extracted_data)
                logger.debug(f"[{request_id}] Extracted data preview: {data_str}")
            
        except asyncio.TimeoutError as e:
            llm_time = time.time() - llm_start
            logger.error(f"[{request_id}] LLM processing timed out after {llm_time:.2f}s")
            logger.error(f"[{request_id}] Timeout error: {str(e)}")
            
            return {
                "status": "error",
                "error": "LLM processing timed out after 150 seconds",
                "raw_error": str(e),
                "markdown_file": file_path,
                "scraping_method": "inhouse_playwright" if request.use_inhouse_scraping else "firecrawl",
                "processing_time": llm_time
            }
        except Exception as e:
            llm_time = time.time() - llm_start
            logger.error(f"[{request_id}] LLM processing failed after {llm_time:.2f}s")
            logger.error(f"[{request_id}] LLM Error: {str(e)}")
            logger.error(f"[{request_id}] Raw LLM error traceback: {traceback.format_exc()}")
            
            return {
                "status": "error",
                "error": f"LLM processing failed: {str(e)}",
                "raw_error": traceback.format_exc(),
                "markdown_file": file_path,
                "scraping_method": "inhouse_playwright" if request.use_inhouse_scraping else "firecrawl",
                "processing_time": llm_time
            }
        
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] ===== REQUEST COMPLETED SUCCESSFULLY =====")
        logger.info(f"[{request_id}] Total processing time: {total_time:.2f}s")
        
        return {
            "status": "success",
            "request_id": request_id,
            "extracted_data": extracted_data,
            "extraction_file": extraction_file_path,
            "markdown_file": file_path,
            "model_used": request.model,
            "scraping_method": "inhouse_playwright" if request.use_inhouse_scraping else "firecrawl",
            "debug_mode": settings.DEBUG_MODE,
            "processing_time": {
                "total": f"{total_time:.2f}s",
                "scraping": f"{scrape_time:.2f}s",
                "llm": f"{llm_time:.2f}s"
            }
        }
        
    except httpx.RequestError as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] HTTP Request Error: {str(e)}")
        logger.error(f"[{request_id}] Raw HTTP error: {traceback.format_exc()}")
        
        return {
            "status": "error",
            "request_id": request_id,
            "error": f"HTTP request error: {str(e)}",
            "raw_error": traceback.format_exc(),
            "processing_time": f"{total_time:.2f}s"
        }
        
    except HTTPException as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] HTTP Exception: {e.detail}")
        raise
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] ===== REQUEST FAILED =====")
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        logger.error(f"[{request_id}] Raw error traceback: {traceback.format_exc()}")
        logger.error(f"[{request_id}] Total time before failure: {total_time:.2f}s")
        
        return {
            "status": "error",
            "request_id": request_id,
            "error": str(e),
            "raw_error": traceback.format_exc(),
            "processing_time": f"{total_time:.2f}s"
        } 