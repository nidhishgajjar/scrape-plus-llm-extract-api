import random
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
from fake_useragent import UserAgent

router = APIRouter()
logger = setup_logger(__name__)

ua = UserAgent(browsers=['chrome', 'edge', 'firefox'])

@router.get("/")
async def root():
    return {"message": "Health check", "status": "ok"}

@router.get("/health")
async def health():
    return {"status": "healthy", "message": "Service is running"}

async def perform_enhanced_scraping(url: str, request_id: str, delay_ms: int = 5000):
    """
    Reusable enhanced scraping function with anti-detection measures.
    Returns: (html_content, error_dict or None)
    """
    logger.info(f"[{request_id}] Starting enhanced Playwright scraping...")
    
    async with async_playwright() as p:
        logger.debug(f"[{request_id}] Launching browser...")
        
        # Enhanced browser launch arguments for stealth
        browser = await p.chromium.launch(
            headless=True,  # Consider headless=False for really tough sites
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-extensions',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-web-security',
                '--disable-features=BlockInsecurePrivateNetworkRequests',
                '--disable-features=OutOfBlinkCors',
                '--window-size=1920,1080',
                '--start-maximized',
                # Additional stealth arguments
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-features=TranslateUI',
                '--disable-ipc-flooding-protection',
                '--password-store=basic',
                '--use-mock-keychain',
                '--force-color-profile=srgb',
                '--disable-features=UserAgentClientHint',
                '--disable-features=WebRtcHideLocalIpsWithMdns',
                '--disable-webgl',
                '--use-gl=swiftshader',
                '--disable-accelerated-2d-canvas',
                '--disable-features=AudioServiceOutOfProcess',
                '--disable-features=IsolateOrigins',
                '--disable-site-isolation-trials',
                '--disable-features=site-per-process',
                '--disable-web-security',
                '--disable-features=CrossSiteDocumentBlockingIfIsolating',
                '--disable-features=CrossSiteDocumentBlockingAlways',
            ]
        )

        # Create context with additional settings
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            screen={"width": 1920, "height": 1080},
            device_scale_factor=1,
            has_touch=False,
            is_mobile=False,
            java_script_enabled=True,
            locale='en-US',
            timezone_id='America/New_York',
            geolocation=None,
            permissions=[],
            color_scheme='light',
            reduced_motion='no-preference',
        )
        
        page = await context.new_page()
        
        # Set realistic headers
        await page.set_extra_http_headers({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'User-Agent': ua.random,
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Comprehensive anti-detection script
        await page.add_init_script("""
            // Remove webdriver
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            // Mock permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            
            // Mock plugins to look like real browser
            Object.defineProperty(navigator, 'plugins', {
                get: () => {
                    return [
                        {
                            0: {type: "application/x-google-chrome-pdf", suffixes: "pdf", description: "Portable Document Format"},
                            description: "Portable Document Format",
                            filename: "internal-pdf-viewer",
                            length: 1,
                            name: "Chrome PDF Plugin"
                        },
                        {
                            0: {type: "application/x-nacl", suffixes: "", description: "Native Client Executable"},
                            1: {type: "application/x-pnacl", suffixes: "", description: "Portable Native Client Executable"},
                            description: "Native Client",
                            filename: "internal-nacl-plugin",
                            length: 2,
                            name: "Native Client"
                        }
                    ];
                },
            });
            
            // Mock languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            
            // Mock hardware concurrency
            Object.defineProperty(navigator, 'hardwareConcurrency', {
                get: () => 8,
            });
            
            // Mock device memory
            Object.defineProperty(navigator, 'deviceMemory', {
                get: () => 8,
            });
            
            // Mock WebGL vendor
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                if (parameter === 37445) {
                    return 'Intel Inc.';
                }
                if (parameter === 37446) {
                    return 'Intel Iris OpenGL Engine';
                }
                return getParameter(parameter);
            };
            
            // Mock battery API
            if (typeof navigator.getBattery !== 'undefined') {
                navigator.getBattery = () => Promise.resolve({
                    charging: true,
                    chargingTime: 0,
                    dischargingTime: Infinity,
                    level: 1,
                });
            }
            
            // Override chrome detection
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };
            
            // Mock connection API
            Object.defineProperty(navigator, 'connection', {
                get: () => ({
                    effectiveType: '4g',
                    rtt: 50,
                    downlink: 10,
                    saveData: false,
                }),
            });
            
            // Remove automation controlled flag
            Object.defineProperty(navigator, 'automationControlled', {
                get: () => undefined,
            });
            
            // Mock platform
            Object.defineProperty(navigator, 'platform', {
                get: () => 'Win32',
            });
            
            // Mock vendor
            Object.defineProperty(navigator, 'vendor', {
                get: () => 'Google Inc.',
            });
            
            // Mock app version
            Object.defineProperty(navigator, 'appVersion', {
                get: () => '5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            });
            
            // Override toString methods to avoid detection
            window.navigator.permissions.query.toString = () => 'function query() { [native code] }';
            if (window.navigator.getBattery) {
                window.navigator.getBattery.toString = () => 'function getBattery() { [native code] }';
            }
        """)

        try:
            # Navigate with extended timeout for Cloudflare challenges
            logger.info(f"[{request_id}] Navigating to URL with 30s timeout...")
            await page.goto(url, 
                wait_until='domcontentloaded',  # or 'networkidle' for SPAs
                timeout=45000  # 45 second timeout for slow sites/Cloudflare
            )
            
            # Check for Cloudflare challenge
            page_title = await page.title()
            page_content = await page.content()
            
            if any(indicator in page_title.lower() for indicator in ['just a moment', 'cloudflare', 'checking your browser', 'challenge']):
                logger.warning(f"[{request_id}] Cloudflare challenge detected, waiting...")
                
                # Wait for challenge to complete
                try:
                    # Wait for either navigation or specific element that indicates page loaded
                    await page.wait_for_selector('body', state='visible', timeout=15000)
                    await page.wait_for_load_state('networkidle', timeout=15000)
                    
                    # Additional wait for JavaScript challenges
                    await page.wait_for_timeout(5000)
                    
                    # Re-check content
                    page_content = await page.content()
                    page_title = await page.title()
                    logger.info(f"[{request_id}] Challenge appears to be resolved")
                    
                except Exception as e:
                    logger.warning(f"[{request_id}] Challenge wait failed: {str(e)}")
            
            # Check for other bot detection systems
            if any(indicator in page_content.lower() for indicator in ['recaptcha', 'hcaptcha', 'robot', 'bot detected']):
                logger.warning(f"[{request_id}] Bot detection system detected on page")
            
            logger.info(f"[{request_id}] Page loaded successfully - Title: {page_title}")
            
            # Wait for dynamic content (but don't fail if it times out)
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
                logger.debug(f"[{request_id}] Network idle state reached")
            except PlaywrightTimeout:
                logger.debug(f"[{request_id}] Network idle timeout - continuing anyway")
            
            # Apply delay if specified
            if delay_ms > 0:
                logger.info(f"[{request_id}] Applying delay: {delay_ms}ms")
                await page.wait_for_timeout(delay_ms)
            
            # Scroll with timeout and human-like behavior
            logger.debug(f"[{request_id}] Starting page scrolling...")
            try:
                await asyncio.wait_for(
                    human_like_scroll(page, max_scroll_time=15),  # Pass max scroll time to function
                    timeout=20  # 20 second timeout for scrolling
                )
                logger.debug(f"[{request_id}] Scrolling completed")
            except asyncio.TimeoutError:
                logger.warning(f"[{request_id}] Scrolling timeout reached, continuing with partial scroll")
                # Continue anyway - we may have scrolled enough
            
            # Final wait for any lazy-loaded content
            await page.wait_for_timeout(2000)
            
            logger.info(f"[{request_id}] Extracting page content...")
            html_content = await page.content()
            logger.info(f"[{request_id}] HTML content extracted (size: {len(html_content)} chars)")
            
            return html_content, None
            
        except PlaywrightTimeout as e:
            logger.error(f"[{request_id}] Page loading timed out: {str(e)}")
            
            # Try to get partial content
            partial_content = None
            try:
                partial_content = await page.content()
            except:
                pass
                
            return None, {
                "status": "error",
                "error": "Page loading timed out - possible bot protection",
                "raw_error": str(e),
                "partial_content": partial_content
            }
        finally:
            await context.close()
            await browser.close()
            logger.debug(f"[{request_id}] Browser closed")

@router.get("/scrape")
async def scrape_url(url: str):
    start_time = time.time()
    request_id = f"{int(time.time() * 1000)}"
    
    logger.info(f"[{request_id}] ===== SCRAPE REQUEST RECEIVED =====")
    logger.info(f"[{request_id}] URL: {url}")
    
    try:
        scrape_start = time.time()
        
        # Use the enhanced scraping function
        html_content, error = await perform_enhanced_scraping(url, request_id)
        
        if error:
            return error

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


async def human_like_scroll(page, max_scroll_time=15):
    """
    Scroll the page in a more human-like manner with variable speeds and pauses
    Limited by max_scroll_time to prevent timeouts
    """
    import time
    start_time = time.time()
    viewport_height = page.viewport_size['height']
    total_height = await page.evaluate('document.body.scrollHeight')
    current_position = 0
    
    while current_position < total_height:
        # Check if we've exceeded max scroll time
        if time.time() - start_time > max_scroll_time:
            logger.debug(f"Reached max scroll time of {max_scroll_time}s, stopping scroll")
            break
            
        # Random scroll distance (between 300-700 pixels)
        scroll_distance = random.randint(300, 700)
        
        # Smooth scroll with random speed
        await page.evaluate(f"""
            window.scrollBy({{
                top: {scroll_distance},
                behavior: 'smooth'
            }});
        """)
        
        # Shorter random wait between scrolls (reduced from 500-2000 to 200-800)
        await page.wait_for_timeout(random.randint(200, 800))
        
        # Occasionally scroll up a bit (like a human re-reading) - reduced frequency
        if random.random() < 0.05:  # Reduced from 10% to 5% chance
            scroll_up = random.randint(50, 200)
            await page.evaluate(f"window.scrollBy(0, -{scroll_up})")
            await page.wait_for_timeout(random.randint(200, 400))  # Reduced wait time
        
        # Update position
        current_position = await page.evaluate('window.pageYOffset')
        
        # Check if we've reached near the bottom
        if current_position + viewport_height >= total_height - 100:
            break
    
    # Ensure we're at the bottom
    await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
    await page.wait_for_timeout(1000)
class ExtractRequest(BaseModel):
    url: str
    extraction_prompt: str
    output_format: Dict[str, Any]
    model: ModelType = "gpt-4o-mini"
    use_inhouse_scraping: bool = False
    delay_page_load: int = 5000  # Delay in milliseconds for both scraping methods

@router.post("/scrape/llm-extract")
async def scrape_and_extract(request: ExtractRequest):
    start_time = time.time()
    request_id = f"{int(time.time() * 1000)}"  # Simple request ID based on timestamp
    
    logger.info(f"[{request_id}] REQUEST: {request.url}")
    
    try:
        settings = get_settings()
        # Debug mode check
        
        if request.use_inhouse_scraping:
            # Use enhanced in-house scraping with anti-detection measures
            scrape_start = time.time()
            
            # Use the enhanced scraping function
            html_content, error = await perform_enhanced_scraping(
                request.url, 
                request_id, 
                delay_ms=request.delay_page_load
            )
            
            if error:
                return error
            
            # Converting HTML to Markdown
            markdown_content = await convert_html_to_markdown(html_content)
            
            if markdown_content is None:
                logger.error(f"[{request_id}] Failed to convert HTML to markdown")
                raise HTTPException(status_code=500, detail="Failed to convert HTML to markdown")
            
            scrape_time = time.time() - scrape_start
            
            # Log markdown preview to help debug null extractions
            markdown_preview = markdown_content[:300] + "..." if len(markdown_content) > 300 else markdown_content
            logger.info(f"[{request_id}] SCRAPED: {len(markdown_content)} chars | Preview: {markdown_preview}")
            
        else:
            # Use Firecrawl approach
            # Starting Firecrawl scraping
            scrape_start = time.time()
            
            try:
                app = FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY'))
                
                # Add delay parameter to Firecrawl
                firecrawl_params = {
                    'formats': ['markdown'],
                    'actions': [
                        {"type": "wait", "milliseconds": request.delay_page_load}
                    ]
                }
                
                logger.info(f"[{request_id}] Using Firecrawl with delay: {request.delay_page_load}ms")
                
                response = app.scrape_url(
                    url=request.url,
                    params=firecrawl_params
                )
                markdown_content = response['markdown']
                
                scrape_time = time.time() - scrape_start
                
                # Log markdown preview to help debug null extractions
                markdown_preview = markdown_content[:300] + "..." if len(markdown_content) > 300 else markdown_content
                logger.info(f"[{request_id}] SCRAPED: {len(markdown_content)} chars | Preview: {markdown_preview}")
                
            except Exception as e:
                logger.error(f"[{request_id}] Firecrawl scraping failed: {str(e)}")
                logger.error(f"[{request_id}] Raw Firecrawl error: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Firecrawl scraping failed: {str(e)}")
        
        # save the markdown to a file
        file_path = "File saving disabled in production mode"
        if settings.DEBUG_MODE:
            file_path = save_markdown_to_file(markdown_content)
        else:
            file_path = "File saving disabled in production mode"
        
        # Process with LLM
        # Start LLM processing
        
        llm_start = time.time()
        llm_processor = LLMProcessor(model=request.model, request_id=request_id)
        
        try:
            # Sending request to LLM
            extracted_data, extraction_file_path = await asyncio.wait_for(
                llm_processor.extract_information(
                    content=markdown_content,
                    extraction_prompt=request.extraction_prompt,
                    output_format=request.output_format
                ),
                timeout=150  # 2.5 minute timeout for LLM processing
            )
            
            llm_time = time.time() - llm_start
            # LLM processing completed
            
            # Check if LLM processing returned an error
            if isinstance(extracted_data, dict) and "error" in extracted_data:
                logger.error(f"[{request_id}] LLM processing returned error: {extracted_data['error']}")
                return {
                    "status": "error",
                    "request_id": request_id,
                    "error": extracted_data['error'],
                    "raw_error": extracted_data.get('raw_error', 'No raw error provided'),
                    "markdown_file": file_path,
                    "scraping_method": "inhouse_playwright" if request.use_inhouse_scraping else "firecrawl",
                    "processing_time": llm_time,
                    "llm_error_type": extraction_file_path  # This contains the error type
                }
            
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
        logger.info(f"[{request_id}] COMPLETED in {total_time:.2f}s")
        
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