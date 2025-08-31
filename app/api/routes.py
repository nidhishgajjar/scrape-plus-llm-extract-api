import random
from fastapi import APIRouter, HTTPException
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout, Browser
from firecrawl import FirecrawlApp
from app.services.scraper import scroll_to_bottom, save_markdown_to_file
from app.services.markdown_converter import convert_html_to_markdown
from app.services.llm_processor import LLMProcessor, ModelType
from typing import Dict, Any, Optional, Set, Literal
from pydantic import BaseModel
import os
import asyncio
import httpx
from app.config import get_settings
import time
import traceback
import signal
import atexit
from app.utils.logger import setup_logger
from fake_useragent import UserAgent
from app.utils.resource_manager import resource_manager

router = APIRouter()
logger = setup_logger(__name__)

ua = UserAgent(browsers=['chrome', 'edge', 'firefox'])

# Track active requests and browsers for cleanup
active_requests: Set[str] = set()
active_browsers: Set[Browser] = set()
shutdown_event = asyncio.Event()

def cleanup_handler(signum=None, frame=None):
    """Handle graceful shutdown on SIGTERM/SIGINT"""
    if signum:
        logger.warning(f"Received shutdown signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()
    
    # Give active requests time to complete
    if active_requests:
        logger.info(f"Waiting for {len(active_requests)} active requests to complete...")
    
    # Note: Actual browser cleanup will happen in the finally blocks
    logger.info("Shutdown handler completed")

# Register cleanup handlers
signal.signal(signal.SIGTERM, cleanup_handler)
signal.signal(signal.SIGINT, cleanup_handler)
atexit.register(cleanup_handler)

@router.get("/")
async def root():
    return {"message": "Health check", "status": "ok"}

@router.get("/health")
async def health():
    """Health check with resource status"""
    status = resource_manager.get_status()
    
    # Determine health based on resource usage
    if status["memory_percent"] > 90:
        return {
            "status": "degraded",
            "message": "High memory usage",
            "resources": status
        }
    
    return {
        "status": "healthy",
        "message": "Service is running",
        "resources": status
    }

async def perform_enhanced_scraping(url: str, request_id: str, delay_ms: int = 5000, enable_scrolling: bool = False, scrolling_type: Literal["human", "bot"] = "bot"):
    """
    Reusable enhanced scraping function with anti-detection measures.
    Returns: (html_content, error_dict or None)
    """
    logger.info(f"[{request_id}] Starting enhanced Playwright scraping (scrolling={'enabled (' + scrolling_type + ')' if enable_scrolling else 'disabled'})...")
    
    # Track this request
    active_requests.add(request_id)
    browser = None
    
    try:
        # Check for shutdown
        if shutdown_event.is_set():
            logger.warning(f"[{request_id}] Rejecting request due to shutdown")
            return None, {"status": "error", "error": "Service is shutting down"}
        
        # Wait for browser slot (will queue if necessary)
        if not await resource_manager.acquire_browser_slot(timeout=300.0):  # 5 minutes timeout
            logger.error(f"[{request_id}] Timeout waiting for browser slot after 5 minutes")
            return None, {
                "status": "error",
                "error": "Browser launch timeout - server overloaded",
                "message": "Waited 5 minutes for browser resources. Please try again later."
            }
        
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
            
                # Optional scrolling to load dynamic content
                if enable_scrolling:
                    if scrolling_type == "human":
                        logger.debug(f"[{request_id}] Starting human-like page scrolling...")
                        try:
                            await asyncio.wait_for(
                                human_like_scroll(page, max_scroll_time=15),  # Pass max scroll time to function
                                timeout=20  # 20 second timeout for scrolling
                            )
                            logger.debug(f"[{request_id}] Human scrolling completed")
                        except asyncio.TimeoutError:
                            logger.warning(f"[{request_id}] Scrolling timeout reached, continuing with partial scroll")
                    else:  # bot scrolling - fast and efficient
                        logger.debug(f"[{request_id}] Starting bot scrolling (fast mode)...")
                        await bot_scroll(page)
                        logger.debug(f"[{request_id}] Bot scrolling completed")
                else:
                    # Simple quick scroll to trigger lazy loading (minimal impact)
                    logger.debug(f"[{request_id}] Skipping scrolling as per configuration")
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(500)  # Brief wait for any lazy-loaded content
            
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
            except Exception as e:
                logger.error(f"[{request_id}] Unexpected error: {str(e)}")
                return None, {
                    "status": "error",
                    "error": f"Scraping failed: {str(e)}",
                    "raw_error": str(e)
                }
            finally:
                try:
                    await context.close()
                    await browser.close()
                    if browser in active_browsers:
                        active_browsers.remove(browser)
                except:
                    pass
                finally:
                    resource_manager.release_browser_slot()
                    logger.debug(f"[{request_id}] Browser closed and slot released")
    
    except Exception as e:
        logger.error(f"[{request_id}] Failed to launch browser: {str(e)}")
        return None, {
            "status": "error",
            "error": "Failed to launch browser",
            "raw_error": str(e)
        }
    finally:
        # Clean up request tracking
        if request_id in active_requests:
            active_requests.remove(request_id)
        logger.debug(f"[{request_id}] Request completed")

@router.get("/scrape")
async def scrape_url(
    url: str, 
    enable_scrolling: bool = False, 
    scrolling_type: Literal["human", "bot"] = "bot",
    delay_ms: int = 5000
):
    start_time = time.time()
    request_id = f"{int(time.time() * 1000)}"
    
    logger.info(f"[{request_id}] ===== SCRAPE REQUEST RECEIVED =====")
    logger.info(f"[{request_id}] URL: {url}")
    
    try:
        scrape_start = time.time()
        
        # Use the enhanced scraping function
        html_content, error = await perform_enhanced_scraping(
            url, 
            request_id,
            delay_ms=delay_ms,
            enable_scrolling=enable_scrolling,
            scrolling_type=scrolling_type
        )
        
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


class ScrapeRequest(BaseModel):
    url: str
    enable_scrolling: bool = False
    scrolling_type: Literal["human", "bot"] = "bot"
    delay_ms: int = 5000


@router.post("/scrape")
async def scrape_url_post(request: ScrapeRequest):
    """POST endpoint for scraping with JSON payload"""
    return await scrape_url(
        url=request.url,
        enable_scrolling=request.enable_scrolling,
        scrolling_type=request.scrolling_type,
        delay_ms=request.delay_ms
    )


async def bot_scroll(page):
    """
    Fast bot scrolling - scrolls to bottom in chunks quickly
    """
    logger.debug("Bot scrolling: Starting fast scroll")
    
    # Get page height
    total_height = await page.evaluate('document.body.scrollHeight')
    viewport_height = page.viewport_size['height']
    current_position = 0
    
    # Scroll in large chunks quickly
    while current_position < total_height:
        # Scroll down by viewport height
        await page.evaluate(f'window.scrollBy(0, {viewport_height})')
        await page.wait_for_timeout(200)  # Small delay to let content load
        
        current_position = await page.evaluate('window.pageYOffset')
        # Update total height in case new content loaded
        new_height = await page.evaluate('document.body.scrollHeight')
        
        if new_height == total_height:
            # No new content loaded, we're done
            break
        total_height = new_height
    
    # Final scroll to absolute bottom
    await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
    await page.wait_for_timeout(500)  # Wait for any final lazy loading
    logger.debug(f"Bot scrolling: Completed, scrolled to {total_height}px")


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
    enable_scrolling: bool = False  # Default to no scrolling
    scrolling_type: Literal["human", "bot"] = "bot"  # Type of scrolling
    delay_page_load: int = 5000  # Delay in milliseconds for both scraping methods

@router.post("/scrape/llm-extract")
async def scrape_and_extract(request: ExtractRequest):
    start_time = time.time()
    request_id = f"{int(time.time() * 1000)}"  # Simple request ID based on timestamp
    
    # Accept request and add to queue
    logger.info(f"[{request_id}] REQUEST: {request.url} - Queueing...")
    
    # Wait in queue (will timeout after 2 minutes if queue is full)
    if not await resource_manager.acquire_request_slot(timeout=120.0):
        logger.error(f"[{request_id}] Request timeout in queue")
        return {
            "status": "error",
            "error": "Request timeout - server overloaded",
            "message": "The server queue was full for too long. Please try again later.",
            "resources": resource_manager.get_status()
        }
    
    try:
        logger.info(f"[{request_id}] Processing request: {request.url}")
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing request: {str(e)}")
        return {
            "status": "error",
            "error": f"Error processing request: {str(e)}",
            "message": "An error occurred while processing your request.",
            "resources": resource_manager.get_status()
        }
    
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
                delay_ms=request.delay_page_load,
                enable_scrolling=request.enable_scrolling,
                scrolling_type=request.scrolling_type
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
    finally:
        # Always release the request slot
        resource_manager.release_request_slot()
        logger.debug(f"[{request_id}] Request completed, slot released") 