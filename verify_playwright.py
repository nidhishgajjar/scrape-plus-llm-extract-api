#!/usr/bin/env python3
"""
Verification script to test Playwright installation on Render
"""

import asyncio
import sys
import os
from playwright.async_api import async_playwright

async def verify_playwright_installation():
    """Test if Playwright browsers are properly installed and working"""
    
    print("🔍 Verifying Playwright installation...")
    
    try:
        async with async_playwright() as p:
            print("✅ Playwright module loaded successfully")
            
            # Test browser launch with cloud-friendly args
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-extensions',
                    '--disable-web-security',
                    '--single-process'
                ]
            )
            print("✅ Chromium browser launched successfully")
            
            # Test page creation and navigation
            page = await browser.new_page()
            await page.goto('data:text/html,<h1>Playwright Test</h1>')
            title = await page.title()
            print(f"✅ Page navigation successful: {title}")
            
            # Clean up
            await browser.close()
            print("✅ Browser closed successfully")
            
        print("🎉 All Playwright verification tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Playwright verification failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Print environment info for debugging
        print("\n🔧 Environment Information:")
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        print(f"PLAYWRIGHT_BROWSERS_PATH: {os.environ.get('PLAYWRIGHT_BROWSERS_PATH', 'Not set')}")
        
        return False

if __name__ == "__main__":
    success = asyncio.run(verify_playwright_installation())
    sys.exit(0 if success else 1) 