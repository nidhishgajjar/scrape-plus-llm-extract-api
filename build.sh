#!/bin/bash

# Build script for Render deployment
set -e

echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Installing Playwright browsers..."
playwright install chromium --with-deps

echo "Installing additional system dependencies for Playwright..."
playwright install-deps chromium

echo "Verifying Playwright installation..."
python -c "
import asyncio
from playwright.async_api import async_playwright

async def test_browser():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto('data:text/html,<h1>Test</h1>')
        title = await page.title()
        await browser.close()
        print(f'Browser test successful: {title}')

asyncio.run(test_browser())
"

echo "Build completed successfully!" 