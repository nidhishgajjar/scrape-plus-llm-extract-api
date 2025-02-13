from playwright.async_api import Page
import logging
import os
from datetime import datetime

async def scroll_to_bottom(page: Page) -> None:
    try:
        last_height = await page.evaluate('document.documentElement.scrollHeight')
        
        while True:
            await page.evaluate('window.scrollTo(0, document.documentElement.scrollHeight)')
            await page.wait_for_timeout(2000)
            
            new_height = await page.evaluate('document.documentElement.scrollHeight')
            
            if new_height == last_height:
                break
                
            last_height = new_height
            
    except Exception as e:
        logging.error(f"Error during scrolling: {str(e)}")

def save_markdown_to_file(markdown_content: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scraped_{timestamp}.md"
    
    os.makedirs("scraped", exist_ok=True)
    file_path = os.path.join("scraped", filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    return file_path 