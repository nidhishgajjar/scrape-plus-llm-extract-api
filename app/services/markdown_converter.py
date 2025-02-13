import html2text
import logging

async def convert_html_to_markdown(html_content: str, ignore_links: bool = False, inline_links: bool = True) -> str | None:
    if html_content is None:
        return None
    try:
        h = html2text.HTML2Text()
        h.ignore_links = ignore_links
        h.ignore_images = True
        h.inline_links = inline_links
        h.skip_internal_links = True
        h.single_line_break = True
        markdown_content = h.handle(html_content)
        
        return markdown_content
    except Exception as e:
        logging.error(f"Error converting HTML to Markdown: {str(e)}")
        return None 