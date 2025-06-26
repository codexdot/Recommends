"""
Download helper utilities for ensuring reliable file downloads in Streamlit.
"""
import streamlit as st
import base64
import os
from typing import Optional

def create_download_link(file_path: str, download_name: str, link_text: str = "Download File") -> str:
    """
    Create a download link for a file that forces download to user's device.
    
    Args:
        file_path: Path to the file to download
        download_name: Name for the downloaded file
        link_text: Text to display for the download link
    
    Returns:
        HTML string with download link
    """
    if not os.path.exists(file_path):
        return f"<p style='color: red;'>File not found: {file_path}</p>"
    
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        b64_data = base64.b64encode(file_data).decode()
        file_size = len(file_data)
        
        # Create download link with proper MIME type
        mime_type = "application/zip" if download_name.endswith('.zip') else "application/octet-stream"
        
        href = f"""
        <a href="data:{mime_type};base64,{b64_data}" 
           download="{download_name}" 
           style="
               display: inline-block;
               padding: 0.5rem 1rem;
               background-color: #0066cc;
               color: white;
               text-decoration: none;
               border-radius: 0.25rem;
               margin: 0.25rem 0;
           "
           target="_blank">
           {link_text} ({file_size} bytes)
        </a>
        """
        return href
        
    except Exception as e:
        return f"<p style='color: red;'>Error creating download link: {str(e)}</p>"

def force_download_button(file_path: str, button_label: str, file_name: Optional[str] = None) -> bool:
    """
    Create a Streamlit download button that forces file download.
    
    Args:
        file_path: Path to the file
        button_label: Label for the button
        file_name: Custom filename for download (optional)
    
    Returns:
        True if download button was clicked, False otherwise
    """
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        
        download_name = file_name or os.path.basename(file_path)
        mime_type = "application/zip" if download_name.endswith('.zip') else "application/octet-stream"
        
        return st.download_button(
            label=button_label,
            data=file_bytes,
            file_name=download_name,
            mime=mime_type,
            key=f"download_{hash(file_path)}",
            help=f"Download {download_name} to your device"
        )
        
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return False