import requests
import os
from pydantic import HttpUrl
from .base import Tool
from typing import Dict, Any, Union

class AresInternetTool(Tool):
    """
    Tool to search real-time information from the internet using the Traversaal Ares API.
    """
    
    name: str = "ares_internet_search_tool"
    description: str = "Tool to search real-time relevant content from the internet. Use this tool for any factual information or recent events."
    arg: str = "A single string parameter that will be searched on the internet to find relevant content"

    # Specific Parameters
    url: HttpUrl = "https://api-ares.traversaal.ai/live/predict"
    x_api_key: str = None
    
    def __init__(self, **data):
        """
        Initialize the AresInternetTool with the provided parameters.
        Gets API key from environment if not provided.
        """
        super().__init__(**data)
        if self.x_api_key is None:
            self.x_api_key = os.environ.get("TRAVERSAAL_ARES_API_KEY")
            if not self.x_api_key:
                raise ValueError("TRAVERSAAL_ARES_API_KEY environment variable not set")

    def run(self, prompt: Union[str, Dict, Any]) -> str:
        """
        Run a search query through the Ares API and return the results.
        
        Args:
            prompt: Search query string or object. If not a string, will be converted to string.
            
        Returns:
            Response text from the Ares API or error message
        """
        print(f"Calling Ares Internet Search Tool with prompt: {prompt}")
        
        # Make sure prompt is a string
        if not isinstance(prompt, str):
            prompt = str(prompt)
            
        # Clean up quotation marks from the prompt if they exist
        prompt = prompt.strip("'\"")
        
        # Create the API payload
        payload = {"query": [prompt]}
        
        try:
            # Make the API request
            print(f"Sending request to Ares API with query: {prompt}")
            response = requests.post(
                self.url, 
                json=payload, 
                headers={
                    "x-api-key": self.x_api_key,
                    "content-type": "application/json"
                },
                timeout=45  # Extended timeout
            )
            
            # Print response status code for debugging
            print(f"DEBUG - Ares API Response Status: {response.status_code}")
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Log the raw response for debugging
            print(f"DEBUG - Raw Response Preview: {response.text[:200]}...")
            
            # Try to parse the JSON response
            try:
                result = response.json()
                
                # Check for the expected structure
                if 'data' in result and 'response_text' in result['data']:
                    response_text = result['data']['response_text']
                    
                    # Check if the response has actual content
                    if not response_text or response_text.strip() == "":
                        return "No information found for this query. Please try a different search term."
                    
                    print(f"DEBUG - Found response_text (length: {len(response_text)})")
                    print(f"DEBUG - Response preview: {response_text[:200]}...")
                    
                    # Return the successful response
                    return response_text
                else:
                    # Detailed error for missing expected structure
                    error_msg = "Error: Unexpected API response structure."
                    if 'data' in result:
                        error_msg += f" Data keys: {list(result['data'].keys())}"
                    else:
                        error_msg += f" Response keys: {list(result.keys())}"
                    
                    print(f"DEBUG - {error_msg}")
                    return error_msg
                
            except ValueError as e:
                # Handle JSON parsing errors
                print(f"DEBUG - JSON parsing failed: {e}")
                return f"Error: Could not parse API response - {e}. Raw response preview: {response.text[:200]}"
                
        except requests.exceptions.RequestException as e:
            print(f"DEBUG - HTTP request failed: {e}")
            return f"Error: HTTP request failed - {e}"
        except Exception as e:
            import traceback
            print(f"DEBUG - Unexpected error in Ares tool: {str(e)}")
            print(traceback.format_exc())
            return f"Error: Unexpected error in Ares tool - {e}"
# Add this class to ares_tool.py for testing
class MockAresInternetTool(Tool):
    """
    Mock version of AresInternetTool for testing.
    """
    
    name: str = "ares_internet_search_tool"
    description: str = "Tool to search real-time relevant content from the internet. Use this tool for any factual information or recent events."
    arg: str = "A single string parameter that will be searched on the internet to find relevant content"
    
    def run(self, prompt: Union[str, Dict, Any]) -> str:
        """
        Return mock data based on the query.
        """
        prompt = str(prompt).lower().strip("'\"")
        print(f"Calling Mock Ares Internet Search Tool with prompt: {prompt}")
        
        # For the conference query, return mock data
        if "optimized ai" in prompt and "conference" in prompt and "2025" in prompt:
            return """
### Overview of Optimized AI 2025 Conference

- **Event Name**: Optimized AI Conference 2025
- **Dates**: April 14-16, 2025
  - **Conference Talks**: April 14-15
  - **Workshops**: April 16
- **Location**: San Francisco, CA

- **Description**: 
  - The Optimized AI Conference is the premier event focused on AI in production. It aims to bring together over 100 industry leaders, venture capitalists, and executive decision-makers across 12 specialized tracks. The conference is designed to facilitate the scaling of real-world AI solutions more rapidly.
  - Attendees will have the opportunity to explore cutting-edge use cases, tools, and strategies that help transition from pilot projects to full enterprise deployment.

- **Key Features**:
  - **Tracks**: 12 focused tracks covering various aspects of AI.
  - **Speakers**: Insights from experts at leading AI companies, including NVIDIA, Google, and OpenAI.
  - **Networking**: Opportunities to connect with industry leaders and peers.
  - **Workshops**: Practical sessions aimed at enhancing understanding and application of AI technologies.

This conference is an essential event for anyone involved in AI, providing valuable insights and practical knowledge to optimize AI applications in various industries.
            """
        # For other queries, return a generic response
        return f"Here is information about: {prompt}\n\nThis is mock data for testing purposes."