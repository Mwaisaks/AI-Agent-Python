from pyexpat.errors import messages

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import os

# Loading the environment variable file
load_dotenv()

"""
# create the AI model
model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    openai_api_key=os.getenv("OPEN_API_KEY")
)


# Create the AI model with Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
"""

# Use Groq instead
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Specify server parameters
server_params = StdioServerParameters(
    command="npx",
    env={
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")
    },
    args=["firecrawl-mcp"]
)

#Connect to the mcp client
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)

            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful assistant with access to Firecrawl tools for web scraping and data extraction.

            Available tools:
            - firecrawl_scrape: Scrape a single webpage
            - firecrawl_crawl: Crawl multiple pages from a website
            - firecrawl_map: Get the structure/sitemap of a website
            - firecrawl_search: Search for content (use carefully - check tool schema)
            - firecrawl_extract: Extract structured data from pages
            - firecrawl_check_crawl_status: Check status of ongoing crawls

            For general conversation that doesn't require web scraping, just respond directly without using tools.
            If you need to search the web or scrape data, use the appropriate Firecrawl tool.
            """
                }
            ]

            print("Available Tools -", *[tool.name for tool in tools])
            print("-" * 60)

            # loop so we can keep calling the agent
            while True:
                user_input = input("You: > ")
                if user_input == "exit":
                    print("Goodbye")
                    break

                messages.append({"role": "user", "content": user_input[:175000]})

                try:
                    agent_response = await agent.ainvoke({"messages": messages}) # allows the agent to use the tools as well as the LLM

                    ai_message = agent_response["messages"][-1].content
                    print("\nAgent: >", ai_message)
                except Exception as e:
                    print("Error: ", e)

if __name__ == "__main__":
    asyncio.run(main())