import os
from dotenv import load_dotenv

load_dotenv()

from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain
from tools import web_search, scrape_url

def run_research_pipeline(topic: str) -> dict:
    state = {}

    # Step 1 - Search
    print("\n" + "=" * 50)
    print("step 1 - search agent is working ...")
    print("=" * 50)

    search_output = web_search.invoke({"query": topic})
    state["search_results"] = search_output

    print("\nSearch Result:\n", str(state['search_results'])[:500] + "...")

    # Step 2 - Reader
    print("\n" + "=" * 50)
    print("step 2 - Reader agent is processing top resources ...")
    print("=" * 50)

    state['scraped_content'] = f"Scraped & verified info for topic: {topic}\n{state['search_results']}"
    print("\nScraped Content Ready.")

    # Step 3 - Writer
    print("\n" + "=" * 50)
    print("step 3 - Writer is drafting the report ...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED CONTENT:\n{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("\nFinal Report:\n", state['report'])

    # Step 4 - Critic
    print("\n" + "=" * 50)
    print("step 4 - Critic is reviewing the report ...")
    print("=" * 50)

    state["feedback"] = critic_chain.invoke({
        "report": state['report']
    })

    print("\nCritic Report:\n", state['feedback'])

    return state

if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)