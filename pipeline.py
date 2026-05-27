import re
import time
from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
    planner_chain
)

def run_research_pipeline(topic: str, research_mode: str) -> dict:

    state = {}

    # step 0 - planner agent
    print("\n"+"="*50)
    print("Step 0 - planner agent is creating research strategy...")
    print("="*50)

    state["research_plan"] = planner_chain.invoke({
        "topic": topic
    })

    print("\nResearch Plan:\n")
    print(state["research_plan"])

    # step 1 - search agent
    print("\n"+"="*50)
    print("Step 1 - search agent is working...")
    print("="*50)

    search_agent = build_search_agent()

    search_results = search_agent.invoke({
        "messages": [
            (
                "user",
                f"""
Find recent and reliable information for this topic.

TOPIC:
{topic}

RESEARCH PLAN:
{state['research_plan']}
"""
            )
        ]
    })

    state['search_results'] = str(search_results["messages"][-1].content)

    print("search results: ", state['search_results'])

    #step 2 - reader agent
    print("\n"+"="*50)
    print("Step 2 - reader agent is scraping top resources  ...")
    print("="*50)

    # Extract URLs from search results
    urls = re.findall(r'https?://\S+', str(state['search_results']))

    # Keep only top 5 URLs
    top_urls = urls[:3]

    print("\nTop URLs Found:\n", top_urls)

    reader_agent = build_reader_agent()

    all_scraped_content = []

    # Scrape multiple URLs
    for idx, url in enumerate(top_urls):

        print(f"\nScraping URL {idx+1}: {url}")

        try:
            reader_result = reader_agent.invoke({
                "messages": [
                    (
                        "user",
                        f"Scrape and summarize this webpage:\n{url}"
                    )
                ]
            })

            scraped_text = reader_result["messages"][-1].content

            all_scraped_content.append(
                f"\nSOURCE {idx+1}: {url}\n\n{scraped_text}"
            )

        except Exception as e:

            print(f"Error scraping {url}: {e}")

    # Merge all research
    state['scraped_content'] = "\n\n".join(all_scraped_content)

    print("\nscraped content: \n ", state['scraped_content']) 

    #step 3 

    print("\n"+"="*50)
    print("Step 3 - writer agent is writing a research report  ...")
    print("="*50)

    research_combined = (
            f"SEARCH RESULTS:\n{state['search_results']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
        )


    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
        "mode": research_mode
    })
    print("\nGenerated Research Report:\n", state["report"])

    #step 4 - critic agent
    print("\n"+"="*50)
    print("Step 4 - critic agent is evaluating the report  ...")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
            "report": state["report"]
       })

    print("\nCritic Feedback:\n", state["feedback"])

    return state

if __name__ == "__main__":
    topic = input("Enter a research topic: ")
    run_research_pipeline(topic)