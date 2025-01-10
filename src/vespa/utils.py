from IPython.display import HTML, display


def display_query_results(query, response, hits=5):
    query_time = response.json.get("timing", {}).get("searchtime", -1)
    query_time = round(query_time, 2)
    count = response.json.get("root", {}).get("fields", {}).get("totalCount", 0)
    html_content = f"<h3>Query text: '{query}', query time {query_time}s, count={count}, top results:</h3>"

    for i, hit in enumerate(response.hits[:hits]):
        title = hit["fields"]["title"]
        url = hit["fields"]["url"]
        page = hit["fields"]["page_number"]
        image = hit["fields"]["image"]
        score = hit["relevance"]

        html_content += f"<h4>PDF Result {i + 1}</h4>"
        html_content += f'<p><strong>Title:</strong> <a href="{url}">{title}</a>, page {page+1} with score {score:.2f}</p>'
        html_content += (
            f'<img src="data:image/png;base64,{image}" style="max-width:100%;">'
        )

    display(HTML(html_content))
