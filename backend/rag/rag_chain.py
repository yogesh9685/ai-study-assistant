def get_sources(documents):
    sources = []
    seen = set()

    for document in documents:
        source = document.metadata.get("source")
        page = document.metadata.get("page")

        key = (source, page)

        if key not in seen:
            sources.append({
                "source": source,
                "page": page
            })
            seen.add(key)

    return sources