# How to get clean web data for chatbots and LLMs

This is a companion repository to go with [František Nesveda](https://github.com/fnesveda)'s Dev World 2025 talk: How to get clean web data for chatbots and LLMs.

The presentation slides are available here in [PDF](./how-to-get-clean-web-data-for-chatbots-and-llms.pdf) and [PPTX](how-to-get-clean-web-data-for-chatbots-and-llms.pptx) formats.

## Chatbot examples

To run the chatbot examples, you need to have Node.js installed and install dependencies with:

```bash
npm install
```

To run the chatbots, you will need to export your OpenAI API key as an environment variable or use an alternative way of setting this env var:

```bash
export OPENAI_API_KEY=your-api-key
```

Finally, run them with:

```bash
node tesla-chatbot.js
```

```bash
node bmw-chatbot.js
```

For more information on how they work. Reference [Crawlee](https://crawlee.dev) and [LangChain JS](https://js.langchain.com) documentation.

## Useful links

- [Apify](https://apify.com)
- [Website Content Crawler](https://apify.com/apify/website-content-crawler)
- [Crawlee](https://crawlee.dev)
- [LangChain](https://www.langchain.com)
- [Mozilla Readability](https://github.com/mozilla/readability)
- [Scrapy](https://scrapy.org)
- [Puppeteer](https://pptr.dev)
- [Playwright](https://playwright.dev)
- [Selenium](https://www.selenium.dev)
- [curl impersonate](https://github.com/lwthiker/curl-impersonate)
- [Impit](https://github.com/apify/impit)
