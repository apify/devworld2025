import { CSVLoader } from '@langchain/community/document_loaders/fs/csv';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { OpenAIEmbeddings, ChatOpenAI } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Readability } from '@mozilla/readability';
import { Actor } from 'apify';
import { CheerioCrawler, Dataset } from 'crawlee';
import { JSDOM } from 'jsdom';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import TurndownService from 'turndown';

const { OPENAI_API_KEY } = process.env;
if (!OPENAI_API_KEY) throw new Error('OPENAI_API_KEY environment variable is required.');

const URL = 'https://www.tesla.com/ownersmanual/model3/en_us/';
const QUESTION = 'Explain what autopilot is and provide step by step instructions to turning it on.';

const turndownService = new TurndownService();

// If you're getting 403 errors, you might have to use proxies.
// This is the simplest way to add them to the crawler, but it requires
// an Apify account with proxies enabled.

// const proxyConfiguration = await Actor.createProxyConfiguration({
//     countryCode: 'US',
// })

const crawler = new CheerioCrawler({
    // proxyConfiguration,
    requestHandler: async ({ body, request, enqueueLinks }) => {
        console.log('Parsing', request.url);

        const dom = new JSDOM(body, { url: request.loadedUrl });
        const reader = new Readability(dom.window.document, {
            charThreshold: 200,
        });

        const { content } = reader.parse();
        const markdown = turndownService.turndown(content);

        await Dataset.pushData({
            request,
            markdown,
        });

        await enqueueLinks({
            globs: [`${URL}/**`],
        });
    },
});

await crawler.run([URL]);

console.log('Exporting crawled results to CSV.');
await Dataset.exportToCSV('tesla-docs');

const loader = new CSVLoader(
    'storage/key_value_stores/default/tesla-docs.csv',
    'markdown',
);

console.log('Loading results to LangChain.');
const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
});

console.log('Splitting markdown pages into LangChain documents.');
const splitDocs = await textSplitter.splitDocuments(docs);

console.log('Creating a vector database.');
const embeddings = new OpenAIEmbeddings();
const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings,
);

const model = new ChatOpenAI({
    model: 'gpt-4o',
    apiKey: OPENAI_API_KEY,
    temperature: 0,
});

const questionAnsweringPrompt = ChatPromptTemplate.fromMessages([
    ['system', 'Answer the user\'s questions based on the below context:\n\n{context}'],
    ['human', '{input}'],
]);

const combineDocsChain = await createStuffDocumentsChain({
    llm: model,
    prompt: new ChatPromptTemplate(),
});

const chain = await createRetrievalChain({
    retriever: vectorStore.asRetriever(),
    combineDocsChain,
});

const res = await chain.invoke({ input: QUESTION });

console.log(res.answer);
