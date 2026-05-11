import { tavily } from "@tavily/core";
import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get('query');

  if (!query) {
    return NextResponse.json({ error: 'Query parameter is required' }, { status: 400 });
  }

  // Use process.env directly in the route handler so it stays on the server
  const apiKey = process.env.TAVILY_API_KEY;

  if (!apiKey) {
    console.error('TAVILY_API_KEY is not set in environment variables.');
    return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
  }

  try {
    console.log(`[API Proxy] Securely calling Tavily for query: "${query}"`);
    
    // Initialize the official Tavily client
    const tvly = tavily({ apiKey: apiKey });
    
    const searchResults = await tvly.search(query, {
      searchDepth: 'basic',
      maxResults: 5,
    });

    // Process searchResults to extract relevant snippets
    const snippets = searchResults.results
      .map((result: any) => `Title: ${result.title}\nURL: ${result.url}\nContent: ${result.content}`)
      .join('\n\n');

    return NextResponse.json({ success: true, results: snippets });
  } catch (error: any) {
    console.error('[API Proxy] Error calling Tavily:', error);
    return NextResponse.json({ error: error.message || 'Failed to perform web search' }, { status: 500 });
  }
}