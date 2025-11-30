import { NextResponse } from 'next/server';

// Cache for 1 hour
export const revalidate = 3600;

const HF_DATASET = "JasonYan777/novelty-ranked-preprints";
const HF_API_URL = `https://datasets-server.huggingface.co/rows?dataset=${HF_DATASET}&config=default&split=train`;

async function fetchWithRetry(url, retries = 5) {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url, { next: { revalidate: 3600 } });
      
      if (response.ok) return response;
      
      // If rate limited (429) or server error (5xx)
      if (response.status === 429 || response.status >= 500) {
        console.warn(`Fetch attempt ${i + 1} failed: ${response.status}. Retrying...`);
        // Exponential backoff with jitter: 1s, 2s, 4s, 8s, 16s + random jitter
        const backoff = 1000 * Math.pow(2, i) + Math.random() * 1000;
        await new Promise(r => setTimeout(r, backoff));
        
        // If this was the last attempt, throw the error
        if (i === retries - 1) {
          throw new Error(`Failed to fetch after ${retries} attempts: ${response.status}`);
        }
        continue;
      }
      
      throw new Error(`Failed to fetch: ${response.status}`);
    } catch (e) {
      if (i === retries - 1) throw e;
      const backoff = 1000 * Math.pow(2, i) + Math.random() * 1000;
      await new Promise(r => setTimeout(r, backoff));
    }
  }
  throw new Error(`Failed to fetch after ${retries} attempts`);
}

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const offset = searchParams.get('offset') || '0';
  const limit = searchParams.get('limit') || '100';

  try {
    const response = await fetchWithRetry(`${HF_API_URL}&offset=${offset}&length=${limit}`);
    const data = await response.json();
    
    return NextResponse.json(data);
  } catch (error) {
    console.error("API Error:", error);
    return NextResponse.json({ error: "Failed to fetch data" }, { status: 500 });
  }
}
